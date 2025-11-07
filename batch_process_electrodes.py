#!/usr/bin/env python
"""
Batch processing script for electrode localization across multiple subjects.

This script processes all subjects in the FreeSurfer directory through three stages:
1. Resample segmentation from conformed to original T1w space
2. Localize electrodes to brain regions using the resampled segmentation
3. Filter electrodes to identify those in perisylvian regions

Usage:
    python batch_process_electrodes.py --freesurfer-dir ./Freesurfer --bids-root /path/to/bids

    # Process specific subjects only
    python batch_process_electrodes.py --subjects sub-01,sub-05,sub-10

    # Skip resampling if already done
    python batch_process_electrodes.py --skip-resample

    # Skip perisylvian filtering
    python batch_process_electrodes.py --skip-filter
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import subprocess

# Import the processing functions from the individual scripts
try:
    from resample_segmentation import resample_segmentation
    from find_electrode_location import localize_electrodes
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Please ensure resample_segmentation.py and find_electrode_location.py are in the same directory")
    sys.exit(1)


class BatchProcessor:
    """Handles batch processing of electrode localization for multiple subjects."""

    def __init__(self, freesurfer_dir: str, bids_root: str, output_dir: str,
                 max_distance: float, skip_resample: bool, skip_filter: bool):
        self.freesurfer_dir = Path(freesurfer_dir)
        self.bids_root = bids_root
        self.output_dir = Path(output_dir)
        self.max_distance = max_distance
        self.skip_resample = skip_resample
        self.skip_filter = skip_filter

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.success_count = 0
        self.fail_count = 0
        self.failed_subjects = []

    def find_subjects(self, subject_list: List[str] = None) -> List[str]:
        """
        Find all subjects to process.

        Args:
            subject_list: Optional list of specific subjects to process

        Returns:
            List of subject IDs
        """
        if subject_list:
            return subject_list

        # Find all subject directories in FreeSurfer directory
        subjects = []
        if not self.freesurfer_dir.exists():
            raise FileNotFoundError(f"FreeSurfer directory not found: {self.freesurfer_dir}")

        for item in self.freesurfer_dir.iterdir():
            if item.is_dir() and item.name.startswith("sub-"):
                subjects.append(item.name)

        if not subjects:
            raise ValueError(f"No subjects found in {self.freesurfer_dir}")

        return sorted(subjects)

    def process_subject(self, subject: str) -> Tuple[bool, str]:
        """
        Process a single subject through all stages.

        Args:
            subject: Subject ID (e.g., "sub-05")

        Returns:
            (success, error_message) tuple
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING {subject}")
        print(f"{'='*70}")

        try:
            # Stage 1: Resample segmentation
            if not self.skip_resample:
                print("\nStage 1/3: Resampling segmentation...")
                resample_segmentation(
                    subject_id=subject,
                    freesurfer_dir=str(self.freesurfer_dir)
                )
                print("[OK] Resampling completed")
            else:
                print("\nStage 1/3: Skipping resampling (using existing files)")

            # Stage 2: Localize electrodes
            print("\nStage 2/3: Localizing electrodes...")
            seg_file = self.freesurfer_dir / subject / "aparc.DKTatlas+aseg.deep_original_space.nii.gz"

            if not seg_file.exists():
                raise FileNotFoundError(
                    f"Resampled segmentation not found: {seg_file}\n"
                    f"Please run without --skip-resample first"
                )

            localize_electrodes(
                subject_id=subject,
                bids_root=self.bids_root,
                seg_file_path=str(seg_file),
                output_dir=str(self.output_dir),
                max_distance_mm=self.max_distance
            )
            print("[OK] Electrode localization completed")

            # Stage 3: Filter perisylvian electrodes
            if not self.skip_filter:
                print("\nStage 3/3: Filtering perisylvian electrodes...")
                input_csv = self.output_dir / f"{subject}_electrode_locations.csv"
                output_csv = self.output_dir / f"{subject}_perisylvian_electrodes.csv"

                # Call filter script as subprocess to avoid import issues
                result = subprocess.run(
                    [
                        sys.executable,
                        "filter_perisylvian_electrodes.py",
                        "--input", str(input_csv),
                        "--output", str(output_csv)
                    ],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Filtering failed:\n{result.stderr}")

                print("[OK] Perisylvian filtering completed")
            else:
                print("\nStage 3/3: Skipping perisylvian filtering")

            print(f"\n[SUCCESS] {subject} processing completed!")
            return True, ""

        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            print(f"\n[ERROR] {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"\n[ERROR] {error_msg}")
            return False, error_msg

    def run(self, subjects: List[str]):
        """
        Run batch processing on all subjects.

        Args:
            subjects: List of subject IDs to process
        """
        total = len(subjects)

        print("="*70)
        print("BATCH ELECTRODE LOCALIZATION")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  FreeSurfer directory: {self.freesurfer_dir}")
        print(f"  BIDS root: {self.bids_root}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Max search distance: {self.max_distance} mm")
        print(f"  Skip resample: {self.skip_resample}")
        print(f"  Skip filter: {self.skip_filter}")
        print(f"\nSubjects to process ({total} total):")
        for subject in subjects:
            print(f"  - {subject}")
        print("\n" + "="*70)

        # Process each subject
        for i, subject in enumerate(subjects, 1):
            print(f"\n\nProcessing subject {i}/{total}: {subject}")
            success, error_msg = self.process_subject(subject)

            if success:
                self.success_count += 1
            else:
                self.fail_count += 1
                self.failed_subjects.append((subject, error_msg))

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print final processing summary."""
        print("\n\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Total subjects: {self.success_count + self.fail_count}")
        print(f"  Successfully processed: {self.success_count}")
        print(f"  Failed: {self.fail_count}")

        if self.failed_subjects:
            print(f"\nFailed subjects:")
            for subject, error in self.failed_subjects:
                print(f"  - {subject}: {error}")

        print(f"\nOutput files location: {self.output_dir}")

        if self.fail_count == 0:
            print("\n[SUCCESS] All subjects processed successfully!")
        else:
            print("\n[WARNING] Some subjects failed to process")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process electrode localization for multiple subjects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects in FreeSurfer directory
  python batch_process_electrodes.py

  # Process specific subjects
  python batch_process_electrodes.py --subjects sub-01,sub-05,sub-10

  # Custom directories
  python batch_process_electrodes.py --freesurfer-dir ./data/FreeSurfer --output-dir ./results

  # Skip resampling (if already done)
  python batch_process_electrodes.py --skip-resample

  # Skip perisylvian filtering
  python batch_process_electrodes.py --skip-filter
        """
    )

    parser.add_argument(
        "--freesurfer-dir",
        type=str,
        default="./Freesurfer",
        help="Root directory containing FreeSurfer outputs (default: ./Freesurfer)"
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default="C:/ds003688/ds003688",
        help="Root directory of BIDS dataset (default: C:/ds003688/ds003688)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./electrode_results",
        help="Directory to save output CSV files (default: ./electrode_results)"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=5.0,
        help="Maximum search distance in mm for nearest-neighbor (default: 5.0)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects to process (e.g., sub-01,sub-05). If not specified, processes all subjects in FreeSurfer directory"
    )
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Skip resampling step (use existing resampled segmentation files)"
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip perisylvian filtering step"
    )

    args = parser.parse_args()

    # Parse subject list
    subject_list = None
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(",")]

    # Create processor
    processor = BatchProcessor(
        freesurfer_dir=args.freesurfer_dir,
        bids_root=args.bids_root,
        output_dir=args.output_dir,
        max_distance=args.max_distance,
        skip_resample=args.skip_resample,
        skip_filter=args.skip_filter
    )

    # Find subjects
    try:
        subjects = processor.find_subjects(subject_list)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Run batch processing
    processor.run(subjects)

    # Exit with appropriate code
    sys.exit(0 if processor.fail_count == 0 else 1)


if __name__ == "__main__":
    main()
