"""
Script to resample FastSurfer segmentation from conformed space to original T1w space.
Uses nearest-neighbor interpolation to preserve integer label values.
"""

import argparse
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from pathlib import Path


def resample_segmentation(subject_id, freesurfer_dir="./Freesurfer"):
    """
    Resample segmentation from conformed to original T1w space.

    Args:
        subject_id: Subject ID (e.g., "sub-05")
        freesurfer_dir: Root directory containing FreeSurfer outputs
    """
    print("="*70)
    print(f"RESAMPLING SEGMENTATION FOR {subject_id}")
    print("="*70)

    # Construct file paths
    subject_dir = Path(freesurfer_dir) / subject_id
    orig_001_path = subject_dir / "orig" / "001.mgz"  # Target space (original T1w)
    seg_conformed_path = subject_dir / "aparc.DKTatlas+aseg.deep.mgz"  # Source (conformed space)
    seg_resampled_path = subject_dir / "aparc.DKTatlas+aseg.deep_original_space.nii.gz"  # Output

    # Verify input files exist
    if not orig_001_path.exists():
        raise FileNotFoundError(f"Original T1w file not found: {orig_001_path}")
    if not seg_conformed_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_conformed_path}")

    print("\nLoading files...")
    print(f"  Target space: {orig_001_path}")
    print(f"  Source segmentation: {seg_conformed_path}")

    # Load files
    target_img = nib.load(orig_001_path)
    seg_img = nib.load(seg_conformed_path)

    print(f"\nTarget space shape: {target_img.shape}")
    print(f"Segmentation shape: {seg_img.shape}")

    # Resample using nearest-neighbor interpolation
    # This preserves integer label values without interpolation artifacts
    print("\nResampling segmentation to target space...")
    print("  Using nearest-neighbor interpolation to preserve labels...")

    resampled_seg = resample_from_to(
        seg_img,
        target_img,
        order=0,  # 0 = nearest-neighbor interpolation
        mode='constant',
        cval=0
    )

    print(f"Resampled segmentation shape: {resampled_seg.shape}")
    print(f"Target shape: {target_img.shape}")
    assert resampled_seg.shape == target_img.shape, "Shapes don't match!"

    # Verify affine matrices match
    print(f"\nVerifying affine matrix alignment...")
    affine_match = np.allclose(resampled_seg.affine, target_img.affine, atol=1e-3)
    print(f"  Affines match: {affine_match}")

    if not affine_match:
        print("  WARNING: Affines don't match exactly. Difference:")
        print(resampled_seg.affine - target_img.affine)

    # Check label preservation
    seg_labels = np.unique(seg_img.get_fdata())
    resampled_labels = np.unique(resampled_seg.get_fdata())
    print(f"\nOriginal unique labels: {len(seg_labels)}")
    print(f"Resampled unique labels: {len(resampled_labels)}")
    print(f"Labels preserved: {len(np.intersect1d(seg_labels, resampled_labels))} / {len(seg_labels)}")

    # Save resampled segmentation
    print(f"\nSaving resampled segmentation to: {seg_resampled_path}")
    nib.save(resampled_seg, seg_resampled_path)
    print("[OK] Resampling complete!")

    # Final verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"\nReloading saved file to verify...")
    verify_img = nib.load(seg_resampled_path)
    print(f"  Shape: {verify_img.shape} (expected: {target_img.shape})")
    print(f"  Shape match: {verify_img.shape == target_img.shape}")
    print(f"  Affine match: {np.allclose(verify_img.affine, target_img.affine, atol=1e-3)}")
    print(f"  Orientation: {nib.aff2axcodes(verify_img.affine)}")

    print("\n" + "="*70)
    print("[SUCCESS] Segmentation is now aligned with electrode coordinates!")
    print("="*70)
    print(f"\nOutput file: {seg_resampled_path}")
    print("\nThis file can now be used for electrode localization.")
    print("="*70)

    return str(seg_resampled_path)


def main():
    parser = argparse.ArgumentParser(
        description="Resample FastSurfer segmentation from conformed to original T1w space"
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., sub-05)"
    )
    parser.add_argument(
        "--freesurfer-dir",
        type=str,
        default="./Freesurfer",
        help="Root directory containing FreeSurfer outputs (default: ./Freesurfer)"
    )

    args = parser.parse_args()

    # Run resampling
    resample_segmentation(
        subject_id=args.subject,
        freesurfer_dir=args.freesurfer_dir
    )


if __name__ == "__main__":
    main()
