"""
Test script for the modified full_pipeline.py

This script tests the modified pipeline on sub-05 to verify:
1. CSV-based electrode loading works correctly
2. Preprocessed data is saved properly
3. ROI mapping is preserved
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import the modified pipeline
from full_pipeline import run_subject_pipeline

def verify_preprocessed_data(subject_id, output_dir):
    """
    Verify that preprocessed data was saved correctly

    Parameters:
    -----------
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory

    Returns:
    --------
    success : bool
        True if all files exist and contain expected data
    """
    print("\n" + "="*80)
    print("VERIFYING PREPROCESSED DATA")
    print("="*80)

    preprocessed_dir = Path(output_dir) / 'preprocessed_data'

    # Check for theta power file
    theta_file = preprocessed_dir / f'{subject_id}_theta_power.npz'
    roi_file = preprocessed_dir / f'{subject_id}_roi_mapping.csv'

    if not theta_file.exists():
        print(f"\n❌ ERROR: Theta power file not found: {theta_file}")
        return False

    if not roi_file.exists():
        print(f"\n❌ ERROR: ROI mapping file not found: {roi_file}")
        return False

    # Load and verify theta power data
    print(f"\n✓ Loading theta power data from: {theta_file}")
    data = np.load(theta_file)

    print("\nTheta power file contents:")
    print(f"  Arrays: {list(data.keys())}")

    speech_power = data['speech_theta_power']
    music_power = data['music_theta_power']
    channel_names = data['channel_names']
    times = data['times']
    freqs = data['freqs']
    subj_id = str(data['subject_id'])

    print(f"\n  speech_theta_power shape: {speech_power.shape}")
    print(f"  music_theta_power shape: {music_power.shape}")
    print(f"  Number of channels: {len(channel_names)}")
    print(f"  Time points: {len(times)}")
    print(f"  Time range: {times[0]:.2f}s to {times[-1]:.2f}s")
    print(f"  Frequencies: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
    print(f"  Subject ID: {subj_id}")

    # Verify shapes match
    n_channels, n_times = speech_power.shape

    if music_power.shape != (n_channels, n_times):
        print(f"\n❌ ERROR: Shape mismatch between speech and music data")
        return False

    if len(channel_names) != n_channels:
        print(f"\n❌ ERROR: Number of channel names doesn't match data shape")
        return False

    # Load and verify ROI mapping
    print(f"\n✓ Loading ROI mapping from: {roi_file}")
    roi_df = pd.read_csv(roi_file)

    print(f"\n  Total electrodes with ROI labels: {len(roi_df)}")
    print(f"  Columns: {list(roi_df.columns)}")
    print(f"  Unique ROIs: {roi_df['region'].nunique()}")

    # Verify all channel names are in ROI mapping
    channels_in_roi = set(roi_df['electrode_name'].tolist())
    channels_in_data = set(channel_names)

    if channels_in_roi != channels_in_data:
        missing_in_roi = channels_in_data - channels_in_roi
        missing_in_data = channels_in_roi - channels_in_data

        if missing_in_roi:
            print(f"\n⚠ WARNING: Channels in data but not in ROI mapping: {missing_in_roi}")
        if missing_in_data:
            print(f"\n⚠ WARNING: Channels in ROI mapping but not in data: {missing_in_data}")
    else:
        print(f"\n✓ All {len(channels_in_data)} channels have ROI labels")

    # Display ROI distribution
    print(f"\nROI distribution:")
    roi_counts = roi_df['region'].value_counts()
    for roi, count in roi_counts.items():
        print(f"  {roi}: {count} electrodes")

    # Show sample channels
    print(f"\nSample channels:")
    print(roi_df.head(10).to_string(index=False))

    print("\n" + "="*80)
    print("✓ VERIFICATION SUCCESSFUL")
    print("="*80)

    return True


if __name__ == "__main__":
    # Configuration
    SUBJECT_ID = "sub-05"
    DATA_DIR = r"C:\DS003688\DS003688"
    ELECTRODE_RESULTS_DIR = "./electrode_results"
    OUTPUT_DIR = "./test_pipeline_output"

    print("="*80)
    print("TESTING MODIFIED PIPELINE")
    print("="*80)
    print(f"\nSubject: {SUBJECT_ID}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Electrode results directory: {ELECTRODE_RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Run the pipeline
    print("\n" + "="*80)
    print("RUNNING PIPELINE")
    print("="*80)

    success = run_subject_pipeline(
        subject_id=SUBJECT_ID,
        data_dir=DATA_DIR,
        electrode_results_dir=ELECTRODE_RESULTS_DIR,
        output_base_dir=OUTPUT_DIR,
        save_preprocessed=True
    )

    if not success:
        print("\n❌ Pipeline failed!")
        exit(1)

    # Verify the output
    verification_success = verify_preprocessed_data(SUBJECT_ID, OUTPUT_DIR)

    if verification_success:
        print("\n✓ All checks passed!")
        print(f"\nPreprocessed data is ready for ROI group analysis at:")
        print(f"  {OUTPUT_DIR}/preprocessed_data/")
    else:
        print("\n❌ Verification failed!")
        exit(1)
