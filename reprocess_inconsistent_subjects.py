"""
Reprocess subjects with inconsistent temporal resolution

This script identifies and reprocesses subjects that don't match
the target 32 Hz sampling rate (961 timepoints).
"""

import numpy as np
from pathlib import Path
from full_pipeline import run_subject_pipeline

# Configuration
DATA_DIR = r"C:\DS003688\DS003688"
ELECTRODE_RESULTS_DIR = "./electrode_results"
OUTPUT_DIR = "./results_theta"

preprocessed_dir = Path(OUTPUT_DIR) / "preprocessed_data"

# Check all subjects for temporal consistency
print("="*80)
print("CHECKING TEMPORAL CONSISTENCY")
print("="*80)

target_timepoints = 961
target_sfreq = 32.0

subjects_to_reprocess = []

npz_files = sorted(preprocessed_dir.glob("*_theta_power.npz"))

for npz_file in npz_files:
    data = np.load(npz_file)
    subject_id = str(data['subject_id'])
    times = data['times']
    n_timepoints = len(times)

    if n_timepoints != target_timepoints:
        print(f"\n❌ {subject_id}: {n_timepoints} timepoints (NEEDS REPROCESSING)")
        subjects_to_reprocess.append(subject_id)
    else:
        print(f"✓ {subject_id}: {n_timepoints} timepoints (OK)")

if not subjects_to_reprocess:
    print("\n" + "="*80)
    print("✓ All subjects have consistent temporal resolution!")
    print("="*80)
    exit(0)

print(f"\n{'='*80}")
print(f"REPROCESSING {len(subjects_to_reprocess)} SUBJECTS")
print(f"{'='*80}")
print(f"\nSubjects to reprocess: {', '.join(subjects_to_reprocess)}")

user_input = input("\nProceed with reprocessing? (yes/no): ")

if user_input.lower() != 'yes':
    print("Aborted.")
    exit(0)

# Reprocess each subject
successful = []
failed = []

for idx, subject_id in enumerate(subjects_to_reprocess, 1):
    print(f"\n\n{'='*80}")
    print(f"REPROCESSING {idx}/{len(subjects_to_reprocess)}: {subject_id}")
    print(f"{'='*80}")

    success = run_subject_pipeline(
        subject_id=subject_id,
        data_dir=DATA_DIR,
        electrode_results_dir=ELECTRODE_RESULTS_DIR,
        output_base_dir=OUTPUT_DIR,
        save_preprocessed=True
    )

    if success:
        successful.append(subject_id)
    else:
        failed.append(subject_id)

# Summary
print("\n" + "="*80)
print("REPROCESSING COMPLETE")
print("="*80)
print(f"\nSuccessful: {len(successful)}/{len(subjects_to_reprocess)}")
print(f"Failed: {len(failed)}/{len(subjects_to_reprocess)}")

if successful:
    print("\n✓ Successfully reprocessed:")
    for subj in successful:
        print(f"  - {subj}")

if failed:
    print("\n❌ Failed:")
    for subj in failed:
        print(f"  - {subj}")

# Verify all subjects now have consistent temporal resolution
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

all_consistent = True
for npz_file in sorted(preprocessed_dir.glob("*_theta_power.npz")):
    data = np.load(npz_file)
    subject_id = str(data['subject_id'])
    n_timepoints = len(data['times'])

    if n_timepoints != target_timepoints:
        print(f"❌ {subject_id}: {n_timepoints} timepoints (STILL INCONSISTENT!)")
        all_consistent = False
    else:
        print(f"✓ {subject_id}: {n_timepoints} timepoints")

if all_consistent:
    print("\n" + "="*80)
    print("✓ SUCCESS! All subjects now have consistent temporal resolution")
    print(f"  Common time grid: 0-30s at {target_sfreq} Hz ({target_timepoints} timepoints)")
    print("  Group analysis (roi_group_analysis.py) is now safe to run!")
    print("="*80)
else:
    print("\n" + "="*80)
    print("⚠ WARNING: Some subjects still have inconsistent timepoints")
    print("="*80)
