"""
Quick script to check temporal consistency across subjects
"""
import numpy as np
from pathlib import Path

preprocessed_dir = Path("./results_theta/preprocessed_data")
npz_files = sorted(preprocessed_dir.glob("*_theta_power.npz"))

print("="*80)
print("TEMPORAL CONSISTENCY CHECK")
print("="*80)

for npz_file in npz_files:
    data = np.load(npz_file)
    subject_id = str(data['subject_id'])
    times = data['times']
    speech_shape = data['speech_theta_power'].shape

    print(f"\n{subject_id}:")
    print(f"  Shape: {speech_shape}")
    print(f"  Time points: {len(times)}")
    print(f"  Time range: {times[0]:.3f}s to {times[-1]:.3f}s")
    print(f"  Sampling rate: {1/(times[1]-times[0]):.2f} Hz")
