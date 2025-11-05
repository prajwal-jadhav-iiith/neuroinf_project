"""
Investigate why many electrodes are labeled as "Unknown"
"""

import nibabel as nib
import numpy as np
import pandas as pd

print("="*70)
print("INVESTIGATION: Unknown Electrode Labels")
print("="*70)

# Load segmentation file
seg_path = "./aparc.DKTatlas+aseg.deep_original_space.nii.gz"
print(f"\nLoading segmentation: {seg_path}")
seg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata().astype(int)

# Get all unique labels in the segmentation
unique_labels = np.unique(seg_data)
print(f"\nTotal unique labels in segmentation: {len(unique_labels)}")
print(f"\nFirst 50 unique labels found:")
print(unique_labels[:50])

# Count voxels per label
print("\n" + "="*70)
print("Top 20 labels by voxel count:")
print("="*70)
labels, counts = np.unique(seg_data, return_counts=True)
sorted_indices = np.argsort(counts)[::-1][:20]
for idx in sorted_indices:
    label = labels[idx]
    count = counts[idx]
    percentage = (count / seg_data.size) * 100
    print(f"Label {label:5d}: {count:10d} voxels ({percentage:5.2f}%)")

# Load the detailed electrode results to see what labels electrodes are getting
print("\n" + "="*70)
print("Checking electrode label assignments:")
print("="*70)
detailed_csv = "./sub-05_electrode_locations_detailed.csv"
df = pd.read_csv(detailed_csv)

print(f"\nElectrode label statistics:")
print(f"Total electrodes: {len(df)}")
print(f"Electrodes with label 0 (Unknown): {(df['label_id'] == 0).sum()}")
print(f"Electrodes with label != 0: {(df['label_id'] != 0).sum()}")

print(f"\nUnique label IDs assigned to electrodes:")
electrode_labels = df['label_id'].unique()
print(sorted(electrode_labels))

# Check a few "Unknown" electrodes in detail
print("\n" + "="*70)
print("Sampling 5 'Unknown' electrodes - checking neighborhood:")
print("="*70)
unknown_electrodes = df[df['label_id'] == 0].head(5)

for idx, row in unknown_electrodes.iterrows():
    x, y, z = int(row['x_voxel']), int(row['y_voxel']), int(row['z_voxel'])
    print(f"\nElectrode {row['electrode_name']} at voxel ({x}, {y}, {z}):")
    print(f"  Coordinates (mm): ({row['x_mm']:.2f}, {row['y_mm']:.2f}, {row['z_mm']:.2f})")

    # Check if coordinates are within bounds
    if (0 <= x < seg_data.shape[0] and
        0 <= y < seg_data.shape[1] and
        0 <= z < seg_data.shape[2]):

        # Get label at electrode location
        label_at_electrode = seg_data[x, y, z]
        print(f"  Label at electrode: {label_at_electrode}")

        # Check 3x3x3 neighborhood
        x_min, x_max = max(0, x-1), min(seg_data.shape[0], x+2)
        y_min, y_max = max(0, y-1), min(seg_data.shape[1], y+2)
        z_min, z_max = max(0, z-1), min(seg_data.shape[2], z+2)

        neighborhood = seg_data[x_min:x_max, y_min:y_max, z_min:z_max]
        unique_in_neighborhood = np.unique(neighborhood)
        print(f"  Unique labels in 3x3x3 neighborhood: {unique_in_neighborhood}")
    else:
        print(f"  ERROR: Voxel coordinates out of bounds!")
        print(f"  Segmentation shape: {seg_data.shape}")

print("\n" + "="*70)
