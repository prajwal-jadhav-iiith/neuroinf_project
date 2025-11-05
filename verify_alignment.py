"""
Script to verify alignment between T1w.nii.gz and FreeSurfer segmentation file.
"""

import nibabel as nib
import numpy as np

# File paths
t1w_path = "C:/ds003688/ds003688/sub-05/ses-mri3t/anat/sub-05_ses-mri3t_run-1_T1w.nii.gz"
seg_path = "./aparc.DKTatlas+aseg.deep.nii.gz"

# Load both files
print("Loading T1w file...")
t1w_img = nib.load(t1w_path)
print(f"T1w loaded: {t1w_path}")

print("\nLoading segmentation file...")
seg_img = nib.load(seg_path)
print(f"Segmentation loaded: {seg_path}")

# Compare basic properties
print("\n" + "="*60)
print("COMPARISON REPORT")
print("="*60)

print("\n1. Image Shapes:")
print(f"   T1w shape:          {t1w_img.shape}")
print(f"   Segmentation shape: {seg_img.shape}")
print(f"   Shapes match: {t1w_img.shape == seg_img.shape}")

print("\n2. Voxel Dimensions:")
t1w_voxel_sizes = t1w_img.header.get_zooms()
seg_voxel_sizes = seg_img.header.get_zooms()
print(f"   T1w voxel sizes (mm):          {t1w_voxel_sizes}")
print(f"   Segmentation voxel sizes (mm): {seg_voxel_sizes}")
print(f"   Voxel sizes match: {np.allclose(t1w_voxel_sizes, seg_voxel_sizes)}")

print("\n3. Affine Transformation Matrices:")
print(f"\n   T1w affine matrix:")
print(t1w_img.affine)
print(f"\n   Segmentation affine matrix:")
print(seg_img.affine)
print(f"\n   Affine matrices match: {np.allclose(t1w_img.affine, seg_img.affine)}")

if not np.allclose(t1w_img.affine, seg_img.affine):
    print("\n   Affine matrix difference:")
    diff = t1w_img.affine - seg_img.affine
    print(diff)
    print(f"   Maximum absolute difference: {np.max(np.abs(diff))}")

print("\n4. Data Types:")
print(f"   T1w data type:          {t1w_img.get_data_dtype()}")
print(f"   Segmentation data type: {seg_img.get_data_dtype()}")

print("\n5. Orientation:")
print(f"   T1w orientation:          {nib.aff2axcodes(t1w_img.affine)}")
print(f"   Segmentation orientation: {nib.aff2axcodes(seg_img.affine)}")

# Check if files are aligned
print("\n" + "="*60)
print("ALIGNMENT STATUS")
print("="*60)

shapes_match = t1w_img.shape == seg_img.shape
affines_match = np.allclose(t1w_img.affine, seg_img.affine, atol=1e-3)
voxels_match = np.allclose(t1w_voxel_sizes, seg_voxel_sizes, atol=1e-3)

if shapes_match and affines_match and voxels_match:
    print("\n[OK] FILES ARE ALIGNED!")
    print("  The T1w and segmentation files share the same coordinate space.")
    print("  Electrode coordinates can be safely mapped to segmentation labels.")
else:
    print("\n[ERROR] FILES ARE NOT PROPERLY ALIGNED!")
    if not shapes_match:
        print("  - Image dimensions do not match")
    if not affines_match:
        print("  - Affine transformation matrices do not match")
    if not voxels_match:
        print("  - Voxel sizes do not match")
    print("\n  WARNING: Electrode localization may produce incorrect results.")
    print("  Consider re-registering the segmentation to the T1w image.")

print("\n" + "="*60)
