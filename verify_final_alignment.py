"""
Final verification: Check alignment between BIDS T1w and resampled segmentation.
"""

import nibabel as nib
import numpy as np

# File paths
t1w_path = "C:/ds003688/ds003688/sub-05/ses-mri3t/anat/sub-05_ses-mri3t_run-1_T1w.nii.gz"
seg_path = "./aparc.DKTatlas+aseg.deep_original_space.nii.gz"

print("="*70)
print("FINAL ALIGNMENT VERIFICATION")
print("="*70)

# Load both files
print("\nLoading files...")
t1w_img = nib.load(t1w_path)
seg_img = nib.load(seg_path)

print(f"  T1w: {t1w_path}")
print(f"  Segmentation: {seg_path}")

# Compare properties
print("\n" + "="*70)
print("ALIGNMENT CHECK")
print("="*70)

print("\n1. Image Shapes:")
print(f"   T1w:          {t1w_img.shape}")
print(f"   Segmentation: {seg_img.shape}")
shapes_match = t1w_img.shape == seg_img.shape
print(f"   Match: {shapes_match}")

print("\n2. Voxel Dimensions:")
t1w_voxels = t1w_img.header.get_zooms()
seg_voxels = seg_img.header.get_zooms()
print(f"   T1w:          {t1w_voxels}")
print(f"   Segmentation: {seg_voxels}")
voxels_match = np.allclose(t1w_voxels, seg_voxels, atol=1e-3)
print(f"   Match: {voxels_match}")

print("\n3. Affine Transformation Matrices:")
print(f"\n   T1w affine:")
print(t1w_img.affine)
print(f"\n   Segmentation affine:")
print(seg_img.affine)
affines_match = np.allclose(t1w_img.affine, seg_img.affine, atol=1e-3)
print(f"\n   Match: {affines_match}")

if not affines_match:
    diff = t1w_img.affine - seg_img.affine
    print(f"\n   Difference matrix:")
    print(diff)
    print(f"   Max absolute difference: {np.max(np.abs(diff))}")

print("\n4. Orientation:")
print(f"   T1w:          {nib.aff2axcodes(t1w_img.affine)}")
print(f"   Segmentation: {nib.aff2axcodes(seg_img.affine)}")

# Final verdict
print("\n" + "="*70)
print("RESULT")
print("="*70)

if shapes_match and affines_match and voxels_match:
    print("\n[SUCCESS] FILES ARE PERFECTLY ALIGNED!")
    print("\nThe resampled segmentation is now in the same coordinate space")
    print("as the T1w image and electrode coordinates.")
    print("\nYou can now proceed with electrode localization using:")
    print(f"  - Electrodes: BIDS electrodes.tsv (ACPC coordinates)")
    print(f"  - Segmentation: {seg_path}")
    print(f"  - Reference: {t1w_path}")
else:
    print("\n[ERROR] Alignment issue detected!")
    if not shapes_match:
        print("  - Shapes don't match")
    if not affines_match:
        print("  - Affine matrices don't match")
    if not voxels_match:
        print("  - Voxel dimensions don't match")

print("\n" + "="*70)
