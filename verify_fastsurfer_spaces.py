"""
Script to verify FastSurfer file relationships and coordinate spaces.
"""

import nibabel as nib
import numpy as np

# File paths
bids_t1w_path = "C:/ds003688/ds003688/sub-05/ses-mri3t/anat/sub-05_ses-mri3t_run-1_T1w.nii.gz"
orig_001_path = "./orig/001.mgz"
orig_mgz_path = "./orig.mgz"
seg_path = "./aparc.DKTatlas+aseg.deep.mgz"

print("="*70)
print("FASTSURFER FILE RELATIONSHIP VERIFICATION")
print("="*70)

# Load all files
print("\nLoading files...")
bids_t1w = nib.load(bids_t1w_path)
orig_001 = nib.load(orig_001_path)
orig_mgz = nib.load(orig_mgz_path)
seg_mgz = nib.load(seg_path)
print("All files loaded successfully!")

# Check 1: Does orig/001.mgz match BIDS T1w?
print("\n" + "="*70)
print("CHECK 1: Does orig/001.mgz match BIDS T1w?")
print("="*70)
print(f"\nBIDS T1w shape:  {bids_t1w.shape}")
print(f"orig/001.mgz shape: {orig_001.shape}")
print(f"Shapes match: {bids_t1w.shape == orig_001.shape}")

print(f"\nBIDS T1w affine:")
print(bids_t1w.affine)
print(f"\norig/001.mgz affine:")
print(orig_001.affine)
affine_match_001 = np.allclose(bids_t1w.affine, orig_001.affine, atol=1e-3)
print(f"\nAffines match: {affine_match_001}")

if affine_match_001 and bids_t1w.shape == orig_001.shape:
    print("\n[OK] orig/001.mgz IS the original BIDS T1w file (unconformed)")
else:
    print("\n[WARNING] orig/001.mgz differs from BIDS T1w")
    if not affine_match_001:
        print("Affine difference:")
        print(bids_t1w.affine - orig_001.affine)

# Check 2: Relationship between orig/001.mgz and orig.mgz
print("\n" + "="*70)
print("CHECK 2: Relationship between orig/001.mgz (unconformed) and orig.mgz (conformed)")
print("="*70)
print(f"\norig/001.mgz (original) shape:  {orig_001.shape}")
print(f"orig.mgz (conformed) shape:     {orig_mgz.shape}")
print(f"\norig/001.mgz orientation: {nib.aff2axcodes(orig_001.affine)}")
print(f"orig.mgz orientation:     {nib.aff2axcodes(orig_mgz.affine)}")

is_conformed = (orig_mgz.shape[0] == 256 and
                orig_mgz.shape[1] == 256 and
                orig_mgz.shape[2] == 256)
print(f"\norig.mgz is conformed (256^3): {is_conformed}")

# Check 3: Relationship between orig.mgz and segmentation
print("\n" + "="*70)
print("CHECK 3: Relationship between orig.mgz and segmentation")
print("="*70)
print(f"\norig.mgz shape:      {orig_mgz.shape}")
print(f"segmentation shape:  {seg_mgz.shape}")
print(f"Shapes match: {orig_mgz.shape == seg_mgz.shape}")

print(f"\norig.mgz affine:")
print(orig_mgz.affine)
print(f"\nsegmentation affine:")
print(seg_mgz.affine)
seg_affine_match = np.allclose(orig_mgz.affine, seg_mgz.affine, atol=1e-3)
print(f"\nAffines match: {seg_affine_match}")

if seg_affine_match and orig_mgz.shape == seg_mgz.shape:
    print("\n[OK] Segmentation is in conformed space (same as orig.mgz)")
else:
    print("\n[WARNING] Segmentation space differs from orig.mgz")

# Final Summary
print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATION")
print("="*70)

if affine_match_001 and seg_affine_match and is_conformed:
    print("\n[SOLUTION FOUND]")
    print("1. orig/001.mgz = BIDS T1w (original space)")
    print("2. orig.mgz = conformed version (256^3)")
    print("3. segmentation is in conformed space")
    print("\nRECOMMENDATION:")
    print("Resample segmentation from conformed space (orig.mgz)")
    print("to original space (orig/001.mgz) using nibabel.")
    print("This will align segmentation with electrode coordinates.")
else:
    print("\n[ISSUE DETECTED]")
    print("File relationships are not as expected.")
    print("Please review the output above for details.")

print("\n" + "="*70)
