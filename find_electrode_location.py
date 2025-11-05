"""
Electrode Localization Script

This script maps electrode coordinates from BIDS iEEG data to brain regions
using FastSurfer/FreeSurfer segmentation files (DKT atlas).

For electrodes landing on unlabeled voxels (e.g., on cortical surface),
the script finds the nearest labeled region within a specified search radius.

Usage:
    python find_electrode_location.py --subject sub-05 --max-distance 5

Output:
    CSV file with electrode names and their corresponding brain regions
"""

import argparse
import os
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt


# FreeSurfer DKT Atlas Color Lookup Table
# This maps segmentation label IDs to anatomical region names
DKT_LOOKUP_TABLE = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    77: "WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    # DKT Cortical Labels - Left Hemisphere
    1002: "ctx-lh-caudalanteriorcingulate",
    1003: "ctx-lh-caudalmiddlefrontal",
    1005: "ctx-lh-cuneus",
    1006: "ctx-lh-entorhinal",
    1007: "ctx-lh-fusiform",
    1008: "ctx-lh-inferiorparietal",
    1009: "ctx-lh-inferiortemporal",
    1010: "ctx-lh-isthmuscingulate",
    1011: "ctx-lh-lateraloccipital",
    1012: "ctx-lh-lateralorbitofrontal",
    1013: "ctx-lh-lingual",
    1014: "ctx-lh-medialorbitofrontal",
    1015: "ctx-lh-middletemporal",
    1016: "ctx-lh-parahippocampal",
    1017: "ctx-lh-paracentral",
    1018: "ctx-lh-parsopercularis",
    1019: "ctx-lh-parsorbitalis",
    1020: "ctx-lh-parstriangularis",
    1021: "ctx-lh-pericalcarine",
    1022: "ctx-lh-postcentral",
    1023: "ctx-lh-posteriorcingulate",
    1024: "ctx-lh-precentral",
    1025: "ctx-lh-precuneus",
    1026: "ctx-lh-rostralanteriorcingulate",
    1027: "ctx-lh-rostralmiddlefrontal",
    1028: "ctx-lh-superiorfrontal",
    1029: "ctx-lh-superiorparietal",
    1030: "ctx-lh-superiortemporal",
    1031: "ctx-lh-supramarginal",
    1034: "ctx-lh-transversetemporal",
    1035: "ctx-lh-insula",
    # DKT Cortical Labels - Right Hemisphere
    2002: "ctx-rh-caudalanteriorcingulate",
    2003: "ctx-rh-caudalmiddlefrontal",
    2005: "ctx-rh-cuneus",
    2006: "ctx-rh-entorhinal",
    2007: "ctx-rh-fusiform",
    2008: "ctx-rh-inferiorparietal",
    2009: "ctx-rh-inferiortemporal",
    2010: "ctx-rh-isthmuscingulate",
    2011: "ctx-rh-lateraloccipital",
    2012: "ctx-rh-lateralorbitofrontal",
    2013: "ctx-rh-lingual",
    2014: "ctx-rh-medialorbitofrontal",
    2015: "ctx-rh-middletemporal",
    2016: "ctx-rh-parahippocampal",
    2017: "ctx-rh-paracentral",
    2018: "ctx-rh-parsopercularis",
    2019: "ctx-rh-parsorbitalis",
    2020: "ctx-rh-parstriangularis",
    2021: "ctx-rh-pericalcarine",
    2022: "ctx-rh-postcentral",
    2023: "ctx-rh-posteriorcingulate",
    2024: "ctx-rh-precentral",
    2025: "ctx-rh-precuneus",
    2026: "ctx-rh-rostralanteriorcingulate",
    2027: "ctx-rh-rostralmiddlefrontal",
    2028: "ctx-rh-superiorfrontal",
    2029: "ctx-rh-superiorparietal",
    2030: "ctx-rh-superiortemporal",
    2031: "ctx-rh-supramarginal",
    2034: "ctx-rh-transversetemporal",
    2035: "ctx-rh-insula",
}


def load_electrode_coordinates(electrodes_tsv_path):
    """
    Load electrode coordinates from BIDS electrodes.tsv file.

    Args:
        electrodes_tsv_path: Path to the electrodes.tsv file

    Returns:
        pandas DataFrame with electrode names and coordinates
    """
    df = pd.read_csv(electrodes_tsv_path, sep='\t')

    # Verify required columns exist
    required_cols = ['name', 'x', 'y', 'z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in electrodes.tsv: {missing_cols}")

    return df


def acpc_to_voxel(acpc_coords, affine):
    """
    Transform ACPC (mm) coordinates to voxel indices.

    Args:
        acpc_coords: Nx3 array of ACPC coordinates in mm
        affine: 4x4 affine transformation matrix from the NIfTI file

    Returns:
        Nx3 array of voxel indices (integers)
    """
    # Add homogeneous coordinate (make Nx4)
    acpc_homogeneous = np.column_stack([acpc_coords, np.ones(len(acpc_coords))])

    # Apply inverse affine to get voxel coordinates
    inv_affine = np.linalg.inv(affine)
    voxel_coords = acpc_homogeneous @ inv_affine.T

    # Remove homogeneous coordinate and round to nearest integer
    voxel_indices = np.round(voxel_coords[:, :3]).astype(int)

    return voxel_indices


def find_nearest_labeled_voxel(voxel_idx, seg_data, max_distance_voxels=10):
    """
    Find the nearest labeled (non-zero) voxel to the given voxel index.

    Args:
        voxel_idx: (x, y, z) voxel indices
        seg_data: 3D segmentation array
        max_distance_voxels: Maximum search radius in voxels

    Returns:
        (label_id, distance_voxels) or (None, None) if no label found
    """
    x, y, z = voxel_idx
    img_shape = seg_data.shape

    # Check if voxel is within bounds
    if not (0 <= x < img_shape[0] and
            0 <= y < img_shape[1] and
            0 <= z < img_shape[2]):
        return None, None

    # If already labeled (non-zero), return it
    label = seg_data[x, y, z]
    if label != 0:
        return int(label), 0.0

    # Search in expanding sphere
    for radius in range(1, max_distance_voxels + 1):
        # Create a bounding box for the search region
        x_min = max(0, x - radius)
        x_max = min(img_shape[0], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(img_shape[1], y + radius + 1)
        z_min = max(0, z - radius)
        z_max = min(img_shape[2], z + radius + 1)

        # Get all voxels in this region
        xx, yy, zz = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max),
            np.arange(z_min, z_max),
            indexing='ij'
        )

        # Calculate distances
        distances = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)

        # Find voxels at approximately this radius
        mask = (distances <= radius) & (distances > radius - 1)

        if not np.any(mask):
            continue

        # Get labels at these voxels
        candidate_labels = seg_data[xx[mask], yy[mask], zz[mask]]
        candidate_distances = distances[mask]

        # Find non-zero labels
        nonzero_mask = candidate_labels != 0
        if np.any(nonzero_mask):
            # Return the closest non-zero label
            closest_idx = np.argmin(candidate_distances[nonzero_mask])
            closest_label = candidate_labels[nonzero_mask][closest_idx]
            closest_distance = candidate_distances[nonzero_mask][closest_idx]
            return int(closest_label), float(closest_distance)

    # No label found within max_distance
    return None, None


def get_region_label(voxel_idx, seg_data, img_shape, max_distance_voxels=10):
    """
    Get the segmentation label at a given voxel index.
    If unlabeled (0), find the nearest labeled voxel.

    Args:
        voxel_idx: (x, y, z) voxel indices
        seg_data: 3D segmentation array
        img_shape: Shape of the segmentation volume
        max_distance_voxels: Maximum search distance in voxels

    Returns:
        (label_id, distance_voxels) tuple
    """
    return find_nearest_labeled_voxel(voxel_idx, seg_data, max_distance_voxels)


def label_to_region_name(label_id):
    """
    Convert segmentation label ID to anatomical region name.

    Args:
        label_id: Integer label ID from segmentation

    Returns:
        String name of the anatomical region
    """
    if label_id is None:
        return "Out-of-bounds"

    return DKT_LOOKUP_TABLE.get(label_id, f"Unknown-Label-{label_id}")


def localize_electrodes(subject_id, bids_root, seg_file_path, output_dir=".", max_distance_mm=5.0):
    """
    Main function to localize electrodes to brain regions.

    Args:
        subject_id: Subject ID (e.g., "sub-05")
        bids_root: Root directory of BIDS dataset
        seg_file_path: Path to the aligned segmentation file
        output_dir: Directory to save output CSV
        max_distance_mm: Maximum search distance in mm for nearest neighbor
    """
    print("="*70)
    print(f"ELECTRODE LOCALIZATION FOR {subject_id}")
    print("="*70)

    # Construct paths
    bids_root = Path(bids_root)
    subject_dir = bids_root / subject_id

    # Find electrodes file (look for clinical acquisition)
    ieeg_dir = subject_dir / "ses-iemu" / "ieeg"
    electrodes_files = list(ieeg_dir.glob("*electrodes.tsv"))

    if not electrodes_files:
        raise FileNotFoundError(f"No electrodes.tsv file found in {ieeg_dir}")

    electrodes_file = electrodes_files[0]  # Use first match
    print(f"\nElectrodes file: {electrodes_file}")
    print(f"Segmentation file: {seg_file_path}")
    print(f"Max search distance: {max_distance_mm} mm")

    # Load data
    print("\nLoading electrode coordinates...")
    electrodes_df = load_electrode_coordinates(electrodes_file)
    print(f"  Found {len(electrodes_df)} electrodes")

    print("\nLoading segmentation volume...")
    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata().astype(int)
    affine = seg_img.affine

    # Calculate voxel size to convert mm to voxels
    voxel_sizes = seg_img.header.get_zooms()[:3]
    avg_voxel_size = np.mean(voxel_sizes)
    max_distance_voxels = int(np.ceil(max_distance_mm / avg_voxel_size))

    print(f"  Segmentation shape: {seg_data.shape}")
    print(f"  Voxel size: {voxel_sizes} mm")
    print(f"  Max search distance: {max_distance_voxels} voxels")
    print(f"  Coordinate system: ACPC (mm)")

    # Extract electrode coordinates
    acpc_coords = electrodes_df[['x', 'y', 'z']].values

    # Transform to voxel coordinates
    print("\nTransforming coordinates to voxel space...")
    voxel_coords = acpc_to_voxel(acpc_coords, affine)

    # Localize each electrode
    print("\nLocalizing electrodes to brain regions...")
    print("(Using nearest-neighbor search for unlabeled voxels)")
    regions = []
    label_ids = []
    distances_mm = []
    method_used = []

    for i, (electrode_name, voxel_idx) in enumerate(zip(electrodes_df['name'], voxel_coords)):
        label_id, distance_voxels = get_region_label(
            voxel_idx, seg_data, seg_data.shape, max_distance_voxels
        )
        region_name = label_to_region_name(label_id)

        regions.append(region_name)
        label_ids.append(label_id if label_id is not None else -1)

        if distance_voxels is not None:
            distance_mm = distance_voxels * avg_voxel_size
            distances_mm.append(distance_mm)
            if distance_voxels == 0:
                method_used.append("direct")
            else:
                method_used.append("nearest-neighbor")
        else:
            distances_mm.append(-1)
            method_used.append("failed")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(electrodes_df)} electrodes")

    print(f"  Completed: {len(electrodes_df)}/{len(electrodes_df)} electrodes")

    # Create output DataFrame
    output_df = pd.DataFrame({
        'electrode_name': electrodes_df['name'],
        'region': regions,
        'label_id': label_ids,
        'distance_mm': distances_mm,
        'method': method_used,
        'x_mm': electrodes_df['x'],
        'y_mm': electrodes_df['y'],
        'z_mm': electrodes_df['z'],
        'x_voxel': voxel_coords[:, 0],
        'y_voxel': voxel_coords[:, 1],
        'z_voxel': voxel_coords[:, 2],
    })

    # Save to CSV
    output_file = Path(output_dir) / f"{subject_id}_electrode_locations.csv"
    output_df[['electrode_name', 'region']].to_csv(output_file, index=False)

    # Also save detailed version
    detailed_file = Path(output_dir) / f"{subject_id}_electrode_locations_detailed.csv"
    output_df.to_csv(detailed_file, index=False)

    print(f"\n{'-'*70}")
    print("RESULTS")
    print(f"{'-'*70}")
    print(f"\nOutput files created:")
    print(f"  1. {output_file}")
    print(f"     (electrode_name, region)")
    print(f"  2. {detailed_file}")
    print(f"     (includes coordinates, distances, and method used)")

    # Print summary statistics
    print(f"\nSummary:")
    unique_regions = output_df['region'].value_counts()
    print(f"  Total electrodes: {len(output_df)}")
    print(f"  Unique regions: {len(unique_regions)}")

    # Method statistics
    direct_count = (output_df['method'] == 'direct').sum()
    nn_count = (output_df['method'] == 'nearest-neighbor').sum()
    failed_count = (output_df['method'] == 'failed').sum()

    print(f"\nLocalization method:")
    print(f"  Direct hit: {direct_count} electrodes")
    print(f"  Nearest-neighbor: {nn_count} electrodes")
    print(f"  Failed: {failed_count} electrodes")

    if nn_count > 0:
        nn_distances = output_df[output_df['method'] == 'nearest-neighbor']['distance_mm']
        print(f"\nNearest-neighbor distances:")
        print(f"  Mean: {nn_distances.mean():.2f} mm")
        print(f"  Median: {nn_distances.median():.2f} mm")
        print(f"  Max: {nn_distances.max():.2f} mm")

    print(f"\nTop 10 regions by electrode count:")
    for region, count in unique_regions.head(10).items():
        print(f"    {region}: {count}")

    print("\n" + "="*70)
    print("LOCALIZATION COMPLETE!")
    print("="*70)

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Localize iEEG electrodes to brain regions using FastSurfer segmentation"
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., sub-05)"
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default="C:/ds003688/ds003688",
        help="Root directory of BIDS dataset"
    )
    parser.add_argument(
        "--seg-file",
        type=str,
        default="./aparc.DKTatlas+aseg.deep_original_space.nii.gz",
        help="Path to aligned segmentation file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output CSV files"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=5.0,
        help="Maximum search distance in mm for nearest-neighbor (default: 5.0)"
    )

    args = parser.parse_args()

    # Run localization
    localize_electrodes(
        subject_id=args.subject,
        bids_root=args.bids_root,
        seg_file_path=args.seg_file,
        output_dir=args.output_dir,
        max_distance_mm=args.max_distance
    )


if __name__ == "__main__":
    main()
