"""
Filter Perisylvian Electrodes Script

This script reads electrode localization data and filters for electrodes
located in the perisylvian area (regions surrounding the Sylvian fissure).

The perisylvian area includes regions involved in language processing and
auditory function, surrounding the lateral sulcus.

Usage:
    python filter_perisylvian_electrodes.py --subject sub-05
    python filter_perisylvian_electrodes.py --input electrode_file.csv --output perisylvian_electrodes.csv

Output:
    CSV file with perisylvian electrode names and their regions
"""

import argparse
import pandas as pd
from pathlib import Path


# Define perisylvian regions based on DKT atlas
# These regions surround the Sylvian fissure (lateral sulcus)
PERISYLVIAN_REGIONS = {
    # Temporal regions (superior bank of Sylvian fissure)
    "ctx-lh-superiortemporal",
    "ctx-rh-superiortemporal",
    "ctx-lh-transversetemporal",  # Heschl's gyrus (primary auditory cortex)
    "ctx-rh-transversetemporal",
    "ctx-lh-middletemporal",  # Sometimes included in perisylvian
    "ctx-rh-middletemporal",

    # Inferior frontal regions (anterior bank of Sylvian fissure)
    "ctx-lh-parsopercularis",  # Broca's area (pars opercularis)
    "ctx-rh-parsopercularis",
    "ctx-lh-parstriangularis",  # Broca's area (pars triangularis)
    "ctx-rh-parstriangularis",
    "ctx-lh-parsorbitalis",
    "ctx-rh-parsorbitalis",

    # Inferior parietal regions (posterior bank of Sylvian fissure)
    "ctx-lh-supramarginal",  # Part of Wernicke's area
    "ctx-rh-supramarginal",
    "ctx-lh-inferiorparietal",
    "ctx-rh-inferiorparietal",

    # Insula (deep to Sylvian fissure)
    "ctx-lh-insula",
    "ctx-rh-insula",

    # Pre/postcentral gyri (ventral portions border Sylvian fissure)
    "ctx-lh-precentral",  # Ventral motor cortex
    "ctx-rh-precentral",
    "ctx-lh-postcentral",  # Ventral somatosensory cortex
    "ctx-rh-postcentral",
}


# Optional: Define core perisylvian regions (more conservative)
CORE_PERISYLVIAN_REGIONS = {
    "ctx-lh-superiortemporal",
    "ctx-rh-superiortemporal",
    "ctx-lh-transversetemporal",
    "ctx-rh-transversetemporal",
    "ctx-lh-parsopercularis",
    "ctx-rh-parsopercularis",
    "ctx-lh-parstriangularis",
    "ctx-rh-parstriangularis",
    "ctx-lh-supramarginal",
    "ctx-rh-supramarginal",
    "ctx-lh-insula",
    "ctx-rh-insula",
}


def filter_perisylvian_electrodes(input_csv, output_csv, use_core_only=False):
    """
    Filter electrodes located in perisylvian regions.

    Args:
        input_csv: Path to electrode locations CSV file
        output_csv: Path to save filtered perisylvian electrodes
        use_core_only: If True, use only core perisylvian regions (more conservative)

    Returns:
        DataFrame with perisylvian electrodes
    """
    print("="*70)
    print("FILTERING PERISYLVIAN ELECTRODES")
    print("="*70)

    # Select region set
    if use_core_only:
        regions_to_filter = CORE_PERISYLVIAN_REGIONS
        print("\nUsing CORE perisylvian regions (conservative)")
    else:
        regions_to_filter = PERISYLVIAN_REGIONS
        print("\nUsing EXTENDED perisylvian regions (inclusive)")

    print(f"\nPerisylvian regions defined ({len(regions_to_filter)} total):")
    for region in sorted(regions_to_filter):
        print(f"  - {region}")

    # Load electrode data
    print(f"\nLoading electrode data from: {input_csv}")
    df = pd.read_csv(input_csv)

    if 'electrode_name' not in df.columns or 'region' not in df.columns:
        raise ValueError("Input CSV must contain 'electrode_name' and 'region' columns")

    print(f"  Total electrodes: {len(df)}")

    # Filter for perisylvian electrodes
    perisylvian_mask = df['region'].isin(regions_to_filter)
    perisylvian_df = df[perisylvian_mask].copy()

    print(f"\n{'-'*70}")
    print("RESULTS")
    print(f"{'-'*70}")
    print(f"\nPerisylvian electrodes found: {len(perisylvian_df)}/{len(df)}")
    print(f"Percentage: {100 * len(perisylvian_df) / len(df):.1f}%")

    if len(perisylvian_df) == 0:
        print("\nNo perisylvian electrodes found!")
        print("Check if region names match the expected format.")
        return None

    # Region breakdown
    print(f"\nBreakdown by region:")
    region_counts = perisylvian_df['region'].value_counts()
    for region, count in region_counts.items():
        print(f"  {region}: {count}")

    # Save to CSV (only electrode_name and region columns)
    output_df = perisylvian_df[['electrode_name', 'region']]
    output_df.to_csv(output_csv, index=False)

    print(f"\n{'-'*70}")
    print(f"Output saved to: {output_csv}")
    print(f"{'-'*70}")

    # Print electrode names
    print(f"\nPerisylvian electrode names:")
    electrode_list = output_df['electrode_name'].tolist()
    print(f"  {', '.join(electrode_list)}")

    print("\n" + "="*70)
    print("FILTERING COMPLETE!")
    print("="*70)

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter electrodes in perisylvian regions"
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject ID (e.g., sub-05). Will use {subject}_electrode_locations.csv as input"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file with electrode locations (alternative to --subject)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for perisylvian electrodes (optional, auto-generated if not specified)"
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Use only core perisylvian regions (more conservative filtering)"
    )

    args = parser.parse_args()

    # Determine input file
    if args.subject:
        input_csv = f"{args.subject}_electrode_locations.csv"
        if args.output is None:
            output_csv = f"{args.subject}_perisylvian_electrodes.csv"
        else:
            output_csv = args.output
    elif args.input:
        input_csv = args.input
        if args.output is None:
            # Auto-generate output name
            input_path = Path(args.input)
            output_csv = input_path.parent / f"{input_path.stem}_perisylvian.csv"
        else:
            output_csv = args.output
    else:
        parser.error("Either --subject or --input must be specified")

    # Run filtering
    filter_perisylvian_electrodes(input_csv, output_csv, use_core_only=args.core_only)


if __name__ == "__main__":
    main()
