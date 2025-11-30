"""
Diagnostic script to check ROI coverage and data availability
"""
import pandas as pd
from pathlib import Path
import numpy as np

def diagnose_roi_data(band='theta'):
    """Check ROI coverage and data availability"""

    data_dir = Path(f'./results_{band}/preprocessed_data')

    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist!")
        print(f"Run full_pipeline.py with --band {band} first")
        return

    # Find all .npz files
    npz_files = list(data_dir.glob('*_power.npz'))

    if len(npz_files) == 0:
        print(f"ERROR: No power files found in {data_dir}")
        return

    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC REPORT: {band.upper()} BAND")
    print(f"{'='*80}\n")

    print(f"Found {len(npz_files)} subjects with preprocessed data\n")

    # Collect ROI coverage information
    roi_subject_count = {}
    roi_electrode_count = {}

    for npz_file in npz_files:
        subject_id = npz_file.stem.replace(f'_{band}_power', '')

        # Load data
        data = np.load(npz_file)

        # Check ROI mapping
        roi_csv = data_dir / f'{subject_id}_roi_mapping.csv'

        if not roi_csv.exists():
            print(f"WARNING: No ROI mapping for {subject_id}")
            continue

        roi_df = pd.read_csv(roi_csv)

        # Count ROIs
        for roi in roi_df['region'].unique():
            if roi not in roi_subject_count:
                roi_subject_count[roi] = 0
                roi_electrode_count[roi] = 0

            roi_subject_count[roi] += 1
            roi_electrode_count[roi] += len(roi_df[roi_df['region'] == roi])

    # Create summary
    coverage_data = []
    for roi in sorted(roi_subject_count.keys()):
        coverage_data.append({
            'ROI': roi,
            'N_Subjects': roi_subject_count[roi],
            'Total_Electrodes': roi_electrode_count[roi],
            'Avg_Electrodes_Per_Subject': roi_electrode_count[roi] / roi_subject_count[roi]
        })

    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('N_Subjects', ascending=False)

    print("ROI COVERAGE SUMMARY:")
    print("-" * 80)
    print(f"Total unique ROIs: {len(coverage_df)}")
    print(f"ROIs with ≥2 subjects: {len(coverage_df[coverage_df['N_Subjects'] >= 2])}")
    print(f"ROIs with ≥3 subjects: {len(coverage_df[coverage_df['N_Subjects'] >= 3])}")
    print(f"ROIs with ≥5 subjects: {len(coverage_df[coverage_df['N_Subjects'] >= 5])}\n")

    print("TOP 20 ROIs BY SUBJECT COUNT:")
    print("-" * 80)
    print(coverage_df.head(20).to_string(index=False))

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")

    max_coverage = coverage_df['N_Subjects'].max()

    if max_coverage < 2:
        print("❌ PROBLEM: No ROI has ≥2 subjects!")
        print("   → Use --min-subjects 1 to analyze single-subject ROIs")
    elif len(coverage_df[coverage_df['N_Subjects'] >= 2]) < 5:
        print("⚠️  WARNING: Very few ROIs with ≥2 subjects")
        print(f"   → Only {len(coverage_df[coverage_df['N_Subjects'] >= 2])} ROIs available for group analysis")
        print("   → Consider --min-subjects 1 to include more ROIs")
    else:
        print(f"✓ {len(coverage_df[coverage_df['N_Subjects'] >= 2])} ROIs available for group analysis")

    # Check individual subject effect sizes
    print(f"\n{'='*80}")
    print("CHECKING INDIVIDUAL SUBJECT RESULTS:")
    print(f"{'='*80}")

    results_dir = Path(f'./results_{band}')
    subjects_with_effects = 0
    total_subjects = 0

    for subject_dir in results_dir.glob('sub-*'):
        if not subject_dir.is_dir():
            continue

        total_subjects += 1
        summary_csv = subject_dir / f'{subject_dir.name}_cluster_summary.csv'

        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            if len(df) > 0:
                subjects_with_effects += 1
                print(f"  {subject_dir.name}: {len(df)} clusters found")
            else:
                print(f"  {subject_dir.name}: No significant clusters")
        else:
            print(f"  {subject_dir.name}: No cluster summary file")

    print(f"\nSummary: {subjects_with_effects}/{total_subjects} subjects have significant within-subject effects")

    if subjects_with_effects == 0:
        print("\n❌ CRITICAL: No subjects show significant within-subject effects!")
        print("   This explains why group analysis finds nothing.")
        print("   Suggestions:")
        print("   1. Check if full_pipeline.py completed successfully")
        print("   2. Try more liberal precluster threshold in full_pipeline.py")
        print("   3. Verify your data quality")
    elif subjects_with_effects < total_subjects / 2:
        print(f"\n⚠️  WARNING: Only {subjects_with_effects}/{total_subjects} subjects show effects")
        print("   Group-level detection will be difficult with sparse effects")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    import sys
    band = sys.argv[1] if len(sys.argv) > 1 else 'theta'
    diagnose_roi_data(band)
