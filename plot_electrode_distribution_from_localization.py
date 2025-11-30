"""
Visualize electrode distribution in perisylvian regions from electrode localization data

Uses the CSV files from electrode localization (Stage 1-2) to show
distribution of electrodes across perisylvian brain regions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict

def load_all_perisylvian_electrodes(electrode_dir='electrode_results'):
    """
    Load all perisylvian electrode data from CSV files

    Parameters:
    -----------
    electrode_dir : str or Path
        Directory containing electrode CSV files

    Returns:
    --------
    all_data : pd.DataFrame
        Combined data from all subjects
    """
    electrode_dir = Path(electrode_dir)

    # Find all perisylvian electrode CSV files
    csv_files = sorted(electrode_dir.glob('*_perisylvian_electrodes.csv'))

    if len(csv_files) == 0:
        print(f"No perisylvian electrode files found in {electrode_dir}")
        return None

    print(f"Found {len(csv_files)} subjects with perisylvian electrode data\n")

    # Load and combine all data
    all_dfs = []
    for csv_file in csv_files:
        subject_id = csv_file.stem.replace('_perisylvian_electrodes', '')

        try:
            df = pd.read_csv(csv_file)
            df['subject'] = subject_id
            all_dfs.append(df)
            print(f"  [OK] {subject_id}: {len(df)} perisylvian electrodes")

        except Exception as e:
            print(f"  [ERROR] {subject_id}: {str(e)}")
            continue

    if len(all_dfs) == 0:
        print("No data loaded!")
        return None

    # Combine all subjects
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n[OK] Loaded {len(combined_df)} total perisylvian electrodes from {len(all_dfs)} subjects")

    return combined_df


def categorize_regions(df):
    """
    Add anatomical category and hemisphere information
    """
    df = df.copy()

    # Define categories
    categories = {
        'Temporal': ['superiortemporal', 'middletemporal', 'transversetemporal'],
        'Inferior Frontal': ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
        'Inferior Parietal': ['supramarginal', 'inferiorparietal'],
        'Sensorimotor': ['precentral', 'postcentral'],
        'Insula': ['insula']
    }

    df['category'] = ''
    df['hemisphere'] = ''
    df['region_short'] = ''

    for idx, row in df.iterrows():
        region = row['region']

        # Extract hemisphere
        if 'ctx-lh-' in region:
            df.at[idx, 'hemisphere'] = 'Left'
            region_name = region.replace('ctx-lh-', '')
        elif 'ctx-rh-' in region:
            df.at[idx, 'hemisphere'] = 'Right'
            region_name = region.replace('ctx-rh-', '')
        else:
            continue

        df.at[idx, 'region_short'] = region_name

        # Assign category
        for cat, keywords in categories.items():
            if any(kw in region_name for kw in keywords):
                df.at[idx, 'category'] = cat
                break

    return df


def create_electrode_distribution_plot(df, output_dir='.'):
    """
    Create comprehensive visualization of electrode distribution
    """
    # Set up the plot
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle('Perisylvian Electrode Distribution (from Electrode Localization)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Color scheme
    colors = {
        'Temporal': '#FF6B6B',
        'Inferior Frontal': '#4ECDC4',
        'Inferior Parietal': '#95E1D3',
        'Sensorimotor': '#FFD93D',
        'Insula': '#C7B7E8'
    }

    # Summary by region
    region_summary = df.groupby(['region', 'category', 'hemisphere']).agg({
        'electrode_name': 'count',
        'subject': 'nunique'
    }).reset_index()
    region_summary.columns = ['region', 'category', 'hemisphere', 'n_electrodes', 'n_subjects']
    region_summary['region_short'] = region_summary['region'].str.replace('ctx-lh-', '').str.replace('ctx-rh-', '')

    # 1. Total electrodes by category and hemisphere (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    category_summary = df.groupby(['category', 'hemisphere']).size().reset_index(name='n_electrodes')
    categories_order = ['Temporal', 'Inferior Frontal', 'Inferior Parietal', 'Sensorimotor', 'Insula']

    x = np.arange(len(categories_order))
    width = 0.35

    left_data = []
    right_data = []
    for cat in categories_order:
        left_val = category_summary[(category_summary['category'] == cat) &
                                     (category_summary['hemisphere'] == 'Left')]['n_electrodes'].sum()
        right_val = category_summary[(category_summary['category'] == cat) &
                                      (category_summary['hemisphere'] == 'Right')]['n_electrodes'].sum()
        left_data.append(left_val)
        right_data.append(right_val)

    bars1 = ax1.bar(x - width/2, left_data, width, label='Left Hemisphere',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, right_data, width, label='Right Hemisphere',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Anatomical Region', fontweight='bold')
    ax1.set_ylabel('Number of Electrodes', fontweight='bold')
    ax1.set_title('A. Total Electrodes by Category and Hemisphere', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories_order, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Subject coverage by region (top right)
    ax2 = fig.add_subplot(gs[0, 1])

    region_summary_sorted = region_summary.sort_values('n_subjects', ascending=True)

    y_pos = np.arange(len(region_summary_sorted))
    bar_colors = [colors[cat] for cat in region_summary_sorted['category']]

    bars = ax2.barh(y_pos, region_summary_sorted['n_subjects'], color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(y_pos)
    labels = [f"{row['region_short']} ({row['hemisphere'][0]})"
              for _, row in region_summary_sorted.iterrows()]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Number of Subjects', fontweight='bold')
    ax2.set_title('B. Subject Coverage per Region', fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, region_summary_sorted['n_subjects'])):
        ax2.text(val + 0.2, i, str(int(val)), va='center', fontsize=8)

    # 3. Total electrodes per region (middle row, full width)
    ax3 = fig.add_subplot(gs[1, :])

    region_summary_sorted = region_summary.sort_values('n_electrodes', ascending=True)

    y_pos = np.arange(len(region_summary_sorted))
    bar_colors = [colors[cat] for cat in region_summary_sorted['category']]

    bars = ax3.barh(y_pos, region_summary_sorted['n_electrodes'], color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)

    ax3.set_yticks(y_pos)
    labels = [f"{row['region_short']} ({row['hemisphere'][0]})"
              for _, row in region_summary_sorted.iterrows()]
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Total Electrodes (All Subjects)', fontweight='bold')
    ax3.set_title('C. Total Electrode Count per Region', fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, region_summary_sorted['n_electrodes'])):
        ax3.text(val + 2, i, str(int(val)), va='center', fontsize=8, fontweight='bold')

    # 4. Average electrodes per subject (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    region_summary['avg_per_subject'] = region_summary['n_electrodes'] / region_summary['n_subjects']
    region_summary_sorted = region_summary.sort_values('avg_per_subject', ascending=True)

    y_pos = np.arange(len(region_summary_sorted))
    bar_colors = [colors[cat] for cat in region_summary_sorted['category']]

    bars = ax4.barh(y_pos, region_summary_sorted['avg_per_subject'],
                    color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax4.set_yticks(y_pos)
    labels = [f"{row['region_short']} ({row['hemisphere'][0]})"
              for _, row in region_summary_sorted.iterrows()]
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('Average Electrodes per Subject', fontweight='bold')
    ax4.set_title('D. Average Electrode Density', fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Summary by category (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])

    cat_summary = df.groupby('category').agg({
        'electrode_name': 'count',
        'subject': 'nunique',
        'region': 'nunique'
    }).reset_index()
    cat_summary.columns = ['category', 'n_electrodes', 'n_subjects', 'n_regions']
    cat_summary = cat_summary.sort_values('n_electrodes', ascending=False)

    x = np.arange(len(cat_summary))
    bar_colors = [colors[cat] for cat in cat_summary['category']]

    bars = ax5.bar(x, cat_summary['n_electrodes'], color=bar_colors,
                   alpha=0.8, edgecolor='black', linewidth=1)

    ax5.set_xticks(x)
    ax5.set_xticklabels(cat_summary['category'], rotation=45, ha='right')
    ax5.set_ylabel('Total Electrodes', fontweight='bold')
    ax5.set_title('E. Summary by Anatomical Category', fontweight='bold', pad=10)
    ax5.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, cat_summary.itertuples())):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({row.n_regions} regions)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[cat], alpha=0.8,
                                     edgecolor='black', linewidth=0.5, label=cat)
                      for cat in categories_order]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10,
              title='Anatomical Categories', title_fontsize=11)

    # Save figure
    output_dir = Path(output_dir)
    output_file = output_dir / 'perisylvian_electrode_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot saved: {output_file}")

    plt.show()
    plt.close()

    return region_summary, cat_summary


def print_summary(df, region_summary, cat_summary):
    """
    Print detailed summary statistics
    """
    print("\n" + "="*80)
    print("PERISYLVIAN ELECTRODE DISTRIBUTION SUMMARY")
    print("="*80)

    n_subjects = df['subject'].nunique()
    n_electrodes = len(df)
    n_regions = df['region'].nunique()

    print(f"\nOverall Statistics:")
    print(f"  Total subjects: {n_subjects}")
    print(f"  Total perisylvian electrodes: {n_electrodes}")
    print(f"  Unique perisylvian regions: {n_regions}")
    print(f"  Average electrodes per subject: {n_electrodes / n_subjects:.1f}")

    print("\nBy Anatomical Category:")
    print("-"*80)
    for _, row in cat_summary.iterrows():
        print(f"  {row['category']:<25} | {int(row['n_electrodes']):>4} electrodes | "
              f"{int(row['n_regions']):>2} regions | {int(row['n_subjects']):>2} subjects")

    print("\nTop 10 Regions by Electrode Count:")
    print("-"*80)
    top_regions = region_summary.nlargest(10, 'n_electrodes')
    for _, row in top_regions.iterrows():
        print(f"  {row['region']:<35} | {int(row['n_electrodes']):>4} electrodes | "
              f"{int(row['n_subjects']):>2} subjects | "
              f"{row['avg_per_subject']:>5.1f} avg/subject")

    print("\n" + "="*80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize perisylvian electrode distribution from localization data'
    )
    parser.add_argument('--electrode-dir', type=str, default='electrode_results',
                       help='Directory with electrode CSV files (default: electrode_results)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')

    args = parser.parse_args()

    print("="*80)
    print("PERISYLVIAN ELECTRODE DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Electrode directory: {args.electrode_dir}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Load data
    df = load_all_perisylvian_electrodes(args.electrode_dir)

    if df is None or len(df) == 0:
        print("Failed to load data. Exiting.")
        return

    # Categorize regions
    df = categorize_regions(df)

    # Create visualization
    region_summary, cat_summary = create_electrode_distribution_plot(df, args.output_dir)

    # Print summary
    print_summary(df, region_summary, cat_summary)

    # Save summary to CSV
    output_dir = Path(args.output_dir)
    region_summary.to_csv(output_dir / 'perisylvian_electrode_summary_by_region.csv', index=False)
    cat_summary.to_csv(output_dir / 'perisylvian_electrode_summary_by_category.csv', index=False)
    print("[OK] Summary CSVs saved")

    print("\n[OK] Analysis complete!")


if __name__ == '__main__':
    main()
