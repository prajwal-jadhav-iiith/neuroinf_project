"""
Visualize electrode distribution in perisylvian regions

Creates a comprehensive visualization showing electrode counts
across perisylvian brain regions, grouped by anatomical category.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def categorize_perisylvian_rois(df):
    """
    Categorize ROIs into anatomical regions and add metadata
    """
    # Define categories
    categories = {
        'Temporal': ['superiortemporal', 'middletemporal', 'transversetemporal'],
        'Inferior Frontal': ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
        'Inferior Parietal': ['supramarginal', 'inferiorparietal'],
        'Sensorimotor': ['precentral', 'postcentral'],
        'Insula': ['insula']
    }

    # Add category and hemisphere columns
    df['category'] = ''
    df['hemisphere'] = ''
    df['region_name'] = ''

    for idx, row in df.iterrows():
        roi = row['roi']

        # Extract hemisphere
        if 'ctx-lh-' in roi:
            df.at[idx, 'hemisphere'] = 'Left'
            region = roi.replace('ctx-lh-', '')
        elif 'ctx-rh-' in roi:
            df.at[idx, 'hemisphere'] = 'Right'
            region = roi.replace('ctx-rh-', '')
        else:
            continue

        df.at[idx, 'region_name'] = region

        # Assign category
        for cat, keywords in categories.items():
            if any(kw in region for kw in keywords):
                df.at[idx, 'category'] = cat
                break

    return df


def create_perisylvian_distribution_plot(csv_file, band='theta', output_dir='.'):
    """
    Create comprehensive visualization of electrode distribution
    """
    # Read data
    df = pd.read_csv(csv_file)
    df = categorize_perisylvian_rois(df)

    # Remove any uncategorized ROIs
    df = df[df['category'] != '']

    # Set up the plot
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Electrode Distribution in Perisylvian Regions ({band.upper()} Band)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Color scheme
    colors = {
        'Temporal': '#FF6B6B',
        'Inferior Frontal': '#4ECDC4',
        'Inferior Parietal': '#95E1D3',
        'Sensorimotor': '#FFD93D',
        'Insula': '#C7B7E8'
    }

    # 1. Total electrodes by category and hemisphere (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    category_data = df.groupby(['category', 'hemisphere'])['total_electrodes'].sum().reset_index()
    categories_order = ['Temporal', 'Inferior Frontal', 'Inferior Parietal', 'Sensorimotor', 'Insula']

    x = np.arange(len(categories_order))
    width = 0.35

    left_data = []
    right_data = []
    for cat in categories_order:
        left_val = category_data[(category_data['category'] == cat) &
                                  (category_data['hemisphere'] == 'Left')]['total_electrodes'].sum()
        right_val = category_data[(category_data['category'] == cat) &
                                   (category_data['hemisphere'] == 'Right')]['total_electrodes'].sum()
        left_data.append(left_val)
        right_data.append(right_val)

    bars1 = ax1.bar(x - width/2, left_data, width, label='Left Hemisphere',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, right_data, width, label='Right Hemisphere',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Anatomical Region', fontweight='bold')
    ax1.set_ylabel('Total Electrodes', fontweight='bold')
    ax1.set_title('A. Total Electrodes by Category and Hemisphere', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories_order, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Subject coverage by region (top right)
    ax2 = fig.add_subplot(gs[0, 1])

    df_sorted = df.sort_values('n_subjects', ascending=True)

    y_pos = np.arange(len(df_sorted))
    bar_colors = [colors[cat] for cat in df_sorted['category']]

    bars = ax2.barh(y_pos, df_sorted['n_subjects'], color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)

    # Highlight significant ROI
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['n_significant_timepoints'] > 0:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)

    ax2.set_yticks(y_pos)
    labels = [f"{row['region_name']} ({row['hemisphere'][0]})"
              for _, row in df_sorted.iterrows()]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Number of Subjects', fontweight='bold')
    ax2.set_title('B. Subject Coverage per ROI', fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['n_subjects'])):
        ax2.text(val + 0.2, i, str(int(val)), va='center', fontsize=8)

    # 3. Total electrodes per ROI (middle row, full width)
    ax3 = fig.add_subplot(gs[1, :])

    df_sorted = df.sort_values('total_electrodes', ascending=True)

    y_pos = np.arange(len(df_sorted))
    bar_colors = [colors[cat] for cat in df_sorted['category']]

    bars = ax3.barh(y_pos, df_sorted['total_electrodes'], color=bar_colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)

    # Highlight significant ROI
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['n_significant_timepoints'] > 0:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
            # Add star
            ax3.text(row['total_electrodes'] + 5, i, '★',
                    fontsize=16, color='red', va='center')

    ax3.set_yticks(y_pos)
    labels = [f"{row['region_name']} ({row['hemisphere'][0]})"
              for _, row in df_sorted.iterrows()]
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Total Electrodes (All Subjects)', fontweight='bold')
    ax3.set_title('C. Total Electrode Count per ROI (★ = Significant Effect)',
                  fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['total_electrodes'])):
        ax3.text(val + 2, i, str(int(val)), va='center', fontsize=8, fontweight='bold')

    # 4. Average electrodes per subject (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    df_sorted = df.sort_values('avg_electrodes_per_subject', ascending=True)

    y_pos = np.arange(len(df_sorted))
    bar_colors = [colors[cat] for cat in df_sorted['category']]

    bars = ax4.barh(y_pos, df_sorted['avg_electrodes_per_subject'],
                    color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Highlight significant ROI
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['n_significant_timepoints'] > 0:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)

    ax4.set_yticks(y_pos)
    labels = [f"{row['region_name']} ({row['hemisphere'][0]})"
              for _, row in df_sorted.iterrows()]
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('Average Electrodes per Subject', fontweight='bold')
    ax4.set_title('D. Average Electrode Density', fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Summary by category (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])

    summary = df.groupby('category').agg({
        'total_electrodes': 'sum',
        'n_subjects': 'max',
        'roi': 'count'
    }).reset_index()
    summary.columns = ['category', 'total_electrodes', 'max_subjects', 'n_rois']
    summary = summary.sort_values('total_electrodes', ascending=False)

    x = np.arange(len(summary))
    bar_colors = [colors[cat] for cat in summary['category']]

    bars = ax5.bar(x, summary['total_electrodes'], color=bar_colors,
                   alpha=0.8, edgecolor='black', linewidth=1)

    ax5.set_xticks(x)
    ax5.set_xticklabels(summary['category'], rotation=45, ha='right')
    ax5.set_ylabel('Total Electrodes', fontweight='bold')
    ax5.set_title('E. Summary by Anatomical Category', fontweight='bold', pad=10)
    ax5.grid(axis='y', alpha=0.3)

    # Add value labels and ROI counts
    for i, (bar, row) in enumerate(zip(bars, summary.itertuples())):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({row.n_rois} ROIs)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create legend for categories (at bottom)
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[cat], alpha=0.8,
                                     edgecolor='black', linewidth=0.5, label=cat)
                      for cat in categories_order]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10,
              title='Anatomical Categories', title_fontsize=11)

    # Save figure
    output_dir = Path(output_dir)
    output_file = output_dir / f'perisylvian_electrode_distribution_{band}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot saved: {output_file}")

    plt.show()
    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print(f"PERISYLVIAN ELECTRODE DISTRIBUTION SUMMARY ({band.upper()} BAND)")
    print("="*80)

    print("\nBy Anatomical Category:")
    print("-"*80)
    for _, row in summary.iterrows():
        print(f"  {row['category']:<25} | {int(row['total_electrodes']):>4} electrodes | "
              f"{int(row['n_rois']):>2} ROIs | {int(row['max_subjects']):>2} max subjects")

    print(f"\nGrand Total: {int(summary['total_electrodes'].sum())} electrodes "
          f"across {int(summary['n_rois'].sum())} perisylvian ROIs")

    # Show significant ROIs
    sig_rois = df[df['n_significant_timepoints'] > 0]
    if len(sig_rois) > 0:
        print(f"\nROIs with Significant Effects:")
        print("-"*80)
        for _, row in sig_rois.iterrows():
            print(f"  ★ {row['roi']:<35} | {int(row['total_electrodes']):>3} electrodes | "
                  f"{int(row['n_subjects']):>2} subjects | "
                  f"{int(row['n_significant_timepoints'])} sig. timepoints")

    print("\n" + "="*80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize electrode distribution in perisylvian regions'
    )
    parser.add_argument('--band', type=str, default='theta',
                       choices=['theta', 'alpha'],
                       help='Frequency band (default: theta)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to summary CSV (default: roi_group_results_{band}/summary_ttest.csv)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')

    args = parser.parse_args()

    # Set default CSV path if not specified
    if args.csv is None:
        args.csv = f'roi_group_results_{args.band}/summary_ttest.csv'

    print("="*80)
    print("PERISYLVIAN ELECTRODE DISTRIBUTION VISUALIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Frequency band: {args.band}")
    print(f"  Input CSV: {args.csv}")
    print(f"  Output directory: {args.output_dir}")

    # Create visualization
    create_perisylvian_distribution_plot(args.csv, args.band, args.output_dir)

    print("\n[OK] Analysis complete!")


if __name__ == '__main__':
    main()
