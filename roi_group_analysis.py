"""
ROI-Specific Group Analysis for iEEG Data

This script performs group-level statistical comparisons between speech and music
conditions for specific anatomical regions of interest (ROIs).

Key Features:
- Loads preprocessed theta power data from multiple subjects
- Groups electrodes by anatomical ROI (e.g., superior temporal gyrus)
- Averages within-subject across electrodes in each ROI
- Performs group-level paired t-tests or cluster-based permutation tests
- Generates ROI-specific visualizations and reports

Author: Generated for neuroinformatics project
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.ndimage import label
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class ROIGroupAnalyzer:
    """
    Main class for ROI-based group analysis
    """

    def __init__(self, preprocessed_data_dir, output_dir='./roi_group_results'):
        """
        Initialize the ROI Group Analyzer

        Parameters:
        -----------
        preprocessed_data_dir : str
            Directory containing preprocessed theta power data from full_pipeline.py
        output_dir : str
            Directory to save group analysis results
        """
        self.preprocessed_data_dir = Path(preprocessed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated by load_all_subjects()
        self.subjects = []
        self.subject_data = {}  # {subject_id: {speech_theta, music_theta, channels, roi_mapping}}

        # Will be populated by build_roi_dataset()
        self.roi_data = {}  # {roi_name: {subjects, speech_timeseries, music_timeseries}}

        # Common time vector (assumes all subjects have same time vector)
        self.times = None

    def load_all_subjects(self):
        """
        Load preprocessed data for all subjects

        Returns:
        --------
        n_subjects : int
            Number of subjects loaded
        """
        print("\n" + "="*80)
        print("LOADING PREPROCESSED DATA")
        print("="*80)

        # Find all theta power files
        theta_files = list(self.preprocessed_data_dir.glob("*_theta_power.npz"))

        if len(theta_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed theta power files found in {self.preprocessed_data_dir}\n"
                "Please run full_pipeline.py first to generate preprocessed data."
            )

        print(f"\nFound {len(theta_files)} subjects with preprocessed data")

        for theta_file in sorted(theta_files):
            # Extract subject ID from filename
            subject_id = theta_file.stem.replace('_theta_power', '')

            # Load theta power data
            data = np.load(theta_file)

            # Load ROI mapping
            roi_file = self.preprocessed_data_dir / f"{subject_id}_roi_mapping.csv"

            if not roi_file.exists():
                print(f"  ⚠ WARNING: Skipping {subject_id} - no ROI mapping file found")
                continue

            roi_df = pd.read_csv(roi_file)

            # Store data
            self.subject_data[subject_id] = {
                'speech_theta': data['speech_theta_power'],
                'music_theta': data['music_theta_power'],
                'channel_names': data['channel_names'],
                'roi_mapping': roi_df,
                'times': data['times'],
                'freqs': data['freqs']
            }

            self.subjects.append(subject_id)

            # Set common time vector from first subject
            if self.times is None:
                self.times = data['times']

            print(f"  [OK] Loaded {subject_id}: {len(data['channel_names'])} channels, "
                  f"{roi_df['region'].nunique()} ROIs")

        print(f"\n[OK] Successfully loaded {len(self.subjects)} subjects")

        return len(self.subjects)

    def get_all_rois(self):
        """
        Get list of all unique ROIs across all subjects

        Returns:
        --------
        roi_list : list
            Sorted list of unique ROI names
        roi_coverage : pd.DataFrame
            DataFrame with columns: roi, n_subjects, total_electrodes
        """
        all_rois = set()

        for subject_id, data in self.subject_data.items():
            roi_df = data['roi_mapping']
            all_rois.update(roi_df['region'].unique())

        roi_list = sorted(list(all_rois))

        # Create coverage summary
        coverage_data = []
        for roi in roi_list:
            n_subjects = 0
            total_electrodes = 0

            for subject_id, data in self.subject_data.items():
                roi_df = data['roi_mapping']
                roi_electrodes = roi_df[roi_df['region'] == roi]

                if len(roi_electrodes) > 0:
                    n_subjects += 1
                    total_electrodes += len(roi_electrodes)

            coverage_data.append({
                'roi': roi,
                'n_subjects': n_subjects,
                'total_electrodes': total_electrodes,
                'avg_electrodes_per_subject': total_electrodes / n_subjects if n_subjects > 0 else 0
            })

        coverage_df = pd.DataFrame(coverage_data)
        coverage_df = coverage_df.sort_values('n_subjects', ascending=False)

        return roi_list, coverage_df

    def build_roi_dataset(self, roi_name, min_subjects=2):
        """
        Build dataset for a specific ROI across all subjects

        For each subject:
        1. Find all electrodes in the ROI
        2. Average theta power across those electrodes
        3. Get one representative time-series per subject per condition

        Parameters:
        -----------
        roi_name : str
            Name of the ROI (e.g., 'ctx-lh-superiortemporal')
        min_subjects : int
            Minimum number of subjects required for this ROI

        Returns:
        --------
        roi_dataset : dict or None
            Dictionary with keys:
            - 'subjects': list of subject IDs
            - 'speech_timeseries': array (n_subjects, n_times)
            - 'music_timeseries': array (n_subjects, n_times)
            - 'n_electrodes_per_subject': list of electrode counts
            Returns None if insufficient subjects
        """
        speech_timeseries_list = []
        music_timeseries_list = []
        subjects_with_roi = []
        n_electrodes_per_subject = []

        for subject_id, data in self.subject_data.items():
            roi_df = data['roi_mapping']
            speech_theta = data['speech_theta']
            music_theta = data['music_theta']
            channel_names = data['channel_names']

            # Find electrodes in this ROI
            roi_electrodes = roi_df[roi_df['region'] == roi_name]['electrode_name'].tolist()

            if len(roi_electrodes) == 0:
                continue  # This subject doesn't have electrodes in this ROI

            # Get indices of these electrodes in the data
            channel_indices = [i for i, ch in enumerate(channel_names) if ch in roi_electrodes]

            if len(channel_indices) == 0:
                continue

            # Average across electrodes for this subject
            subject_speech = speech_theta[channel_indices, :].mean(axis=0)  # (n_times,)
            subject_music = music_theta[channel_indices, :].mean(axis=0)

            speech_timeseries_list.append(subject_speech)
            music_timeseries_list.append(subject_music)
            subjects_with_roi.append(subject_id)
            n_electrodes_per_subject.append(len(channel_indices))

        # Check if we have enough subjects
        if len(subjects_with_roi) < min_subjects:
            return None

        # Convert to arrays
        speech_array = np.array(speech_timeseries_list)  # (n_subjects, n_times)
        music_array = np.array(music_timeseries_list)

        return {
            'subjects': subjects_with_roi,
            'speech_timeseries': speech_array,
            'music_timeseries': music_array,
            'n_electrodes_per_subject': n_electrodes_per_subject
        }

    def paired_ttest_timeseries(self, speech_data, music_data):
        """
        Perform paired t-test at each timepoint

        Parameters:
        -----------
        speech_data : np.ndarray
            Shape: (n_subjects, n_times)
        music_data : np.ndarray
            Shape: (n_subjects, n_times)

        Returns:
        --------
        results : dict
            Dictionary with:
            - 't_values': t-statistic at each timepoint
            - 'p_values': uncorrected p-values
            - 'p_values_fdr': FDR-corrected p-values
            - 'significant_mask_fdr': boolean mask (corrected)
        """
        n_subjects, n_times = speech_data.shape

        t_values = np.zeros(n_times)
        p_values = np.zeros(n_times)

        # T-test at each timepoint
        for t_idx in range(n_times):
            t_val, p_val = stats.ttest_rel(speech_data[:, t_idx], music_data[:, t_idx])
            t_values[t_idx] = t_val
            p_values[t_idx] = p_val

        # FDR correction (Benjamini-Hochberg)
        from statsmodels.stats.multitest import multipletests
        reject_fdr, p_values_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        return {
            't_values': t_values,
            'p_values': p_values,
            'p_values_fdr': p_values_fdr,
            'significant_mask_fdr': reject_fdr,
            'n_subjects': n_subjects
        }

    def cluster_permutation_test_1d(self, speech_data, music_data, n_permutations=5000,
                                   alpha=0.05, tail='two'):
        """
        Cluster-based permutation test for 1D time-series

        Parameters:
        -----------
        speech_data : np.ndarray
            Shape: (n_subjects, n_times)
        music_data : np.ndarray
            Shape: (n_subjects, n_times)
        n_permutations : int
            Number of permutations
        alpha : float
            Significance level
        tail : str
            'two', 'positive', or 'negative'

        Returns:
        --------
        results : dict
            Dictionary with cluster test results
        """
        from mne.stats import permutation_cluster_1samp_test

        # Compute difference: speech - music
        differences = speech_data - music_data  # (n_subjects, n_times)

        # Set threshold
        if tail == 'two':
            threshold = stats.t.ppf(1 - alpha/2, len(differences) - 1)
        elif tail == 'positive':
            threshold = stats.t.ppf(1 - alpha, len(differences) - 1)
        else:  # negative
            threshold = stats.t.ppf(alpha, len(differences) - 1)

        # Run cluster test
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            differences,
            n_permutations=n_permutations,
            threshold=threshold,
            tail=0 if tail == 'two' else (1 if tail == 'positive' else -1),
            out_type='mask',
            verbose=False
        )

        # Find significant clusters
        sig_clusters = []
        for cluster_idx, cluster_mask in enumerate(clusters):
            if cluster_p_values[cluster_idx] < alpha:
                time_indices = np.where(cluster_mask)[0]
                sig_clusters.append({
                    'time_indices': time_indices,
                    'start_time': self.times[time_indices[0]],
                    'end_time': self.times[time_indices[-1]],
                    'duration': self.times[time_indices[-1]] - self.times[time_indices[0]],
                    'p_value': cluster_p_values[cluster_idx],
                    'cluster_stat': np.sum(np.abs(T_obs[cluster_mask]))
                })

        return {
            't_obs': T_obs,
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'significant_clusters': sig_clusters,
            'n_subjects': len(differences),
            'threshold': threshold
        }

    def analyze_roi(self, roi_name, method='cluster', min_subjects=3, n_permutations=5000):
        """
        Perform complete analysis for a single ROI

        Parameters:
        -----------
        roi_name : str
            ROI to analyze
        method : str
            'ttest' or 'cluster'
        min_subjects : int
            Minimum subjects required
        n_permutations : int
            For cluster method

        Returns:
        --------
        results : dict or None
            Analysis results or None if insufficient data
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING ROI: {roi_name}")
        print(f"{'='*80}")

        # Build dataset for this ROI
        roi_dataset = self.build_roi_dataset(roi_name, min_subjects=min_subjects)

        if roi_dataset is None:
            print(f"  ⚠ Skipping {roi_name}: insufficient subjects (< {min_subjects})")
            return None

        n_subjects = len(roi_dataset['subjects'])
        print(f"\n  Subjects with this ROI: {n_subjects}")
        print(f"  Subjects: {', '.join(roi_dataset['subjects'])}")
        print(f"  Electrodes per subject: {roi_dataset['n_electrodes_per_subject']}")
        print(f"  Total electrodes: {sum(roi_dataset['n_electrodes_per_subject'])}")

        speech_data = roi_dataset['speech_timeseries']
        music_data = roi_dataset['music_timeseries']

        # Perform statistical test
        if method == 'ttest':
            print(f"\n  Running paired t-test with FDR correction...")
            stats_results = self.paired_ttest_timeseries(speech_data, music_data)
        elif method == 'cluster':
            print(f"\n  Running cluster-based permutation test ({n_permutations} permutations)...")
            stats_results = self.cluster_permutation_test_1d(
                speech_data, music_data, n_permutations=n_permutations
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Combine results
        results = {
            'roi_name': roi_name,
            'method': method,
            'roi_dataset': roi_dataset,
            'stats': stats_results
        }

        # Print summary
        if method == 'ttest':
            n_sig = np.sum(stats_results['significant_mask_fdr'])
            print(f"\n  Significant timepoints (FDR-corrected): {n_sig}/{len(self.times)}")
        elif method == 'cluster':
            n_sig_clusters = len(stats_results['significant_clusters'])
            print(f"\n  Significant clusters: {n_sig_clusters}")
            for i, cluster in enumerate(stats_results['significant_clusters'], 1):
                print(f"    Cluster {i}: {cluster['start_time']:.2f}-{cluster['end_time']:.2f}s, "
                      f"p={cluster['p_value']:.4f}")

        return results

    def plot_roi_results(self, results, save=True):
        """
        Visualize results for a single ROI

        Parameters:
        -----------
        results : dict
            Results from analyze_roi()
        save : bool
            Whether to save the figure
        """
        roi_name = results['roi_name']
        method = results['method']
        roi_dataset = results['roi_dataset']
        stats_results = results['stats']

        speech_data = roi_dataset['speech_timeseries']
        music_data = roi_dataset['music_timeseries']
        n_subjects = len(roi_dataset['subjects'])

        # Compute group averages and SEM
        speech_mean = speech_data.mean(axis=0)
        speech_sem = speech_data.std(axis=0) / np.sqrt(n_subjects)
        music_mean = music_data.mean(axis=0)
        music_sem = music_data.std(axis=0) / np.sqrt(n_subjects)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # --- Top panel: Time-series with error bars ---
        ax = axes[0]

        # Plot group averages
        ax.plot(self.times, speech_mean, 'b-', linewidth=2, label='Speech')
        ax.fill_between(self.times, speech_mean - speech_sem, speech_mean + speech_sem,
                        alpha=0.3, color='blue')

        ax.plot(self.times, music_mean, 'r-', linewidth=2, label='Music')
        ax.fill_between(self.times, music_mean - music_sem, music_mean + music_sem,
                        alpha=0.3, color='red')

        # Highlight significant time windows
        if method == 'ttest':
            sig_mask = stats_results['significant_mask_fdr']
            if np.any(sig_mask):
                y_min, y_max = ax.get_ylim()
                ax.fill_between(self.times, y_min, y_max, where=sig_mask,
                               alpha=0.2, color='green', label='Significant (FDR)')
        elif method == 'cluster':
            for cluster in stats_results['significant_clusters']:
                ax.axvspan(cluster['start_time'], cluster['end_time'],
                          alpha=0.2, color='green', label='Significant cluster')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Theta Power (a.u.)', fontsize=12)
        ax.set_title(f'{roi_name}\nGroup Average (n={n_subjects} subjects)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # --- Bottom panel: Statistical results ---
        ax = axes[1]

        if method == 'ttest':
            # Plot t-values
            t_values = stats_results['t_values']
            ax.plot(self.times, t_values, 'k-', linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

            # Mark significance
            sig_mask = stats_results['significant_mask_fdr']
            if np.any(sig_mask):
                ax.scatter(self.times[sig_mask], t_values[sig_mask],
                          color='green', s=20, zorder=5, label='Significant (FDR)')

            ax.set_ylabel('T-statistic', fontsize=12)
            ax.set_title('Paired T-Test Results (Speech vs Music)', fontsize=12)

        elif method == 'cluster':
            # Plot t-values
            t_obs = stats_results['t_obs']
            ax.plot(self.times, t_obs, 'k-', linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

            # Mark significant clusters
            for cluster in stats_results['significant_clusters']:
                time_idx = cluster['time_indices']
                ax.scatter(self.times[time_idx], t_obs[time_idx],
                          color='green', s=20, zorder=5)

            # Mark threshold
            threshold = stats_results['threshold']
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5,
                      label=f'Threshold (±{threshold:.2f})')
            ax.axhline(y=-threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_ylabel('T-statistic', fontsize=12)
            ax.set_title('Cluster-Based Permutation Test Results', fontsize=12)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"{roi_name}_{method}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  [OK] Figure saved to: {save_path}")

        plt.close()

    def run_all_rois(self, method='cluster', min_subjects=3, n_permutations=5000,
                    top_n=None):
        """
        Run analysis for all ROIs with sufficient coverage

        Parameters:
        -----------
        method : str
            'ttest' or 'cluster'
        min_subjects : int
            Minimum subjects required
        n_permutations : int
            For cluster method
        top_n : int or None
            If provided, only analyze top N ROIs by subject coverage

        Returns:
        --------
        all_results : dict
            Dictionary mapping ROI names to results
        """
        print("\n" + "="*80)
        print("RUNNING GROUP ANALYSIS FOR ALL ROIs")
        print("="*80)

        # Get all ROIs
        roi_list, coverage_df = self.get_all_rois()

        print(f"\nTotal unique ROIs: {len(roi_list)}")
        print(f"\nROI coverage summary:")
        print(coverage_df.to_string(index=False))

        # Filter ROIs by minimum subjects
        valid_rois = coverage_df[coverage_df['n_subjects'] >= min_subjects]['roi'].tolist()
        print(f"\n[OK] ROIs with >= {min_subjects} subjects: {len(valid_rois)}")

        if top_n is not None:
            valid_rois = valid_rois[:top_n]
            print(f"  Analyzing top {top_n} ROIs by coverage")

        # Analyze each ROI
        all_results = {}
        successful = 0
        failed = 0

        for roi_name in valid_rois:
            try:
                results = self.analyze_roi(roi_name, method=method, min_subjects=min_subjects,
                                         n_permutations=n_permutations)

                if results is not None:
                    all_results[roi_name] = results
                    self.plot_roi_results(results, save=True)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"\n  ❌ ERROR analyzing {roi_name}: {str(e)}")
                failed += 1

        # Create summary report
        self.create_summary_report(all_results, method)

        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Results saved to: {self.output_dir}")

        return all_results

    def create_summary_report(self, all_results, method):
        """
        Create summary report across all ROIs

        Parameters:
        -----------
        all_results : dict
            Dictionary of results from run_all_rois()
        method : str
            Analysis method used
        """
        summary_data = []

        for roi_name, results in all_results.items():
            roi_dataset = results['roi_dataset']
            stats_results = results['stats']

            row = {
                'roi': roi_name,
                'n_subjects': len(roi_dataset['subjects']),
                'total_electrodes': sum(roi_dataset['n_electrodes_per_subject']),
                'avg_electrodes_per_subject': np.mean(roi_dataset['n_electrodes_per_subject'])
            }

            if method == 'ttest':
                row['n_significant_timepoints'] = np.sum(stats_results['significant_mask_fdr'])
                row['proportion_significant'] = np.mean(stats_results['significant_mask_fdr'])
                row['max_t_value'] = np.max(np.abs(stats_results['t_values']))

            elif method == 'cluster':
                row['n_significant_clusters'] = len(stats_results['significant_clusters'])

                if len(stats_results['significant_clusters']) > 0:
                    row['min_cluster_p'] = min([c['p_value'] for c in stats_results['significant_clusters']])
                    row['max_cluster_duration'] = max([c['duration'] for c in stats_results['significant_clusters']])
                else:
                    row['min_cluster_p'] = 1.0
                    row['max_cluster_duration'] = 0.0

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Sort by significance
        if method == 'ttest':
            summary_df = summary_df.sort_values('proportion_significant', ascending=False)
        elif method == 'cluster':
            summary_df = summary_df.sort_values(['n_significant_clusters', 'min_cluster_p'],
                                               ascending=[False, True])

        # Save to CSV
        csv_path = self.output_dir / f'summary_{method}.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"\n[OK] Summary table saved to: {csv_path}")

        # Print top results
        print(f"\n{'='*80}")
        print(f"TOP ROIs WITH SIGNIFICANT EFFECTS")
        print(f"{'='*80}")

        if method == 'ttest':
            top_rois = summary_df[summary_df['proportion_significant'] > 0].head(10)
        elif method == 'cluster':
            top_rois = summary_df[summary_df['n_significant_clusters'] > 0].head(10)

        if len(top_rois) > 0:
            print(top_rois.to_string(index=False))
        else:
            print("No ROIs with significant effects found.")


def main():
    """
    Main execution function
    """
    # Configuration
    PREPROCESSED_DATA_DIR = "./test_pipeline_output/preprocessed_data"
    OUTPUT_DIR = "./roi_group_results"
    METHOD = "ttest"  # 'ttest' or 'cluster' (use ttest for single subject testing)
    MIN_SUBJECTS = 1  # Minimum subjects required per ROI (set to 1 for testing)
    N_PERMUTATIONS = 5000

    print("="*80)
    print("ROI-SPECIFIC GROUP ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Preprocessed data: {PREPROCESSED_DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Method: {METHOD}")
    print(f"  Minimum subjects per ROI: {MIN_SUBJECTS}")
    print(f"  Permutations (if cluster): {N_PERMUTATIONS}")

    # Initialize analyzer
    analyzer = ROIGroupAnalyzer(PREPROCESSED_DATA_DIR, OUTPUT_DIR)

    # Load all subjects
    n_subjects = analyzer.load_all_subjects()

    if n_subjects < 1:
        print("\n[ERROR] Need at least 1 subject for analysis")
        return

    if n_subjects == 1:
        print("\n[WARNING] Only 1 subject loaded - results will show single-subject patterns")
        print("  For true group analysis, process multiple subjects through full_pipeline.py")

    # Run analysis for all ROIs
    results = analyzer.run_all_rois(
        method=METHOD,
        min_subjects=MIN_SUBJECTS,
        n_permutations=N_PERMUTATIONS,
        top_n=None  # Set to a number to limit analysis to top N ROIs
    )

    print(f"\n[OK] Analysis complete!")
    print(f"  Total ROIs analyzed: {len(results)}")
    print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
