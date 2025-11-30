"""
ROI-Specific Group Analysis for iEEG Data

This script performs group-level statistical comparisons between speech and music
conditions for specific anatomical regions of interest (ROIs).

Key Features:
- Loads preprocessed power data (theta or alpha band) from multiple subjects
- Groups electrodes by anatomical ROI (e.g., superior temporal gyrus)
- Averages within-subject across electrodes in each ROI
- Performs group-level paired t-tests or cluster-based permutation tests
- Generates ROI-specific visualizations and reports

Usage:
    python roi_group_analysis.py --band theta
    python roi_group_analysis.py --band alpha
    python roi_group_analysis.py --band alpha --method cluster --n-permutations 10000

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
import argparse

warnings.filterwarnings('ignore')


class ROIGroupAnalyzer:
    """
    Main class for ROI-based group analysis
    """

    def __init__(self, preprocessed_data_dir, output_dir='./roi_group_results', band='theta'):
        """
        Initialize the ROI Group Analyzer

        Parameters:
        -----------
        preprocessed_data_dir : str
            Directory containing preprocessed power data from full_pipeline.py
        output_dir : str
            Directory to save group analysis results
        band : str
            Frequency band to analyze: 'theta' or 'alpha' (default: 'theta')
        """
        self.preprocessed_data_dir = Path(preprocessed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.band = band

        # Will be populated by load_all_subjects()
        self.subjects = []
        self.subject_data = {}  # {subject_id: {speech_power, music_power, channels, roi_mapping}}

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
        print(f"LOADING PREPROCESSED DATA ({self.band.upper()} BAND)")
        print("="*80)

        # Find all power files for the specified band
        power_files = list(self.preprocessed_data_dir.glob(f"*_{self.band}_power.npz"))

        if len(power_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed {self.band} power files found in {self.preprocessed_data_dir}\n"
                f"Please run full_pipeline.py with --band {self.band} first to generate preprocessed data."
            )

        print(f"\nFound {len(power_files)} subjects with preprocessed {self.band} data")

        for power_file in sorted(power_files):
            # Extract subject ID from filename
            subject_id = power_file.stem.replace(f'_{self.band}_power', '')

            # Load power data
            data = np.load(power_file)

            # Load ROI mapping
            roi_file = self.preprocessed_data_dir / f"{subject_id}_roi_mapping.csv"

            if not roi_file.exists():
                print(f"  ⚠ WARNING: Skipping {subject_id} - no ROI mapping file found")
                continue

            roi_df = pd.read_csv(roi_file)

            # Store data with dynamic keys
            speech_key = f'speech_{self.band}_power'
            music_key = f'music_{self.band}_power'

            # Store data
            self.subject_data[subject_id] = {
                'speech_power': data[speech_key],
                'music_power': data[music_key],
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
        2. Average power across those electrodes
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
            speech_power = data['speech_power']
            music_power = data['music_power']
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
            subject_speech = speech_power[channel_indices, :].mean(axis=0)  # (n_times,)
            subject_music = music_power[channel_indices, :].mean(axis=0)

            speech_timeseries_list.append(subject_speech)
            music_timeseries_list.append(subject_music)
            subjects_with_roi.append(subject_id)
            n_electrodes_per_subject.append(len(channel_indices))

        # Check if we have enough subjects
        if len(subjects_with_roi) < min_subjects:
            return None

        # Validate that all time series have the same length
        time_lengths = [len(ts) for ts in speech_timeseries_list]
        if len(set(time_lengths)) > 1:
            print(f"\n  ⚠️  WARNING: Time series length mismatch in ROI '{roi_name}':")
            for subj, length in zip(subjects_with_roi, time_lengths):
                print(f"      {subj}: {length} timepoints")
            print(f"\n  This indicates subjects were processed with different parameters.")
            print(f"  Please reprocess all subjects with: python full_pipeline.py --force-reprocess")
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
                                   alpha=0.05, tail='two', precluster_p=0.05):
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
            Significance level for cluster-level test
        tail : str
            'two', 'positive', or 'negative'
        precluster_p : float
            P-value threshold for forming clusters (liberal threshold)
            Default 0.05, use 0.1-0.2 for more sensitive detection

        Returns:
        --------
        results : dict
            Dictionary with cluster test results
        """
        from mne.stats import permutation_cluster_1samp_test

        # Compute difference: speech - music
        differences = speech_data - music_data  # (n_subjects, n_times)

        print(f"    [DEBUG] Data shapes: speech={speech_data.shape}, music={music_data.shape}")
        print(f"    [DEBUG] Differences shape: {differences.shape}")
        print(f"    [DEBUG] Times shape: {len(self.times)}")

        # Compute t-statistics manually to check
        t_manual = (differences.mean(axis=0)) / (differences.std(axis=0, ddof=1) / np.sqrt(len(differences)))
        print(f"    [DEBUG] T-stats range: [{np.min(t_manual):.3f}, {np.max(t_manual):.3f}]")
        print(f"    [DEBUG] T-stats > 0: {np.sum(t_manual > 0)}, T-stats < 0: {np.sum(t_manual < 0)}")

        # Set precluster threshold (more liberal for better cluster formation)
        if tail == 'two':
            threshold = stats.t.ppf(1 - precluster_p/2, len(differences) - 1)
        elif tail == 'positive':
            threshold = stats.t.ppf(1 - precluster_p, len(differences) - 1)
        else:  # negative
            threshold = stats.t.ppf(precluster_p, len(differences) - 1)

        print(f"    [DEBUG] Precluster threshold: t = ±{threshold:.3f} (p = {precluster_p})")
        print(f"    [DEBUG] Timepoints exceeding threshold: {np.sum(np.abs(t_manual) > threshold)}/{len(t_manual)}")

        # Run cluster test
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            differences,
            n_permutations=n_permutations,
            threshold=threshold,
            tail=0 if tail == 'two' else (1 if tail == 'positive' else -1),
            out_type='mask',
            verbose=False
        )

        print(f"    [DEBUG] MNE returned: T_obs shape={T_obs.shape}, n_clusters={len(clusters)}")
        print(f"    [DEBUG] T_obs range: [{np.min(T_obs):.3f}, {np.max(T_obs):.3f}]")
        if len(clusters) > 0:
            # Calculate cluster sizes properly (handle tuples and slices)
            cluster_sizes = []
            for c in clusters[:5]:
                # Unpack tuple if needed
                if isinstance(c, tuple):
                    c = c[0]

                if isinstance(c, slice):
                    size = c.stop - c.start
                else:
                    size = np.sum(c)
                cluster_sizes.append(size)
            print(f"    [DEBUG] First 5 cluster sizes (timepoints): {cluster_sizes}")
            print(f"    [DEBUG] First 5 cluster p-values: {cluster_p_values[:5]}")

        # Find significant clusters
        sig_clusters = []
        for cluster_idx, cluster_mask in enumerate(clusters):
            if cluster_p_values[cluster_idx] < alpha:
                # DEBUG: Show what we received
                print(f"    [DEBUG] Cluster {cluster_idx+1} (p={cluster_p_values[cluster_idx]:.4f}): "
                      f"type={type(cluster_mask)}, value={cluster_mask}")

                # Handle tuple wrapping (MNE returns tuples for 1D data)
                if isinstance(cluster_mask, tuple):
                    cluster_mask = cluster_mask[0]  # Unpack single-element tuple
                    print(f"    [DEBUG]   -> Unpacked tuple, inner type={type(cluster_mask)}")

                # Handle both slice objects and boolean arrays
                if isinstance(cluster_mask, slice):
                    # Convert slice to array of indices
                    time_indices = np.arange(cluster_mask.start, cluster_mask.stop,
                                            cluster_mask.step if cluster_mask.step else 1)
                    print(f"    [DEBUG]   -> Converted slice to {len(time_indices)} indices: {time_indices[:10]}...")
                else:
                    # Boolean mask - get True indices
                    time_indices = np.where(cluster_mask)[0]
                    print(f"    [DEBUG]   -> Boolean mask, found {len(time_indices)} True values")

                # DEBUG: Print cluster info
                print(f"    [DEBUG] Cluster {cluster_idx+1}: {len(time_indices)} timepoints, "
                      f"indices: {time_indices[:5]}...{time_indices[-5:] if len(time_indices) > 5 else time_indices}")

                if len(time_indices) == 0:
                    print(f"    [WARNING] Cluster {cluster_idx+1} has no timepoints! Skipping.")
                    continue

                sig_clusters.append({
                    'time_indices': time_indices,
                    'start_time': self.times[time_indices[0]],
                    'end_time': self.times[time_indices[-1]],
                    'duration': self.times[time_indices[-1]] - self.times[time_indices[0]],
                    'p_value': cluster_p_values[cluster_idx],
                    'cluster_stat': np.sum(np.abs(T_obs[time_indices]))  # Use time_indices, not cluster_mask
                })

                print(f"    [DEBUG] Cluster times: {self.times[time_indices[0]]:.3f}s to {self.times[time_indices[-1]]:.3f}s")

        return {
            't_obs': T_obs,
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'significant_clusters': sig_clusters,
            'n_subjects': len(differences),
            'threshold': threshold
        }

    def analyze_roi(self, roi_name, method='cluster', min_subjects=3, n_permutations=5000,
                   precluster_p=0.1):
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
        precluster_p : float
            P-value threshold for cluster formation (default 0.1 for better sensitivity)

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
            print(f"\n  Running cluster-based permutation test ({n_permutations} permutations, precluster_p={precluster_p})...")
            stats_results = self.cluster_permutation_test_1d(
                speech_data, music_data, n_permutations=n_permutations, precluster_p=precluster_p
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

        print(f"\n  [DEBUG] Starting plot for {roi_name}, method={method}, save={save}")

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

        # Determine y-limits first for proper shading
        y_min_data = min(np.min(speech_mean - speech_sem), np.min(music_mean - music_sem))
        y_max_data = max(np.max(speech_mean + speech_sem), np.max(music_mean + music_sem))
        y_range = y_max_data - y_min_data
        y_min = y_min_data - 0.1 * y_range
        y_max = y_max_data + 0.1 * y_range
        ax.set_ylim(y_min, y_max)

        # Highlight significant time windows FIRST (so they appear behind data)
        if method == 'ttest':
            sig_mask = stats_results['significant_mask_fdr']
            if np.any(sig_mask):
                ax.fill_between(self.times, y_min, y_max, where=sig_mask,
                               alpha=0.2, color='green', label='Significant (FDR)', zorder=1)
        elif method == 'cluster':
            for i, cluster in enumerate(stats_results['significant_clusters']):
                label = 'Significant cluster' if i == 0 else None  # Only label first cluster
                ax.axvspan(cluster['start_time'], cluster['end_time'],
                          alpha=0.3, color='green', label=label, zorder=2)

        # Plot group averages ON TOP of shaded regions
        ax.plot(self.times, speech_mean, 'b-', linewidth=2, label='Speech', zorder=3)
        ax.fill_between(self.times, speech_mean - speech_sem, speech_mean + speech_sem,
                        alpha=0.3, color='blue', zorder=3)

        ax.plot(self.times, music_mean, 'r-', linewidth=2, label='Music', zorder=3)
        ax.fill_between(self.times, music_mean - music_sem, music_mean + music_sem,
                        alpha=0.3, color='red', zorder=3)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'{self.band.capitalize()} Power (a.u.)', fontsize=12)
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
            print(f"\n  [DEBUG] Saving to: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [OK] Figure saved successfully!")
        else:
            print(f"\n  [DEBUG] save=False, not saving figure")

        plt.close()
        print(f"  [DEBUG] Plot closed for {roi_name}")

    def run_all_rois(self, method='cluster', min_subjects=3, n_permutations=5000,
                    precluster_p=0.1, top_n=None):
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
        precluster_p : float
            P-value threshold for cluster formation (default 0.1)
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
                                         n_permutations=n_permutations, precluster_p=precluster_p)

                if results is not None:
                    all_results[roi_name] = results

                    # Plot with error handling
                    try:
                        self.plot_roi_results(results, save=True)
                        successful += 1
                    except Exception as plot_error:
                        print(f"\n  ⚠️  WARNING: Plotting failed for {roi_name}: {str(plot_error)}")
                        import traceback
                        traceback.print_exc()
                        successful += 1  # Still count as successful analysis
                else:
                    failed += 1

            except Exception as e:
                print(f"\n  ❌ ERROR analyzing {roi_name}: {str(e)}")
                import traceback
                traceback.print_exc()
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


def generate_roi_analysis_summary(results, band, output_dir):
    """
    Generate comprehensive summary and interpretation of ROI group analysis results

    Parameters:
    -----------
    results : dict
        Dictionary of ROI results from run_all_rois()
    band : str
        Frequency band analyzed ('theta' or 'alpha')
    output_dir : Path
        Output directory for saving summary

    Returns:
    --------
    summary_text : str
        Formatted summary text
    """
    summary_lines = []

    # Header
    summary_lines.append("\n" + "="*80)
    summary_lines.append(f"ROI GROUP ANALYSIS SUMMARY: {band.upper()} BAND")
    summary_lines.append("="*80)
    summary_lines.append("")

    # Collect significant ROIs
    significant_rois = []
    for roi_name, roi_result in results.items():
        if roi_result is None:
            continue

        stats = roi_result.get('stats')  # Fixed: key is 'stats', not 'stats_results'
        if stats is None:
            continue

        # Check for significance
        is_significant = False
        if 'significant_clusters' in stats and len(stats['significant_clusters']) > 0:
            is_significant = True
        elif 'significant_mask_fdr' in stats and np.sum(stats['significant_mask_fdr']) > 0:
            is_significant = True

        if is_significant:
            n_subjects = len(roi_result['roi_dataset']['subjects'])
            significant_rois.append({
                'roi': roi_name,
                'n_subjects': n_subjects,
                'stats': stats,
                'result': roi_result
            })

    # Overall statistics
    total_rois = len(results)
    n_significant = len(significant_rois)
    n_not_significant = total_rois - n_significant

    summary_lines.append("OVERALL RESULTS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"• Total ROIs analyzed: {total_rois}")
    summary_lines.append(f"• ROIs with significant effects: {n_significant} ({100*n_significant/total_rois:.1f}%)")
    summary_lines.append(f"• ROIs without significant effects: {n_not_significant}")
    summary_lines.append("")

    if n_significant == 0:
        summary_lines.append("CONCLUSION:")
        summary_lines.append("-" * 80)
        summary_lines.append(f"• No ROIs show significant differences between Speech and Music")
        summary_lines.append(f"• This suggests similar {band} power responses across conditions")
        summary_lines.append(f"• Consider:")
        summary_lines.append(f"  - Lower precluster threshold for more liberal testing")
        summary_lines.append(f"  - Check individual subject results for heterogeneous patterns")
        summary_lines.append(f"  - Analyze different frequency bands")
    else:
        # Categorize ROIs by anatomical region
        roi_categories = {
            'Temporal': [],
            'Frontal': [],
            'Parietal': [],
            'Insula': [],
            'Other': []
        }

        for roi_info in significant_rois:
            roi = roi_info['roi']
            if 'temporal' in roi.lower() or 'heschl' in roi.lower():
                roi_categories['Temporal'].append(roi_info)
            elif 'frontal' in roi.lower() or 'pars' in roi.lower():
                roi_categories['Frontal'].append(roi_info)
            elif 'parietal' in roi.lower() or 'supramarginal' in roi.lower() or 'angular' in roi.lower():
                roi_categories['Parietal'].append(roi_info)
            elif 'insula' in roi.lower():
                roi_categories['Insula'].append(roi_info)
            else:
                roi_categories['Other'].append(roi_info)

        # Report by anatomical category
        summary_lines.append("SIGNIFICANT ROIs BY ANATOMICAL REGION:")
        summary_lines.append("-" * 80)

        for category, rois in roi_categories.items():
            if len(rois) > 0:
                summary_lines.append(f"\n{category} Cortex ({len(rois)} ROIs):")
                for roi_info in rois:
                    roi_name = roi_info['roi']
                    n_subj = roi_info['n_subjects']

                    # Determine effect direction
                    stats = roi_info['stats']
                    result = roi_info['result']
                    method = result.get('method', 'unknown')

                    if method == 'cluster' and 'significant_clusters' in stats and len(stats['significant_clusters']) > 0:
                        # For cluster method: count clusters and infer direction from t_obs
                        n_clusters = len(stats['significant_clusters'])
                        t_obs = stats.get('t_obs', np.array([]))

                        # Get t-values at significant timepoints
                        sig_t_values = []
                        for cluster in stats['significant_clusters']:
                            time_indices = cluster['time_indices']
                            sig_t_values.extend(t_obs[time_indices])

                        # Determine predominant direction
                        pos_count = sum(1 for t in sig_t_values if t > 0)
                        neg_count = sum(1 for t in sig_t_values if t < 0)

                        if pos_count > neg_count * 1.5:
                            direction = "Speech > Music"
                        elif neg_count > pos_count * 1.5:
                            direction = "Music > Speech"
                        else:
                            direction = "Mixed"

                        summary_lines.append(f"  • {roi_name} (n={n_subj}): {n_clusters} clusters - {direction}")

                    elif method == 'ttest' and 'significant_mask_fdr' in stats:
                        # For t-test method: determine direction from t-values
                        t_values = stats.get('t_values', np.array([]))
                        sig_mask = stats['significant_mask_fdr']
                        sig_t_values = t_values[sig_mask]

                        n_sig = np.sum(sig_mask)

                        if len(sig_t_values) > 0:
                            # Determine predominant direction
                            pos_count = np.sum(sig_t_values > 0)
                            neg_count = np.sum(sig_t_values < 0)

                            if pos_count > neg_count * 1.5:
                                direction = "Speech > Music"
                            elif neg_count > pos_count * 1.5:
                                direction = "Music > Speech"
                            else:
                                direction = "Mixed"

                            summary_lines.append(f"  • {roi_name} (n={n_subj}): {n_sig} timepoints - {direction}")
                        else:
                            summary_lines.append(f"  • {roi_name} (n={n_subj}): Significant effect")

                    else:
                        summary_lines.append(f"  • {roi_name} (n={n_subj}): Significant effect")

        # Interpretation
        summary_lines.append("")
        summary_lines.append("ANATOMICAL INTERPRETATION:")
        summary_lines.append("-" * 80)

        if len(roi_categories['Temporal']) > 0:
            summary_lines.append(f"\n• TEMPORAL REGIONS ({len(roi_categories['Temporal'])} ROIs):")
            summary_lines.append(f"  - Function: Auditory processing, speech perception, music processing")
            summary_lines.append(f"  - Finding: Differential {band} power suggests condition-specific auditory encoding")
            if band == 'theta':
                summary_lines.append(f"  - Theta interpretation: Temporal chunking and segmentation of auditory streams")
            elif band == 'alpha':
                summary_lines.append(f"  - Alpha interpretation: Attentional gating and inhibitory control in auditory cortex")

        if len(roi_categories['Frontal']) > 0:
            summary_lines.append(f"\n• FRONTAL REGIONS ({len(roi_categories['Frontal'])} ROIs):")
            summary_lines.append(f"  - Function: Speech production (Broca's area), motor planning, cognitive control")
            summary_lines.append(f"  - Finding: Differential {band} power suggests condition-specific motor/cognitive engagement")
            if band == 'theta':
                summary_lines.append(f"  - Theta interpretation: Working memory and task-related cognitive control")
            elif band == 'alpha':
                summary_lines.append(f"  - Alpha interpretation: Motor inhibition and preparation states")

        if len(roi_categories['Parietal']) > 0:
            summary_lines.append(f"\n• PARIETAL REGIONS ({len(roi_categories['Parietal'])} ROIs):")
            summary_lines.append(f"  - Function: Multisensory integration, language comprehension (Wernicke's area)")
            summary_lines.append(f"  - Finding: Differential {band} power suggests condition-specific integration processes")
            if band == 'theta':
                summary_lines.append(f"  - Theta interpretation: Semantic processing and comprehension")
            elif band == 'alpha':
                summary_lines.append(f"  - Alpha interpretation: Attention allocation and sensory gating")

        if len(roi_categories['Insula']) > 0:
            summary_lines.append(f"\n• INSULAR REGIONS ({len(roi_categories['Insula'])} ROIs):")
            summary_lines.append(f"  - Function: Interoception, emotional processing, salience detection")
            summary_lines.append(f"  - Finding: Differential {band} power suggests condition-specific salience or affect")

        # Overall conclusion
        summary_lines.append("")
        summary_lines.append("GROUP-LEVEL CONCLUSION:")
        summary_lines.append("-" * 80)

        # Count predominant directions across all significant ROIs
        total_speech_dominant = 0
        total_music_dominant = 0
        total_mixed = 0

        for roi_info in significant_rois:
            stats = roi_info['stats']
            result = roi_info['result']
            method = result.get('method', 'unknown')

            # Determine direction based on method
            if method == 'cluster' and 'significant_clusters' in stats and len(stats['significant_clusters']) > 0:
                # For cluster method: infer direction from t_obs
                t_obs = stats.get('t_obs', np.array([]))
                sig_t_values = []
                for cluster in stats['significant_clusters']:
                    time_indices = cluster['time_indices']
                    sig_t_values.extend(t_obs[time_indices])

                pos_count = sum(1 for t in sig_t_values if t > 0)
                neg_count = sum(1 for t in sig_t_values if t < 0)

                if pos_count > neg_count * 1.5:
                    total_speech_dominant += 1
                elif neg_count > pos_count * 1.5:
                    total_music_dominant += 1
                else:
                    total_mixed += 1

            elif method == 'ttest' and 'significant_mask_fdr' in stats:
                # For t-test method: determine direction from t-values
                t_values = stats.get('t_values', np.array([]))
                sig_mask = stats['significant_mask_fdr']
                sig_t_values = t_values[sig_mask]

                if len(sig_t_values) > 0:
                    pos_count = np.sum(sig_t_values > 0)
                    neg_count = np.sum(sig_t_values < 0)

                    if pos_count > neg_count * 1.5:
                        total_speech_dominant += 1
                    elif neg_count > pos_count * 1.5:
                        total_music_dominant += 1
                    else:
                        total_mixed += 1

        summary_lines.append(f"• {n_significant} ROIs show significant speech vs. music differences in {band} band")

        if total_music_dominant > total_speech_dominant * 1.5:
            summary_lines.append(f"• PREDOMINANT PATTERN: Music > Speech ({total_music_dominant}/{n_significant} ROIs)")
            summary_lines.append(f"• Music stimuli elicit stronger {band} oscillations in perisylvian cortex")
            summary_lines.append(f"• This suggests enhanced neural engagement for musical vs. speech stimuli")
        elif total_speech_dominant > total_music_dominant * 1.5:
            summary_lines.append(f"• PREDOMINANT PATTERN: Speech > Music ({total_speech_dominant}/{n_significant} ROIs)")
            summary_lines.append(f"• Speech stimuli elicit stronger {band} oscillations in perisylvian cortex")
            summary_lines.append(f"• This suggests enhanced neural engagement for speech vs. musical stimuli")
        else:
            summary_lines.append(f"• HETEROGENEOUS PATTERN: Both directions represented")
            summary_lines.append(f"  - Music > Speech: {total_music_dominant} ROIs")
            summary_lines.append(f"  - Speech > Music: {total_speech_dominant} ROIs")
            summary_lines.append(f"  - Mixed: {total_mixed} ROIs")
            summary_lines.append(f"• Different ROIs show distinct functional specialization")

    summary_lines.append("")
    summary_lines.append("RECOMMENDATIONS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"• Examine individual ROI plots in: {output_dir}")
    summary_lines.append(f"• Check summary CSV for detailed statistics")
    summary_lines.append(f"• Consider follow-up analyses:")
    summary_lines.append(f"  - Connectivity analysis between significant ROIs")
    summary_lines.append(f"  - Single-trial variability in high-effect ROIs")
    summary_lines.append(f"  - Cross-frequency coupling (if multiple bands analyzed)")
    summary_lines.append("")
    summary_lines.append("="*80)

    # Create final text
    summary_text = "\n".join(summary_lines)

    # Print to console
    print(summary_text)

    # Save to file
    summary_file = output_dir / f'roi_analysis_conclusion_{band}.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"\n[OK] Summary saved to: {summary_file}")

    return summary_text


def main():
    """
    Main execution function
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='ROI-Specific Group Analysis for iEEG Data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--band',
        type=str,
        choices=['theta', 'alpha'],
        default='theta',
        help="Frequency band to analyze: 'theta' (4-8 Hz) or 'alpha' (8-12 Hz). Default: theta"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help="Directory containing preprocessed data (default: ./results_{band}/preprocessed_data)"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Directory to save results (default: ./roi_group_results_{band})"
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['ttest', 'cluster'],
        default='cluster',
        help="Statistical method: 'ttest' or 'cluster' (default: cluster)"
    )

    parser.add_argument(
        '--min-subjects',
        type=int,
        default=2,
        help="Minimum subjects required per ROI (default: 2)"
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help="Number of permutations for cluster test (default: 5000)"
    )

    parser.add_argument(
        '--precluster-p',
        type=float,
        default=0.1,
        help="Precluster p-value threshold (default: 0.1)"
    )

    args = parser.parse_args()

    # Set default directories based on band if not specified
    BAND = args.band
    PREPROCESSED_DATA_DIR = args.data_dir or f"./results_{BAND}/preprocessed_data"
    OUTPUT_DIR = args.output_dir or f"./roi_group_results_{BAND}"
    METHOD = args.method
    MIN_SUBJECTS = args.min_subjects
    N_PERMUTATIONS = args.n_permutations
    PRECLUSTER_P = args.precluster_p

    print("="*80)
    print(f"ROI-SPECIFIC GROUP ANALYSIS ({BAND.upper()} BAND)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Frequency band: {BAND}")
    print(f"  Preprocessed data: {PREPROCESSED_DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Method: {METHOD}")
    print(f"  Minimum subjects per ROI: {MIN_SUBJECTS}")
    print(f"  Permutations (if cluster): {N_PERMUTATIONS}")
    print(f"  Precluster threshold (if cluster): p={PRECLUSTER_P}")

    # Initialize analyzer
    analyzer = ROIGroupAnalyzer(PREPROCESSED_DATA_DIR, OUTPUT_DIR, band=BAND)

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
        precluster_p=PRECLUSTER_P,
        top_n=None  # Set to a number to limit analysis to top N ROIs
    )

    print(f"\n[OK] Analysis complete!")
    print(f"  Total ROIs analyzed: {len(results)}")
    print(f"  Results saved to: {OUTPUT_DIR}")

    # Generate comprehensive summary and interpretation
    generate_roi_analysis_summary(results, BAND, analyzer.output_dir)


if __name__ == "__main__":
    main()
