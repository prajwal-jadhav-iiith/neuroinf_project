"""
Quick script to inspect data for a specific ROI
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

def inspect_roi(roi_name='ctx-lh-superiortemporal', band='theta'):
    """Inspect data quality and alignment for a specific ROI"""

    data_dir = Path(f'./results_{band}/preprocessed_data')

    print(f"\n{'='*80}")
    print(f"INSPECTING ROI: {roi_name} ({band.upper()} BAND)")
    print(f"{'='*80}\n")

    # Collect data for this ROI across subjects
    speech_data = []
    music_data = []
    subject_ids = []

    for npz_file in sorted(data_dir.glob(f'*_{band}_power.npz')):
        subject_id = npz_file.stem.replace(f'_{band}_power', '')

        # Load data
        data = np.load(npz_file)
        roi_csv = data_dir / f'{subject_id}_roi_mapping.csv'

        if not roi_csv.exists():
            continue

        roi_df = pd.read_csv(roi_csv)

        # Check if this subject has electrodes in the ROI
        roi_channels = roi_df[roi_df['region'] == roi_name]['electrode_name'].tolist()

        if len(roi_channels) == 0:
            continue

        # Get channel indices
        all_channels = list(data['channel_names'])
        channel_indices = [all_channels.index(ch) for ch in roi_channels if ch in all_channels]

        if len(channel_indices) == 0:
            continue

        # Load power data
        speech_key = f'speech_{band}_power'
        music_key = f'music_{band}_power'

        speech_power = data[speech_key][channel_indices, :].mean(axis=0)  # Average across electrodes
        music_power = data[music_key][channel_indices, :].mean(axis=0)

        speech_data.append(speech_power)
        music_data.append(music_power)
        subject_ids.append(subject_id)

        print(f"  {subject_id}: {len(channel_indices)} electrodes in {roi_name}")

    if len(subject_ids) == 0:
        print(f"\nNo subjects found with data in {roi_name}")
        return

    print(f"\nTotal subjects with {roi_name}: {len(subject_ids)}")

    # Stack data
    speech_array = np.array(speech_data)  # (n_subjects, n_times)
    music_array = np.array(music_data)

    print(f"Data shape: {speech_array.shape}")

    # Compute paired t-test at each timepoint
    t_values = np.zeros(speech_array.shape[1])
    p_values = np.zeros(speech_array.shape[1])

    for t_idx in range(speech_array.shape[1]):
        t_val, p_val = stats.ttest_rel(speech_array[:, t_idx], music_array[:, t_idx])
        t_values[t_idx] = t_val
        p_values[t_idx] = p_val

    # Count significant timepoints
    sig_uncorrected = np.sum(p_values < 0.05)
    sig_bonferroni = np.sum(p_values < 0.05/len(p_values))

    print(f"\nUncorrected significance:")
    print(f"  Timepoints with p<0.05: {sig_uncorrected}/{len(p_values)} ({100*sig_uncorrected/len(p_values):.1f}%)")
    print(f"  Max |t-value|: {np.max(np.abs(t_values)):.3f}")
    print(f"  Min p-value: {np.min(p_values):.6f}")

    print(f"\nBonferroni corrected:")
    print(f"  Significant timepoints: {sig_bonferroni}/{len(p_values)}")

    # FDR correction
    from scipy.stats import false_discovery_control
    fdr_adjusted_pvalues = false_discovery_control(p_values, method='bh')
    fdr_rejected = fdr_adjusted_pvalues < 0.05  # Compare to alpha threshold
    sig_fdr = np.sum(fdr_rejected)  # Count significant timepoints

    print(f"\nFDR corrected (Benjamini-Hochberg):")
    print(f"  Significant timepoints: {sig_fdr}/{len(p_values)}")

    # Plot
    times = np.linspace(0, 30, len(t_values))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Mean timecourses
    ax = axes[0]
    speech_mean = speech_array.mean(axis=0)
    speech_sem = speech_array.std(axis=0) / np.sqrt(len(subject_ids))
    music_mean = music_array.mean(axis=0)
    music_sem = music_array.std(axis=0) / np.sqrt(len(subject_ids))

    ax.plot(times, speech_mean, 'b-', label='Speech', linewidth=2)
    ax.fill_between(times, speech_mean - speech_sem, speech_mean + speech_sem, alpha=0.3, color='blue')
    ax.plot(times, music_mean, 'r-', label='Music', linewidth=2)
    ax.fill_between(times, music_mean - music_sem, music_mean + music_sem, alpha=0.3, color='red')
    ax.set_ylabel(f'{band.capitalize()} Power')
    ax.set_title(f'{roi_name} - Group Average (n={len(subject_ids)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: T-values
    ax = axes[1]
    ax.plot(times, t_values, 'k-', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Mark significant regions
    if sig_fdr > 0:
        ax.fill_between(times, -10, 10, where=fdr_rejected, alpha=0.3, color='green', label='FDR sig')

    ax.set_ylabel('T-statistic')
    ax.set_title('Speech vs Music (paired t-test)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: P-values
    ax = axes[2]
    ax.plot(times, -np.log10(p_values), 'k-', linewidth=1.5)
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(-np.log10(0.05/len(p_values)), color='red', linestyle='--', alpha=0.7, label='Bonferroni')
    ax.set_ylabel('-log10(p-value)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Statistical Significance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'inspect_{roi_name}_{band}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: inspect_{roi_name}_{band}.png")
    plt.close()

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS:")
    print(f"{'='*80}")

    if sig_fdr > 0:
        print(f"GOOD: {sig_fdr} timepoints show FDR-corrected significance")
        print(f"  This ROI should show significant effects in group analysis")
        print(f"  If roi_group_analysis.py finds nothing, the precluster threshold is too strict")
    elif sig_uncorrected > len(p_values) * 0.1:
        print(f"MARGINAL: {sig_uncorrected} timepoints significant (uncorrected)")
        print(f"  No FDR-corrected significance, but trends present")
        print(f"  Try: --precluster-p 0.3 or --method ttest")
    else:
        print(f"WEAK: Only {sig_uncorrected} timepoints significant (uncorrected)")
        print(f"  This ROI shows weak group-level effects")
        print(f"  This is normal - effects may be heterogeneous across subjects")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    import sys
    roi = sys.argv[1] if len(sys.argv) > 1 else 'ctx-lh-superiortemporal'
    band = sys.argv[2] if len(sys.argv) > 2 else 'theta'
    inspect_roi(roi, band)
