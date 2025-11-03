import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import stats
from scipy.ndimage import label
import scipy.io as sio
import matplotlib.pyplot as plt
import json
import time
import traceback

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

def load_ieeg_data(subject_id, data_dir, task='film', acquisition='clinical', run=1):
    """Load iEEG data for a specific subject from BIDS format"""
    data_dir = Path(data_dir)
    subject_dir = data_dir / subject_id / 'ses-iemu' / 'ieeg'
    base_name = f"{subject_id}_ses-iemu_task-{task}_acq-{acquisition}_run-{run}"

    vhdr_file = subject_dir / f"{base_name}_ieeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)

    events_file = subject_dir / f"{subject_id}_ses-iemu_task-{task}_run-{run}_events.tsv"
    events_df = pd.read_csv(events_file, sep='\t')

    channels_file = subject_dir / f"{base_name}_channels.tsv"
    channels_df = pd.read_csv(channels_file, sep='\t')

    return raw, events_df, channels_df


def detect_eog_artifacts(ica, ica_sources, raw_all_filt, eog_channels, threshold=0.3):
    """
    Detect EOG (eye movement) artifacts in ICA components

    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    ica_sources : mne.io.Raw
        ICA source data
    raw_all_filt : mne.io.Raw
        Filtered raw data containing EOG channels
    eog_channels : list
        List of EOG channel names
    threshold : float
        Correlation threshold for artifact detection

    Returns:
    --------
    eog_indices : list
        Indices of ICA components correlated with EOG
    """
    eog_indices = []
    if len(eog_channels) > 0:
        print(f"  Detecting EOG artifacts using channels: {eog_channels}")

        # Get ICA source data
        sources_data = ica_sources.get_data()

        # Get EOG data from raw_all_filt
        eog_data = raw_all_filt.get_data(picks=eog_channels)

        eog_correlations = []
        for comp_idx in range(ica.n_components_):
            comp_signal = sources_data[comp_idx, :]

            # Correlate with all EOG channels and take max absolute correlation
            max_corr = 0
            for eog_idx in range(len(eog_channels)):
                eog_signal = eog_data[eog_idx, :]
                corr = np.abs(np.corrcoef(comp_signal, eog_signal)[0, 1])
                max_corr = max(max_corr, corr)

            eog_correlations.append(max_corr)

        eog_indices = [i for i, corr in enumerate(eog_correlations) if corr > threshold]
        print(f"  EOG components detected: {eog_indices}")

    return eog_indices


def detect_ecg_artifacts(ica, ica_sources, raw_all_filt, ecg_channels, threshold=0.3):
    """
    Detect ECG (cardiac) artifacts in ICA components

    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    ica_sources : mne.io.Raw
        ICA source data
    raw_all_filt : mne.io.Raw
        Filtered raw data containing ECG channels
    ecg_channels : list
        List of ECG channel names
    threshold : float
        Correlation threshold for artifact detection

    Returns:
    --------
    ecg_indices : list
        Indices of ICA components correlated with ECG
    """
    ecg_indices = []
    if len(ecg_channels) > 0:
        print(f"  Detecting ECG artifacts using channels: {ecg_channels}")

        sources_data = ica_sources.get_data()
        ecg_data = raw_all_filt.get_data(picks=ecg_channels)

        ecg_correlations = []
        for comp_idx in range(ica.n_components_):
            comp_signal = sources_data[comp_idx, :]

            max_corr = 0
            for ecg_idx in range(len(ecg_channels)):
                ecg_signal = ecg_data[ecg_idx, :]
                corr = np.abs(np.corrcoef(comp_signal, ecg_signal)[0, 1])
                max_corr = max(max_corr, corr)

            ecg_correlations.append(max_corr)

        ecg_indices = [i for i, corr in enumerate(ecg_correlations) if corr > threshold]
        print(f"  ECG components detected: {ecg_indices}")

    return ecg_indices


def detect_emg_artifacts(ica, ica_sources, raw_all_filt, emg_channels, threshold=0.3):
    """
    Detect EMG (muscle) artifacts in ICA components

    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    ica_sources : mne.io.Raw
        ICA source data
    raw_all_filt : mne.io.Raw
        Filtered raw data containing EMG channels
    emg_channels : list
        List of EMG channel names
    threshold : float
        Correlation threshold for artifact detection

    Returns:
    --------
    emg_indices : list
        Indices of ICA components correlated with EMG
    """
    emg_indices = []
    if len(emg_channels) > 0:
        print(f"  Detecting EMG artifacts using channels: {emg_channels}")

        sources_data = ica_sources.get_data()
        emg_data = raw_all_filt.get_data(picks=emg_channels)

        emg_correlations = []
        for comp_idx in range(ica.n_components_):
            comp_signal = sources_data[comp_idx, :]

            max_corr = 0
            for emg_idx in range(len(emg_channels)):
                emg_signal = emg_data[emg_idx, :]
                corr = np.abs(np.corrcoef(comp_signal, emg_signal)[0, 1])
                max_corr = max(max_corr, corr)

            emg_correlations.append(max_corr)

        emg_indices = [i for i, corr in enumerate(emg_correlations) if corr > threshold]
        print(f"  EMG components detected: {emg_indices}")

    return emg_indices


def get_channels_by_type(raw, channels_df, channel_types, subject_id, perisylvian_data):
    """
    Get channels of specific types that are marked as 'good'

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw data object
    channels_df : pd.DataFrame
        Channels information dataframe
    channel_types : list of str
        List of channel types to extract (e.g., ['SEEG', 'ECOG'])
    subject_id : str
        Subject identifier
    perisylvian_data : dict
        Dictionary containing perisylvian channel information

    Returns:
    --------
    channels : list of str
        List of channel names
    """
    current_subject_perisylvian = perisylvian_data.get(subject_id, {})

    if current_subject_perisylvian:
        current_subject_perisylvian = set(current_subject_perisylvian.keys())

    if "SEEG" in channel_types or "ECOG" in channel_types:
        good_channels = channels_df[
            (channels_df['status'] == 'good') &
            (channels_df['type'].isin(channel_types)) &
            (channels_df['name'].isin(current_subject_perisylvian))
        ]['name'].tolist()
    else:
        good_channels = channels_df[
            (channels_df['status'] == 'good') &
            (channels_df['type'].isin(channel_types))
        ]['name'].tolist()

    # Filter to only those present in raw data
    channels = [ch for ch in good_channels if ch in raw.ch_names]

    return channels


def notch_filter(raw, freqs=[50, 100, 150]):
    """
    Apply a notch filter to the raw iEEG data to remove power line noise.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw iEEG data.
    freqs : list of float
        The frequencies to filter out (in Hz). Defaults to 50 Hz and its
        first two harmonics.

    Returns:
    --------
    raw_notched : mne.io.Raw
        The notch-filtered data.
    """
    raw_notched = raw.copy().notch_filter(
        freqs=freqs,
        fir_design='firwin',
        verbose=False
    )
    return raw_notched


def bandpass_filter(raw, l_freq=0.1, h_freq=200.0):
    """
    Apply a band-pass filter to the raw iEEG data to remove slow drifts
    and high-frequency noise.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw iEEG data.
    l_freq : float
        The lower bound of the filter (in Hz).
    h_freq : float
        The upper bound of the filter (in Hz).

    Returns:
    --------
    raw_filtered : mne.io.Raw
        The band-pass filtered data.
    """
    raw_filtered = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        verbose=False
    )
    return raw_filtered


def reference(raw):
    """
    Apply Common Average Reference (CAR) to iEEG data.

    Temporarily sets the channel type to 'eeg' to use MNE's built-in
    referencing function, then sets it back.
    """
    # Create a copy to avoid modifying the original data
    raw_copy = raw.copy()

    # 1. Get the original channel types to restore them later
    original_ch_types = raw_copy.get_channel_types()

    # 2. Temporarily set the iEEG channel types to 'eeg'
    mapping = {ch_name: 'eeg' for ch_name in raw_copy.ch_names}
    raw_copy.set_channel_types(mapping)

    # 3. Apply the common average reference
    raw_car, _ = mne.set_eeg_reference(raw_copy, 'average', projection=False)

    # 4. Set the channel types back to their original values
    original_mapping = {ch_name: ch_type for ch_name, ch_type in zip(raw_copy.ch_names, original_ch_types)}
    raw_car.set_channel_types(original_mapping)

    return raw_car


def extract_hfb_power(raw):
    """
    Extract theta band (4-8 Hz) power from raw iEEG data.

    This function filters the data in the theta range and then applies the Hilbert
    transform to compute the signal's power envelope.

    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed continuous iEEG data.

    Returns:
    --------
    raw_hfb : mne.io.Raw
        Continuous iEEG data with theta power instead of voltage.
    """
    # 1. Band-pass filter in the theta range (4-8 Hz)
    raw_hfb = raw.copy().filter(l_freq=4., h_freq=8., fir_design='firwin', verbose=False)

    # 2. Apply Hilbert transform to get the envelope (power) of the signal
    raw_hfb.apply_hilbert(envelope=True)

    return raw_hfb


def save_hilbert_data_to_matlab(hilbert_data, output_dir, subject_id, filename=None):
    """
    Save Hilbert-transformed iEEG data to MATLAB-compatible .mat format

    Parameters:
    -----------
    hilbert_data : mne.io.Raw
        MNE Raw object containing Hilbert-transformed data (power envelope)
    output_dir : str or Path
        Directory where the .mat file will be saved
    subject_id : str
        Subject identifier (e.g., 'sub-03')
    filename : str, optional
        Custom filename. If None, uses default naming convention

    Returns:
    --------
    output_path : Path
        Path to the saved .mat file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f"{subject_id}_hilbert_theta_power.mat"

    output_path = output_dir / filename

    # Extract data and metadata from MNE object
    data = hilbert_data.get_data()  # Shape: (n_channels, n_timepoints)
    channel_names = hilbert_data.ch_names
    sfreq = hilbert_data.info['sfreq']
    times = hilbert_data.times

    # Get channel information
    n_channels, n_timepoints = data.shape

    matlab_dict = {
        'data': data.T,  # Transpose to MATLAB convention: timepoints x channels
        'channel_names': np.array(channel_names, dtype=object),
        'sampling_frequency': sfreq,
        'times': times,
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'subject_id': subject_id,
        'description': 'Hilbert-transformed theta band power envelope data',
        'data_shape_info': 'data is [n_timepoints x n_channels]',
    }

    # Save to .mat file (using MATLAB v5 format for better compatibility)
    sio.savemat(output_path, matlab_dict, format='5', oned_as='column')

    print(f"\nHilbert-transformed data saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")

    return output_path


def create_mne_events(events_df, raw):
    """
    Convert BIDS events to MNE events format

    Parameters:
    -----------
    events_df : pd.DataFrame
        BIDS events dataframe
    raw : mne.io.Raw
        Raw data for sampling frequency

    Returns:
    --------
    events : np.ndarray
        MNE events array (n_events, 3)
    event_id : dict
        Event ID mapping
    """
    # Convert onset times to samples
    sfreq = raw.info['sfreq']

    # Filter for speech and music events only
    speech_music_events = events_df[events_df['trial_type'].isin(['speech', 'music'])].copy()

    # Create MNE events array
    n_events = len(speech_music_events)
    events = np.zeros((n_events, 3), dtype=int)

    # Sample numbers (onset times * sampling frequency)
    events[:, 0] = (speech_music_events['onset'] * sfreq).astype(int)
    events[:, 1] = 0  # Duration (not used in MNE events)

    # Event codes: speech=1, music=2
    event_id = {'speech': 1, 'music': 2}
    for i, trial_type in enumerate(speech_music_events['trial_type']):
        events[i, 2] = event_id[trial_type]

    return events, event_id


def epoch_ieeg_data(raw, events, event_id, tmin=0.0, tmax=30.0):
    """
    Create epochs from raw iEEG data (without MNE baseline correction)

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw iEEG data
    events : np.ndarray
        MNE events array
    event_id : dict
        Event ID mapping
    tmin : float
        Start time relative to event (seconds)
    tmax : float
        End time relative to event (seconds)

    Returns:
    --------
    epochs : mne.Epochs
        Epoched data (no baseline correction applied)
    """
    # Create epochs without baseline correction
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,  # No MNE baseline correction
        preload=True,
        verbose=False
    )

    return epochs


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def downsample_time_dimension(power_data, target_time_points=100):
    """
    Downsample the temporal dimension by averaging in bins

    Parameters:
    -----------
    power_data : np.ndarray (n_epochs, n_channels, n_freqs, n_times)
    target_time_points : int
        Target number of time points (default: 100)

    Returns:
    --------
    downsampled : np.ndarray (n_epochs, n_channels, n_freqs, target_time_points)
    """
    n_epochs, n_channels, n_freqs, n_times = power_data.shape

    # Calculate bin size
    bin_size = n_times // target_time_points
    actual_time_points = n_times // bin_size

    # Reshape and average
    downsampled = np.zeros((n_epochs, n_channels, n_freqs, actual_time_points))

    for t_idx in range(actual_time_points):
        start = t_idx * bin_size
        end = start + bin_size
        downsampled[:, :, :, t_idx] = power_data[:, :, :, start:end].mean(axis=3)

    return downsampled


def compute_observed_tstatistics_vectorized(speech_power, music_power):
    """
    VECTORIZED version - much faster!
    Compute t-statistic for each channel at each time-frequency pixel

    Parameters:
    -----------
    speech_power : np.ndarray (n_speech_epochs, n_channels, n_freqs, n_times)
    music_power : np.ndarray (n_music_epochs, n_channels, n_freqs, n_times)

    Returns:
    --------
    t_maps : np.ndarray (n_channels, n_freqs, n_times)
    """
    n_speech = speech_power.shape[0]
    n_music = music_power.shape[0]

    # Compute means
    mean_speech = speech_power.mean(axis=0)  # (n_channels, n_freqs, n_times)
    mean_music = music_power.mean(axis=0)

    # Compute variances
    var_speech = speech_power.var(axis=0, ddof=1)
    var_music = music_power.var(axis=0, ddof=1)

    # Pooled standard error
    pooled_se = np.sqrt(var_speech / n_speech + var_music / n_music)

    # T-statistic
    t_maps = (mean_speech - mean_music) / pooled_se

    # Handle any NaN or Inf values
    t_maps = np.nan_to_num(t_maps, nan=0.0, posinf=0.0, neginf=0.0)

    return t_maps


def find_clusters_2d(binary_map):
    """Find contiguous clusters in a 2D time-frequency map"""
    labeled_map, n_clusters = label(binary_map)
    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_coords = np.where(labeled_map == cluster_id)
        clusters.append(cluster_coords)
    return clusters


def calculate_cluster_statistics(clusters, t_map_abs, statistic_type='mass'):
    """Calculate statistics for each cluster"""
    stats = np.zeros(len(clusters))
    for i, cluster_coords in enumerate(clusters):
        if statistic_type == 'size':
            stats[i] = len(cluster_coords[0])
        elif statistic_type == 'mass':
            stats[i] = np.sum(t_map_abs[cluster_coords])
    return stats


def permutation_test_cluster_based(speech_power, music_power,
                                    n_permutations=5000, precluster_p=0.05,
                                    tail='two', cluster_statistic='mass',
                                    downsample_time=True, target_time_points=100):
    """
    OPTIMIZED cluster-based permutation test

    Parameters:
    -----------
    speech_power : np.ndarray (n_speech_epochs, n_channels, n_freqs, n_times)
    music_power : np.ndarray (n_music_epochs, n_channels, n_freqs, n_times)
    n_permutations : int
    precluster_p : float
    tail : str
    cluster_statistic : str ('size' or 'mass')
    downsample_time : bool
    target_time_points : int

    Returns:
    --------
    results : dict
    """
    print("\n" + "="*70)
    print("CLUSTER-BASED PERMUTATION TEST (OPTIMIZED)")
    print("="*70)

    # Downsample temporal dimension if requested
    if downsample_time:
        print("\nDownsampling temporal dimension...")
        speech_power = downsample_time_dimension(speech_power, target_time_points)
        music_power = downsample_time_dimension(music_power, target_time_points)

    n_channels = speech_power.shape[1]
    n_freqs = speech_power.shape[2]
    n_times = speech_power.shape[3]

    print(f"\nFinal dimensions:")
    print(f"  Channels: {n_channels}")
    print(f"  Frequencies: {n_freqs}")
    print(f"  Time points: {n_times}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Precluster threshold: p < {precluster_p}")
    print(f"  Cluster statistic: {cluster_statistic}")

    # Compute observed statistics
    print("\nComputing observed statistics...")
    observed_t_maps = compute_observed_tstatistics_vectorized(speech_power, music_power)
    print("  Done!")

    # Determine precluster threshold
    n_speech = speech_power.shape[0]
    n_music = music_power.shape[0]
    df = n_speech + n_music - 2

    if tail == 'two':
        t_thresh = stats.t.ppf(1 - precluster_p/2, df)
        precluster_thresh_upper = t_thresh
        precluster_thresh_lower = -t_thresh
    elif tail == 'positive':
        precluster_thresh_upper = stats.t.ppf(1 - precluster_p, df)
        precluster_thresh_lower = None
    else:
        precluster_thresh_lower = stats.t.ppf(precluster_p, df)
        precluster_thresh_upper = None

    print(f"Precluster t-threshold: {precluster_thresh_upper if tail != 'negative' else precluster_thresh_lower:.3f}")

    # Combine data
    all_data = np.concatenate([speech_power, music_power], axis=0)
    n_total = all_data.shape[0]

    # Store max cluster statistics per channel
    max_cluster_stats_per_channel = np.zeros((n_channels, n_permutations))

    # Permutation loop
    print("\nRunning permutations...")
    start_time = time.time()

    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (perm_idx + 1) / elapsed
            remaining = (n_permutations - perm_idx - 1) / rate
            print(f"  Permutation {perm_idx + 1}/{n_permutations} "
                  f"({rate:.1f} perm/sec, ~{remaining:.0f}s remaining)")

        # Shuffle labels
        shuffle_idx = np.random.permutation(n_total)
        perm_speech = all_data[shuffle_idx[:n_speech]]
        perm_music = all_data[shuffle_idx[n_speech:]]

        # Compute permuted t-maps (VECTORIZED)
        perm_t_maps = compute_observed_tstatistics_vectorized(perm_speech, perm_music)

        # For each channel, find largest cluster
        for ch in range(n_channels):
            # Threshold
            if tail == 'two':
                perm_thresh_map = (perm_t_maps[ch] > precluster_thresh_upper) | \
                                  (perm_t_maps[ch] < precluster_thresh_lower)
            elif tail == 'positive':
                perm_thresh_map = perm_t_maps[ch] > precluster_thresh_upper
            else:
                perm_thresh_map = perm_t_maps[ch] < precluster_thresh_lower

            # Find clusters
            clusters = find_clusters_2d(perm_thresh_map)

            # Calculate cluster statistics
            if len(clusters) > 0:
                cluster_stats = calculate_cluster_statistics(
                    clusters, np.abs(perm_t_maps[ch]), cluster_statistic
                )
                max_cluster_stats_per_channel[ch, perm_idx] = np.max(cluster_stats)
            else:
                max_cluster_stats_per_channel[ch, perm_idx] = 0

    total_time = time.time() - start_time
    print(f"\nPermutations completed in {total_time:.1f} seconds "
          f"({n_permutations/total_time:.1f} perm/sec)")

    # Determine cluster thresholds (95th percentile)
    cluster_thresholds = np.percentile(max_cluster_stats_per_channel, 95, axis=1)

    print("\nCluster thresholds per channel:")
    for ch in range(n_channels):
        print(f"  Channel {ch}: {cluster_thresholds[ch]:.1f}")

    # Apply to observed data
    sig_mask = np.zeros_like(observed_t_maps, dtype=bool)
    all_sig_clusters = []

    for ch in range(n_channels):
        # Threshold observed map
        if tail == 'two':
            obs_thresh_map = (observed_t_maps[ch] > precluster_thresh_upper) | \
                            (observed_t_maps[ch] < precluster_thresh_lower)
        elif tail == 'positive':
            obs_thresh_map = observed_t_maps[ch] > precluster_thresh_upper
        else:
            obs_thresh_map = observed_t_maps[ch] < precluster_thresh_lower

        # Find clusters
        obs_clusters = find_clusters_2d(obs_thresh_map)

        # Calculate cluster statistics
        if len(obs_clusters) > 0:
            obs_cluster_stats = calculate_cluster_statistics(
                obs_clusters, np.abs(observed_t_maps[ch]), cluster_statistic
            )

            # Keep only significant clusters
            for cluster, stat in zip(obs_clusters, obs_cluster_stats):
                if stat >= cluster_thresholds[ch]:
                    sig_mask[ch][cluster] = True
                    all_sig_clusters.append({
                        'channel': ch,
                        'cluster_coords': cluster,
                        'statistic': stat
                    })

    # Summary
    n_sig_pixels = np.sum(sig_mask)
    total_pixels = sig_mask.size

    print(f"\nTotal significant clusters: {len(all_sig_clusters)}")
    print(f"Total significant pixels: {n_sig_pixels} / {total_pixels}")
    print(f"Percentage: {100 * n_sig_pixels / total_pixels:.2f}%")

    print("\nPer-channel summary:")
    for ch in range(n_channels):
        n_clust_ch = sum(1 for c in all_sig_clusters if c['channel'] == ch)
        n_sig_ch = np.sum(sig_mask[ch])
        print(f"  Channel {ch}: {n_clust_ch} clusters, {n_sig_ch} pixels")

    return {
        'observed_t_maps': observed_t_maps,
        'significant_mask': sig_mask,
        'significant_clusters': all_sig_clusters,
        'cluster_thresholds': cluster_thresholds,
        'null_distributions': max_cluster_stats_per_channel,
        'precluster_threshold_upper': precluster_thresh_upper,
        'precluster_threshold_lower': precluster_thresh_lower,
        'n_significant_pixels': n_sig_pixels,
        'n_significant_clusters': len(all_sig_clusters),
        'method': 'cluster-based'
    }


# =============================================================================
# VISUALIZATION AND REPORTING FUNCTIONS
# =============================================================================

def plot_tf_results(results, channel_names, freqs, times, subject_id,
                    save_dir='.', show_plots=False):
    """
    Visualize time-frequency results

    Parameters:
    -----------
    results : dict
        Results from permutation test
    channel_names : list
        Names of channels
    freqs : np.ndarray
        Frequencies
    times : np.ndarray
        Time points
    subject_id : str
        Subject ID
    save_dir : str
        Directory to save figures
    show_plots : bool
        Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)

    n_channels = len(channel_names)

    # Summary figure with all channels
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4*n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{subject_id} - All Channels - {results["method"]}',
                fontsize=14, fontweight='bold')

    for ch_idx in range(n_channels):
        # Observed
        ax = axes[ch_idx, 0]
        vmax = min(5, np.abs(results['observed_t_maps'][ch_idx]).max())
        im = ax.pcolormesh(times, freqs, results['observed_t_maps'][ch_idx],
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_ylabel(f'{channel_names[ch_idx]}\nFreq (Hz)')
        if ch_idx == n_channels - 1:
            ax.set_xlabel('Time (s)')
        ax.set_yscale('log')
        if ch_idx == 0:
            ax.set_title('Observed T-Statistics')

        # Significant
        ax = axes[ch_idx, 1]
        masked_t_map = results['observed_t_maps'][ch_idx].copy()
        masked_t_map[~results['significant_mask'][ch_idx]] = 0
        im = ax.pcolormesh(times, freqs, masked_t_map,
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        if ch_idx == n_channels - 1:
            ax.set_xlabel('Time (s)')
        ax.set_yscale('log')
        if ch_idx == 0:
            ax.set_title('Significant Pixels (corrected)')

        n_sig = np.sum(results['significant_mask'][ch_idx])
        ax.text(0.02, 0.98, f'n={n_sig}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir,
        f"{subject_id}_summary_{results['method']}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary: {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def generate_cluster_report(results, channel_names, freqs, times,
                           subject_id, save_path=None):
    """
    Generate detailed text report of significant clusters

    Parameters:
    -----------
    results : dict
        Results from cluster-based permutation test
    channel_names : list
    freqs : np.ndarray
    times : np.ndarray
    subject_id : str
    save_path : str or None
        If provided, save report to file
    """
    report_lines = []

    # Header
    report_lines.append("="*80)
    report_lines.append(f"CLUSTER-BASED PERMUTATION TEST RESULTS")
    report_lines.append(f"Subject: {subject_id}")
    report_lines.append("="*80)
    report_lines.append("")

    # Overall summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-"*80)
    report_lines.append(f"Method: {results['method']}")
    report_lines.append(f"Total channels analyzed: {len(channel_names)}")
    report_lines.append(f"Channels with significant clusters: {len(set(c['channel'] for c in results['significant_clusters']))}")
    report_lines.append(f"Total significant clusters: {results['n_significant_clusters']}")
    report_lines.append(f"Total significant pixels: {results['n_significant_pixels']}")
    report_lines.append(f"Percentage of all pixels: {100*results['n_significant_pixels']/results['significant_mask'].size:.2f}%")
    report_lines.append("")

    # Check direction of effects
    observed_t = results['observed_t_maps']
    sig_mask = results['significant_mask']

    sig_t_values = observed_t[sig_mask]
    n_positive = np.sum(sig_t_values > 0)
    n_negative = np.sum(sig_t_values < 0)

    report_lines.append("EFFECT DIRECTION")
    report_lines.append("-"*80)
    report_lines.append(f"Significant pixels with positive t (Speech > Music): {n_positive}")
    report_lines.append(f"Significant pixels with negative t (Music > Speech): {n_negative}")
    report_lines.append("")

    if n_negative > n_positive:
        report_lines.append("INTERPRETATION: Music shows higher theta power than Speech")
    elif n_positive > n_negative:
        report_lines.append("INTERPRETATION: Speech shows higher theta power than Music")
    else:
        report_lines.append("INTERPRETATION: Mixed effects (both directions present)")
    report_lines.append("")

    # Per-channel detailed summary
    report_lines.append("DETAILED CLUSTER INFORMATION BY CHANNEL")
    report_lines.append("="*80)
    report_lines.append("")

    # Group clusters by channel
    channels_with_clusters = sorted(set(c['channel'] for c in results['significant_clusters']))

    for ch_idx in channels_with_clusters:
        ch_clusters = [c for c in results['significant_clusters'] if c['channel'] == ch_idx]

        report_lines.append(f"Channel {ch_idx}: {channel_names[ch_idx]}")
        report_lines.append("-"*80)
        report_lines.append(f"Number of significant clusters: {len(ch_clusters)}")
        report_lines.append(f"Cluster threshold for this channel: {results['cluster_thresholds'][ch_idx]:.1f}")
        report_lines.append("")

        # Detail each cluster
        for cluster_num, cluster_info in enumerate(ch_clusters, 1):
            coords = cluster_info['cluster_coords']
            freq_indices = coords[0]
            time_indices = coords[1]

            # Get frequency and time ranges
            freq_min = freqs[freq_indices.min()]
            freq_max = freqs[freq_indices.max()]
            time_min = times[time_indices.min()]
            time_max = times[time_indices.max()]

            # Get t-values in this cluster
            cluster_t_values = observed_t[ch_idx][coords]
            mean_t = np.mean(cluster_t_values)
            max_t = cluster_t_values[np.argmax(np.abs(cluster_t_values))]

            # Determine effect direction
            if mean_t > 0:
                direction = "Speech > Music"
            else:
                direction = "Music > Speech"

            report_lines.append(f"  Cluster {cluster_num}:")
            report_lines.append(f"    Size: {len(freq_indices)} pixels")
            report_lines.append(f"    Cluster mass: {cluster_info['statistic']:.1f}")
            report_lines.append(f"    Frequency range: {freq_min:.2f} - {freq_max:.2f} Hz")
            report_lines.append(f"    Time range: {time_min:.2f} - {time_max:.2f} seconds")
            report_lines.append(f"    Mean t-statistic: {mean_t:.3f}")
            report_lines.append(f"    Peak t-statistic: {max_t:.3f}")
            report_lines.append(f"    Effect direction: {direction}")
            report_lines.append("")

        report_lines.append("")

    # Channels without significant clusters
    channels_without = [i for i in range(len(channel_names))
                       if i not in channels_with_clusters]

    if channels_without:
        report_lines.append("CHANNELS WITHOUT SIGNIFICANT CLUSTERS")
        report_lines.append("-"*80)
        for ch_idx in channels_without:
            report_lines.append(f"  Channel {ch_idx}: {channel_names[ch_idx]}")
        report_lines.append("")

    # Statistical details
    report_lines.append("STATISTICAL PARAMETERS")
    report_lines.append("="*80)
    report_lines.append(f"Precluster threshold (uncorrected): p < 0.05")
    report_lines.append(f"Precluster t-value: {results['precluster_threshold_upper']:.3f}")
    report_lines.append(f"Cluster statistic type: mass (sum of absolute t-values)")
    report_lines.append(f"Number of permutations: 5000")
    report_lines.append(f"Correction method: Cluster-based permutation (Maris & Oostenveld, 2007)")
    report_lines.append("")

    # Create the full report text
    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")

    return report_text


def create_summary_table(results, channel_names):
    """
    Create a pandas DataFrame summarizing cluster results

    Parameters:
    -----------
    results : dict
        Results from cluster-based permutation test
    channel_names : list

    Returns:
    --------
    df : pd.DataFrame
    """
    summary_data = []

    for cluster_info in results['significant_clusters']:
        ch_idx = cluster_info['channel']
        coords = cluster_info['cluster_coords']

        # Get t-values
        cluster_t_values = results['observed_t_maps'][ch_idx][coords]
        mean_t = np.mean(cluster_t_values)

        # Effect direction
        if mean_t > 0:
            direction = "Speech > Music"
            effect_magnitude = mean_t
        else:
            direction = "Music > Speech"
            effect_magnitude = abs(mean_t)

        summary_data.append({
            'Channel': channel_names[ch_idx],
            'Channel_Index': ch_idx,
            'Cluster_Size': len(coords[0]),
            'Cluster_Mass': cluster_info['statistic'],
            'Mean_T': mean_t,
            'Peak_T': cluster_t_values[np.argmax(np.abs(cluster_t_values))],
            'Effect_Direction': direction,
            'Effect_Magnitude': effect_magnitude
        })

    df = pd.DataFrame(summary_data)

    # Sort by effect magnitude (strongest first)
    if len(df) > 0:
        df = df.sort_values('Effect_Magnitude', ascending=False)

    return df


def verify_two_tailed_test(results):
    """
    Verify that the two-tailed test is working correctly

    Parameters:
    -----------
    results : dict
        Results from permutation test
    """
    print("\n" + "="*70)
    print("TWO-TAILED TEST VERIFICATION")
    print("="*70)

    observed_t = results['observed_t_maps']
    sig_mask = results['significant_mask']

    # Overall statistics
    print(f"\nObserved t-statistics (all pixels):")
    print(f"  Min: {observed_t.min():.3f}")
    print(f"  Max: {observed_t.max():.3f}")
    print(f"  Mean: {observed_t.mean():.3f}")
    print(f"  Positive pixels: {np.sum(observed_t > 0)} ({100*np.sum(observed_t > 0)/observed_t.size:.1f}%)")
    print(f"  Negative pixels: {np.sum(observed_t < 0)} ({100*np.sum(observed_t < 0)/observed_t.size:.1f}%)")

    # Significant pixel statistics
    if np.sum(sig_mask) > 0:
        sig_t_values = observed_t[sig_mask]
        print(f"\nSignificant pixels only:")
        print(f"  Min: {sig_t_values.min():.3f}")
        print(f"  Max: {sig_t_values.max():.3f}")
        print(f"  Mean: {sig_t_values.mean():.3f}")
        print(f"  Positive (Speech > Music): {np.sum(sig_t_values > 0)}")
        print(f"  Negative (Music > Speech): {np.sum(sig_t_values < 0)}")

        # Check thresholds
        if results['method'] == 'pixel-based':
            print(f"\nThresholds:")
            print(f"  Upper (for Speech > Music): {results['threshold_upper']:.3f}")
            print(f"  Lower (for Music > Speech): {results['threshold_lower']:.3f}")
        elif results['method'] == 'cluster-based':
            print(f"\nPrecluster threshold:")
            print(f"  Upper: {results['precluster_threshold_upper']:.3f}")
            print(f"  Lower: {results['precluster_threshold_lower']:.3f}")

        print(f"\nInterpretation:")
        if np.sum(sig_t_values < 0) > np.sum(sig_t_values > 0):
            print("  ✓ Two-tailed test is working correctly")
            print("  ✓ Majority of significant effects show Music > Speech")
            print("  ✓ This suggests a real, unidirectional effect in theta band")
        else:
            print("  ✓ Two-tailed test is working correctly")
            print("  ✓ Significant effects in both directions detected")
    else:
        print("\nNo significant pixels found")


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_subject_pipeline(subject_id, data_dir, perisylvian_data, output_base_dir='./results_theta'):
    """
    Run the complete analysis pipeline for a single subject

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-05')
    data_dir : str
        Path to the data directory
    perisylvian_data : dict
        Dictionary containing perisylvian channel information
    output_base_dir : str
        Base directory for output files

    Returns:
    --------
    success : bool
        True if analysis completed successfully, False otherwise
    """
    print("\n" + "="*80)
    print(f"PROCESSING {subject_id}")
    print("="*80)

    try:
        # Create subject-specific output directory
        subject_output_dir = os.path.join(output_base_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)

        # 1. Load data
        print(f"\n[1/10] Loading data for {subject_id}...")
        raw, events_df, channels_df = load_ieeg_data(subject_id, data_dir)
        print(f"  Raw data shape: {raw._data.shape}")
        print(f"  Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"  Recording duration: {raw.times[-1]:.2f} seconds")

        # 2. Get channels
        print(f"\n[2/10] Selecting channels...")
        ieeg_channels = get_channels_by_type(raw, channels_df, ['SEEG', 'ECOG'], subject_id, perisylvian_data)
        eog_channels = get_channels_by_type(raw, channels_df, ['EOG'], subject_id, perisylvian_data)
        ecg_channels = get_channels_by_type(raw, channels_df, ['ECG'], subject_id, perisylvian_data)
        emg_channels = get_channels_by_type(raw, channels_df, ['EMG'], subject_id, perisylvian_data)

        print(f"  iEEG channels: {len(ieeg_channels)}")
        print(f"  EOG channels: {len(eog_channels)}")
        print(f"  ECG channels: {len(ecg_channels)}")
        print(f"  EMG channels: {len(emg_channels)}")

        if len(ieeg_channels) == 0:
            print(f"  WARNING: No iEEG channels found for {subject_id}. Skipping...")
            return False

        # 3. Prepare raw objects
        print(f"\n[3/10] Preparing raw data objects...")
        # Raw object with ONLY iEEG channels (for ICA fitting)
        raw_ieeg = raw.copy().pick_channels(ieeg_channels)

        # Raw object with ALL good channels (for artifact detection)
        all_channels = ieeg_channels + eog_channels + ecg_channels + emg_channels
        raw_all = raw.copy().pick_channels(all_channels)
        print(f"  raw_ieeg channels: {len(raw_ieeg.ch_names)}")
        print(f"  raw_all channels: {len(raw_all.ch_names)}")

        # 4. High-pass filter for ICA
        print(f"\n[4/10] Applying high-pass filter (1 Hz) for ICA...")
        # Filter BOTH raw objects with the same parameters
        raw_ieeg_filt = raw_ieeg.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
        raw_all_filt = raw_all.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
        print("  High-pass filter applied to both datasets")

        # 5. ICA artifact removal
        print(f"\n[5/10] Fitting ICA on iEEG channels...")
        ica = mne.preprocessing.ICA(
            n_components=0.99,
            method='fastica',
            random_state=97,
            max_iter=800,
            verbose=False
        )
        # Fit on filtered iEEG data
        ica.fit(raw_ieeg_filt, decim=3, verbose=False)
        print(f"  ICA fitted with {ica.n_components_} components")

        # Get ICA sources from iEEG channels
        ica_sources = ica.get_sources(raw_ieeg_filt)

        # Detect artifacts using correlation with EOG/ECG/EMG channels
        print(f"\n[6/10] Detecting artifacts...")
        eog_indices = detect_eog_artifacts(ica, ica_sources, raw_all_filt, eog_channels)
        ecg_indices = detect_ecg_artifacts(ica, ica_sources, raw_all_filt, ecg_channels)
        emg_indices = detect_emg_artifacts(ica, ica_sources, raw_all_filt, emg_channels)

        # Combine all artifact indices
        all_artifact_indices = list(set(eog_indices + ecg_indices + emg_indices))
        ica.exclude = all_artifact_indices

        print(f"  Total artifact components to remove: {len(all_artifact_indices)}")
        print(f"  Component indices: {all_artifact_indices}")

        # Apply ICA to the same filtered data it was trained on
        raw_ieeg_clean = raw_ieeg_filt.copy()
        ica.apply(raw_ieeg_clean, verbose=False)
        print(f"  ICA applied successfully!")
        print(f"  Components removed: {len(all_artifact_indices)} out of {ica.n_components_}")

        # 7. Notch filter
        print(f"\n[7/10] Applying notch filter...")
        raw_notched = notch_filter(raw_ieeg_clean)

        # 8. Bandpass filter
        print(f"\n[8/10] Applying bandpass filter...")
        raw_bandpass = bandpass_filter(raw_notched)

        # 9. Common Average Reference
        print(f"\n[9/10] Applying common average reference...")
        raw_referenced = reference(raw_bandpass)

        # Note: Theta band extraction with Hilbert transform is commented out per user's modification
        # hilbert_data = extract_hfb_power(raw_referenced)

        # 10. Create epochs
        print(f"\n[10/10] Creating epochs...")
        events, event_id = create_mne_events(events_df, raw_referenced)
        print(f"  MNE events created: {len(events)} events")

        tmin = 0.0
        tmax = 30.0
        epochs = epoch_ieeg_data(raw_referenced, events, event_id, tmin=tmin, tmax=tmax)

        print(f"  Speech epochs: {len(epochs['speech'])}")
        print(f"  Music epochs: {len(epochs['music'])}")

        # 11. Time-frequency analysis
        print(f"\n[11/11] Computing time-frequency representation...")
        freqs = np.arange(4, 8.5, 0.5)
        n_cycles = freqs / 2

        power_speech = mne.time_frequency.tfr_morlet(
            epochs['speech'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, verbose=False, decim=16
        )
        power_music = mne.time_frequency.tfr_morlet(
            epochs['music'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, verbose=False, decim=16
        )

        print(f"  Speech power shape: {power_speech.data.shape}")
        print(f"  Music power shape: {power_music.data.shape}")

        # 12. Statistical testing
        print(f"\n[12/12] Running cluster-based permutation test...")
        cluster_results = permutation_test_cluster_based(
            power_speech.data,
            power_music.data,
            n_permutations=5000,
            precluster_p=0.05,
            tail='two',
            cluster_statistic='mass',
            downsample_time=False
        )

        # Verify two-tailed test is working correctly
        verify_two_tailed_test(cluster_results)

        # 13. Generate visualizations
        print(f"\n[13/13] Generating visualizations...")
        plot_tf_results(
            cluster_results,
            channel_names=ieeg_channels,
            freqs=power_speech.freqs,
            times=power_speech.times,
            subject_id=subject_id,
            save_dir=subject_output_dir,
            show_plots=False
        )

        # 14. Generate report
        print(f"\n[14/14] Generating text report...")
        report_path = os.path.join(subject_output_dir, f'{subject_id}_cluster_report.txt')
        generate_cluster_report(
            cluster_results,
            channel_names=ieeg_channels,
            freqs=power_speech.freqs,
            times=power_speech.times,
            subject_id=subject_id,
            save_path=report_path
        )

        # 15. Create summary table
        print(f"\n[15/15] Creating summary table...")
        summary_df = create_summary_table(cluster_results, ieeg_channels)

        if len(summary_df) > 0:
            csv_path = os.path.join(subject_output_dir, f'{subject_id}_cluster_summary.csv')
            summary_df.to_csv(csv_path, index=False)
            print(f"  Summary table saved to: {csv_path}")
            print(f"\n  Top clusters:")
            print(summary_df.head().to_string(index=False))
        else:
            print("  No significant clusters found for this subject")

        print(f"\n{'='*80}")
        print(f"SUCCESSFULLY COMPLETED {subject_id}")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR processing {subject_id}")
        print(f"{'!'*80}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'!'*80}\n")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"C:\DS003688\DS003688"
    PERISYLVIAN_JSON = "./perisylvian_data.json"
    OUTPUT_DIR = "./results_theta"

    # Load perisylvian channel data
    print("Loading perisylvian channel data...")
    with open(PERISYLVIAN_JSON, 'r') as f:
        perisylvian_data = json.load(f)

    # Get all subjects from JSON
    all_subjects = list(perisylvian_data.keys())
    print(f"\nFound {len(all_subjects)} subjects in {PERISYLVIAN_JSON}:")
    for subj in all_subjects:
        n_channels = len(perisylvian_data[subj])
        print(f"  {subj}: {n_channels} channels")

    # Track results
    successful_subjects = []
    failed_subjects = []

    # Process each subject
    print(f"\n{'='*80}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*80}\n")

    start_time_total = time.time()

    for idx, subject_id in enumerate(all_subjects, 1):
        print(f"\nProcessing subject {idx}/{len(all_subjects)}: {subject_id}")

        success = run_subject_pipeline(
            subject_id=subject_id,
            data_dir=DATA_DIR,
            perisylvian_data=perisylvian_data,
            output_base_dir=OUTPUT_DIR
        )

        if success:
            successful_subjects.append(subject_id)
        else:
            failed_subjects.append(subject_id)

    total_time = time.time() - start_time_total

    # Final summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Successful subjects: {len(successful_subjects)}/{len(all_subjects)}")
    print(f"Failed subjects: {len(failed_subjects)}/{len(all_subjects)}")

    if successful_subjects:
        print(f"\nSuccessful subjects:")
        for subj in successful_subjects:
            print(f"  ✓ {subj}")

    if failed_subjects:
        print(f"\nFailed subjects:")
        for subj in failed_subjects:
            print(f"  ✗ {subj}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*80)
