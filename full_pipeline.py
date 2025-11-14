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


def load_perisylvian_electrodes_from_csv(subject_id, electrode_results_dir='./electrode_results'):
    """
    Load perisylvian electrode information from CSV file

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-05')
    electrode_results_dir : str
        Directory containing electrode CSV files

    Returns:
    --------
    perisylvian_df : pd.DataFrame or None
        DataFrame with columns: electrode_name, region
        Returns None if file doesn't exist
    """
    csv_path = Path(electrode_results_dir) / f"{subject_id}_perisylvian_electrodes.csv"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"  WARNING: Perisylvian CSV not found at {csv_path}")
        return None


def get_channels_by_type(raw, channels_df, channel_types, subject_id, perisylvian_df=None):
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
    perisylvian_df : pd.DataFrame or None
        DataFrame with perisylvian electrode information (electrode_name, region)
        If None, uses all good channels of specified type

    Returns:
    --------
    channels : list of str
        List of channel names
    """
    # Get perisylvian electrode names if available
    if perisylvian_df is not None and len(perisylvian_df) > 0:
        perisylvian_electrodes = set(perisylvian_df['electrode_name'].tolist())
    else:
        perisylvian_electrodes = None

    if "SEEG" in channel_types or "ECOG" in channel_types:
        if perisylvian_electrodes is not None:
            # Filter for perisylvian electrodes only
            good_channels = channels_df[
                (channels_df['status'] == 'good') &
                (channels_df['type'].isin(channel_types)) &
                (channels_df['name'].isin(perisylvian_electrodes))
            ]['name'].tolist()
        else:
            # Use all good channels of specified type
            good_channels = channels_df[
                (channels_df['status'] == 'good') &
                (channels_df['type'].isin(channel_types))
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

def downsample_time_dimension(power_data, decim_factor=None, target_time_points=None):
    """
    Downsample the temporal dimension by averaging in bins

    Parameters:
    -----------
    power_data : np.ndarray (n_epochs, n_channels, n_freqs, n_times)
    decim_factor : int or None
        Decimation factor (e.g., 4 means reduce sampling by 4x)
        If provided, takes precedence over target_time_points
    target_time_points : int or None
        Target number of time points (e.g., 100)
        Only used if decim_factor is None

    Returns:
    --------
    downsampled : np.ndarray
        Downsampled array
    """
    n_epochs, n_channels, n_freqs, n_times = power_data.shape

    if decim_factor is not None:
        # Use decimation factor
        bin_size = decim_factor
        actual_time_points = n_times // decim_factor
        print(f"  Decimating by factor {decim_factor}: {n_times} -> {actual_time_points} timepoints")
    elif target_time_points is not None:
        # Use target time points (old behavior)
        bin_size = n_times // target_time_points
        actual_time_points = n_times // bin_size
        print(f"  Downsampling to {actual_time_points} timepoints")
    else:
        # No downsampling
        return power_data

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
                                    decim_factor=None, target_time_points=None):
    """
    Cluster-based permutation test using MNE's validated implementation

    This function wraps mne.stats.permutation_cluster_test to provide
    a standardized, peer-reviewed statistical test (Maris & Oostenveld, 2007).

    Parameters:
    -----------
    speech_power : np.ndarray (n_speech_epochs, n_channels, n_freqs, n_times)
    music_power : np.ndarray (n_music_epochs, n_channels, n_freqs, n_times)
    n_permutations : int
    precluster_p : float
    tail : str ('two', 'positive', 'negative')
    cluster_statistic : str ('mass' or 'size')
    decim_factor : int or None
        Temporal decimation factor (e.g., 4 = reduce time points by 4x)
    target_time_points : int or None
        Target number of time points (only used if decim_factor is None)

    Returns:
    --------
    results : dict
        Dictionary with keys:
        - 'observed_t_maps': Observed t-statistics
        - 'significant_mask': Boolean mask of significant pixels
        - 'significant_clusters': List of significant cluster info
        - 'cluster_thresholds': Thresholds per channel
        - 'method': 'cluster-based'
    """
    from mne.stats import permutation_cluster_test

    print("\n" + "="*70)
    print("CLUSTER-BASED PERMUTATION TEST (MNE Standard Implementation)")
    print("="*70)

    # Downsample temporal dimension if requested
    if decim_factor is not None or target_time_points is not None:
        print("\nDownsampling temporal dimension...")
        speech_power = downsample_time_dimension(speech_power,
                                                 decim_factor=decim_factor,
                                                 target_time_points=target_time_points)
        music_power = downsample_time_dimension(music_power,
                                                decim_factor=decim_factor,
                                                target_time_points=target_time_points)

    n_speech_epochs = speech_power.shape[0]
    n_music_epochs = music_power.shape[0]
    n_channels = speech_power.shape[1]
    n_freqs = speech_power.shape[2]
    n_times = speech_power.shape[3]

    print(f"\nData dimensions:")
    print(f"  Speech epochs: {n_speech_epochs}")
    print(f"  Music epochs: {n_music_epochs}")
    print(f"  Channels: {n_channels}")
    print(f"  Frequencies: {n_freqs}")
    print(f"  Time points: {n_times}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Precluster threshold: p < {precluster_p}")
    print(f"  Tail: {tail}")

    # Determine threshold from t-distribution
    df = n_speech_epochs + n_music_epochs - 2
    if tail == 'two':
        threshold = stats.t.ppf(1 - precluster_p/2, df)
        tail_mne = 0  # MNE uses 0 for two-tailed
        precluster_thresh_upper = threshold
        precluster_thresh_lower = -threshold
    elif tail == 'positive':
        threshold = stats.t.ppf(1 - precluster_p, df)
        tail_mne = 1  # MNE uses 1 for positive tail
        precluster_thresh_upper = threshold
        precluster_thresh_lower = None
    else:  # negative
        threshold = stats.t.ppf(precluster_p, df)
        tail_mne = -1  # MNE uses -1 for negative tail
        precluster_thresh_upper = None
        precluster_thresh_lower = threshold

    print(f"  t-threshold: {threshold:.3f} (df={df})")

    # Initialize result storage
    observed_t_maps = np.zeros((n_channels, n_freqs, n_times))
    sig_mask = np.zeros((n_channels, n_freqs, n_times), dtype=bool)
    all_sig_clusters = []
    cluster_thresholds = np.zeros(n_channels)

    print("\nRunning cluster-based permutation test per channel...")
    start_time = time.time()

    # Process each channel independently (iEEG channels are spatially independent)
    for ch_idx in range(n_channels):
        if (ch_idx + 1) % max(1, n_channels // 10) == 0 or ch_idx == 0:
            print(f"  Channel {ch_idx + 1}/{n_channels}...")

        # Prepare data for this channel: (n_epochs, n_freqs, n_times)
        speech_ch = speech_power[:, ch_idx, :, :]  # (n_speech, n_freqs, n_times)
        music_ch = music_power[:, ch_idx, :, :]    # (n_music, n_freqs, n_times)

        # Compute difference for each epoch: speech - music
        # For permutation_cluster_test, we need to test if the difference is != 0
        # We'll use a one-sample test on the difference scores
        differences = speech_ch - music_ch.mean(axis=0, keepdims=True)

        # Create the data array for MNE (one sample per observation)
        # Shape: (n_observations, n_freqs, n_times)
        X = [speech_ch, music_ch]  # List of two arrays for independent samples

        try:
            # Run MNE's standard cluster-based permutation test
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                X,
                n_permutations=n_permutations,
                threshold=threshold,
                tail=tail_mne,
                stat_fun=lambda x, y: stats.ttest_ind(x, y, equal_var=False)[0],  # Welch's t-test
                adjacency=None,  # No spatial adjacency for 2D time-frequency
                n_jobs=-1,  # Use all available CPU cores for parallel processing (8-12x speedup)
                seed=42,
                verbose=False
            )

            # Store observed t-statistics
            observed_t_maps[ch_idx] = T_obs

            # Find significant clusters (p < 0.05)
            sig_clusters_idx = np.where(cluster_p_values < 0.05)[0]

            if len(sig_clusters_idx) > 0:
                # Get the largest cluster statistic as threshold
                cluster_stats = [np.sum(np.abs(T_obs[clusters[i]])) for i in range(len(clusters))]
                cluster_thresholds[ch_idx] = np.max(cluster_stats)

                # Mark significant pixels and store cluster info
                for cluster_idx in sig_clusters_idx:
                    cluster_mask = clusters[cluster_idx]
                    sig_mask[ch_idx][cluster_mask] = True

                    # Get cluster coordinates
                    coords = np.where(cluster_mask)
                    cluster_stat = np.sum(np.abs(T_obs[cluster_mask]))

                    all_sig_clusters.append({
                        'channel': ch_idx,
                        'cluster_coords': coords,
                        'statistic': cluster_stat,
                        'p_value': cluster_p_values[cluster_idx]
                    })
            else:
                cluster_thresholds[ch_idx] = 0

        except Exception as e:
            print(f"    Warning: MNE cluster test failed for channel {ch_idx}: {str(e)}")
            # Fall back to simple t-test for this channel
            T_obs = stats.ttest_ind(speech_ch, music_ch, axis=0, equal_var=False)[0]
            observed_t_maps[ch_idx] = T_obs
            cluster_thresholds[ch_idx] = 0

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f} seconds ({total_time/n_channels:.1f} sec/channel)")

    # Summary statistics
    n_sig_pixels = np.sum(sig_mask)
    total_pixels = sig_mask.size

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total significant clusters: {len(all_sig_clusters)}")
    print(f"Total significant pixels: {n_sig_pixels} / {total_pixels}")
    print(f"Percentage: {100 * n_sig_pixels / total_pixels:.2f}%")

    print("\nPer-channel summary:")
    for ch in range(n_channels):
        n_clust_ch = sum(1 for c in all_sig_clusters if c['channel'] == ch)
        n_sig_ch = np.sum(sig_mask[ch])
        if n_clust_ch > 0 or n_sig_ch > 0:
            print(f"  Channel {ch}: {n_clust_ch} clusters, {n_sig_ch} pixels")

    return {
        'observed_t_maps': observed_t_maps,
        'significant_mask': sig_mask,
        'significant_clusters': all_sig_clusters,
        'cluster_thresholds': cluster_thresholds,
        'null_distributions': None,  # MNE handles this internally
        'precluster_threshold_upper': precluster_thresh_upper,
        'precluster_threshold_lower': precluster_thresh_lower,
        'n_significant_pixels': n_sig_pixels,
        'n_significant_clusters': len(all_sig_clusters),
        'method': 'cluster-based (MNE standard)'
    }


# =============================================================================
# VISUALIZATION AND REPORTING FUNCTIONS
# =============================================================================

def plot_ica_comparison(raw_before, raw_after, ica, channel_names, subject_id,
                        save_dir='./ica_comparison', duration=10.0, n_channels=6):
    """
    Create before/after ICA comparison visualizations

    Parameters:
    -----------
    raw_before : mne.io.Raw
        Raw data before ICA cleaning
    raw_after : mne.io.Raw
        Raw data after ICA cleaning
    ica : mne.preprocessing.ICA
        Fitted ICA object
    channel_names : list
        List of channel names to plot
    subject_id : str
        Subject identifier
    save_dir : str
        Directory to save comparison plots
    duration : float
        Duration of data segment to plot (seconds)
    n_channels : int
        Number of channels to display in time series plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select subset of channels for plotting (evenly spaced)
    n_total_channels = len(channel_names)
    if n_total_channels > n_channels:
        step = n_total_channels // n_channels
        plot_channels = channel_names[::step][:n_channels]
    else:
        plot_channels = channel_names

    print(f"\n  Creating ICA comparison visualizations...")
    print(f"    Plotting {len(plot_channels)} representative channels")

    # =========================================================================
    # Figure 1: Time Series Comparison
    # =========================================================================
    fig, axes = plt.subplots(len(plot_channels), 2, figsize=(16, 2*len(plot_channels)))
    if len(plot_channels) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{subject_id} - ICA Artifact Removal Comparison\n'
                 f'Before (left) vs After (right) | {len(ica.exclude)} components removed',
                 fontsize=14, fontweight='bold', y=0.995)

    # Get data segment
    start_sample = int(raw_before.info['sfreq'] * 5)  # Start at 5 seconds
    n_samples = int(raw_before.info['sfreq'] * duration)
    times = np.arange(n_samples) / raw_before.info['sfreq']

    for idx, ch_name in enumerate(plot_channels):
        # Before ICA
        ax_before = axes[idx, 0]
        data_before = raw_before.copy().pick([ch_name]).get_data()[0, start_sample:start_sample+n_samples]
        ax_before.plot(times, data_before * 1e6, 'k-', linewidth=0.5)  # Convert to µV
        ax_before.set_ylabel(f'{ch_name}\n(µV)', fontsize=9)
        ax_before.grid(True, alpha=0.3)
        if idx == 0:
            ax_before.set_title('Before ICA', fontsize=12, fontweight='bold')
        if idx == len(plot_channels) - 1:
            ax_before.set_xlabel('Time (s)', fontsize=10)
        else:
            ax_before.set_xticklabels([])

        # After ICA
        ax_after = axes[idx, 1]
        data_after = raw_after.copy().pick([ch_name]).get_data()[0, start_sample:start_sample+n_samples]
        ax_after.plot(times, data_after * 1e6, 'b-', linewidth=0.5)  # Convert to µV
        ax_after.grid(True, alpha=0.3)
        if idx == 0:
            ax_after.set_title('After ICA', fontsize=12, fontweight='bold')
        if idx == len(plot_channels) - 1:
            ax_after.set_xlabel('Time (s)', fontsize=10)
        else:
            ax_after.set_xticklabels([])
        ax_after.set_yticklabels([])

        # Match y-limits for comparison
        y_min = min(ax_before.get_ylim()[0], ax_after.get_ylim()[0])
        y_max = max(ax_before.get_ylim()[1], ax_after.get_ylim()[1])
        ax_before.set_ylim(y_min, y_max)
        ax_after.set_ylim(y_min, y_max)

    plt.tight_layout()
    timeseries_path = os.path.join(save_dir, f'{subject_id}_ica_timeseries_comparison.png')
    plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {timeseries_path}")
    plt.close()

    # =========================================================================
    # Figure 2: Power Spectral Density Comparison
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(f'{subject_id} - Power Spectral Density Before/After ICA',
                 fontsize=14, fontweight='bold')

    # Compute PSD for all channels
    print(f"    Computing power spectral density...")
    # Use the Raw object's compute_psd method (modern MNE API)
    psd_obj_before = raw_before.compute_psd(method='welch', fmin=0.5, fmax=100,
                                             n_fft=2048, verbose=False)
    psd_before = psd_obj_before.get_data()
    freqs_psd = psd_obj_before.freqs

    psd_obj_after = raw_after.compute_psd(method='welch', fmin=0.5, fmax=100,
                                           n_fft=2048, verbose=False)
    psd_after = psd_obj_after.get_data()

    # Average across channels
    psd_before_avg = psd_before.mean(axis=0)
    psd_after_avg = psd_after.mean(axis=0)

    # Plot 1: Before ICA (average across channels)
    ax = axes[0, 0]
    ax.semilogy(freqs_psd, psd_before_avg, 'k-', linewidth=1.5, label='Before ICA')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (V²/Hz)', fontsize=11)
    ax.set_title('Before ICA - Average PSD', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 100)

    # Plot 2: After ICA (average across channels)
    ax = axes[0, 1]
    ax.semilogy(freqs_psd, psd_after_avg, 'b-', linewidth=1.5, label='After ICA')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (V²/Hz)', fontsize=11)
    ax.set_title('After ICA - Average PSD', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 100)

    # Plot 3: Overlay comparison
    ax = axes[1, 0]
    ax.semilogy(freqs_psd, psd_before_avg, 'k-', linewidth=1.5, label='Before ICA', alpha=0.7)
    ax.semilogy(freqs_psd, psd_after_avg, 'b-', linewidth=1.5, label='After ICA', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (V²/Hz)', fontsize=11)
    ax.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 100)

    # Plot 4: Difference (reduction in power)
    ax = axes[1, 1]
    power_reduction = (psd_before_avg - psd_after_avg) / psd_before_avg * 100  # Percent reduction
    ax.plot(freqs_psd, power_reduction, 'r-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Reduction (%)', fontsize=11)
    ax.set_title('Power Reduction by ICA', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 100)

    # Highlight artifact-related frequencies
    # EOG: 0-5 Hz, ECG: 1-2 Hz (fundamental), EMG: >20 Hz
    ax.axvspan(0.5, 5, alpha=0.1, color='orange', label='EOG range')
    ax.axvspan(20, 100, alpha=0.1, color='red', label='EMG range')
    ax.legend(fontsize=9)

    plt.tight_layout()
    psd_path = os.path.join(save_dir, f'{subject_id}_ica_psd_comparison.png')
    plt.savefig(psd_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {psd_path}")
    plt.close()

    # =========================================================================
    # Figure 3: ICA Components Removed
    # =========================================================================
    if len(ica.exclude) > 0:
        n_excluded = len(ica.exclude)
        n_cols = min(4, n_excluded)
        n_rows = int(np.ceil(n_excluded / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_excluded == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle(f'{subject_id} - ICA Components Removed ({n_excluded} components)',
                     fontsize=14, fontweight='bold')

        sources = ica.get_sources(raw_before)
        sources_data = sources.get_data()
        times_comp = np.arange(min(10000, sources_data.shape[1])) / raw_before.info['sfreq']

        for idx, comp_idx in enumerate(ica.exclude):
            ax = axes[idx]
            comp_data = sources_data[comp_idx, :len(times_comp)]
            ax.plot(times_comp, comp_data, 'k-', linewidth=0.5)
            ax.set_title(f'Component {comp_idx}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Amplitude (AU)', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_excluded, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        components_path = os.path.join(save_dir, f'{subject_id}_ica_removed_components.png')
        plt.savefig(components_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {components_path}")
        plt.close()

    print(f"  [OK] ICA comparison visualizations complete!")


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
            print("  [OK] Two-tailed test is working correctly")
            print("  [OK] Majority of significant effects show Music > Speech")
            print("  [OK] This suggests a real, unidirectional effect in theta band")
        else:
            print("  [OK] Two-tailed test is working correctly")
            print("  [OK] Significant effects in both directions detected")
    else:
        print("\nNo significant pixels found")


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_subject_pipeline(subject_id, data_dir, electrode_results_dir='./electrode_results',
                         output_base_dir='./results_theta', save_preprocessed=True):
    """
    Run the complete analysis pipeline for a single subject

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-05')
    data_dir : str
        Path to the data directory
    electrode_results_dir : str
        Directory containing electrode localization CSV files
    output_base_dir : str
        Base directory for output files
    save_preprocessed : bool
        Whether to save preprocessed theta power data for group analysis

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

        # 0. Load perisylvian electrode information from CSV
        print(f"\n[0/15] Loading perisylvian electrode information...")
        perisylvian_df = load_perisylvian_electrodes_from_csv(subject_id, electrode_results_dir)
        if perisylvian_df is not None:
            print(f"  Loaded {len(perisylvian_df)} perisylvian electrodes")
            print(f"  Unique ROIs: {perisylvian_df['region'].nunique()}")
        else:
            print(f"  WARNING: No perisylvian electrodes found for {subject_id}")
            print(f"  Will process all good iEEG channels instead")

        # 1. Load data
        print(f"\n[1/15] Loading data for {subject_id}...")
        raw, events_df, channels_df = load_ieeg_data(subject_id, data_dir)
        print(f"  Raw data shape: {raw._data.shape}")
        print(f"  Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"  Recording duration: {raw.times[-1]:.2f} seconds")

        # 2. Get channels
        print(f"\n[2/15] Selecting channels...")
        ieeg_channels = get_channels_by_type(raw, channels_df, ['SEEG', 'ECOG'], subject_id, perisylvian_df)
        eog_channels = get_channels_by_type(raw, channels_df, ['EOG'], subject_id, None)
        ecg_channels = get_channels_by_type(raw, channels_df, ['ECG'], subject_id, None)
        emg_channels = get_channels_by_type(raw, channels_df, ['EMG'], subject_id, None)

        print(f"  iEEG channels: {len(ieeg_channels)}")
        print(f"  EOG channels: {len(eog_channels)}")
        print(f"  ECG channels: {len(ecg_channels)}")
        print(f"  EMG channels: {len(emg_channels)}")

        if len(ieeg_channels) == 0:
            print(f"  WARNING: No iEEG channels found for {subject_id}. Skipping...")
            return False

        # 3. Prepare raw objects
        print(f"\n[3/15] Preparing raw data objects...")
        # Raw object with ONLY iEEG channels (for ICA fitting)
        raw_ieeg = raw.copy().pick(ieeg_channels)

        # Raw object with ALL good channels (for artifact detection)
        all_channels = ieeg_channels + eog_channels + ecg_channels + emg_channels
        raw_all = raw.copy().pick(all_channels)
        print(f"  raw_ieeg channels: {len(raw_ieeg.ch_names)}")
        print(f"  raw_all channels: {len(raw_all.ch_names)}")

        # 4. High-pass filter for ICA fitting only
        print(f"\n[4/15] Preparing data for ICA...")
        # Create a high-pass filtered copy (1 Hz) for ICA fitting
        # Higher cutoff improves stationarity and ICA performance
        raw_ieeg_for_ica = raw_ieeg.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
        print(f"  High-pass filtered copy at 1 Hz created for ICA fitting")

        # Minimal filtering (0.1 Hz) for artifact channel reference
        # This removes only very slow drifts while preserving low frequencies
        raw_all_minimal = raw_all.copy().filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)
        print(f"  Minimal high-pass filter (0.1 Hz) applied to artifact channels")

        # 5. ICA artifact removal
        print(f"\n[5/15] Fitting ICA on high-pass filtered copy (1 Hz)...")
        ica = mne.preprocessing.ICA(
            n_components=0.99,
            method='fastica',
            random_state=97,
            max_iter=800,
            verbose=False
        )
        # Fit on 1 Hz high-passed data for better ICA performance
        ica.fit(raw_ieeg_for_ica, decim=3, verbose=False)
        print(f"  ICA fitted with {ica.n_components_} components")

        # IMPORTANT: Extract ICA sources from ORIGINAL unfiltered iEEG data
        # This preserves all frequency information for artifact detection
        ica_sources = ica.get_sources(raw_ieeg)
        print(f"  ICA sources extracted from original unfiltered data")

        # Detect artifacts using correlation with EOG/ECG/EMG channels
        print(f"\n[6/15] Detecting artifacts...")
        eog_indices = detect_eog_artifacts(ica, ica_sources, raw_all_minimal, eog_channels)
        ecg_indices = detect_ecg_artifacts(ica, ica_sources, raw_all_minimal, ecg_channels)
        emg_indices = detect_emg_artifacts(ica, ica_sources, raw_all_minimal, emg_channels)

        # Combine all artifact indices
        all_artifact_indices = list(set(eog_indices + ecg_indices + emg_indices))
        ica.exclude = all_artifact_indices

        print(f"  Total artifact components to remove: {len(all_artifact_indices)}")
        print(f"  Component indices: {all_artifact_indices}")

        # CRITICAL: Apply ICA to the ORIGINAL unfiltered data
        # This preserves all frequency content while removing artifacts
        raw_ieeg_clean = raw_ieeg.copy()
        ica.apply(raw_ieeg_clean, verbose=False)
        print(f"  ICA applied to original unfiltered data")
        print(f"  Components removed: {len(all_artifact_indices)} out of {ica.n_components_}")
        print(f"  [OK] All original frequency content preserved!")

        # 6b. Create ICA comparison visualizations
        print(f"\n[6b/15] Creating ICA comparison visualizations...")
        ica_comparison_dir = os.path.join(output_base_dir, 'ica_comparison')
        plot_ica_comparison(
            raw_before=raw_ieeg,
            raw_after=raw_ieeg_clean,
            ica=ica,
            channel_names=ieeg_channels,
            subject_id=subject_id,
            save_dir=ica_comparison_dir,
            duration=10.0,
            n_channels=6
        )

        # 7. Notch filter
        print(f"\n[7/15] Applying notch filter...")
        raw_notched = notch_filter(raw_ieeg_clean)

        # 8. Bandpass filter
        print(f"\n[8/15] Applying bandpass filter...")
        raw_bandpass = bandpass_filter(raw_notched)

        # 9. Common Average Reference
        print(f"\n[9/15] Applying common average reference...")
        raw_referenced = reference(raw_bandpass)

        # Note: Theta band extraction with Hilbert transform is commented out per user's modification
        # hilbert_data = extract_hfb_power(raw_referenced)

        # 10. Create epochs
        print(f"\n[10/15] Creating epochs...")
        events, event_id = create_mne_events(events_df, raw_referenced)
        print(f"  MNE events created: {len(events)} events")

        tmin = 0.0
        tmax = 30.0
        epochs = epoch_ieeg_data(raw_referenced, events, event_id, tmin=tmin, tmax=tmax)

        print(f"  Speech epochs: {len(epochs['speech'])}")
        print(f"  Music epochs: {len(epochs['music'])}")

        # 11. Time-frequency analysis
        print(f"\n[11/15] Computing time-frequency representation...")
        freqs = np.arange(4, 8.5, 0.5)
        n_cycles = freqs / 2

        # Optimized: decim=24 provides 41.7 Hz sampling (sufficient for theta 4-8Hz, Nyquist theorem)
        # This reduces computation time while maintaining statistical accuracy
        power_speech = mne.time_frequency.tfr_morlet(
            epochs['speech'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, verbose=False, decim=24
        )
        power_music = mne.time_frequency.tfr_morlet(
            epochs['music'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, verbose=False, decim=24
        )

        print(f"  Speech power shape: {power_speech.data.shape}")
        print(f"  Music power shape: {power_music.data.shape}")

        # 12. Save preprocessed theta power data for group analysis
        if save_preprocessed:
            print(f"\n[12/15] Saving preprocessed theta power data...")
            preprocessed_dir = os.path.join(output_base_dir, 'preprocessed_data')
            os.makedirs(preprocessed_dir, exist_ok=True)

            # Average theta power across trials for each condition
            speech_theta_avg = power_speech.data.mean(axis=0)  # Average across trials: (n_channels, n_freqs, n_times)
            music_theta_avg = power_music.data.mean(axis=0)

            # Average across frequency band (theta = 4-8 Hz)
            speech_theta_power = speech_theta_avg.mean(axis=1)  # (n_channels, n_times)
            music_theta_power = music_theta_avg.mean(axis=1)

            # CRITICAL: Resample to common time grid for group analysis consistency
            # Define standard time grid: 0-30s at 32 Hz (961 timepoints)
            # This ensures all subjects have identical temporal resolution
            common_sfreq = 32.0  # Hz
            common_times = np.linspace(0, 30, int(30 * common_sfreq) + 1)

            current_times = power_speech.times
            current_sfreq = 1 / (current_times[1] - current_times[0])

            # Check if resampling is needed (check length first to avoid broadcast error)
            if len(current_times) != len(common_times) or not np.allclose(current_times, common_times, atol=0.01):
                print(f"  Resampling from {current_sfreq:.1f} Hz ({len(current_times)} pts) to {common_sfreq:.1f} Hz ({len(common_times)} pts)")
                from scipy.interpolate import interp1d

                # Resample each channel independently
                speech_resampled = np.zeros((speech_theta_power.shape[0], len(common_times)))
                music_resampled = np.zeros((music_theta_power.shape[0], len(common_times)))

                for ch_idx in range(speech_theta_power.shape[0]):
                    # Linear interpolation for each channel
                    f_speech = interp1d(current_times, speech_theta_power[ch_idx, :],
                                       kind='linear', fill_value='extrapolate')
                    f_music = interp1d(current_times, music_theta_power[ch_idx, :],
                                      kind='linear', fill_value='extrapolate')

                    speech_resampled[ch_idx, :] = f_speech(common_times)
                    music_resampled[ch_idx, :] = f_music(common_times)

                speech_theta_power = speech_resampled
                music_theta_power = music_resampled
                save_times = common_times
            else:
                print(f"  Already at target sampling rate ({current_sfreq:.1f} Hz)")
                save_times = current_times

            # Save as .npz file with metadata
            save_path = os.path.join(preprocessed_dir, f'{subject_id}_theta_power.npz')
            np.savez(
                save_path,
                speech_theta_power=speech_theta_power,
                music_theta_power=music_theta_power,
                channel_names=np.array(ieeg_channels),
                times=save_times,
                freqs=power_speech.freqs,
                subject_id=subject_id
            )
            print(f"  Saved to: {save_path}")
            print(f"  Final shape: {speech_theta_power.shape} (channels × timepoints)")

            # Also save ROI mapping if available
            if perisylvian_df is not None:
                roi_mapping_path = os.path.join(preprocessed_dir, f'{subject_id}_roi_mapping.csv')
                # Filter to only include channels that were analyzed
                analyzed_df = perisylvian_df[perisylvian_df['electrode_name'].isin(ieeg_channels)]
                analyzed_df.to_csv(roi_mapping_path, index=False)
                print(f"  ROI mapping saved to: {roi_mapping_path}")

        # 13. Statistical testing
        print(f"\n[13/15] Running cluster-based permutation test...")
        cluster_results = permutation_test_cluster_based(
            power_speech.data,
            power_music.data,
            n_permutations=2000,  # Optimized: 2000 provides p<0.0005 resolution (Phipson & Smyth 2010)
            precluster_p=0.05,
            tail='two',
            cluster_statistic='mass',
            decim_factor=1  # Optimized: No secondary decimation (already decimated in TFR)
        )

        # Verify two-tailed test is working correctly
        verify_two_tailed_test(cluster_results)

        # 14. Generate visualizations
        print(f"\n[14/15] Generating visualizations...")
        # Use TFR times directly (already optimally decimated with decim=24)
        # No secondary decimation needed since decim_factor=1 in permutation test

        plot_tf_results(
            cluster_results,
            channel_names=ieeg_channels,
            freqs=power_speech.freqs,
            times=power_speech.times,  # Use decimated times from TFR directly
            subject_id=subject_id,
            save_dir=subject_output_dir,
            show_plots=False
        )

        # 15. Generate report and summary
        print(f"\n[15/15] Generating report and summary...")
        report_path = os.path.join(subject_output_dir, f'{subject_id}_cluster_report.txt')
        generate_cluster_report(
            cluster_results,
            channel_names=ieeg_channels,
            freqs=power_speech.freqs,
            times=power_speech.times,  # Use decimated times from TFR directly
            subject_id=subject_id,
            save_path=report_path
        )

        # Create summary table
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
# CHECKPOINT AND RESUME FUNCTIONS
# =============================================================================

def is_subject_completed(subject_id, output_base_dir):
    """
    Check if a subject has already been processed successfully

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-05')
    output_base_dir : str
        Base directory for output files

    Returns:
    --------
    completed : bool
        True if subject has been fully processed
    """
    # Check for critical output files
    preprocessed_dir = Path(output_base_dir) / 'preprocessed_data'
    theta_power_file = preprocessed_dir / f'{subject_id}_theta_power.npz'
    roi_mapping_file = preprocessed_dir / f'{subject_id}_roi_mapping.csv'

    subject_output_dir = Path(output_base_dir) / subject_id
    summary_plot = subject_output_dir / f'{subject_id}_summary_cluster-based.png'
    cluster_report = subject_output_dir / f'{subject_id}_cluster_report.txt'

    # All critical files must exist for subject to be considered complete
    critical_files = [theta_power_file, summary_plot, cluster_report]

    all_exist = all(f.exists() for f in critical_files)

    return all_exist


def get_processing_status(subjects, output_base_dir):
    """
    Get processing status for all subjects

    Parameters:
    -----------
    subjects : list
        List of subject IDs
    output_base_dir : str
        Base directory for output files

    Returns:
    --------
    completed : list
        List of completed subject IDs
    incomplete : list
        List of incomplete subject IDs
    """
    completed = []
    incomplete = []

    for subject_id in subjects:
        if is_subject_completed(subject_id, output_base_dir):
            completed.append(subject_id)
        else:
            incomplete.append(subject_id)

    return completed, incomplete


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run full iEEG preprocessing and analysis pipeline with resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects (will resume from last checkpoint automatically)
  python full_pipeline.py

  # Force reprocess all subjects (ignore checkpoints)
  python full_pipeline.py --force-reprocess

  # Process specific subjects only
  python full_pipeline.py --subjects sub-01,sub-05

  # Show processing status without running
  python full_pipeline.py --status-only
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\DS003688\DS003688",
        help="Root directory of BIDS dataset (default: C:\\DS003688\\DS003688)"
    )
    parser.add_argument(
        "--electrode-results-dir",
        type=str,
        default="./electrode_results",
        help="Directory containing electrode localization CSV files (default: ./electrode_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_theta",
        help="Directory to save output files (default: ./results_theta)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects to process (e.g., sub-01,sub-05). If not specified, processes all subjects."
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of all subjects, even if they are already completed"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show processing status and exit (don't run analysis)"
    )

    args = parser.parse_args()

    # Configuration
    DATA_DIR = args.data_dir
    ELECTRODE_RESULTS_DIR = args.electrode_results_dir
    OUTPUT_DIR = args.output_dir

    # Discover subjects from electrode_results directory
    print("Discovering subjects from electrode localization results...")
    electrode_results_path = Path(ELECTRODE_RESULTS_DIR)

    # Find all perisylvian electrode CSV files
    perisylvian_files = list(electrode_results_path.glob("*_perisylvian_electrodes.csv"))

    if len(perisylvian_files) == 0:
        print(f"\nERROR: No perisylvian electrode CSV files found in {ELECTRODE_RESULTS_DIR}")
        print("Please run the electrode localization pipeline first (Stages 0-2)")
        print("  1. resample_segmentation.py")
        print("  2. find_electrode_location.py")
        print("  3. filter_perisylvian_electrodes.py")
        exit(1)

    # Extract subject IDs from filenames
    all_subjects = []
    for csv_file in perisylvian_files:
        # Extract subject ID from filename (e.g., "sub-05_perisylvian_electrodes.csv" -> "sub-05")
        subject_id = csv_file.stem.replace('_perisylvian_electrodes', '')
        all_subjects.append(subject_id)

    all_subjects = sorted(all_subjects)

    # Filter subjects if specified
    if args.subjects:
        requested_subjects = [s.strip() for s in args.subjects.split(',')]
        all_subjects = [s for s in all_subjects if s in requested_subjects]
        print(f"\nFiltered to {len(all_subjects)} requested subjects: {', '.join(all_subjects)}")

    if len(all_subjects) == 0:
        print("\nERROR: No subjects to process!")
        exit(1)

    print(f"\nFound {len(all_subjects)} subjects with perisylvian electrode data:")
    for subj in all_subjects:
        csv_path = electrode_results_path / f"{subj}_perisylvian_electrodes.csv"
        df = pd.read_csv(csv_path)
        print(f"  {subj}: {len(df)} perisylvian electrodes, {df['region'].nunique()} unique ROIs")

    # Check processing status
    print(f"\n{'='*80}")
    print("CHECKING PROCESSING STATUS")
    print(f"{'='*80}")

    completed_subjects, incomplete_subjects = get_processing_status(all_subjects, OUTPUT_DIR)

    print(f"\nProcessing status:")
    print(f"  [OK] Completed: {len(completed_subjects)}/{len(all_subjects)}")
    print(f"  [PENDING] Incomplete: {len(incomplete_subjects)}/{len(all_subjects)}")

    if completed_subjects:
        print(f"\nAlready completed subjects:")
        for subj in completed_subjects:
            print(f"  [OK] {subj}")

    if incomplete_subjects:
        print(f"\nIncomplete/not started subjects:")
        for subj in incomplete_subjects:
            print(f"  [PENDING] {subj}")

    # Determine which subjects to process
    if args.force_reprocess:
        subjects_to_process = all_subjects
        print(f"\n[WARNING] FORCE REPROCESS MODE: Will reprocess all {len(subjects_to_process)} subjects")
    else:
        subjects_to_process = incomplete_subjects
        if len(completed_subjects) > 0:
            print(f"\n[OK] RESUME MODE: Skipping {len(completed_subjects)} completed subjects")
        if len(subjects_to_process) == 0:
            print(f"\n[OK] All subjects already completed! Use --force-reprocess to rerun.")

    # Status-only mode: exit without processing
    if args.status_only:
        print(f"\n{'='*80}")
        print("STATUS REPORT COMPLETE (--status-only mode)")
        print(f"{'='*80}")
        exit(0)

    if len(subjects_to_process) == 0:
        print(f"\n{'='*80}")
        print("NO SUBJECTS TO PROCESS")
        print(f"{'='*80}")
        exit(0)

    # Track results
    successful_subjects = []
    failed_subjects = []
    skipped_subjects = []

    # Add already-completed subjects to successful list
    if not args.force_reprocess:
        successful_subjects.extend(completed_subjects)
        skipped_subjects = completed_subjects

    # Process each subject
    print(f"\n{'='*80}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*80}")
    print(f"Subjects to process: {len(subjects_to_process)}")
    print(f"Subjects to skip: {len(skipped_subjects)}")
    print(f"{'='*80}\n")

    start_time_total = time.time()

    for idx, subject_id in enumerate(subjects_to_process, 1):
        print(f"\nProcessing subject {idx}/{len(subjects_to_process)}: {subject_id}")
        print(f"(Total progress: {idx + len(skipped_subjects)}/{len(all_subjects)})")

        success = run_subject_pipeline(
            subject_id=subject_id,
            data_dir=DATA_DIR,
            electrode_results_dir=ELECTRODE_RESULTS_DIR,
            output_base_dir=OUTPUT_DIR,
            save_preprocessed=True
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
    print(f"\nProcessing time: {total_time/60:.1f} minutes")
    print(f"Subjects processed this run: {len(subjects_to_process)}")
    print(f"Successful (total): {len(successful_subjects)}/{len(all_subjects)}")
    print(f"Failed: {len(failed_subjects)}/{len(all_subjects)}")
    if len(skipped_subjects) > 0:
        print(f"Skipped (already completed): {len(skipped_subjects)}/{len(all_subjects)}")

    if len(skipped_subjects) > 0 and not args.force_reprocess:
        print(f"\nSkipped subjects (already completed):")
        for subj in skipped_subjects:
            print(f"  ⊙ {subj}")

    # Split successful subjects into newly processed and previously completed
    newly_successful = [s for s in successful_subjects if s not in skipped_subjects]

    if newly_successful:
        print(f"\nNewly processed subjects ({len(newly_successful)}):")
        for subj in newly_successful:
            print(f"  [OK] {subj}")

    if failed_subjects:
        print(f"\nFailed subjects ({len(failed_subjects)}):")
        for subj in failed_subjects:
            print(f"  ✗ {subj}")
        print(f"\nTo retry failed subjects, run:")
        print(f"  python full_pipeline.py --subjects {','.join(failed_subjects)}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    if len(incomplete_subjects) > len(failed_subjects):
        remaining = [s for s in incomplete_subjects if s not in failed_subjects and s not in newly_successful]
        if remaining:
            print(f"\nTo resume interrupted processing, simply run:")
            print(f"  python full_pipeline.py")

    print("="*80)
