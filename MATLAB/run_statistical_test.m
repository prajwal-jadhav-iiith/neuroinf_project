load('C:\Users\rahma\NIProject\derivatives\output\sub-02_epochs_ROI_auto.mat');

% parameters 

min_freq = 4;
max_freq = 8;
cfreqs = min_freq:0.25:max_freq;
%% calculating speech power


% define the outputs for speech epochs
n_trials = size(epochs_speech_roi, 3);
n_channels = size(epochs_speech_roi, 1);
n_freqs = length(cfreqs);
n_time = size(epochs_speech_roi, 2);

tf_speech = zeros(n_freqs, n_channels, n_time, n_trials);


for trial = 1:n_trials
    trial_data = squeeze(epochs_speech_roi(:,:,trial));

    [cfx,~]= morletWaveletTransform(trial_data,fs,cfreqs,6,2);
    tf_speech(:,:,:,trial)= cfx;
end

power_speech = abs(tf_speech).^2;


times = linspace(-2000, 32000, n_time); %ms

% imagesc(n_time, cfreqs, squeeze(power_speech(:,1,:,3)));
% axis xy   % puts low freq at bottom, early time on left
% xlabel('Time (ms)')
% ylabel('Frequency (Hz)')
% colorbar
% title('TF power: channel 1, trial 1');

% average power across trials 
power_speech_mean = mean(power_speech,4);

imagesc(times, cfreqs, squeeze(power_speech_mean(:,1,:)));
axis xy   % puts low freq at bottom, early time on left
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
colorbar

%% calculating music power 

% defining outputs for music 
n_trials = size(epochs_music_roi, 3);
n_channels = size(epochs_music_roi, 1);
n_freqs = length(cfreqs);
n_time = size(epochs_music_roi, 2);

tf_music = zeros(n_freqs,n_channels,n_time,n_trials);

for trial = 1:n_trials
    trial_data = squeeze(epochs_music_roi(:,:,trial));
    [cfs,~] = morletWaveletTransform(trial_data,fs,cfreqs,6,2);
    tf_music(:,:,:,trial) = cfs;
end

power_music = abs(tf_music).^2;
power_music_mean = mean(power_music,4);

times_music = linspace(-2000, 32000, n_time); %ms
figure;
imagesc(times_music, cfreqs, squeeze(power_music_mean(:,1,:)));
axis xy   % puts low freq at bottom, early time on left
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
colorbar
title('TF power: channel 1, music');

%% log power transform 
log_power_speech = log10(power_speech + 1e-12);
log_power_music = log10(power_music + 1e-12);

% Plotting the log power for speech and music for comparison
figure;

subplot(2,1,1);
imagesc(times, cfreqs, squeeze(log_power_speech(:,1,:,1)));
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
colorbar;
title('Log Power: Speech');

subplot(2,1,2);
imagesc(times, cfreqs, squeeze(log_power_music(:,1,:,1)));
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
colorbar;
title('Log Power: Music');

% creating a log difference map
log_difference_map = mean(log_power_speech,4) - mean(log_power_music,4);

% Visualizing the log difference map
figure;
imagesc(times, cfreqs, squeeze(log_difference_map(:,1,:)));
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
colorbar;
title('Log Power Difference: Speech - Music');

%% paired sample t-test for observed value

[~,~,~,obs_stats] = ttest(log_power_speech,log_power_music,'Dim',4);
obs_tval = obs_stats.tstat;

figure;
imagesc(times,cfreqs,squeeze(obs_tval(:,1,:)));
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
colorbar;
title('t-values for channel 1 : speech vs music');

%% Pixel based correction for multiple comparison
n_perm = 1000;
max_tval = zeros(n_perm,1);
min_tval = zeros(n_perm,1);

for ev = 1:n_perm
    %flipping the sign of the difference is the same as changing the label 
    flip_sign = (rand(1,n_trials)>0.5)*2 - 1 ; 
    perm_diff = (log_power_speech - log_power_music).* reshape(flip_sign,1,1,1,[]); %randomly flipping the sign 
    [~,~,~,stats_perm] = ttest(perm_diff,0,'Dim',4); % t-test across the trials 
    max_tval(ev) = max(abs(stats_perm.tstat(:)));
    
end

threshold = prctile(max_tval,95);

pixel_corrected_map = obs_tval;
pixel_corrected_map(abs(obs_tval) < threshold) = 0;

fprintf('Observed max |t|: %.3f\n', max(abs(obs_tval(:))));
fprintf('Threshold (95th percentile): %.3f\n', threshold);
fprintf('Number of survivors: %d\n', sum(abs(obs_tval(:)) >= threshold));

[~,idx] = max(abs(obs_tval(:)));
[f_idx, c_idx, t_idx] = ind2sub(size(obs_tval), idx);
fprintf('Strongest effect at freq=%d, chan=%d, time=%d, t=%.3f\n', ...
        f_idx, c_idx, t_idx, obs_tval(f_idx,c_idx,t_idx));


figure;
imagesc(times, cfreqs, squeeze(pixel_corrected_map(:,c_idx,:)));
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
colorbar;
title('Pixel corrected map');
%% Cluster based correction

% Inputs
% log_power_speech [freq x chan x time x trials]
% log_power_music  [freq x chan x time x trials]
% times, cfreqs

n_trials = size(log_power_speech,4);
n_chan   = size(log_power_speech,2);
n_perm   = 1000;

% 1. Observed t-map
[~,~,~,stats_obs] = ttest(log_power_speech, log_power_music, 'Dim',4);
obs_tval = stats_obs.tstat;

% 2. Cluster-forming threshold (lenient, uncorrected p<0.05)
alpha = 0.05;
df = n_trials - 1;
cluster_thresh = tinv(1 - alpha/2, df);   % two-tailed

% 3. Observed clusters and their statistics
obs_cluster_stats = [];
obs_clusters = cell(n_chan,1);

for ch = 1:n_chan
    mask_chan = squeeze(abs(obs_tval(:,ch,:)) > cluster_thresh);
    vals_chan = squeeze(obs_tval(:,ch,:));
    CC = bwconncomp(mask_chan,4);
    obs_clusters{ch} = CC;
    for c = 1:CC.NumObjects
        idx = CC.PixelIdxList{c};
        obs_cluster_stats(end+1) = sum(vals_chan(idx)); %#ok<AGROW>
    end
end

% 4. Permutation null distribution of max cluster stats
max_cluster_stats = zeros(n_perm,1);

for ev = 1:n_perm
    flip_sign = (rand(1,n_trials) > 0.5)*2 - 1;
    perm_diff = (log_power_speech - log_power_music) .* reshape(flip_sign,1,1,1,[]);
    [~,~,~,stats_perm] = ttest(perm_diff,0,'Dim',4);
    perm_tval = stats_perm.tstat;

    perm_max_stat = 0;
    for ch = 1:n_chan
        mask_chan = squeeze(abs(perm_tval(:,ch,:)) > cluster_thresh);
        vals_chan = squeeze(perm_tval(:,ch,:));
        CC = bwconncomp(mask_chan,4);
        for c = 1:CC.NumObjects
            idx = CC.PixelIdxList{c};
            stat_c = sum(vals_chan(idx));
            if abs(stat_c) > abs(perm_max_stat)
                perm_max_stat = stat_c;
            end
        end
    end
    max_cluster_stats(ev) = perm_max_stat;
end

% 5. Final cluster threshold (95th percentile of null)
cluster_null_thresh = prctile(max_cluster_stats,95);

% 6. Test observed clusters against null
sig_clusters = cell(n_chan,1);
for ch = 1:n_chan
    mask_chan = squeeze(abs(obs_tval(:,ch,:)) > cluster_thresh);
    vals_chan = squeeze(obs_tval(:,ch,:));
    CC = obs_clusters{ch};
    sig_mask = zeros(size(mask_chan));
    for c = 1:CC.NumObjects
        idx = CC.PixelIdxList{c};
        stat_c = sum(vals_chan(idx));
        if abs(stat_c) >= cluster_null_thresh
            sig_mask(idx) = 1; % mark significant cluster
        end
    end
    sig_clusters{ch} = sig_mask;
end

% 7. Plot significant clusters for each channel
for ch = 1:n_chan
    figure;
    imagesc(times, cfreqs, sig_clusters{ch});
    axis xy; colorbar;
    xlabel('Time (ms)'); ylabel('Frequency (Hz)');
    title(sprintf('Significant clusters (chan %d)', ch));
end
