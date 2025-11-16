%% ------------------------ Setup and loading data -------------------------
clear;clc;close all;
% EEGLAB initialization
eeglab_path = 'C:\Users\rahma\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\EEGLAB';
if ~exist('eeglab.m','file')
   disp('EEGlab not found , opening it anyways....');
   cd(eeglab_path);
   eeglab;
   close;
   %returning to project directory
   cd('C:\Users\rahma\NIProject');
   disp('Returned to project folder');
end
% Loading BIDS file
bidsRoot = 'C:\Users\rahma\NIProject\ds003688';
bids_data = bids.layout(bidsRoot, 'tolerant' ,true,'verbose',false);

% Output Directory
output_dir = 'C:\Users\rahma\NIProject\derivatives\output';

% Extracting each individual subject
allSubjectName = {bids_data.subjects.name};
% BIDS query to find ieeg data for a subject performing the task 'film'
subject_to_explore = allSubjectName{4};
task_to_explore = 'film';
f=bids.query(bids_data, 'data', ...
               'sub',subject_to_explore, ...
               'task',task_to_explore, ...
               'modality','ieeg', ...
               'suffix','ieeg');
% Finding the .vhdr header file from query 'f'
% .vhdr file is required to load the ieeg data
header_file_index = contains(f,'.vhdr');
header_file_path = f{header_file_index};
% Splitting the fileparts into folder, filename and the extension
[folderPath, fileName, fileExtension] = fileparts(header_file_path);
fullFileName = [fileName,fileExtension];
% Loading the eeg data using pop_loadbv() function from EEGLAB
EEG = pop_loadbv(folderPath, ... %path to the file
                   fullFileName); % name of the file
data = EEG.data * 1e6; % converting to µV
chan_labels = {EEG.chanlocs.labels};
% Rejecting bad electrodes
% fetching the path for ...._channels.tsv file where the channel status are
% present
chan_status = bids.query(bids_data, 'data', ...
               'sub',subject_to_explore, ...
               'task',task_to_explore, ...
               'modality','ieeg', ...
               'suffix','channels');
% Loading channel status data from the .tsv file
chan_status_path = chan_status{1};
chan_status_data = readtable(chan_status_path,'FileType','text','Delimiter','\t');
% Finding indices of the good channels
% good_channel_indices = find(strcmp(chan_status_data.status,'good'));
% keeping EOG,EMG and ECG data in separate variables
eog_channel_indices = find((strcmp(chan_status_data.type,'EOG')) & strcmp(chan_status_data.status,'good'));
emg_channel_indices = find((strcmp(chan_status_data.type,'EMG')) & strcmp(chan_status_data.status,'good'));
ecg_channel_indices = find((strcmp(chan_status_data.type,'ECG')) & strcmp(chan_status_data.status,'good'));
% Use the indices to store the data from those channels
eog_data = EEG.data(eog_channel_indices, :);
emg_data = EEG.data(emg_channel_indices, :);
ecg_data = EEG.data(ecg_channel_indices, :);
% Keeping only ECog and sEEG data channels
good_channel_indices = find((strcmp(chan_status_data.type,'SEEG')| strcmp(chan_status_data.type,'ECOG')) & strcmp(chan_status_data.status,'good'));
% Keeping only the good channels from sEEG and ECog in 'data' variable and 'chan_labels'
data = data(good_channel_indices,:);
chan_labels = chan_labels(good_channel_indices);
% Updating EEG.data , EEG.chanlocs and EEG.nbchan for consistency
EEG.data = data;
EEG.chanlocs = EEG.chanlocs(good_channel_indices);
EEG.nbchan = length(good_channel_indices);
%% ------------------------ Exploratory Data Analysis ----------------------------------------
% ---------Butterfly plot---------
% time vector in seconds
time_s=(0:EEG.pnts-1)/EEG.srate;
% ----------GFP plot----------------
gfp=var(data,0,1);
% Putting them into one figure with event markers .
figure;
ax1=subplot(2,1,1);
plot(time_s,data);
xlabel('Time(s)');ylabel('Amplitude(µV)');
title('Butterfly plot of the iEEG data ');
grid on;
hold on;
% Marking events on the butterfly plot
eventMarkers = EEG.event;
for i = 1:length(eventMarkers)
 xline(eventMarkers(i).latency/EEG.srate, 'r--', eventMarkers(i).type);
end
ax2=subplot(2,1,2);
plot(time_s,gfp);
xlabel('Time(s)');ylabel('Amplitude(µV)');
title('GFP plot of the iEEG data ');
grid on;
hold on;
% Marking events on the GFP plot
for i = 1:length(eventMarkers)
 xline(eventMarkers(i).latency/EEG.srate, 'r--', eventMarkers(i).type);
end
linkaxes([ax1,ax2],'x');
xlim(ax1,[20,40]);xlim(ax2,[20,40]);
% Found one channel to be distorting too much . I have already removed the
% channels that were marked 'bad' in the previous section .
% The butterflyplot and gfp plot makes it seem that there are more bad
% channels
% the channel with maximum variance should be the noisy channel i hope.
% [maxVal, noisy_channel_idx]= max(var(data,0,2));
% noisy_channel_name = chan_labels{noisy_channel_idx};
%
% % Removing the noisy channel from the data and updating channel labels
%
% channels_to_keep = true(EEG.nbchan,1);
% channels_to_keep(noisy_channel_idx)= false; % marking the channel as false
% data = data(channels_to_keep, :);
% chan_labels = chan_labels(channels_to_keep);
% EEG.data = data;
% EEG.chanlocs = EEG.chanlocs(channels_to_keep);
% EEG.nbchan = sum(channels_to_keep);
% There were multiple such electrodes , so the above method didn't work
% I checked for different subjects and same was with subject 3 . maybe
% because epileptic patients . subject 1's was better than 2,3 and 5's.
% subject 4 doesn't have eeg data.
% there are recordings from emg,eog,trig and misc so those could be the
% cause
%% ------------------------ Preprocessing ---------------------------------------------
% Plotting PSD to see if line noise is there.
psd_raw = mean(spectopo(data,0,EEG.srate,'freqrange',[0 200],'plot','off')); %seeing line noise at 50 Hz and its harmonics ,( didn't see any for sub-01 )
% Cleaning the line noise with CleanLine EEG lab plugin
EEG_noLine = pop_cleanline(EEG,'LineFrequencies',[50 100 150 200]);
data_noLine = EEG_noLine.data;
% Plotting PSD after cleaning line noise
psd_noLine = mean(spectopo(data_noLine,0,EEG_noLine.srate,'freqrange',[0 200],'plot','off'));
%-------------------------------------------------------------------------------------------------------------------
% Applying Bandpass Filter ( now applying a high-pass and then a low-pass)
nyquist = EEG.srate/2;
locutoff = 0.5 ;
hicutoff = 150 ; % defining the low and high cutoff for the bandwidth filters
order = round(3*(EEG.srate/locutoff)); % as mentioned in the textbook how eeglab uses 3 times the lower frequency bound
transition_width = 0.2; % 10% - 25% as mentioned by Mike X Cohen in the book
% designing a bandpass filter with firls
% ffrequencies = [0 (1-transition_width)*locutoff locutoff hicutoff (1+transition_width)*hicutoff nyquist]/nyquist;
% ideal_response = [0 0 1 1 0 0];
% filter_weights = firls(order,ffrequencies,ideal_response); % getting warning that it is badly scaled and the frequency response plots were full of ringing artifacts.
%
% % plotting time-domain , frequency and phase response of the filter
% figure;
% plot(filter_weights);
% xlabel('time samples');ylabel('amplitude');
%
% freqz(filter_weights,1,[],nyquist*2);
% using pop_eegfilt , taking their kernel and comparing to mine to see what's wrong with my filter design
% EEG_filt = pop_eegfilt(EEG_noLine,locutoff,hicutoff);
% ran it and even eeglab suggests to high pass and then low pass .
% Thinking of applying a high-pass filter at 0.5 Hz and then a low-pass filter at 150 Hz
%----------------------------------------------------------------------------------------------------------------------
% Designing the High pass filter
order_hp = round(3*(EEG.srate/locutoff));
freq_hp = [0,0.3,0.5,nyquist]/nyquist;
amp_hp = [0 0 1 1];
filter_hp = firls(order_hp,freq_hp,amp_hp);
% plotting the characteristics of the high pass filter
figure;
sgtitle('High-Pass Filter Characteristics'); % Add an overall title
% ---  (Impulse Response) ---
subplot(3, 1, 1);
plot(filter_hp);
title('Impulse Response (Time Domain)');
xlabel('Filter Taps (Samples)');
ylabel('Amplitude');
grid on;
% using freqz to the frequency response of the filter
[h_hp, f_hp] = freqz(filter_hp, 1, [], EEG.srate);
%plotting the frequency response
subplot(3, 1, 2);
% Plot the magnitude in decibels (dB)
plot(f_hp, 20*log10(abs(h_hp)));
xlim([0 5]); % Zoom to the 0-5 Hz range
title('Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
%plotting the phase characteristics
subplot(3,1,3);
plot(f_hp,angle(h_hp));
title('Phase Response');
xlabel('Frequencies');
ylabel('Phase(radians)');
xlim([0 5]);
grid on;
%----------------------------------------------------------------------------
% Designing the low pass filter
order_lp = round(EEG.srate/2);
freq_lp = [0,150,156,nyquist]/nyquist;
amp_lp = [1 1 0 0];
filter_lp = firls(order_lp,freq_lp,amp_lp);
% plotting the characteristics of the low pass filter
figure;
sgtitle('Low-Pass Filter Characteristics'); % Add an overall title
% --- (Impulse Response) ---
subplot(3, 1, 1);
plot(filter_lp);
title('Impulse Response (Time Domain)');
xlabel('Filter Taps (Samples)');
ylabel('Amplitude');
grid on;
% --- (Frequency Response) ---
subplot(3, 1, 2);
% Get the data from freqz
[h_lp, f_lp] = freqz(filter_lp, 1, [], EEG.srate);
% Plot the magnitude in decibels (dB)
plot(f_lp, 20*log10(abs(h_lp)));
% Add the xlim to zoom in on the cutoff frequency
xlim([140 160]); % Zoom to the 0-5 Hz range
title('Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
subplot(3,1,3);
plot(f_lp,angle(h_lp));
title('Phase Response');
xlabel('Frequencies');
ylabel('Phase(radians)');
xlim([140 160]);
grid on;
%---------------------------------------------------------------------------------------
% Applying the filters
data_filt_hp = filtfilt(filter_hp,1,data_noLine')';
data_filt = filtfilt(filter_lp,1,data_filt_hp')';
%--------------------------------------------------------------------------
% Common Average Referencing
% data_car = data_filt - mean(data_filt, 1);
%---------------------------------------------------------------------------------------
% Plots of the preprocessed data
% Mean PSD comparison
% psd_filt = mean(spectopo(data_filt,0,EEG.srate,'freqrange',[0 200],'plot','off'));
% psd_car = mean(spectopo(data_car,0,EEG.srate,'freqrange',[0 200],'plot','off'));
% figure;
% hold on;
% plot(psd_car,'w');
% plot(psd_filt,'r');
% plot(psd_noLine,'g');
% plot(psd_raw,'b');
% legend('CAR Data','Filtered Data', 'No Line Data', 'Raw Data');
% title('Power Spectral Density Comparison');
% xlabel('Frequency (Hz)');
% ylabel('Power (dB)');
% grid on;
% xlim([0 200]);
% % Butterfly and GFP
% gfp_car=var(data_car,0,1);
% figure;
% ax1=subplot(2,1,1);
% plot(time_s,data_car);
% xlabel('Time(s)');ylabel('Amplitude(µV)');
% title('Butterfly plot of the iEEG data ');
% grid on;
% hold on;
% % Marking events on the butterfly plot
% eventMarkers = EEG.event;
% for i = 1:length(eventMarkers)
%  xline(eventMarkers(i).latency/EEG.srate, 'r--', eventMarkers(i).type);
% end
% ax2=subplot(2,1,2);
% plot(time_s,gfp_car);
% xlabel('Time(s)');ylabel('Amplitude(µV)');
% title('GFP plot of the iEEG data ');
% grid on;
% hold on;
% % Marking events on the GFP plot
% for i = 1:length(eventMarkers)
%  xline(eventMarkers(i).latency/EEG.srate, 'r--', eventMarkers(i).type);
% end
% linkaxes([ax1,ax2],'x');
% xlim(ax1,[20,40]);xlim(ax2,[20,40]);
EEG_filt = EEG_noLine;
EEG_filt.data = data_filt;

%% ------------------------ visual inspection ---------------
figure;
dx1 = subplot(3,1,1);
plot(time_s,data(21,:));
xlabel('Time(s)');ylabel('Amplitude(µV)');
title('Before High-Pass');
grid on;
xlim([40 50]);
dx2 = subplot(3,1,2);
plot(time_s,data_filt_hp(21,:));
xlabel('Time(s)');ylabel('Amplitude(µV)');
title('After High-Pass');
grid on;
xlim([40 50]);
dx3 = subplot(3,1,3);
plot(time_s,data_car(21,:));
xlabel('Time(s)');ylabel('Amplitude(µV)');
title('After High-Pass and Low-pass and CAR');
grid on;
xlim([40 50]);
linkaxes([dx1 dx2 dx3],'xy');
%% ------------------------ ICA -------------------------------------------

EEG_ica = EEG_filt;
EEG_ica.data = double(data_filt);
EEG_ica.nbchan = size(EEG_filt.data,1);
EEG_ica.pnts = size(EEG_filt.data,2);
EEG_ica = eeg_checkset(EEG_ica);

% % compute variance over channels as a function of time
% gfp = var(EEG_ica.data,0,1); 
% % mark times where GFP > mean+5*std as extreme
% th = mean(gfp) + 5*std(gfp);
% bad_samples = find(gfp > th);
% fprintf('Detected %d extreme samples (GFP threshold).\n', numel(bad_samples));
% % visualize to decide if to exclude
% figure; plot((1:length(gfp))/EEG_ica.srate, gfp); hold on; yline(th,'r--'); xlabel('Time (s)');
% % If you want to exclude, create cleaned training data by removing those windows
% % (advanced) — for now, just inspect and proceed if not many.

EEG_ica = pop_runica(EEG_ica,'icatype','runica','extended',1,'interrupt','off');


% compute component activations (components x time)
compAct = (EEG_ica.icaweights * EEG_ica.icasphere) * double(EEG_ica.data);


% Get component activation and EOG signal
              
eog_signal = mean(eog_data,1) * 1e6;  % convert volts to µV
          % average across EOG channels if needed
fs = EEG_ica.srate;                      % sampling rate
t = (1:length(comp_signal)) / fs;        % time axis


for c = 1:min(12, size(compAct,1))
    comp_signal = compAct(c,:);
    t = (1:length(comp_signal)) / fs;
    figure('Name',sprintf('Component %d',c));
    plot(t, comp_signal, 'b'); hold on;
    plot(t, eog_signal, 'r');
    xlabel('Time (s)'); ylabel('Activation (µV)');
    title(sprintf('ICA Component %d vs EOG',c));
    legend('Component','EOG'); grid on;
end



for c = 1:min(12, size(compAct,1))
    signal = compAct(c,:);
    [pxx, f] = pwelch(signal, fs*2, fs, [], fs);  % 2s window, 50% overlap
    figure('Name',sprintf('Component %d Spectrum',c));
    plot(f, 10*log10(pxx));
    xlabel('Frequency (Hz)'); ylabel('Power (dB)');
    title(sprintf('Component %d Power Spectrum', c));
    grid on;
end

eog_signal = mean(eog_data,1);  % average if multiple EOG channels
R = corr(compAct', double(eog_signal)');  % components × 1

% Show top correlated components
[~, idxR] = sort(abs(R), 'descend');
disp([idxR(1:5)', R(idxR(1:5))']);

components_to_remove = []; % didn't find any for sub-02
EEG_cleanICA = pop_subcomp(EEG_ica, components_to_remove, 0);


data_car = double(EEG_cleanICA.data) - mean(double(EEG_cleanICA.data), 1);
EEG_clean = EEG_cleanICA;
EEG_clean.data = data_car;

% Ensure output directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save cleaned EEG and ICA metadata
save(fullfile(output_dir, sprintf('%s_postICA_CAR.mat', subject_to_explore)), ...
     'EEG_clean', 'components_to_remove', 'EEG_ica', 'EEG_cleanICA', '-v7.3');

% Save ICA weights separately
ica_weights = EEG_ica.icaweights;
ica_sphere = EEG_ica.icasphere;
icawinv = EEG_ica.icawinv;
save(fullfile(output_dir, sprintf('%s_ICA_weights.mat', subject_to_explore)), ...
     'ica_weights', 'ica_sphere', 'icawinv', '-v7.3');


%% ------------------------ Epoching by Task --------------------------------------------
% extracting latencies and types of events.
event_latencies = [EEG.event.latency];
event_types = {EEG.event.type};
% Define the epoch window in samples.
% time window of -2000ms to +4000ms around the event.
epoch_window = round([-2 4]*EEG.srate);
% Calculate the length of a single epoch in samples.
epoch_len = diff(epoch_window)+1;
%-------------------------------------------------------------------------------------------
% Epoching around 'speech'
% Find the indices of events with the type 'speech'.
speech_idx = find(strcmp(event_types,'speech'));
% Preallocate a 3D matrix to store the epochs to improve performance.
% The dimensions will be: channels x epoch length x number of target events.
epochs_speech = nan(size(data_car,1), epoch_len, length(speech_idx));
epochs_eog_speech = nan(size(eog_data,1), epoch_len, length(speech_idx));
epochs_emg_speech = nan(size(emg_data,1), epoch_len, length(speech_idx));
epochs_ecg_speech = nan(size(ecg_data,1), epoch_len, length(speech_idx));
% Loop through each 'speech' event to extract the corresponding epoch.
for i = 1:length(speech_idx)
  % Find the center latency of the current event in samples.
  center = round(event_latencies(speech_idx(i)));
  % Define the indices of the data points for the current epoch.
  idx = center+epoch_window(1):center+epoch_window(2);
  % Check if the epoch indices are within the bounds of the data matrix to prevent errors.
  if idx(1)>0 && idx(end)<=size(data_car,2)
      % Extract the epoch data and store it in the preallocated matrix.
      epochs_speech(:,:,i) = data_car(:,idx);
      % Extract the EOG epoch data
      if ~isempty(eog_data) % check if EOG data exists
          epochs_eog_speech(:,:,i) = eog_data(:,idx);
      end
     
      % Extract the EMG epoch data
      if ~isempty(emg_data) % check if EMG data exists
          epochs_emg_speech(:,:,i) = emg_data(:,idx);
      end
       % Extract the ECG epoch data
      if ~isempty(emg_data) %  check if ECG data exists
          epochs_ecg_speech(:,:,i) = ecg_data(:,idx);
      end
  end
end
%-------------------------------------------------------------------------------------------
% Epoching around 'music'
% Find the indices of events with the type 'music'.
music_idx = find(strcmp(event_types,'music'));
% Preallocate a 3D matrix to store the epochs to improve performance.
% The dimensions will be: channels x epoch length x number of target events.
epochs_music = nan(size(data_car,1), epoch_len, length(music_idx)-1); % lenght-1 because the first music event was preceded by 'no task' while
           % while the other 6 were preceded by 'speech' period.
epochs_eog_music = nan(size(eog_data,1), epoch_len, length(music_idx)-1);
epochs_emg_music = nan(size(emg_data,1), epoch_len, length(music_idx)-1);
epochs_ecg_music = nan(size(ecg_data,1), epoch_len, length(music_idx)-1);                                                                   
                                                                 
% Loop through each 'music' event to extract the corresponding epoch.
for i = 2:length(music_idx) % same reason for starting with i=2
  % Find the center latency of the current event in samples.
  center = round(event_latencies(music_idx(i)));
  % Define the indices of the data points for the current epoch.
  idx = center+epoch_window(1):center+epoch_window(2);
  % Check if the epoch indices are within the bounds of the data matrix to prevent errors.
   if idx(1)>0 && idx(end)<=size(data_car,2)
      % Extract the epoch data and store it in the preallocated matrix.
      epochs_music(:,:,i-1) = data_car(:,idx);
      % Extract the EOG epoch data
      if ~isempty(eog_data) % check if EOG data exists
          epochs_eog_music(:,:,i-1) = eog_data(:,idx);
      end
     
      % Extract the EMG epoch data
      if ~isempty(emg_data) % check if EMG data exists
          epochs_emg_music(:,:,i-1) = emg_data(:,idx);
      end
       % Extract the ECG epoch data
      if ~isempty(emg_data) %  check if ECG data exists
          epochs_ecg_music(:,:,i-1) = ecg_data(:,idx);
      end
   end  
end

