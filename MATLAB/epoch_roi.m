% epoch_and_roi_from_json_noaux.m
% Single script: loads cleaned continuous EEG, epochs 'speech' and 'music',
% selects ROI channels using subjects_with_peri_cover.json (no cellfun),
% and saves ROI epochs. No EOG/EMG/ECG extraction or saving.
%
% Edit: current_subject, output_dir, json_path as needed.

clear; clc;

%% ---------------------- User parameters ----------------------
current_subject = 'sub-02';                       % change per run (must match JSON key)
output_dir = 'C:\Users\rahma\NIProject\derivatives\output';             % where postICA and outputs live
json_path = 'C:\Users\rahma\NIProject\subjects_with_peri_cover.json';  % path to your JSON
postica_name = sprintf('%s_postICA_CAR.mat', current_subject); % expected input file
save_name = sprintf('%s_epochs_ROI_auto.mat', current_subject); % output filename

% Epoch window in seconds (example: -2 to +4 s)
epoch_tmin = -2;
epoch_tmax = 32;

% Event labels (adjust to your dataset)
speech_event_label = 'speech';
music_event_label = 'music';


%% ---------------------- Prepare folders and load data ----------------------
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

postica_path = fullfile(output_dir, postica_name);
if ~exist(postica_path, 'file')
    error('Post-ICA file not found: %s', postica_path);
end

S = load(postica_path);
if ~isfield(S, 'EEG_clean')
    error('EEG_clean not found in %s', postica_path);
end
EEG = S.EEG_clean;                  % cleaned continuous EEG struct
fs = EEG.srate;

% Convert epoch window to samples
epoch_window = round([epoch_tmin epoch_tmax] * fs);
epoch_len = diff(epoch_window) + 1;

% Extract continuous data used for epoching
data_car = double(EEG.data);        % channels x time
chan_labels = {EEG.chanlocs.labels};

%% ---------------------- Parse events ----------------------
event_latencies = [EEG.event.latency];    % in samples
event_types = {EEG.event.type};

% Speech and music indices
speech_idx = find(strcmp(event_types, speech_event_label));
music_idx = find(strcmp(event_types, music_event_label));

%% ---------------------- Preallocate epoch arrays ----------------------
n_speech = length(speech_idx);
n_music = max(0, length(music_idx) - 1);   % preserve skip-first-music logic

n_chan = size(data_car, 1);

epochs_speech = nan(n_chan, epoch_len, n_speech);
epochs_music  = nan(n_chan, epoch_len, n_music);

%% ---------------------- Extract epochs (speech) ----------------------
count_s = 0;
for i = 1:n_speech
    center = round(event_latencies(speech_idx(i)));
    idx = center + epoch_window(1) : center + epoch_window(2);
    if idx(1) > 0 && idx(end) <= size(data_car, 2)
        count_s = count_s + 1;
        epochs_speech(:, :, count_s) = data_car(:, idx);
    end
end


%% ---------------------- Extract epochs (music) ----------------------
count_m = 0;
for ii = 2:length(music_idx)   % start at 2 to skip first music event per your note
    center = round(event_latencies(music_idx(ii)));
    idx = center + epoch_window(1) : center + epoch_window(2);
    if idx(1) > 0 && idx(end) <= size(data_car, 2)
        count_m = count_m + 1;
        epochs_music(:, :, count_m) = data_car(:, idx);
    end
end


%% ---------------------- Load JSON ROI mapping  ----------------------

if ~exist(json_path, 'file')
    error('JSON file not found: %s', json_path);
end

fid = fopen(json_path, 'r');
if fid == -1
    % Try to give more diagnostic info and fail gracefully
    errMsg = sprintf('Failed to open JSON file: %s\nCheck path and read permissions.', json_path);
    error(errMsg);
end

raw = fread(fid, inf, '*char')';
fclose(fid);

jsonStruct = jsondecode(raw);


% after jsondecode(raw) -> jsonStruct
lookupKey = matlab.lang.makeValidName(current_subject);   % converts sub-02 -> sub_02
if isfield(jsonStruct, lookupKey)
    subMap = jsonStruct.(lookupKey);
else
    error('No ROI mapping for %s (tried %s)', current_subject, lookupKey);
end

% Build roi_labels and roi_idx_from_json using simple loop
roi_labels = {};
roi_idx_from_json = [];
flds = fieldnames(subMap);
for k = 1:length(flds)
    lbl = flds{k};
    roi_labels{end+1} = lbl;
    roi_idx_from_json(end+1) = subMap.(lbl);
end

% Try to map JSON labels to EEG channel labels (preferred)
roi_idx_by_label = [];
matched_labels = {};
for i = 1:length(roi_labels)
    lbl = roi_labels{i};
    match = find(strcmpi(chan_labels, lbl), 1);
    if ~isempty(match)
        roi_idx_by_label(end+1) = match;
        matched_labels{end+1} = lbl;
    end
end

% If none matched by label, fall back to JSON indices (with bounds check)
if isempty(roi_idx_by_label)
    for i = 1:length(roi_idx_from_json)
        idx = roi_idx_from_json(i);
        if idx >= 1 && idx <= numel(chan_labels)
            roi_idx_by_label(end+1) = idx;
        else
            warning('JSON index %d for label %s out of bounds (1..%d)', idx, roi_labels{i}, numel(chan_labels));
        end
    end
else
    % Also include numeric indices for labels that did not match, if valid
    for i = 1:length(roi_idx_from_json)
        if ~ismember(roi_labels{i}, matched_labels)
            idx = roi_idx_from_json(i);
            if idx >= 1 && idx <= numel(chan_labels)
                roi_idx_by_label(end+1) = idx;
            else
                warning('JSON index %d for label %s out of bounds (1..%d)', idx, roi_labels{i}, numel(chan_labels));
            end
        end
    end
end


%% ---------------------- Extract ROI epochs (during epoching) ----------------------
epochs_speech_roi = epochs_speech(roi_idx_by_label, :, :);
epochs_music_roi  = epochs_music(roi_idx_by_label, :, :);

%% ---------------------- Save outputs ----------------------
outpath = fullfile(output_dir, save_name);
save(outpath, ...
    'epochs_speech', 'epochs_music', ...
    'epochs_speech_roi', 'epochs_music_roi', ...
    'event_latencies', 'event_types', 'speech_idx', 'music_idx', ...
    'epoch_window', 'epoch_len', 'fs', 'roi_labels', 'roi_idx_by_label', 'chan_labels', ...
    '-v7.3');

fprintf('Saved ROI epochs and metadata to %s\n', outpath);

%% ---------------------- Quick QC (plot few ROI-channel epochs) ----------------------
n_roi_ch = length(roi_labels);
t = (epoch_window(1):epoch_window(2)) / fs;
nplot = min(3, size(epochs_speech_roi, 3));
for p = 1:nplot
    figure;
    imagesc(t, 1:n_roi_ch, squeeze(epochs_speech_roi(:, :, p)));
    axis xy;
    xlabel('Time (s)'); ylabel('ROI channel'); title(sprintf('Speech epoch %d (ROI channels)', p));
    colorbar; colormap('jet');
end