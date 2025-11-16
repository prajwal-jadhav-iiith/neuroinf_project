%% run_anatomical_localization_with_viz.m
% Anatomical localization + visualization for OpenNeuro ds003688
% - Loads electrode TSVs (ACPC)
% - Normalizes to MNI
% - Labels with AAL atlas
% - Saves CSVs and a per-subject visualization PNG
%
% Requirements:
%   - MATLAB
%   - FieldTrip (ft_defaults on path)
%   - SPM12
%   - BIDS-Matlab-Tools

clear; clc; close all;
ft_defaults;

%% ------------------------ Section 1: Setup ------------------------
bids_root = 'C:\Users\rahma\NIProject\ds003688'; % <-- update if needed
bids_data = bids.layout(bids_root, 'tolerant', true, 'verbose', false);

output_dir = 'C:\Users\rahma\NIProject\derivatives\anatomical_localization';
fig_dir = fullfile(output_dir, 'figures');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

subjects = {'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', ...
            'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26'};




%% ------------------------ Section 2: Process Each Subject ------------------------
for subj_idx = 1:length(subjects)
    current_subject = subjects{subj_idx};
    fprintf('\n==================================================\n');
    fprintf('Processing Subject: %s\n', current_subject);
    fprintf('==================================================\n');

    %% --- Step 3: Load Electrode Data (ACPC Space) ---
    fprintf('Step 3: Loading electrode data...\n');

    ieeg_sessions = bids.query(bids_data, 'sessions', 'sub', current_subject, 'modality', 'ieeg');
    if isempty(ieeg_sessions)
        fprintf('--> No iEEG session found for %s. Skipping.\n', current_subject);
        continue;
    end
    ieeg_session_label = ieeg_sessions{1};

    elec_files_all = bids.query(bids_data, 'data', 'sub', current_subject, ...
                                'ses', ieeg_session_label, 'suffix', 'electrodes', 'extension', '.tsv');
    coord_files_all = bids.query(bids_data, 'data', 'sub', current_subject, ...
                                'ses', ieeg_session_label, 'suffix', 'coordsystem', 'extension', '.json');

    % Prefer acq-clinical task-film
    elec_file = elec_files_all(contains(elec_files_all, {'acq-clinical', 'task-film'}));
    coords_file = coord_files_all(contains(coord_files_all, {'acq-clinical', 'task-film'}));

    if isempty(elec_file), elec_file = elec_files_all; end
    if isempty(coords_file), coords_file = coord_files_all; end

    if isempty(elec_file)
        fprintf('--> Electrode file missing for %s. Skipping.\n', current_subject);
        continue;
    end

    % Read electrode TSV
    elec_info = tdfread(elec_file{1});
    elec_native = struct();
    elec_native.label = cellstr(elec_info.name);
    elec_native.chanpos = [elec_info.x, elec_info.y, elec_info.z];
    elec_native.elecpos = elec_native.chanpos;

    % Read coordinate system JSON (if exists)
    if ~isempty(coords_file)
        coords_json = jsondecode(fileread(coords_file{1}));
        if isfield(coords_json, 'iEEGCoordinateSystem')
            elec_native.coordsys = lower(coords_json.iEEGCoordinateSystem); % ex: 'acpc'
        else
            elec_native.coordsys = 'acpc';
        end
    else
        elec_native.coordsys = 'acpc';
    end
    fprintf('Loaded %d electrodes (%s space)\n', length(elec_native.label), elec_native.coordsys);

  %% --- Step 4: Normalize to MNI Space ---
fprintf('Step 4: Normalizing to MNI space...\n');

% Find anatomical session
anat_sessions = bids.query(bids_data, 'sessions', 'sub', current_subject, 'modality', 'anat');
if isempty(anat_sessions)
    anat_sessions = {'ses-mri3t'}; % fallback if no session found
end
anat_session_label = anat_sessions{1};

% Locate T1w MRI
mri_file = bids.query(bids_data, 'data', 'sub', current_subject, ...
                      'ses', anat_session_label, 'modality', 'anat', 'suffix', 'T1w');
if isempty(mri_file)
    fprintf('--> MRI not found for %s. Skipping.\n', current_subject);
    return;
end

% Read MRI and enforce ACPC coordsys
mri_native = ft_read_mri(mri_file{1});
mri_native.coordsys = 'acpc';

% Normalize to MNI space
cfg = [];
cfg.spmversion = 'spm12';
cfg.nonlinear  = 'yes';
cfg.inputcoord = 'acpc';
mri_norm = ft_volumenormalise(cfg, mri_native);

% Warp electrode coordinates into MNI
elec_mni = elec_native;
elec_mni.chanpos = ft_warp_apply(mri_norm.params, elec_native.chanpos, 'individual2sn');
elec_mni.elecpos = elec_mni.chanpos;
elec_mni.coordsys = 'mni';

fprintf('Normalization complete for %s.\n', current_subject);

%% --- Step 5: Atlas Lookup & ROI Selection (Left Perisylvian) ---
fprintf('Step 5: Labeling electrodes & selecting left perisylvian ROI...\n');

% Load the AAL atlas
atlas_path = fullfile(fileparts(which('ft_defaults')), ...
                      'template','atlas','aal','ROI_MNI_V4.nii');
if ~exist(atlas_path,'file')
    error('AAL atlas not found at %s. Check your FieldTrip template installation.', atlas_path);
end
atlas = ft_read_atlas(atlas_path);

% Define left hemisphere perisylvian ROI labels (AAL naming convention)
perisylvian_roi = { ...
    'Frontal_Inf_Oper_L', ...
    'Frontal_Inf_Tri_L', ...
    'Rolandic_Oper_L', ...
    'Insula_L', ...
    'Temporal_Sup_L', ...
    'Temporal_Mid_L', ...
    'Heschl_L', ...
    'Supramarginal_L', ...
    'Angular_L' ...
};

% Prepare transform
mni_to_voxel_matrix = inv(atlas.transform);
num_electrodes = length(elec_mni.label);

all_labels = cell(num_electrodes,1);
perisylvian_idx = false(num_electrodes,1);

% Preallocate struct array for electrode info
electrode_info = repmat(struct( ...
    'label','', ...
    'mni_coord',[], ...
    'voxel_coord',[], ...
    'atlas_index',NaN, ...
    'atlas_label',''), num_electrodes, 1);

fprintf('Looking up atlas labels for %d electrodes...\n', num_electrodes);

for i = 1:num_electrodes
    mni_coord = [elec_mni.chanpos(i,:), 1];
    voxel_coord = mni_to_voxel_matrix * mni_coord';
    voxel_indices = round(voxel_coord(1:3));

    atlas_label = 'outside_atlas';
    tissue_index = NaN;

    if all(voxel_indices > 0) && all(voxel_indices' <= atlas.dim)
        tissue_index = atlas.tissue(voxel_indices(1), voxel_indices(2), voxel_indices(3));
        if tissue_index > 0 && tissue_index <= length(atlas.tissuelabel)
            atlas_label = atlas.tissuelabel{tissue_index};
        elseif tissue_index == 0
            atlas_label = 'atlas_zero';
        else
            atlas_label = 'atlas_unmapped';
        end
    end

    % Save info
    electrode_info(i).label       = elec_mni.label{i};
    electrode_info(i).mni_coord   = elec_mni.chanpos(i,:);
    electrode_info(i).voxel_coord = voxel_indices';
    electrode_info(i).atlas_index = tissue_index;
    electrode_info(i).atlas_label = atlas_label;

    all_labels{i} = atlas_label;

    % ROI membership (exact match, case-insensitive)
    if any(strcmpi(atlas_label, perisylvian_roi))
        perisylvian_idx(i) = true;
    end
end

% Final list of channels in ROI
perisylvian_channels = elec_mni.label(perisylvian_idx);

fprintf('→ Found %d electrodes in LEFT perisylvian ROI.\n', sum(perisylvian_idx));

%% --- Step 6: Save Results (Corrected) ---
label_col = elec_mni.label(:);
mni_x_col = elec_mni.chanpos(:,1);
mni_y_col = elec_mni.chanpos(:,2);
mni_z_col = elec_mni.chanpos(:,3);
anat_labels_col = all_labels(:);  % Already aligned above

% Create and save full label table
T_all = table(label_col, mni_x_col, mni_y_col, mni_z_col, anat_labels_col,...
              'VariableNames', {'channel_name', 'mni_x', 'mni_y', 'mni_z', 'anatomical_label'});
out_all = fullfile(output_dir, sprintf('%s_anatomical_labels.csv', current_subject));
writetable(T_all, out_all);

% Create and save ROI table
T_roi = table(perisylvian_channels', 'VariableNames', {'channel_name'});
out_roi = fullfile(output_dir, sprintf('%s_perisylvian_channels.csv', current_subject));
writetable(T_roi, out_roi);

fprintf('Saved CSVs for %s.\n', current_subject);

%% --- Step 7: Visualization (Left Hemisphere View) ---
fprintf('Step 7: Creating visualization for %s...\n', current_subject);

try
    % Marker sizes and colors
    if ~exist('all_ch_size','var'), all_ch_size = 40; end
    if ~exist('all_ch_color','var'), all_ch_color = [0.2 0.6 1]; end
    if ~exist('roi_ch_size','var'), roi_ch_size = 70; end
    if ~exist('roi_ch_color','var'), roi_ch_color = [1 0.2 0.2]; end

    % --- Segment normalized MRI into tissues ---
    cfg = [];
    cfg.output     = {'brain','skull','scalp'};
    cfg.spmversion = 'spm12';
    seg = ft_volumesegment(cfg, mri_norm);

    % --- Create cortical mesh from segmented brain ---
    cfg = [];
    cfg.method      = 'projectmesh';
    cfg.numvertices = 20000; % smoother mesh
    mesh = ft_prepare_mesh(cfg, seg);

    % --- Plot brain mesh and electrodes ---
    fig = figure('Visible','off','Position',[50 50 900 700]);
    ax = axes('Parent',fig,'Position',[0 0 1 1]);

    ft_plot_mesh(mesh, 'facecolor',[0.85 0.85 0.9], ...
                       'facealpha',0.1, ...
                       'edgecolor','none');
    hold on;

    % Plot all electrodes
    h_all = scatter3(elec_mni.chanpos(:,1), ...
                     elec_mni.chanpos(:,2), ...
                     elec_mni.chanpos(:,3), ...
                     all_ch_size, all_ch_color, 'filled');

    % Highlight perisylvian electrodes
    h_roi = [];
    if any(perisylvian_idx)
        h_roi = scatter3(elec_mni.chanpos(perisylvian_idx,1), ...
                         elec_mni.chanpos(perisylvian_idx,2), ...
                         elec_mni.chanpos(perisylvian_idx,3), ...
                         roi_ch_size, roi_ch_color, 'filled', ...
                         'MarkerEdgeColor','k', 'LineWidth',1.2);
    end

    % Aesthetics
    axis equal; grid off; set(gca,'Color','white');
    xlabel('MNI X (mm)'); ylabel('MNI Y (mm)'); zlabel('MNI Z (mm)');
    title(sprintf('%s: LEFT perisylvian electrodes (red) — %d found', ...
          current_subject, sum(perisylvian_idx)), 'Interpreter','none');

    % Left hemisphere lateral view
    view([-90 0]);
    camlight headlight; material shiny; lighting gouraud;

    % Legend
    if ~isempty(h_roi)
        legend([h_all h_roi], {'all electrodes','perisylvian electrodes'}, ...
               'Location','northeastoutside');
    else
        legend(h_all, 'all electrodes', 'Location','northeastoutside');
    end

    % Save figure
    fig_fname = fullfile(fig_dir, sprintf('%s_perisylvian_viz.png', current_subject));
    set(gcf,'Color',[1 1 1]);
    exportgraphics(gcf, fig_fname, 'Resolution', 200);
    close(fig);
    fprintf('Saved visualization to %s\n', fig_fname);

catch VISerr
    warning('Visualization for %s failed: %s', current_subject, VISerr.message);
end
end

fprintf('\n--- Anatomical localization and visualization complete for all subjects. ---\n');
