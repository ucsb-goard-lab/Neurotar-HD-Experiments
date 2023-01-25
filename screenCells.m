%% Screening cells for well matched ones, then saving as idk?
% load processed data
load('processed_data.mat')

% load individual files
[h_fn, h_pn] = uigetfile('*.mat');
heading_data = importdata(strcat(h_pn, h_fn));

[p_fn, p_pn] = uigetfile('*.mat');
platform_data = importdata(strcat(p_pn, p_fn));


% Prepare maps
ap = rescale(heading_data.avg_projection);
am = rescale(heading_data.activity_map);
ap_3d = ap(:, :, [1, 1, 1]);
am_3d = cat(3, am, zeros(size(am)), zeros(size(am)));
input_structure.head_avg_projection = ap_3d + am_3d;

ap = rescale(platform_data.avg_projection);
am = rescale(platform_data.activity_map);
ap_3d = ap(:, :, [1, 1, 1]);
am_3d = cat(3, am, zeros(size(am)), zeros(size(am)));
input_structure.plat_avg_projection = ap_3d + am_3d;

input_structure.head = head;
input_structure.plat = plat;
input_structure.head_cellmasks = heading_data.cellMasks;
input_structure.plat_cellmasks = platform_data.cellMasks;

cellComparator(input_structure)

disp('press any key to continue...')
pause
save('is_matched.mat', 'is_matched');
