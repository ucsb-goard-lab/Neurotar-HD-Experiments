classdef Cue < handle
	properties
		data
		bin_width = 6 % Default value
		is_head_direction
		direction_info

		tuning_curves
		significance_info

		data_type
		
		cue_flag
	end

	methods % Contsructor
		function obj = Cue(neural_data, stimulus_data, data_type)
			if nargin < 1 || isempty(neural_data)
				disp('Choose your neural data file...')
				[fn, pn] = uigetfile('.mat');
				neural_data = importdata([pn, fn]);
			end

			if nargin < 2 || isempty(stimulus_data)
				disp('Choose your stimulus data file...')
				[fn, pn] = uigetfile('.mat');
				stimulus_data = importdata([pn, fn]);
			end

			if nargin < 3 || isempty(data_type)
				data_type = 'spikes';
			end

			trimmed_stimulus_data = obj.trimStimulusData(stimulus_data, size(neural_data.raw_F, 2)); % this is in response to some issues we had previously, but it trims stimulus data to be the same size as neural data
			obj.data_type = data_type; % spikes or dff
			obj.data = DataStructure(neural_data, trimmed_stimulus_data); % store into a DataStructure for easy access and use
		end
	end

	methods % Main analyses
		function out = getTuningCurves(obj)
			out = obj.binData(obj.getNeuralData(), obj.getHeading());
		end
		% function is_hd = calculateHeadDirection(obj, tuning_curve_trials, iterations, gauss_flag)
		% 	% To determine which cells are heading selective
		% 	if nargin < 2 || isempty(tuning_curve_trials)
		% 		tuning_curve_trials = obj.getTrialResponses();
		% 	end
                %
		% 	if nargin < 3 || isempty(iterations)
		% 		iterations = 100; % this is set low in case you accidentally call it, actually 1000 iterations for real data
		% 	end
                %
		% 	if nargin < 4 || isempty(gauss_flag)
		% 		gauss_flag = 'double';
		% 	end
                %
		% 	if ndims(tuning_curve_trials) < 3
		% 		% Some low quality recordings have very few trials, those are just skipped and not included
		% 		disp('Bad recording, single trials only, skipping...')
		% 		is_hd = false(1, size(tuning_curve_trials, 1));
		% 	else
		% 		is_reliable = false(1, size(tuning_curve_trials, 1));
		% 		for c = 1:size(tuning_curve_trials, 1)
		% 			fprintf('\t%d/%d\n', c, size(tuning_curve_trials, 1))
		% 			bt_CC_data = zeros(1, iterations);
		% 			bt_CC_rand = zeros(1, iterations);
                %
		% 			for iter = 1:iterations
		% 				% Calculating shuffle stuff over many iterations
                %
		% 				% Create the randomized data by circularly shifting across trials
		% 				activity_data = squeeze(tuning_curve_trials(c, :, :));
		% 				activity_rand = zeros(size(activity_data));
                %
		% 				n_trials = size(activity_data, 2);
		% 				for t = 1:n_trials
		% 					activity_rand(:, t) = circshift(activity_data(:, t), randi(size(activity_data, 1)));
		% 				end
		% 				randvec = randperm(n_trials);
                %
		% 				% Calculate between trial CC, based on half-half splits.
		% 				bt_CC_data(iter) = obj.getBetweenTrialCC(activity_data, randvec);
		% 				bt_CC_rand(iter) = obj.getBetweenTrialCC(activity_rand, randvec);
		%
		% 				% Fitting gaussians
		% 				tuning_rand = nanmean(activity_rand, 2);
		% 				[fit_shuff, gof_shuff] = obj.fitGaussians(tuning_rand, gauss_flag);
                %
		% 				% Store important data from each iteration
		% 				amp_vec_shuff(iter) = fit_shuff.a1;
		% 				gof_vec_shuff(iter) = gof_shuff.rsquare;
		% 			end
                %
		% 			% Get the real tuning response and fit Gaussian for the true r-squared
		% 			tuning_resp = nanmean(activity_data, 2);
                %
		% 			[fit_func, gof] = obj.fitGaussians(tuning_resp, gauss_flag);
                %
		% 			% Compare real CC vs shuffled CC for significance
		% 			try
		% 				[~, P] = kstest2(bt_CC_data, bt_CC_rand);
		% 				obj.significance_info.d(c) = obj.cohen_d(bt_CC_data, bt_CC_rand);
		% 				nan_cell_vec(c) = 0;
		% 			catch
		% 				disp(['Error: Unable to calculate reliability for neuron #' num2str(c)])
		% 				nan_cell_vec(c) = 1;
		% 				P = 1;
		% 				d = 0;
		% 			end
                %
		% 			% Include a passable Cohen's D for a strong separation between distributions
		% 			if P < 0.1 && obj.significance_info.d(c) > 0.8 % strong effect
		% 				is_reliable(c) = true;
		% 			end
                %
		% 			% Gaussian information
		% 			obj.significance_info.fit_func{c} = fit_func;
		% 			obj.significance_info.gof(c) = gof;
		% 			obj.significance_info.gof_vec_shuff(c, :) = gof_vec_shuff;
		% 			is_amp(c) = fit_func.a1 > 0;%prctile(amp_vec_shuff, 95);
		% 			is_gof(c) = gof.rsquare > prctile(gof_vec_shuff, 90);
                %
		% 			obj.significance_info.bt_CC_data(c, :) = bt_CC_data;
		% 			obj.significance_info.bt_CC_rand(c, :) = bt_CC_rand;
		% 		end
		% 		is_hd = is_reliable & is_amp & is_gof; % requires a reliable response, amplitude of gauss > 0, and a goodness of fit better than shuffled
                %
		% 	end
		% 	obj.is_head_direction = is_hd;
		% end
                %
		function out = calculatePreferredDirection(obj, method, binned_data)
			% Get the preferred direction of each cell, either vectorsum or just the max
			if nargin < 2 || isempty(method)
				method = 'vectorsum';
				fprintf('No method provided, defaulting to vectorsum\n')
			end

			if nargin < 3 || isempty(binned_data)
				binned_data = obj.binData(); % fold the thing
			end

			binned_data(isnan(binned_data)) = 0;

			out = zeros(size(binned_data, 1), 1);
			switch method
				case 'vectorsum'
					for c = 1:size(binned_data, 1)
						dat = binned_data(c, :);
						dat(dat < 0) = 0; % rectify
						vs = obj.calculateVectorSum(dat);
						out(c) = wrapTo2Pi(vs(:, 1));
					end
				case 'max'
					[out(:, 2), out(:, 1)] = max(binned_data, [], 2);
				case 'fit'
					shift_point = 15;
					x = linspace(0, 2*pi, (360/obj.bin_width) + 1);
					out = zeros(size(binned_data, 1), 2);
					for c = 1:size(binned_data, 1)
						[fit_func, goodness_of_fit] = obj.fitGaussians(binned_data(c, :));
						[~, pref] = max(smoothdata(binned_data(c, :)));
						switch obj.cue_flag
							case 'single'
								out(c, 2) = mod(fit_func.b1 + x(mod(pref-shift_point, 360/obj.bin_width) + 1), 2*pi); % circshift it back, the gaussian fit centers the tuning curve so the peak doesn't end up on the end
								out(c, 1) = fit_func.a1;
							case 'double'
								a = [fit_func.a1, fit_func.a2];
								b = [fit_func.b1, fit_func.b2];
								[~, bigger_peak] = max([fit_func.a1, fit_func.a2]);
								out(c, 2) = mod(b(bigger_peak) + x(mod(pref-shift_point, 360/obj.bin_width) + 1), 2*pi);
								out(c, 1) = a(bigger_peak);
						end
						out(c, 3) = goodness_of_fit.rsquare;
					end
				otherwise
					fprintf('Invalid method provided (methods: ''vectorsum'', ''max'', or ''fit'')\n')
					return
			end
			obj.direction_info = out;
		end

		function flip_score = calculateFlipScore(obj, tuning_curves)
			% Calculating flip score as in Jacob et al 2016
			if nargin < 2 || isempty(tuning_curves)
				tuning_curves = obj.tuning_curves;
			end

			for c = 1:size(tuning_curves, 1)
				for t = 1:size(tuning_curves, 2)
					base_curve = smoothdata(tuning_curves(c, :)');
					shifted_curve = circshift(base_curve, t);
					correlation_curve(c, t) = corr(base_curve, shifted_curve, 'rows', 'complete');
				end
			end
			qtr = size(tuning_curves, 2) / 4; % quarter 360
			window = 0;
			middle_val = mean(correlation_curve(:, 2 * qtr - window : 2 * qtr + window), 2);
			side_val = mean(cat(2, correlation_curve(:, qtr - window : qtr + window), correlation_curve(:, 3 * qtr - window : 3 * qtr + window)), 2);
			flip_score = middle_val - side_val;
		end

		% function out = calculateRayleighVector(obj, data)
		% 	% Calculate folded RVL
		% 	if nargin < 2 || isempty(data)
		% 		data = obj.binData([], [], [], true);
		% 	end
		% 	% fold the tuning curve
		% 	for c = 1:size(data, 1)
		% 		vector_sum = obj.calculateVectorSum(data(c, :));
		% 		out(c) = vector_sum(2); % 1st is direction, second is magnitude
		% 	end
		% end
	end
    
    methods %things that are getting stuff, like tuning curves etc
        function [tf, b, all_fits, all_gof] = calculateTuningFits(obj)
            % Fitting gaussians to tuning curves
            for c = 1:size(obj.tuning_curves, 1)
                [fit, gof] = obj.fitGaussians(obj.tuning_curves(c, :), obj.cue_flag);
                x = linspace(0, 2*pi, size(obj.tuning_curves, 2));
                tf(c, :) = fit(x);
                b(c, :) = [fit.b1, fit.b2];
                all_fits{c} = fit;
                all_gof{c} = gof;
            end
            
        end
        function out = calculateRayleighVector(obj, data)
            % Similar to previous, a little different, because the binned data should only use moving data, not all data
            if nargin < 2 || isempty(data)
                switch obj.cue_flag
                    case 'single'
                        data = obj.binData(obj.getNeuralData(), obj.getHeading());
                    case 'double'
                        data = obj.binData(obj.getNeuralData(), obj.getHeading(), true); % fold it
                end
            end
            for c = 1:size(data, 1)
                vector_sum = obj.calculateVectorSum(data(c, :));
                out(c) = vector_sum(2); % 1st is direction, second is magnitude
            end
        end
        
        % function [tuning_curve_trials, segment] = getTrialResponses(obj, data_structure)
        % 	% Trial by trial response, defined as a full rotation of the cage
        % 	if nargin < 2 || isempty(data_structure)
		% 		data_structure = obj.data;
		% 	end
                %
		% 	% extract heading and timeseries
		% 	heading = [];
		% 	dff = []; % not actually dff, but whatever data_type you're using (generally spikes)
		% 	for d = data_structure
		% 		heading = cat(1, heading, d.get('heading'));
		% 		dff = cat(2, dff, d.get(obj.data_type));
		% 	end
		% 	heading(isnan(heading)) = 0;
		% 	heading = heading(:); % force into a column vector
                %
		% 	% clean up the heading a little bit
		% 	heading_ref = heading(1);
		% 	is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350; % weird spikes from the recorded neurotar stuff, this is to remove those
		% 	ct = 1;
		% 	while any(is_jump)
		% 		if ct == 10
		% 			break
		% 		end
		% 		heading(is_jump) = heading(max(find(is_jump) - 1, 1)); % "interpolate' jumps via nearest neighbor (one way)
		% 		is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350;
		% 		ct = ct + 1;
		% 	end
                %
		% 	% Determine when the heading "jumps" from -180 to 180
		% 	dist_from_ref = pdist2(heading, heading_ref);
		% 	[val, jumps] = findpeaks(dist_from_ref - max(dist_from_ref));
		% 	jumps(abs(val) > 10) = []; % remove tiny bumps that are not because of a revolution
                %
		% 	% Find times when the cage is stopped (rest periods)
		% 	slow_move = find(abs(diff(heading)) < 0.5);
		% 	n = 10; % 10 or more stopped frames
		% 	x = diff(slow_move') == 1;
		% 	f = find([false, x] ~= [x, false]);
		% 	g = find(f(2:2:end) - f(1:2:end-1) > n);
		% 	stops_idx = [f(2*g-1); f(2*g)];
                %
		% 	stops = [];
		% 	for s = 1:size(stops_idx, 2)
		% 		stops = cat(1, stops, slow_move(stops_idx(1, s):stops_idx(2, s)));
		% 	end
                %
		% 	% Either stops or jumps need to be accounted for
		% 	breaks = sort(cat(1, jumps, stops));
                %
		% 	for ii = 1:length(breaks) - 1
		% 		segment{ii} = breaks(ii) + 1:breaks(ii+1);
		% 	end
                %
		% 	segment(cellfun(@length, segment) < 50) = []; % cut out segments too short to feasibly be full rotations
		% 	is_not_full = false(length(segment), 1);
		% 	for ii = 1:length(segment)
		% 		% the following is necessary. range doesn't work b/c sometimes you pick up a lil jump from the previous one, and it leads to bad stuff
		% 		is_not_full(ii) = any(90 - abs(prctile(heading(segment{ii}), [25, 75])) > 50); % should both be around 90
		% 	end
                %
		% 	% Calculate the actual tuning curves using the standard binData method
		% 	segment(is_not_full) = [];
		% 	tuning_curve_trials = zeros(size(dff, 1), 360/obj.bin_width, length(segment));
		% 	for ii = 1:length(segment)
		% 		d = dff(:, segment{ii});
		% 		h = heading(segment{ii});
		% 		tuning_curve_trials(:, :, ii) = obj.binData([], d, h);
		% 	end
		% 	tuning_curve_trials(isnan(tuning_curve_trials)) = 0;
		% end
	end

	methods %Helpers
		% function cc = getBetweenTrialCC(obj, data, random_vec)
		% 	% random_vec needs to be passed in because it's conserved across true and random
		% 	first_half_idx = random_vec(1 : round(size(data, 2) / 2));
		% 	second_half_idx = random_vec(round(size(data, 2) / 2 + 1 : end));
		% 	cc = corr(nanmean(data(:, first_half_idx), 2), nanmean(data(:, second_half_idx), 2));
		% end
                %
		% function d = cohen_d(obj, x1, x2)
		% 	n1       = numel(x1);
		% 	n2       = numel(x2);
		% 	mean_x1  = nanmean(x1);
		% 	mean_x2  = nanmean(x2);
		% 	var_x1   = nanvar(x1);
		% 	var_x2   = nanvar(x2);
		% 	meanDiff = (mean_x1 - mean_x2);
		% 	sv1      = ((n1-1)*var_x1);
		% 	sv2      = ((n2-1)*var_x2);
		% 	numer    =  sv1 + sv2;
		% 	denom    = (n1 + n2 - 2);
		% 	pooledSD =  sqrt(numer / denom); % pooled Standard Deviation
		% 	s        = pooledSD;             % re-name
		% 	d        =  meanDiff / s;        % Cohen's d (for independent samples)
		% end


		function [fit_out, gof_out] = fitGaussians_old(obj, data, rescale_flag, center_flag)
			% Fitting constrained Gaussians to the data
			if nargin < 3 || isempty(rescale_flag)
				rescale_flag = true;
			end

			if nargin < 3 || isempty(center_flag)
				center_flag = false;
			end

			if size(data, 2) > size(data, 1)
				data = transpose(data); % ensure column vector
			end

			x = linspace(0, 2*pi, size(data, 1));
			if rescale_flag
				data = rescale(data);
            end
            
			center_idx = round(length(data)/4);
			if center_flag
			% First center the data so that the peak doesn't end up near the ends
			[~, max_idx] = max(smoothdata(data));
			centered_resp = circshift(data, center_idx - max_idx);
			else
			centered_resp = data;
			end
			centered_resp(isnan(centered_resp)) = 0;
			wiggle = pi/6;
			switch obj.cue_flag
				case 'double'
					% double gauss fit
					fit_func = fittype('a0 + a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2)');
					wiggle = pi/12;
					% Let's set some real parameters here, I think that'll help...
					a0 = min(centered_resp);
					a1 = mean(centered_resp(center_idx-2:center_idx+2));
					a2 = mean(centered_resp(center_idx + length(data)/2 - 2 : center_idx + length(data)/2 + 2));
					b1 = pi/2;
					b2 = 3*pi/2;
					c1 = pi/6;
					c2 = pi/6;
					initial_param = [a0, a1, a2, b1, b2, c1, c2];                     % initial parameters: [a0 a1 b1 c1]
					% upper_bounds = [0.2, 1, 1, pi/2, 3*pi/2 + wiggle, Inf, Inf];
					upper_bounds = [1, 1, 1, pi/2 + wiggle, 3*pi/2 + wiggle, 2*pi, 2*pi]; % updated upper bounds 13Jul2022
					lower_bounds = [0, 0, 0, pi/2 - wiggle, 3*pi/2 - wiggle, 0, 0];
					[double_gauss, double_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
					fit_out = double_gauss; 
					gof_out = double_gof;
				case 'single'
					% single gauss fit
					fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2)'); % define function: gaussian w/ baseline
					initial_param = [0, 0.5, pi/2, pi/6];                     % initial parameters: [a0 a1 b1 c1]
					upper_bounds = [1, 1, pi/2 + wiggle, pi/2];
					lower_bounds = [0, 0, pi/2 - wiggle, 0];
					[single_gauss, single_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
					fit_out = single_gauss; 
					gof_out = single_gof;
			end
			% Don't forget at this point that the location of the peak (c) is wrt the shifted location, so you'll need to unshift it later
		end
		function [fit_out, gof_out] = fitGaussians(obj, data, rescale_flag, center_flag)
			% Fitting constrained Gaussians to the data
			if nargin < 3 || isempty(rescale_flag)
				rescale_flag = true;
			end

			if nargin < 4 || isempty(center_flag)
				center_flag = false;
			end

			if size(data, 2) > size(data, 1)
				data = transpose(data); % ensure column vector
			end

			x = linspace(0, 2*pi, size(data, 1));
			if rescale_flag
				data = rescale(data);
            end
            
			center_idx = round(length(data)/4);
			if center_flag
			% First center the data so that the peak doesn't end up near the ends
			[~, max_idx] = max(smoothdata(data));
			centered_resp = circshift(data, center_idx - max_idx);
			else
			centered_resp = data;
			end
			centered_resp(isnan(centered_resp)) = 0;
			wiggle = pi/6;
			switch obj.cue_flag
				case 'double'
					% double gauss fit
					fit_func = fittype('a0 + a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2)');
					wiggle = pi/12;
					% Let's set some real parameters here, I think that'll help...
					a0 = min(centered_resp);
					a1 = mean(centered_resp(center_idx-2:center_idx+2));
					a2 = mean(centered_resp(center_idx + length(data)/2 - 2 : center_idx + length(data)/2 + 2));
					b1 = pi/2;
					b2 = 3*pi/2;
					c1 = pi/6;
					c2 = pi/6;
					initial_param = [a0, a1, a2, b1, b2, c1, c2];                     % initial parameters: [a0 a1 b1 c1]
					% upper_bounds = [0.2, 1, 1, pi/2, 3*pi/2 + wiggle, Inf, Inf];
					upper_bounds = [0.2, 1, 1, pi/2 + wiggle/2, 3*pi/2 + wiggle/2, 2*pi, 2*pi]; % updated upper bounds 13Jul2022
					% upper_bounds = [0.2, 1, 1, pi/2 + wiggle/2, 3*pi/2 + wiggle/2, pi/2, pi/2]; % updated upper bounds 13Jul2022
					lower_bounds = [0, 0, 0, pi/2 - wiggle/2, 3*pi/2 - wiggle/2, 0, 0];
					[double_gauss, double_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
					fit_out = double_gauss; 
					gof_out = double_gof;
				case 'single'
					% single gauss fit
					fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2)'); % define function: gaussian w/ baseline
					initial_param = [0, 0.5, pi/2, pi/6];                     % initial parameters: [a0 a1 b1 c1]
					upper_bounds = [1, 1, pi/2 + wiggle, pi/2];
					lower_bounds = [0, 0, pi/2 - wiggle, 0];
					[single_gauss, single_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
					fit_out = single_gauss; 
					gof_out = single_gof;
			end
			% Don't forget at this point that the location of the peak (c) is wrt the shifted location, so you'll need to unshift it later
		end

		function out = binData(obj, data, heading, fold_flag)
			if nargin < 2 || isempty(data)
				data = obj.getNeuralData();
			end

			if nargin < 3 || isempty(heading)
				heading = obj.getHeading();
			end

			if nargin < 4 || isempty(fold_flag)
				fold_flag = false; % primarily only used for calculating rvl in dual cue situations
			end

			% Changed this to be bin_width of 3, and a smoothing filter, a la Giocomo et al 2014, Curr Bio
			if fold_flag
				disp('Folding data prior to calculating bins...')
				bin_edges = -360:obj.bin_width:360; % Because alpha is [-180,180];
				groups = discretize(heading * 2, bin_edges); % doubled
			else
				bin_edges = -180:obj.bin_width:180;
				groups = discretize(heading, bin_edges);
			end

			u_groups = 1:length(bin_edges) - 1; % Get all the possible groups
			out = zeros(size(data, 1), length(u_groups));

			for g = 1:length(u_groups)
				for c = 1:size(data, 1)
					temp = data(c, groups == u_groups(g));
					out(c, g) = nanmean(temp);
				end
			end
			out = movmean(out, 15/obj.bin_width, 2); % 15 degree smoothing filter (as in Giocomo et al 2014)

			if fold_flag
				out = nanmean(cat(3, out(:, 1:length(u_groups)/2), out(:, length(u_groups)/2 + 1:end)), 3);
			end
		end

		function out = calculateVectorSum(obj, data)
			% Calculate vector sum of a tuning curve
			getHorz = @(v, theta) v .* cos(theta);
			getVert = @(v, theta) v .* sin(theta);
			getAng = @(vert, horz) atan2(vert, horz);
			getMag = @(vert, horz) sqrt(horz ^ 2 + vert ^ 2);

			theta_step = 2*pi/length(data);
			theta = 0 : theta_step : 2*pi - theta_step;
			h = getHorz(data, theta);
			v = getVert(data, theta);
			% Changed from sum to mean, shouldn't change anything... more similar to Giocomo
			r_h = nanmean(h);
			r_v = nanmean(v);

			m = getMag(r_v, r_h);
			ang = getAng(r_v, r_h);
			out = [ang, m];

		end
                %
		function stimulus_data = trimStimulusData(obj, stimulus_data, trim_length)
			fns = fieldnames(stimulus_data);
			for f = fns'
				stimulus_data.(f{:}) = stimulus_data.(f{:})(1:trim_length);
			end

		end
	end

	methods %visualization
		function visualizeTuning(obj, cells_to_visualize)
			if nargin < 2 || isempty(cells_to_visualize)
				cells_to_visualize = 1:size(obj.getNeuralData(), 1);
			end

			if islogical(cells_to_visualize)
				cells_to_visualize = find(cells_to_visualize); % in case you feed in a logical
			end

			if iscolumn(cells_to_visualize)
				cells_to_visualize = cells_to_visualize';
			end

			figure;
			data = obj.getNeuralData();
			for c = 1:size(data, 1)
				temp = data(c, :);
				temp(temp > prctile(temp, 99)) = prctile(temp, 99);
				data(c, :) = movmean(temp, 20);
			end

			heading = obj.getHeading();
			x = 1:length(heading);
			for c = cells_to_visualize
				subplot(1, 5, [1:4])
				coloredLinePlot(x, heading, data(c, :)); % make sure this is on your path
				title(sprintf('Cell %d', c))
				axis([1, size(data, 2), -180, 180])
				yticks([-180, -90, 0, 90, 180])
				colormap jet
				prettyPlot();

				subplot(1, 5, 5)
				plot(obj.tuning_curves(c, :))
				view([90, -90])
				prettyPlot();
				pause
			end
		end
	end
end
% function reliability = calculateXReliability(obj, data_structure)
		% 	max_lag = 5;
		% 	tuning_curve = obj.getTrialResponses(data_structure);
		% 	tuning_curve(isnan(tuning_curve)) = 0;
		% 	for c = 1:size(tuning_curve, 1)
		% 		for r = 1:size(tuning_curve, 3)
		% 			% determine offset
		% 			current = tuning_curve(c, :, r);
		% 			other = nanmean(tuning_curve(c, :, 1:end ~= r), 3);
		% 			[~, idx] = max(xcorr(current, other, max_lag));
		% 			lag = idx - max_lag;
		% 			% add offset
		% 			switch sign(lag)
		% 				case -1
		% 					other_corrected = padarray(other, [0, abs(lag)], 'post', 'replicate');
		% 					other_corrected = other_corrected(1 - lag:end);
		% 				case 1
		% 					other_corrected = padarray(other, [0, lag], 'pre', 'replicate');
		% 					other_corrected = other_corrected(1:end-lag);
		% 				otherwise
		% 					other_corrected = other;
		% 			end
                %
		% 			%correlate
		% 			reliability(c, r) = corr(current', other_corrected', 'rows', 'complete');
		% 		end
		% 	end
		% end

		% function reliability = calculateReliability(obj, data_structure)
		% 	% Calculate reliability of each cell across trials
		% 	heading = [];
		% 	dff = [];
		% 	for d = data_structure
		% 		heading = cat(1, heading, d.get('heading'));
		% 		dff = cat(2, dff, d.get(obj.data_type));
		% 	end
                %
		% 	nans =  [1; find(isnan(heading)); length(heading)];
                %
		% 	for ii = 1:length(nans) - 1
		% 		segment{ii} = nans(ii):nans(ii+1);
		% 	end
		% 	segment(cellfun(@length, segment) < 10) = [];
                %
		% 	for ii = 1:length(segment)
		% 		d = dff(:, segment{ii});
		% 		h = heading(segment{ii});
                %
		% 		tuning_curve(:, :, ii) = obj.binData([], d, h);
		% 	end
                %
		% 	% reliability, gets a lot of nans because of the method, so gotta account for that...
		% 	for c = 1:size(tuning_curve, 1)
		% 		for r = 1:size(tuning_curve, 3)
		% 			current = tuning_curve(c, :, r);
		% 			other = nanmean(tuning_curve(c, :, 1:end ~= r), 3);
		% 			reliability(c, r) = corr(current', other', 'rows', 'complete');
		% 		end
		% 	end
		% 	reliability = nanmean(reliability, 2);
		% end
                %
		% function is_good_cell = screenDFF(obj, method)
		% 	if nargin < 2 || isempty(method)
		% 		method = 'mse';
		% 	end
                %
		% 	dff = obj.data.get('DFF');
                %
		% 	switch method
		% 		case 'g'
		% 			for ii = 1:size(dff, 1)
		% 				[deconv(ii, :), ~, opt] = deconvolveCa(dff(ii, :), 'ar1', 'foopsi', 'optimize_pars');
		% 				g(ii) = opt.g;
		% 			end
                %
		% 			is_good_cell = g > 0.90;
		% 		case 'mse'
		% 			for ii = 1:size(dff, 1)
		% 				[deconv(ii, :), ~, opt] = deconvolveCa(dff(ii, :), 'ar1', 'foopsi', 'optimize_pars');
		% 			end
                %
		% 			mse = mean((rowrescale(dff) - rowrescale(deconv)).^2, 2);
		% 			is_good_cell = mse < 0.01;
		% 			keyboard
		% 		case 'pca'
		% 			[coeff, score, ~, ~, explained] = pca(dff);
		% 			n_pcs = find(cumsum(explained) > 75, 1, 'first');
		% 			recovered_traces = score(:, 1:n_pcs) * coeff(:, 1:n_pcs)';
                %
		% 			for c = 1:size(dff, 1)
		% 				cc(c) = corr(smooth(recovered_traces(c, :)), smooth(dff(c, :)));
		% 			end
		% 			is_good_cell = cc > 0.75;
		% 		case 'fano'
		% 			ff = @(x) (std(x, [], 2).^2)./mean(x, 2);
		% 			fano = ff(dff);
		% 			%                 1/fano
		% 			keyboard
		% 	end
		% end
                %
% 	function [tuning_curve_trials, segment] = getTrialResponses_old(obj, data_structure)
% 		if nargin < 2 || isempty(data_structure)
% 			data_structure = obj.data;
% 		end
% 		heading = [];
% 		dff = [];
% 		for d = data_structure
% 			heading = cat(1, heading, d.get('heading'));
% 			dff = cat(2, dff, d.get(obj.data_type));
% 		end
% 		heading(isnan(heading)) = 0;
% 		warning('New neurotar extractor has row instead of column vecotrs, probably mess with it later.')
% 		if size(heading, 2) > size(heading, 1)
% 			heading = heading';
% 		end
% 		diff_heading = abs(diff(heading));
%
% 		jumps = [1; find(diff_heading > 200); length(heading)]; % lowered again to 200.. hopefully no false positives 14Feb2022
% 		% 			nans =  [1; find(isnan(heading)); length(heading)];
% 		% breaks (not moving)
% 		stops = find(diff_heading < 0.5);
% 		% 14Feb2022 - consecutive stops only
% 		n = 10;
% 		x = diff(stops') == 1;
% 		f = find([false, x] ~= [x, false]);
% 		g = find(f(2:2:end) - f(1:2:end-1) > n);
% 		stops = [f(2*g-1), f(2*g)];
% 		breaks = sort(cat(1, jumps, stops'));
% 		%{
% 		temp = false(1, length(heading));
% 		temp(breaks) = true;
% 		plot(rescale(heading))
% 		hold on
% 		stem(temp);
% 		axis tight
% 		hold off
% 		%}
% 		keyboard
% 		for ii = 1:length(breaks) - 1
% 			segment{ii} = breaks(ii) + 1:breaks(ii+1);
% 		end
% 		segment(cellfun(@length, segment) < 50) = []; % increased from 10 to 50 14Feb2022 cut out small chunks
% 		for ii = 1:length(segment)
% 			% the following is necessary. range doesn't work b/c sometimes you pick up a lil jump from the previous one, and it leads to bad stuff
% 			is_not_full(ii) = any(90 - abs(prctile(heading(segment{ii}), [25, 75])) > 50); % should both be around 90
% 			% is_not_full(ii) = range(heading(segment{ii})) < 250;
% 		end
%
% 		segment(is_not_full) = [];
% 		tuning_curve_trials = zeros(size(dff, 1), 360/obj.bin_width, length(segment));
% 		for ii = 1:length(segment)
% 			d = dff(:, segment{ii});
% 			h = heading(segment{ii});
% 			tuning_curve_trials(:, :, ii) = obj.binData([], d, h);
% 		end
% 		tuning_curve_trials(isnan(tuning_curve_trials)) = 0;
% end
% Prototype methods
%     methods
%         function out = calculateMinuteCurves(obj, window, increment, display_flag)
%             if nargin < 2 || isempty(window)
%                 window = 300; % 1 minute
%             end

%             if nargin < 3 || isempty(increment)
%                 increment = 300; % 1 minute, nonoverlapping
%             end

%             if nargin < 4 || isempty(display_flag)
%                 display_flag = false;
%             end

%             data = obj.getNeuralData();
%             heading = obj.getHeading();
%             recording_length = min([size(data, 2), size(heading, 1)]);

%             ct = 1;
%             while ((ct - 1) * increment + window) < recording_length
%                 current_start_point = (ct - 1) * increment + 1;
%                 % segment data
%                 current_data = data(:, current_start_point : current_start_point + window - 1);
%                 current_heading = heading(current_start_point : current_start_point + window - 1);
%                 % store it
%                 out(:, :, ct) = obj.binData([], current_data, current_heading);
%                 %increment counter
%                 ct = ct + 1;
%             end

%             if display_flag
%                 tuning_curve = obj.binData();
%                 for c = 1:size(out, 1)
%                     subplot(2, 1, 1)
%                     imagesc(squeeze(out(c, :, :))')
%                     subplot(2, 1, 2)
%                     plot(tuning_curve(c, :))
%                     pause
%                 end
%             end
%         end

%         function [fit_func, goodness_of_fit] = fitDoubleGauss(obj, x, y, coeffs)
%             if nargin < 4 || ~exist('coeffs')
%                 previous_fit = false;
%             else
%                 previous_fit = true;
%             end
%             % shift
%             % [~, max_idx] = max(y);
%             if previous_fit
%                 wiggle = pi / 4;
%                 gauss = 'a1 * exp(-(((x - mu) .^ 2) / (2 * k1 .^ 2))) + a2 * exp(-(((x - (mu + pi)) .^ 2) / (2 * k2 .^ 2)))';
%                 startPoints = [1, 1, 1, 1, coeffs(5)];
%                 [fit_func, goodness_of_fit] = fit(x.', y.', gauss, 'Start', startPoints,...
%                     'Upper', [1, 1, 5, 5, coeffs(5) + wiggle], 'Lower', [0, 0, 0, 0, coeffs(5) - wiggle]);
%             else
%                 gauss = 'a1 * exp(-(((x - mu1) .^ 2) / (2 * k1 .^ 2))) + a2 * exp(-(((x - (mu2)) .^ 2) / (2 * k2 .^ 2))) + c';
%                 startPoints = [1, 1, 0, 1, 1, 1.4, 1.4 + pi]; % this is approx 4/18 * pi
%                 [fit_func, goodness_of_fit] = fit(x.', y.', gauss, 'Start', startPoints,...
%                     'Upper', [Inf, Inf, Inf, pi, pi, 2*pi, 2*pi], 'Lower', [0, 0, -Inf, 0, 0, 0, 0]);
%             end
%         end

%         function calculateContributions(obj)
%             tuning_curve = obj.binData();
%             if isempty(obj.decoder)
%                 obj.decodePopulationActivity();
%             end
%             predicted_heading = obj.decoder.getHeadingTrace(true);
%             nan_predicted_heading = isnan(predicted_heading);
%             t = 1:numel(predicted_heading);
%             predicted_heading(nan_predicted_heading) = interp1(t(~nan_predicted_heading), predicted_heading(~nan_predicted_heading), t(nan_predicted_heading));

%             ts = obj.getNeuralData();
%             ts = rescale(ts, 'InputMin', min(ts, [], 2), 'InputMax', max(ts, [], 2));
%             for t = 1:size(ts, 2)
%                 current_prediction = round(predicted_heading(t));
%                 wts = tuning_curve(:, current_prediction);
%                 contribution(:, t) = ts(:, t) .* wts;
%             end
%             contribution = rescale(contribution, 'InputMin', min(contribution, [],  2), 'InputMax', max(contribution, [], 2));
%             clust_id = obj.classifier.getClusters();
%             obj.contributions.heading = contribution(clust_id == 1, :);
%             obj.contributions.visual = contribution(clust_id == 2, :);
%             obj.contributions.multimodal = contribution(clust_id == 3, :);
%             obj.contributions.other = contribution(clust_id == 4, :);
%         end

%         function out = minCircularDistance(obj, x, y)
%             % ensure p2 >= p1 always
%             % forward dist
%             dist = (y - x);
%             % rev dist
%             out = wrapToPi(y-x); %min([2*pi - dist, dist], [], 2); % circular
%             % for ii = 1:length(dist) % for each possible peak, choose the smaller
%             %     candidates(ii) = min(abs([dist(ii), dist2(ii)]));
%             % end
%             % candidates = candidates';
%         end
%         %% Junk
%         % function [fit_func, goodness_of_fit] = fitVonMises_OLD(obj, x, y, coeffs)
%         %     if nargin < 4 || ~exist('coeffs')
%         %         previous_fit = false;
%         %     else
%         %         previous_fit = true;
%         %     end

%         %     if previous_fit
%         %         wiggle = pi/4;
%         %         vonMises = '(a1 * exp(k1 * cos(x - mu1)) / (2 * pi * besseli(0, k1))) + (a2 * exp(k2 * cos(x - (mu1 + pi))) / (2 * pi * besseli(0, k2)))';
%         %         startPoints = [1, 1, 1, 1, coeffs(5)];
%         %         [fit_func, goodness_of_fit] = fit(x.', y.', vonMises, 'Start', startPoints,...
%         %             'Upper', [Inf, Inf, 20, 20, coeffs(5) + wiggle], 'Lower', [0, 0, 0, 0, coeffs(5) - wiggle]);
%         %     else
%         %         vonMises = '(a1 * exp(k1 * cos(x - mu1)) / (2 * pi * besseli(0, k1))) + (a2 * exp(k2 * cos(x - (mu1 + pi))) / (2 * pi * besseli(0, k2)))';
%         %         [~, pk1] = max(y);
%         %         pk2 = pk1 + length(x)/2;
%         %         startPoints = [1, 1, 1, 1, pk1];
%         %         [fit_func, goodness_of_fit] = fit(x.', y.', vonMises, 'Start', startPoints,...
%         %             'Upper', [Inf, Inf, 20, 20, 2*pi], 'Lower', [0, 0, 0, 0, 0]);
%         %     end
%         % end
%         % function [fit_func, goodness_of_fit] = fitVonMises(obj, x, y, coeffs)
%         %     if nargin < 4 || ~exist('coeffs')
%         %         previous_fit = false;
%         %     else
%         %         previous_fit = true;
%         %     end

%         %     if previous_fit
%         %         wiggle = pi/4;
%         %         vonMises = '(exp(k1 * cos(x - mu1)) / (2 * pi * besseli(0, k1))) + (exp(k2 * cos(x - (mu1 + pi))) / (2 * pi * besseli(0, k2)))';
%         %         startPoints = [1, 1, coeffs(3)];
%         %         [fit_func, goodness_of_fit] = fit(x.', rescale(y.'), vonMises, 'Start', startPoints,...
%         %             'Upper', [20, 20, coeffs(3) + wiggle], 'Lower', [0, 0, coeffs(3) - wiggle]);
%         %     else
%         %         vonMises = '(exp(k1 * cos(x - mu1)) / (2 * pi * besseli(0, k1))) + (exp(k2 * cos(x - (mu1 + pi))) / (2 * pi * besseli(0, k2)))';
%         %         [~, pk1] = max(y);
%         %         pk2 = pk1 + length(x)/2;
%         %         startPoints = [1, 1, 1];
%         %         [fit_func, goodness_of_fit] = fit(x.', rescale(y.'), vonMises, 'Start', startPoints,...
%         %             'Upper', [20, 20, 2*pi], 'Lower', [0, 0, 0]);
%         %     end
%         % end
%     end
% end
% function is_hd = calculateHeadDirection_shuff(obj, tuning_curve_trials, iterations, gauss_flag)
% 	if nargin < 2 || isempty(tuning_curve_trials)
% 		tuning_curve_trials = obj.getTrialResponses();
% 	end
%
% 	if nargin < 3 || isempty(iterations)
% 		iterations = 100;
% 	end
%
% 	if nargin < 4 || isempty(gauss_flag)
% 		gauss_flag = 'double';
% 	end
%
%
% 	if ndims(tuning_curve_trials) < 3
% 		disp('Bad recording, single trials only, skipping...')
% 		is_hd = false(1, size(tuning_curve_trials, 1));
% 	else
% 		is_reliable = false(1, size(tuning_curve_trials, 1));
% 		for c = 1:size(tuning_curve_trials, 1)
% 			fprintf('\t%d/%d\n', c, size(tuning_curve_trials, 1))
% 			bt_CC_data = zeros(1, iterations);
% 			bt_CC_rand = zeros(1, iterations);
%
%
% 			for iter = 1:iterations
% 				% circshift the data
% 				activity_data = squeeze(tuning_curve_trials(c, :, :));
% 				activity_rand = zeros(size(activity_data));
% 				activity_rand2 = zeros(size(activity_data));
%
% 				n_trials = size(activity_data, 2);
% 				for t = 1:n_trials
% 					%% no interpolation of Nans here, but if needed, I guess we can add it?
% 					activity_rand2(:, t) = circshift(activity_data(:, t), randi(size(activity_data, 1)));
% 					activity_rand(:, t) = circshift(activity_data(:, t), randi(size(activity_data, 1)));
% 				end
%
% 				activity_data = activity_rand2;
% 				randvec = randperm(n_trials);
%
% 				bt_CC_data(iter) = obj.getBetweenTrialCC(activity_data, randvec);
% 				bt_CC_rand(iter) = obj.getBetweenTrialCC(activity_rand, randvec);
%
% 				% get random fitz
% 				tuning_rand = nanmean(activity_rand, 2);
% 				switch gauss_flag
% 					case 'single'
% 						[fit_shuff, gof_shuff] = obj.fitGaussians(tuning_rand, 'single');
% 					case 'double'
% 						[fit_shuff, gof_shuff] = obj.fitGaussians(tuning_rand, 'double');
% 				end
% 				amp_vec_shuff(iter) = fit_shuff.a1;
% 				gof_vec_shuff(iter) = gof_shuff.rsquare;
% 			end
%
% 			tuning_resp = nanmean(activity_data, 2);
%
% 			switch gauss_flag
% 				case 'single'
% 					[fit_func, gof] = obj.fitGaussians(tuning_resp, 'single');
% 				case 'double'
% 					[fit_func, gof] = obj.fitGaussians(tuning_resp, 'double');
% 			end
%
% 			% cc thresholding
% 			try
% 				% P = ranksum(bt_CC_data,bt_CC_rand,'tail','right');
% 				[~, P] = kstest2(bt_CC_data, bt_CC_rand);
% 				obj.significance_info.d(c) = obj.cohen_d(bt_CC_data, bt_CC_rand);
% 				nan_cell_vec(c) = 0;
% 			catch
% 				disp(['Error: Unable to calculate reliability for neuron #' num2str(c)])
% 				nan_cell_vec(c) = 1;
% 				P = 1;
% 				d = 0;
% 			end
%
% 			if P < 0.1 && obj.significance_info.d(c) > 0.8 % strong effect
% 				is_reliable(c) = true;
% 			end
% 			% gaussian thresholding
% 			obj.significance_info.fit_func{c} = fit_func;
% 			obj.significance_info.gof(c) = gof;
% 			obj.significance_info.gof_vec_shuff(c, :) = gof_vec_shuff;
%
% 			is_amp(c) = fit_func.a1 > 0;%prctile(amp_vec_shuff, 95);
% 			is_gof(c) = gof.rsquare > prctile(gof_vec_shuff, 90);
%
%
% 			obj.significance_info.bt_CC_data(c, :) = bt_CC_data;
% 			obj.significance_info.bt_CC_rand(c, :) = bt_CC_rand;
% 		end
% 		is_hd = is_reliable & is_amp & is_gof;
%
% 	end
%
% 	obj.is_head_direction = is_hd;
%
% end
