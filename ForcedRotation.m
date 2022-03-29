classdef ForcedRotation < ExperimentStructure
	properties
	end

	methods
		function obj = ForcedRotation()
			obj = obj@ExperimentStructure();
		end
		function is_hd = calculateHeadDirection(obj, iterations)
			% To determine which cells are heading selective
				tuning_curve_trials = obj.getTrialResponses();

			if nargin < 2 || isempty(iterations)
				iterations = 100; % this is set low in case you accidentally call it, actually 1000 iterations for real data
			end

			if ndims(tuning_curve_trials) < 3
				% Some low quality recordings have very few trials, those are just skipped and not included
				disp('Bad recording, single trials only, skipping...')
				is_hd = false(1, size(tuning_curve_trials, 1));
			else
				is_reliable = false(1, size(tuning_curve_trials, 1));
				for c = 1:size(tuning_curve_trials, 1)
					fprintf('\t%d/%d\n', c, size(tuning_curve_trials, 1))
					bt_CC_data = zeros(1, iterations);
					bt_CC_rand = zeros(1, iterations);

					for iter = 1:iterations
						% Calculating shuffle stuff over many iterations

						% Create the randomized data by circularly shifting across trials
						activity_data = squeeze(tuning_curve_trials(c, :, :));
						activity_rand = zeros(size(activity_data));

						n_trials = size(activity_data, 2);
						for t = 1:n_trials
							activity_rand(:, t) = circshift(activity_data(:, t), randi(size(activity_data, 1)));
						end
						randvec = randperm(n_trials);

						% Calculate between trial CC, based on half-half splits.
						bt_CC_data(iter) = obj.getBetweenTrialCC(activity_data, randvec);
						bt_CC_rand(iter) = obj.getBetweenTrialCC(activity_rand, randvec);

						% Fitting gaussians
						tuning_rand = nanmean(activity_rand, 2);
						[fit_shuff, gof_shuff] = obj.fitGaussians(tuning_rand);

						% Store important data from each iteration
						amp_vec_shuff(iter) = fit_shuff.a1;
						gof_vec_shuff(iter) = gof_shuff.rsquare;
					end

					% Get the real tuning response and fit Gaussian for the true r-squared
					tuning_resp = nanmean(activity_data, 2);

					[fit_func, gof] = obj.fitGaussians(tuning_resp);

					% Compare real CC vs shuffled CC for significance
					try
						[~, P] = kstest2(bt_CC_data, bt_CC_rand);
						obj.significance_info.d(c) = obj.cohen_d(bt_CC_data, bt_CC_rand);
						nan_cell_vec(c) = 0;
					catch
						disp(['Error: Unable to calculate reliability for neuron #' num2str(c)])
						nan_cell_vec(c) = 1;
						P = 1;
						d = 0;
					end

					% Include a passable Cohen's D for a strong separation between distributions
					if P < 0.1 && obj.significance_info.d(c) > 0.8 % strong effect
						is_reliable(c) = true;
					end

					% Gaussian information
					obj.significance_info.fit_func{c} = fit_func;
					obj.significance_info.gof(c) = gof;
					obj.significance_info.gof_vec_shuff(c, :) = gof_vec_shuff;
					is_amp(c) = fit_func.a1 > 0;%prctile(amp_vec_shuff, 95);
					is_gof(c) = gof.rsquare > prctile(gof_vec_shuff, 90);

					obj.significance_info.bt_CC_data(c, :) = bt_CC_data;
					obj.significance_info.bt_CC_rand(c, :) = bt_CC_rand;
				end
				is_hd = is_reliable & is_amp & is_gof; % requires a reliable response, amplitude of gauss > 0, and a goodness of fit better than shuffled

			end
			obj.is_head_direction = is_hd;
		end


		% function out = calculateRayleighVector(obj, data)
		% 	% Calculate folded RVL
		% 	if nargin < 2 || isempty(data)
		% 		data = obj.binData([], [], true); % ask for folded data
		% 	end
		% 	% fold the tuning curve
		% 	for c = 1:size(data, 1)
		% 		vector_sum = obj.calculateVectorSum(data(c, :));
		% 		out(c) = vector_sum(2); % 1st is direction, second is magnitude
		% 	end
		% end
	function cc = getBetweenTrialCC(obj, data, random_vec)
			% random_vec needs to be passed in because it's conserved across true and random
			first_half_idx = random_vec(1 : round(size(data, 2) / 2));
			second_half_idx = random_vec(round(size(data, 2) / 2 + 1 : end));
			cc = corr(nanmean(data(:, first_half_idx), 2), nanmean(data(:, second_half_idx), 2));
		end

		function d = cohen_d(obj, x1, x2)
			n1       = numel(x1);
			n2       = numel(x2);
			mean_x1  = nanmean(x1);
			mean_x2  = nanmean(x2);
			var_x1   = nanvar(x1);
			var_x2   = nanvar(x2);
			meanDiff = (mean_x1 - mean_x2);
			sv1      = ((n1-1)*var_x1);
			sv2      = ((n2-1)*var_x2);
			numer    =  sv1 + sv2;
			denom    = (n1 + n2 - 2);
			pooledSD =  sqrt(numer / denom); % pooled Standard Deviation
			s        = pooledSD;             % re-name
			d        =  meanDiff / s;        % Cohen's d (for independent samples)
		end

		% function out = calculateVectorSum(obj, data)
		% 	% Calculate vector sum of a tuning curve
		% 	getHorz = @(v, theta) v .* cos(theta);
		% 	getVert = @(v, theta) v .* sin(theta);
		% 	getAng = @(vert, horz) atan2(vert, horz);
		% 	getMag = @(vert, horz) sqrt(horz ^ 2 + vert ^ 2);
                %
		% 	theta_step = 2*pi/length(data);
		% 	theta = 0 : theta_step : 2*pi - theta_step;
		% 	h = getHorz(data, theta);
		% 	v = getVert(data, theta);
		% 	% Changed from sum to mean, shouldn't change anything... more similar to Giocomo
		% 	r_h = nanmean(h);
		% 	r_v = nanmean(v);
                %
		% 	m = getMag(r_v, r_h);
		% 	ang = getAng(r_v, r_h);
		% 	out = [ang, m];
                %
		% end
                %


		function [tuning_curve_trials, segment] = getTrialResponses(obj, data_structure)
			% Trial by trial response, defined as a full rotation of the cage
			if nargin < 2 || isempty(data_structure)
				data_structure = obj.data;
			end

			% extract heading and timeseries
			heading = [];
			dff = []; % not actually dff, but whatever data_type you're using (generally spikes)
			for d = data_structure
				heading = cat(1, heading, d.get('heading'));
				dff = cat(2, dff, d.get(obj.data_type));
			end
			heading(isnan(heading)) = 0;
			heading = heading(:); % force into a column vector

			% clean up the heading a little bit
			heading_ref = heading(1);
			is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350; % weird spikes from the recorded neurotar stuff, this is to remove those
			ct = 1;
			while any(is_jump)
				if ct == 10
					break
				end
				heading(is_jump) = heading(max(find(is_jump) - 1, 1)); % "interpolate' jumps via nearest neighbor (one way)
				is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350;
				ct = ct + 1;
			end

			% Determine when the heading "jumps" from -180 to 180
			dist_from_ref = pdist2(heading, heading_ref);
			[val, jumps] = findpeaks(dist_from_ref - max(dist_from_ref));
			jumps(abs(val) > 10) = []; % remove tiny bumps that are not because of a revolution

			% Find times when the cage is stopped (rest periods)
			slow_move = find(abs(diff(heading)) < 0.5);
			n = 10; % 10 or more stopped frames
			x = diff(slow_move') == 1;
			f = find([false, x] ~= [x, false]);
			g = find(f(2:2:end) - f(1:2:end-1) > n);
			stops_idx = [f(2*g-1); f(2*g)];

			stops = [];
			for s = 1:size(stops_idx, 2)
				stops = cat(1, stops, slow_move(stops_idx(1, s):stops_idx(2, s)));
			end

			% Either stops or jumps need to be accounted for
			breaks = sort(cat(1, jumps, stops));

			for ii = 1:length(breaks) - 1
				segment{ii} = breaks(ii) + 1:breaks(ii+1);
			end

			segment(cellfun(@length, segment) < 50) = []; % cut out segments too short to feasibly be full rotations
			is_not_full = false(length(segment), 1);
			for ii = 1:length(segment)
				% the following is necessary. range doesn't work b/c sometimes you pick up a lil jump from the previous one, and it leads to bad stuff
				is_not_full(ii) = any(90 - abs(prctile(heading(segment{ii}), [25, 75])) > 50); % should both be around 90
			end

			% Calculate the actual tuning curves using the standard binData method
			segment(is_not_full) = [];
			tuning_curve_trials = zeros(size(dff, 1), 360/obj.bin_width, length(segment));
			for ii = 1:length(segment)
				d = dff(:, segment{ii});
				h = heading(segment{ii});
				tuning_curve_trials(:, :, ii) = obj.binData(d, h);
			end
			tuning_curve_trials(isnan(tuning_curve_trials)) = 0;
		end

	end
end


