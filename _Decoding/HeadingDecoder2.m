classdef HeadingDecoder2 < handle

	% Add some visualization methods later...

	properties
		tuning_wts
		time_series

		memory = 5 % Samples looking back for fitting a line
		window = 0 % Samples around a single point for heading distribution

		n_segments

		heading_distribution
		predicted_heading

		all_peak_centers
		fit_distribution
		target_idx
		scaling

		bin_centers
	end

	methods
		function obj = HeadingDecoder2(tuning_curve, time_series, bin_centers)
			% Concatenating the whole tuning curve for normalization, so that we don't have each phase independently normalized
			% original_dims = size(tuning_curve);
			% tuning_curve = reshape(tuning_curve, size(tuning_curve, 1), []);

% 			rescaled_weights =  rescale(tuning_curve); %rescale(tuning_curve, 'InputMin', min(tuning_curve, [], 2), 'InputMax', max(tuning_curve, [], 2));

			%             rescaled_weights = reshape(rescaled_weights, original_dims);
			% 			rescaled_weights = reshape(rescaled_weights, original_dims);

			obj.tuning_wts = tuning_curve; %rescaled_weights; %smoothdata(rescaled_weights, 2, 'sgolay'); %obj.getTuningWeights(rescaled_weights); %rescaled_weights; % obj.getTuningWeights(rescaled_weights);
			obj.time_series = time_series;
			obj.bin_centers = bin_centers;
			% obj.segments = segments;
			obj.n_segments = size(time_series, 3);
% 			obj.scaling = obj.getScaling();
		end

		function scaling = getScaling(obj)
			all_tc = rowrescale(nanmean(obj.tuning_wts, 3));
            all_tc(isnan(all_tc)) = 0;
			for c = 1:size(all_tc, 1)
                fprintf('%d / %d\n', c, size(all_tc, 1));
				[~, g] = obj.fitGaussians(all_tc(c, :), 'double');
				scaling(c) = g.rsquare;
			end

			
		end
		
		function [fit_out, gof_out] = fitGaussians(obj, data, type)
			if nargin < 3 || isempty(type)
				type = 'double';
			end

			if size(data, 2) > size(data, 1)
				data = transpose(data); % ensure column vector
			end

			x = linspace(0, 2*pi, size(data, 1));
			data = rescale(data);
			[~, max_idx] = max(smoothdata(data));
			centered_resp = circshift(data, round(length(data)/4) - max_idx);
			centered_resp(isnan(centered_resp)) = 0;
			wiggle = pi/6;
			switch type
				case 'double'
					% double gauss fit
					fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2)');

					wiggle = pi/6;
					initial_param = [0, 0.5, 0.5, pi/2, 3*pi/2, pi/3, pi/3];                     % initial parameters: [a0 a1 b1 c1]
					upper_bounds = [1, 1, 1, pi/2 + wiggle, 3*pi/2 + wiggle, 2*pi, 2*pi];
					lower_bounds = [0, 0, 0, pi/2 - wiggle, 3*pi/2 - wiggle, 0, 0];
					[double_gauss, double_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);

					% 					% check peak amplitudes
					% 					if double_gauss.a2 > double_gauss.a1 + 0.05
					% 						keyboard
					% 						% if second peak is greater, do a flippity flop
					% 						a_temp = double_gauss.a2;
					% 						b_temp = double_gauss.b2;
					% 						c_temp = double_gauss.c2;
					% 
					% 						double_gauss.a2 = double_gauss.a1;
					% 						double_gauss.b2 = double_gauss.b1;
					% 						double_gauss.c2 = double_gauss.c1;
					% 
					% 						double_gauss.a1 = a_temp;
					% 						double_gauss.b1 = b_temp;
					% 						double_gauss.c1 = c_temp;
					% 					end

					fit_out = double_gauss; %
					gof_out = double_gof;
				case 'single'
					% single gauss fit
					fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2)'); % define function: gaussian w/ baseline
					initial_param = [0, 0.5, pi/2, pi/6];                     % initial parameters: [a0 a1 b1 c1]
					upper_bounds = [1, 1, pi/2 + wiggle, 2*pi];
					lower_bounds = [0, 0, pi/2 - wiggle, 0];
					[single_gauss, single_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);

					% unshift
					fit_out = single_gauss; %
					gof_out = single_gof;
			end
			%{
			if single_gof.rsquare > double_gof.rsquare
			fit_out = double_gauss;
			gof_out = double_gauss;
			fit_type = 1;
			elseif double_gof.rsquare > single_gof.rsquare
			fit_out = double_gauss; %
			gof_out = double_gof;
			fit_type = 2;
			else % usually because can't calculate reliability, bad cell
			% adjust some junk
			fit_out = single_gauss;
			gof_out = single_gof;

			fit_out.a0 = NaN;
			fit_out.a1 = NaN;
			fit_out.b1 = NaN;
			fit_out.c1 = NaN;

			gof_out.rsquare = 0;
			fit_type = NaN;
			end
			%}

		end
		function decode(obj)
			%             Main pipeline
			obj.calculateHeadingDistribution();
			obj.chooseHeading_old();
			% obj.calculateHeading_beta();
		end

		function out = getHeadingTrace(obj, rescale_flag)
			if nargin < 2 || isempty(rescale_flag)
				rescale_flag = 0;
			end

			for ii = 1:length(obj.predicted_heading)
				if isnan(obj.predicted_heading(ii))
					out(ii) = NaN;
				else
					out(ii) = obj.bin_centers(obj.predicted_heading(ii));
				end
			end
			% 			out = obj.bin_centers(obj.predicted_heading);
			%             if rescale_flag
			%                 out = rescale(obj.predicted_heading) * 71 + 1;
			%             else
			%                 out = obj.predicted_heading;
			%             end
		end

		function out = getHeadingDistribution(obj)
			if isempty(obj.heading_distribution)
				obj.calculateHeadingDistribution();
			end
			out = obj.heading_distribution;
		end

		function prediction_error = calculateDecoderError_prototype(obj, heading, segment)
			warning('Using prototype decoder, 31Aug2020')
			% Calculating error circularly, so that 360 -> 1 = 1
			predicted_heading = obj.predicted_heading(segment);
			true_heading = heading(segment);
			for ii = 1:length(true_heading)
				if predicted_heading(ii) < true_heading(ii)
					t = true_heading(ii);
					true_heading(ii) = predicted_heading(ii);
					predicted_heading(ii) = t;
				end
				dist = predicted_heading(ii) - true_heading(ii) + 1;
				dist2 = true_heading(ii) - 1 + 360 - predicted_heading(ii);
				prediction_error(ii) = min(dist, dist2);
			end
			% cap = prctile(prediction_error, 80); % 90th percentile
			% prediction_error(prediction_error > cap) = cap; % cap it
		end

		%{
		function prediction_error = calculateDecoderError(obj, true_heading)
		% Calculating error circularly, so that 360 -> 1 = 1
		binned_heading = discretize(true_heading, size(obj.tuning_wts, 2));
		predicted_heading = obj.predicted_heading;

		x = linspace(0, 2*pi, size(obj.tuning_wts, 2));

		is_double_nan = isnan(binned_heading) | isnan(predicted_heading)';

		prediction_error_non_nan = angdiff(x(binned_heading(~is_double_nan)), x(predicted_heading(~is_double_nan)));
		prediction_error = nan(1, length(binned_heading));
		prediction_error(~is_double_nan) = prediction_error_non_nan;
		%{
		for ii = 1:length(true_heading)
		if predicted_heading(ii) < true_heading(ii)
		t = true_heading(ii);
		true_heading(ii) = predicted_heading(ii);
		predicted_heading(ii) = t;
	end
	dist = predicted_heading(ii) - true_heading(ii) + 1;
	dist2 = true_heading(ii) - 1 + 360 - predicted_heading(ii);
	prediction_error(ii) = min(dist, dist2);
end
%}

end
%}
end

methods (Access = public)
	%{ 
	%This is more of a standard vector sum thing, but doesn't seem to work that well..
	function phase = getPhases(obj, tuning_curves)
	x = linspace(0, 2*pi, size(tuning_curves, 2)); % set x matrix
	cosine_func = @(a, b, c, d, x) a * cos(b * x + c ) + d; 
	upper_lim = [Inf, 1, 2*pi, Inf]; % b = 3 limits the minimum period to be 2/3 * pi (120deg)
	lower_lim = [0, 1, 0, 0]; % b = 1 limits maximum period to be 2*pi (360deg)
	for c = 1:size(tuning_curves, 1)
		[~, maxidx] = max(tuning_curves(c, :));
		cosine_fit = fit(x', tuning_curves(c, :)', cosine_func, 'Start', [range(tuning_curves(c, :)), 1, x(maxidx), mean(tuning_curves(c, :))], 'Upper', upper_lim, 'Lower', lower_lim);
		phase(c) = cosine_fit.c;
	end
end
function calculateHeading_beta(obj)
	time_series = obj.time_series;
	tuning_wts = obj.tuning_wts;
	tuning_wts(isnan(tuning_wts)) = 0;
	phase = obj.getPhases(tuning_wts);

	for t = 1:size(time_series, 2)
		x = time_series(:, t)' .* cos(phase);
		y = time_series(:, t)' .* sin(phase);
		x(x == 0) = [];
		y(y == 0) = [];
		phi(t) = atan2d(mean(x), mean(y));
	end

	obj.predicted_heading = phi;
end
%}

function tuning_wts = getTuningWeights(obj, tuning_curves)
	tuning_curves(isnan(tuning_curves)) = 0;
	x = linspace(0, 2*pi, size(tuning_curves, 2));
	fixed_gauss = @(a1, a2, b, c1, c2, z, x) a1 * exp(-((x - b)/ c1).^2) + a2 * exp(-((x - (b + pi))/c2).^2) + z;
	for ii = 1:size(tuning_curves, 1)
		y = tuning_curves(ii, :);
		[~, maxidx] = max(y);   

		f = fit(x', y', fixed_gauss, 'StartPoint', [max(y), x(maxidx), pi/6, max(y), pi/6, 0], 'Upper', [1, 2*pi, pi/2, 1, pi/2, 5], 'Lower', [0, 0, 0, 0, 0, 0]);
        tuning_wts(ii, :) = f(x);
    end
end


function calculateHeadingDistribution(obj)
    % time_series = obj.time_series;
    % tuning_wts = obj.tuning_wts;
    
    % gof measure
    for s = 1:obj.n_segments
        time_series = obj.getTs(s);
        tuning_wts = obj.getTw(s);
        %         time_series(time_series == 0) = nan;
        %         tuning_wts(tuning_wts == 0) = nan;
%         tuning_wts = rowrescale(tuning_wts) .* obj.scaling';
        tuning_wts = rowrescale(tuning_wts); %
        % remove nans
        nan_row = any(isnan(tuning_wts)');
        tuning_wts(nan_row, :) = [];
        time_series(nan_row, :) = [];
        heading_distribution = (time_series' * tuning_wts)./sum(time_series' * tuning_wts);
        out = obj.temporalFilter(heading_distribution', 6)';
	% out = heading_distribution;
% if s > 9 && s < 20
%         disp(s)
%         imagesc(rowrescale(out)')
%         pause
% end
%
%         keyboard%                 out = out./sum(out);
        %         heading_distribution = heading_distribution ./ sum(heading_distribution);
        % Calculate the heading distribution based on a linear sum of scaled tuning curve at each time point
        % 		heading_distribution = zeros(size(time_series, 2), size(tuning_wts, 2));
        % 		for t = 1:size(time_series, 2)
        % 			activity = obj.calculateActivity(time_series, t, 'normal');
        % 			is_weak_contributor = activity <= 0;
        % 			if all(is_weak_contributor)
        % 				heading_distribution(t, :) = 0;
        % 			else
        % 				heading_distribution(t, :) = sum(activity(~is_weak_contributor) .* tuning_wts(~is_weak_contributor, :), 1) ./ sum(tuning_wts(~is_weak_contributor, :), 1); % nonzero weights are lost;
        % 			end
        % 		end	% filter it
        % % 		out = obj.temporalFilter(heading_distribution', 0.5)';
        %         subplot(1, 2, 1)
        %         imagesc(rowrescale(heading_distribution)')
        %         subplot(1, 2, 2)
        %         imagesc(rowrescale(out)')
        %
        
        heading_distribution_all{s} = out;
    end
    % Assign heading distribution
    obj.heading_distribution = heading_distribution_all;
end

function [tw] = getTw(obj, s)
    tw = obj.tuning_wts(:, :, s);
    % tw = nanmean(obj.tuning_wts(:, :, 1:end~=s), 3);
	% tw = nanmean(obj.tuning_wts(:, :, setdiff(max([1, s - 5]):min([size(obj.tuning_wts, 3), s + 5]), s)), 3);
end

function [ts] = getTs(obj, s)
    ts = obj.time_series(:, :, s);
% 	ts = obj.time_series(:, obj.segments{s});
end

function out = calculateActivity(obj, time_series, t, calculation_method)
	if nargin < 4 || isempty(calculation_method)
		calculation_method = 'normal';
	end

	switch calculation_method
		case 'normal'
			out = mean(time_series(:, max(t - obj.window, 1) : min(t + obj.window, size(time_series, 2))), 2);
		case 'nonlinear'
			decay = 0.10; % guess
			decay_fcn = @(x) (1 - decay).^x;
			center = time_series(:, t);
			left = time_series(:, max(t - obj.window, 1): t-1);
			right = time_series(:, t+1 : min(t + obj.window, length(time_series)));
			left = fliplr(left) .* decay_fcn(1:size(left, 2)); % fliplr because of ordering
			right = right .* decay_fcn(1:size(right, 2));
			out = mean(cat(2, center, left, right), 2);
	end

	% add a cap?
	%             out(out < prctile(out, 50)) = 0; %prctile(out, 99);
end

function chooseHeading(obj)
	disp('Choosing with max, test both when it comes to normal distributions later, 31Aug2020')
	for s = 1:obj.n_segments
		predicted_heading = [];
		heading_distribution = obj.heading_distribution{s};
		for ii = 1:size(heading_distribution, 1)
			candidates = islocalmax(heading_distribution(ii, :), 'MinProminence', 0.5);
			peak_centers = find(candidates);
			if ~isempty(peak_centers)
				[~, max_idx] = max(heading_distribution(ii, peak_centers));
				predicted_heading(ii) = peak_centers(max_idx);
			else
				predicted_heading(ii) = NaN;
				%{
				if ~obj.checkHeadingDistribution(heading_distribution(ii, :))
				[~, obj.predicted_heading(ii)] = max(heading_distribution(ii, :)); % shift to -180 to 180 range
				%             obj.predicted_heading = rescale(obj.predicted_heading, -180, 180);
			else
				obj.predicted_heading(ii) = NaN;
			end
			%}
		end
	end
	predicted_heading_all{s} = predicted_heading;
end
obj.predicted_heading = predicted_heading_all;
end

function [filt_decoded_probabilities] = temporalFilter(obj, decoded_probabilities, window_size)
	%BAYESIAN_TEMPORAL_FILTER Summary of this function goes here
	%   Detailed explanation goes here
% keyboard
% 	ca_time = 0.1:0.1:size(decoded_probabilities, 2);
% 	Fs = 1/mode(diff(ca_time));
	% window_size = round(window_size*Fs);
	half_win = round(window_size/2);

	%% Pad matrix with zeros
	zero_pad = zeros(size(decoded_probabilities,1),window_size);
	padded_decoded_probabilities = [zero_pad decoded_probabilities zero_pad];

	for step_i = 1:size(decoded_probabilities,2)
		current_window = padded_decoded_probabilities(:,step_i+half_win: step_i-1+window_size+half_win);
		filt_decoded_probabilities(:,step_i) = expm1(sum(log1p(current_window),2,'omitnan'));
		filt_decoded_probabilities(:,step_i) = filt_decoded_probabilities(:,step_i)./sum(filt_decoded_probabilities(:,step_i));
	end

end

function out = checkHeadingDistribution(obj, data)
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
	if m < 0.2
		out = false;
	else
		out = true;
	end
end

function chooseHeading_old(obj)
    choose_method = 'max';
    switch choose_method
        case 'max'
            % From the heading distribution choose the actual heading, largely from Nora
            % 			heading_distribution = obj.heading_distribution;
            
            for s = 1:obj.n_segments
                
                [~, predicted_idx] = max(obj.heading_distribution{s}, [], 2);
                predicted_heading{s} = obj.bin_centers(predicted_idx);
            end
            obj.predicted_heading = predicted_heading;
            
            %{
	obj.predicted_heading = predicted_heading;
	subplot(2, 1, 1)
	imagesc(flipud(rowrescale(cat(2, obj.heading_distribution{:}))));
	subplot(2, 1, 2)
	plot(cat(2, predicted_heading{:}));
            %}
            
        case 'predict'
            for s = 1:obj.n_segments
                heading_distribution = obj.heading_distribution{s};
                predicted_heading = [];
                for ii = 1:size(heading_distribution, 1)
                    %                     threshold = mean(heading_distribution(ii, :) + std(heading_distribution(ii, :)));
                    
                    %                     candidate_peaks = heading_distribution(ii, :) > threshold;
                    %                     candidate_peaks = bwmorph(candidate_peaks, 'clean'); % remove singletons
                    
                    candidates = islocalmax(rescale(movmean(heading_distribution(ii, :), 5)), 'MinProminence', 0.5);
                    peak_centers = find(candidates);
                    %                     peak_centers = fix(obj.getPeakCenters(candidate_peaks)); % used to allow half indices, but now no good because we use the index
                    if ~isempty(peak_centers)
                        % Predict next point
                        if ii <= obj.memory
                            predicted_heading(ii) = min(peak_centers);
                        else
                            x = (ii - obj.memory) : (ii - 1);
                            prediction = obj.predictNextPeak(x, predicted_heading);
                            if numel(peak_centers) > 1
                                target_idx = obj.minCircularDistance(peak_centers, prediction);% min(abs(peak_centers - prediction)); % account for wrapping
                            else
                                target_idx = 1;
                            end
                            predicted_heading(ii) = peak_centers(target_idx);
                            obj.target_idx(ii) = target_idx;
                            temp(ii) = prediction;
                        end
                        
                    else
                        predicted_heading(ii) = NaN;
                    end
                    obj.all_peak_centers{ii} = peak_centers;
                end
                predicted_heading_all{s} = nan(1, length(predicted_heading));
                predicted_heading_all{s}(~isnan(predicted_heading)) = obj.bin_centers(predicted_heading(~isnan(predicted_heading)));
            end
    % 			predicted_heading(isnan(predicted_heading)) = 0;
    %             obj.predicted_heading = predicted_heading + 1;
    obj.predicted_heading = predicted_heading_all;
    end    
end

function prediction = predictNextPeak(obj, x, predicted_heading)
	peak_fit = polyfit(x, predicted_heading(x), 1);
	if peak_fit(:, 1) >= -0.1 && peak_fit(:, 1) <= 0.1
		prediction = mean(predicted_heading(x));
	else
		prediction = polyval(peak_fit, x(end) + 1);
	end
	% wrap
	if prediction > length(obj.bin_centers)
		prediction = prediction - length(obj.bin_centers);
	elseif prediction < 0
		prediction = prediction + length(obj.bin_centers);
	end
end

function peak_centers = getPeakCenters(obj, candidate_peaks)
	% Find rises and falls (because peaks can be consecutive values above threshold, we don't want to count each point as an individual peak)
	change = diff(candidate_peaks(:));
	% n_peaks = ceil((nnz(unique(cumsum([true;diff(change)~=0]).*(change~=0))))/2);
	peak_locs = find(candidate_peaks);
	split = find([0 (diff(peak_locs)) > 1]);
	n_peaks = numel(split) + 1;
	if n_peaks > 1
		for p = 1:n_peaks
			if p == 1
				peak_group{p} = peak_locs(1:split(p) - 1);
			elseif p == numel(split) + 1
				peak_group{p} = peak_locs(split(p - 1):end);
			else
				peak_group{p} = peak_locs(split(p - 1) : split(p) - 1);
			end
			peak_centers = cell2mat(vertcat(cellfun(@(x) mean(x(:)), peak_group, 'UniformOutput', false)));
		end
	else
		peak_centers = mean(peak_locs);
	end
end

function idx = minCircularDistance(obj, x, y)
	% ensure p2 >= p1 always
	% forward dist
	dist = y - x + 1;
	% rev dist
	dist2 = x - 1 + size(obj.heading_distribution, 2) - y;
	for ii = 1:length(dist) % for each possible peak, choose the smaller
		candidates(ii) = min(abs([dist(ii), dist2(ii)]));
	end
	[~, idx] = min(candidates);
end
end
end
