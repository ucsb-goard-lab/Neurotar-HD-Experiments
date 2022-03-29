classdef DualCueAnalyzer < AnalyzerNew
	% A small change on the Analyzer to account for dual cues which have the potential to create dual peaked tuning curves.
	% Essentially, for each method, it overwrites it with a version that folds the tuning curves to ensure no issues
	properties
	end
	methods
		function obj = DualCueAnalyzer(varargin)
			obj = obj@AnalyzerNew(varargin{:});
			obj.cue_flag = 'double';
		end
		% function [fit_out, gof_out] = fitGaussians(obj, data, type, rescale_flag)
		% 	% Fitting constrained Gaussians to the data
		% 	if nargin < 3 || isempty(type)
		% 		type = 'double';
		% 	end
                %
		% 	if nargin < 4 || isempty(rescale_flag)
		% 		rescale_flag = true;
		% 	end
                %
		% 	if size(data, 2) > size(data, 1)
		% 		data = transpose(data); % ensure column vector
		% 	end
                %
		% 	x = linspace(0, 2*pi, size(data, 1));
		% 	if rescale_flag
		% 		data = rescale(data);
		% 	else
		% 		fprintf('No rescaling performed, make sure your data is in the 0 - 1 range\n');
		% 	end
		% 	% First center the data so that the peak doesn't end up near the ends
		% 	[~, max_idx] = max(smoothdata(data));
		% 	centered_resp = circshift(data, round(length(data)/4) - max_idx);
		% 	centered_resp(isnan(centered_resp)) = 0;
                %
		% 	wiggle = pi/6;
		% 	switch type
		% 		case 'double'
		% 			% double gauss fit
		% 			fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2)');
                %
		% 			wiggle = pi/12;
		% 			initial_param = [0, 0.5, 0.5, pi/2, 3*pi/2, pi/6, pi/6];                     % initial parameters: [a0 a1 b1 c1]
		% 			upper_bounds = [1, 1, 1, pi/2 + wiggle, 3*pi/2 + wiggle, 2*pi, 2*pi]; % set constraints
		% 			lower_bounds = [0, 0, 0, pi/2 - wiggle, 3*pi/2 - wiggle, 0, 0];
		% 			[double_gauss, double_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
		% 			fit_out = double_gauss; 
		% 			gof_out = double_gof;
		% 		case 'single'
		% 			% single gauss fit
		% 			fit_func = fittype('a0+a1*exp(-((x-b1)/c1)^2)'); % define function: gaussian w/ baseline
		% 			initial_param = [0, 0.5, pi/2, pi/6];                     % initial parameters: [a0 a1 b1 c1]
		% 			upper_bounds = [1, 1, pi/2 + wiggle, pi/2];
		% 			lower_bounds = [0, 0, pi/2 - wiggle, 0];
		% 			[single_gauss, single_gof] = fit(x', centered_resp, fit_func, 'StartPoint', initial_param, 'Upper', upper_bounds, 'Lower', lower_bounds);
		% 			fit_out = single_gauss; 
		% 			gof_out = single_gof;
		% 	end
		% 	% Don't forget at this point that the location of the peak (c) is wrt the shifted location, so you'll need to unshift it later
		% end



	end
end
