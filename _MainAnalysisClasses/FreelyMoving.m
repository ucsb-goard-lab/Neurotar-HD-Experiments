classdef FreelyMoving < ExperimentStructure
	% Expansion of the standard Analyzer for freely moving recording
	properties
		is_moving
	end

	methods
		function obj = FreelyMoving(varargin)
			obj = obj@ExperimentStructure(varargin{:});
			obj.checkIsMoving();
		end

		function checkIsMoving(obj, threshold)
			% check for times that are greater than a threshold for moving times only
			if nargin < 2 || isempty(threshold)
				threshold = 2;
			end
			speed = obj.data.get('speed');
			obj.is_moving = speed > threshold;
		end

		function out = getNeuralData(obj)
			% Take only moving times
			nd = obj.getNeuralData@ExperimentStructure();
			out = nd(:, obj.is_moving);
		end

		function out = getHeading(obj)
			% Same as above, just for heading
			hd = obj.getHeading@ExperimentStructure();
			out = hd(obj.is_moving);
		end

		function calculateHeadDirection(obj, iterations, thresh)
			if nargin < 2 || isempty(iterations)
				iterations = 1000;
			end

			if nargin < 3 || isempty(thresh)
				thresh = 95;
			end

			real_rvls = obj.calculateRayleighVector();
 			
			neural_data = obj.getNeuralData();
			heading = obj.getHeading();
			for iter = 1:iterations
				fprintf('Iteration: %d/%d\n', iter, iterations)
				shuffled_data = circshift(neural_data, randi(size(neural_data, 2)), 2); % circularly shift the data
				shuffled_tc = obj.binData(shuffled_data, heading); % get shuffled tuning curves
				obj.significance_info.rvl_shuffled(:, iter) = obj.calculateRayleighVector(shuffled_tc); % get shuffled rvl
			end

			obj.is_head_direction = real_rvls' > prctile(obj.significance_info.rvl_shuffled, thresh, 2);
		end
	end
end
