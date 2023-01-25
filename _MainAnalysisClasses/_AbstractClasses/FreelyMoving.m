classdef FreelyMoving < Control
	% Expansion of the standard Analyzer for freely moving recording
	properties
		is_moving
	end

	methods
		function obj = FreelyMoving(varargin)
			obj = obj@Control(varargin{:});
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
			nd = obj.getNeuralData@Control();
			out = nd(:, obj.is_moving);
		end

		function out = getHeading(obj)
			% Same as above, just for heading
			hd = obj.getHeading@Control();
			out = hd(obj.is_moving);
		end
		function out = distanceBin(obj)
			n_distances = 2;
			% get distance first?
			% x = obj.data.get('X');
			% y = obj.data.get('Y');
			% x = x(obj.is_moving);
			% y = y(obj.is_moving);
			% distance = sqrt(x.^2 + y.^2);
			distance = obj.data.get('r');
			distance = distance(obj.is_moving);	
			min_dist = 0;
			max_dist = 120;
			bin_distance = discretize(distance, [0, 60]);%linspace(min_dist, max_dist, n_distances + 1));
			data = obj.getNeuralData();
			heading = obj.getHeading();

			out = zeros(size(data, 1), length(-180:obj.bin_width:180) - 1, n_distances);
			for d = 1:n_distances
				current_data = data(:, bin_distance == d);
				current_heading = heading(bin_distance == d);
				out(:, :, d) = obj.binData(current_data, current_heading);
			end
		end


		% 		function out = getHeading(obj)
		% 			% calculated as a function of proximity to center (for some reviewer stuff)
		% 			x = obj.data.get('X');
		% 			y = obj.data.get('Y');
		% 			dist = sqrt(x.^2 + y.^2);
		%             dist = (rescale(dist) * 360) - 180;
		% 			out = dist(obj.is_moving);
		% 		end

		function calculateHeadDirection(obj, iterations, thresh)
			if nargin < 2 || isempty(iterations)
				iterations = 100;
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
