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
		% function out = calculateRayleighVector(obj, data)
		% 	% Similar to previous, a little different, because the binned data should only use moving data, not all data
		% 	if nargin < 2 || isempty(data)
		% 		data = obj.binData(obj.getNeuralData(), obj.getHeading, true);
		% 	end
		% 	for c = 1:size(data, 1)
		% 		vector_sum = obj.calculateVectorSum(data(c, :));
		% 		out(c) = vector_sum(2); % 1st is direction, second is magnitude
		% 	end
		% end
	end
end
