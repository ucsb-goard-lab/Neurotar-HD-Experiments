classdef FreeRoamSingleExperiment < SingleCueAnalyzer & FreelyMoving
	properties
	end

	methods
		function obj = FreeRoamSingleExperiment(varargin)
			obj@SingleCueAnalyzer(varargin{:});
			obj@FreelyMoving();
		end
	end
end
