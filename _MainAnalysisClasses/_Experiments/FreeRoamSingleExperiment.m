classdef FreeRoamSingleExperiment < SingleCueAnalyzer & FreelyMoving
	properties
	end

	methods
		function obj = FreeRoamSingleExperiment(varargin)
			obj@SingleCueAnalyzer(varargin{:});
			obj@FreelyMoving();

			obj.tuning_curves = obj.getTuningCurves();
		end
	end
end
