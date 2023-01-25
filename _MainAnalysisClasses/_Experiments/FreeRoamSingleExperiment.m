classdef FreeRoamSingleExperiment < SingleCue & FreelyMoving
	properties
	end

	methods
		function obj = FreeRoamSingleExperiment(varargin)
			obj@SingleCue(varargin{:});
			obj@FreelyMoving();

			obj.tuning_curves = obj.getTuningCurves();
		end
	end
end
