classdef FreeRoamDualExperiment < DualCue & FreelyMoving
	properties
	end

	methods
		function obj = FreeRoamDualExperiment(varargin)
			obj@DualCue(varargin{:});
			obj@FreelyMoving();

			obj.tuning_curves = obj.getTuningCurves();
		end
	end
end
