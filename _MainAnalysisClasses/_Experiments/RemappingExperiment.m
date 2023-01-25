classdef RemappingExperiment < ControlledRotation & SingleCue
	properties
	end

	methods
		function obj = RemappingExperiment(varargin)
			obj@SingleCue(varargin{:});
			obj@ControlledRotation();
		end

		function getRecordingParameters(obj)
			obj.n_repeats = 1;
			obj.segment_length = 950;
		end
	end
end
