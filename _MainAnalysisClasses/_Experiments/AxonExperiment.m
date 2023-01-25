classdef AxonExperiment < LightDarkExperiment
	properties
	end

	methods
		function obj = AxonExperiment(varargin)
			obj@LightDarkExperiment(varargin{:});
		end

		function getRecordingParameters(obj)
			obj.n_repeats = 3;
			obj.segment_length = 1250;
		end
	end
end

