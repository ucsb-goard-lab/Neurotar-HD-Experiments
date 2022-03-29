classdef RSCSomaSingleExperiment < LightDarkExperiment
	properties
	end

	methods
		function obj = RSCSomaSingleExperiment(varargin)
			obj@LightDarkExperiment(varargin{:});
		end

		function getRecordingParameters(obj)
			obj.n_repeats = 1;
			obj.segment_length = 950;
		end
	end
end
