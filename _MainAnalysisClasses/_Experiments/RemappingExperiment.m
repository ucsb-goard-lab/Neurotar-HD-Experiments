classdef RemappingExperiment < ForcedRotation & SingleCueAnalyzer
	properties
	end

	methods
		function obj = RemappingExperiment(varargin)
			obj@SingleCueAnalyzer(varargin{:});
			obj@ForcedRotation();
		end

		function getRecordingParameters(obj)
			obj.n_repeats = 1;
			obj.segment_length = 950;
		end
	end
end
