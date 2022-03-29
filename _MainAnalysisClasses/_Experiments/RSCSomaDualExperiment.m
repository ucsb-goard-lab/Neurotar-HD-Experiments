classdef RSCSomaDualExperiment < LightDarkExperiment
	properties
	end

	methods
		function obj = RSCSomaDualExperiment(varargin);
			obj@LightDarkExperiment(varargin{:});
		end

		function getRecordingParameters(obj)
			% this is to (unfortunately) account for differences in older/newer recordings with slightly different parameters
			switch obj.recording_length
				case 1950
					obj.n_repeats = 1;
					obj.segment_length = 950;
				case 7550
					obj.segment_length = 1250;
					obj.n_repeats = 3;
			end
		end
	end
end
