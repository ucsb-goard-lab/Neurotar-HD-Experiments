classdef HeadRotationExperiment < ControlledRotation & SingleCue
	properties
	end
	methods
		function obj = HeadRotationExperiment(varargin)
			obj@SingleCue(varargin{:});
			obj@ControlledRotation();
		end

		function getRecordingParameters(obj)
			switch obj.recording_length
				case 2550
					obj.segment_length = 2550;
					obj.n_repeats = 1;
				case 2600
					obj.segment_length = 2550;
					obj.n_repeats = 1;
				case 5100
					obj.segment_length = 5000;
					obj.n_repeats = 1;
			end
		end
	end
end
