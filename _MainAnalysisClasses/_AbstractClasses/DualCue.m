classdef DualCue < Cue
	% A small change on the Analyzer to account for dual cues which have the potential to create dual peaked tuning curves.
	% Essentially, for each method, it overwrites it with a version that folds the tuning curves to ensure no issues
	properties
	end
	methods
		function obj = DualCue(varargin)
			obj = obj@Cue(varargin{:});
			obj.cue_flag = 'double';
		end
		end
end
