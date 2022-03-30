classdef SingleCue < Cue
    properties
    end

    methods
        function obj = SingleCue(varargin)
            obj = obj@Cue(varargin{:});
	    obj.cue_flag = 'single';
        end
    end
end
