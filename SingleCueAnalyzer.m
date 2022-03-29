classdef SingleCueAnalyzer < AnalyzerNew
    properties
    end

    methods
        function obj = SingleCueAnalyzer(varargin)
            obj = obj@AnalyzerNew(varargin{:});
	    obj.cue_flag = 'single';
        end
    end
end
