classdef ExperimentStructure < handle
	properties
	end

	methods
		function obj = ExperimentStructure()
		end

		function out = getNeuralData(obj);	
			out = obj.data.get(obj.data_type);
		end
		function out = getHeading(obj);
			out = obj.data.get('heading');
		end
	end
end
