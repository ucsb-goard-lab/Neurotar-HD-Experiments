classdef Control < handle
	properties
	end

	methods
		function obj = Control()
		end

		function out = getNeuralData(obj);	
			out = obj.data.get(obj.data_type);
		end
		function out = getHeading(obj);
			out = obj.data.get('heading');
		end
		function calculateHeadDirection(obj)
		end
	end
end
