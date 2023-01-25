classdef Separator < handle
% Given the segment length and number of repeats, this class separates the data into a bunch of datastructures for easier control
	properties
		data_structure
		segment_length
		recording_length
		n_repeats
		n_trials
	end

	methods
		function obj = Separator(data_structure, segment_length, n_repeats)
			obj.data_structure = data_structure;
			obj.segment_length = segment_length;
			obj.recording_length = size(obj.data_structure.get('raw_F'), 2); % raw_F is a pretty safe field to pull for the recording length
			obj.n_repeats = n_repeats;
			obj.n_trials = floor(obj.recording_length/(obj.segment_length * obj.n_repeats));
		end

		function data = run(obj)
			% Main look, separates both neural and stimulus data structures
			neural_data = obj.data_structure.getNeuralData();
			stimulus_data = obj.data_structure.getStimulusData();
			for r = 1:obj.n_repeats
				for t = 1:obj.n_trials
					[n_d] =  obj.separateData(neural_data, r, t); % separate the neural data
					[s_d] =  obj.separateData(stimulus_data, r, t); % separate the stimulus data
					data{r, t} = DataStructure(n_d, s_d);
				end
			end
		end

		function out = separateData(obj, data, rep, trial)
			%% Get all the fields in the data structure
			out = struct();
			struct_fields = fields(data); % for each field
			for i_field = 1:length(struct_fields)
				if ismember(obj.recording_length, size(data.(struct_fields{i_field}))) % some fields shouldn't be separated... 
					out.(struct_fields{i_field}) = obj.separateField(data.(struct_fields{i_field}), rep, trial); %this is to make sure we only separate the right fields
				else
					out.(struct_fields{i_field}) = data.(struct_fields{i_field});
				end
			end
		end
		function out = separateField(obj, data, rep, trial)
			% For separating out a single field
			out = [];
			curr = (rep - 1) * (obj.segment_length * obj.n_trials) + (trial - 1) * obj.segment_length;
			% Prepare data, this is finding the correct index to segment
			inds = repmat({1}, 1, ndims(data));
			for i_dim = 1:ndims(data)
				inds{i_dim} = 1:size(data, i_dim);
			end

			working_dim = find(ismember(size(data), obj.recording_length));
			inds{working_dim} = curr + 1: curr + obj.segment_length;
			curr_data = data(inds{:});
			out = cat(working_dim, out, curr_data);
		end
	end
end
