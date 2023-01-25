classdef DataStructure < handle
    % A simplified method of holding onto a pair of neural data and stimulus data structures, making it easier to pull data from each
    properties
        neural_data
        stimulus_data
    end
    
    methods
        function obj = DataStructure(neural_data, stimulus_data)
            obj.neural_data = neural_data;
            obj.stimulus_data = stimulus_data;
            
            if sum(obj.stimulus_data.heading > 200) > 1
                error('Heading has gone over 200... something is fundamentally wrong here')
            end
            obj.checkForSpikes(); % See if spikes have been inferred
        end
        
        function out = get(obj, field)
            % just a quick way of getting variables out
            if isfield(obj.neural_data, field) && ~isfield(obj.stimulus_data, field) % field only in neural data
                out = obj.neural_data.(field);
            elseif isfield(obj.stimulus_data, field) && ~isfield(obj.neural_data, field) % field only in stimulus data
                out = obj.stimulus_data.(field);
            elseif isfield(obj.neural_data, field) && isfield(obj.stimulus_data, field) % field in both, should really never happen..
                p = obj.resolveOverlappingFields()
                out = obj.(p).(field);
            else
                fprintf('Field %s not found...\n', field)
            end
        end
        
        function set(obj, field, value)
            % for setting fields, not commonly used because we don't want to edit the data in the fields
            if isfield(obj.neural_data, field) && ~isfield(obj.stimulus_data, field)
                obj.neural_data.(field) = value;
            elseif isfield(obj.stimulus_data, field) && ~isfield(obj.neural_data, field)
                obj.stimulus_data.(field) = value;
            elseif isfield(obj.neural_data, field) && isfield(obj.stimulus_data, field)
                p = obj.resolveOverlappingFields();
                obj.(p).(field) = value;
            else
                fprintf('Field %s not found...\n', field)
            end
        end
        
        function out = getNeuralData(obj)
            out = obj.neural_data; % return neural data structure
        end
        
        function out = getStimulusData(obj)
            out = obj.stimulus_data; % return stimulus data structure
        end
        
        function save(obj)
            new_filename = obj.neural_data.filename;
            new_filename = new_filename(1:end-4);
            neural_data = obj.neural_data;
            try
                obj.saveNeuralData();
            catch
                fullfile = pwd;
                backslashes = strfind(fullfile, '\');
                new_filename = strcat(fullfile(backslashes(end) + 1:end));
                save(strcat(new_filename, '_data.mat'), 'neural_data')
            end
            
            stimdat_filename = sprintf('%s_stimulus_data.mat', date);
            stimulus_data = obj.stimulus_data;
            save(stimdat_filename, 'stimulus_data');
        end
        
        function saveNeuralData(obj)
            new_filename = obj.neural_data.filename(1:end-4);
            data = obj.neural_data;
            save(strcat(new_filename, '_data.mat'), 'data')
        end
    end
    
    methods (Access = protected)
        function out = resolveOverlappingFields(obj)
            current_properties = properties(obj);
            out = questdlg('Choose the data you want to modify: ', 'Modify', current_properties{:}, 'neural_data');
        end
        
        function checkForSpikes(obj)
            if ~isfield(obj.neural_data, 'spikes')
                disp('Getting spikes')
                obj.neural_data.spikes = obj.getSpikeData(obj.neural_data);
                obj.save();
            end
        end
        
        function dffDeconv = getSpikeData(obj, data)
            dffDeconv = zeros(size(data.DFF));
            for n = 1:size(data.DFF, 1)
                % get trace and run deconvolution
                trace = data.DFF(n, :);
                [~, spikes, ~] = deconvolveCa(trace, 'ar1' ,'foopsi', 'optimize_pars');
                dffDeconv(n, :) = spikes;
            end
            close all;
        end
    end
end
