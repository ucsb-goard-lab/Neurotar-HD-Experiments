classdef HeadingDecoderPreprocessor < handle
    properties
        expt
        
        n_interp_bins
        n_cells
        window
    end
    
    methods
        function obj = HeadingDecoderPreprocessor(expt)
            obj.expt = expt;
            obj.n_interp_bins = 180; % define here
            obj.window = 15;
            obj.n_cells = numel(obj.expt.is_head_direction);
        end
        
        function [spks_aligned, leave_one_tc, heading_aligned] = run(obj, direction)
            % Direct controls if it's going to be light-on to light-off or
            % light-off to light-on
            switch direction
                case 'forward'
                    last = numel(obj.expt.light_data);
                case 'backward'
                    last = numel(obj.expt.light_data) - 1;
                otherwise
                    error('Did not provide a usable direction: ''forward'' or ''backward''');
            end
            
            if last == 0 % single repeat? no dark -> light
                spks_aligned = nan(obj.n_cells, obj.n_interp_bins, 2*obj.window);
                leave_one_tc = nan(obj.n_cells, 360/obj.expt.bin_width, 2*obj.window);
                heading_aligned = nan(obj.n_interp_bins, 2*obj.window);
                return
            end
            
            leave_one_tc = nan(obj.n_cells, 360/obj.expt.bin_width, 2*obj.window, last);
            heading_aligned = nan(obj.n_interp_bins, 2*obj.window, last);
            spks_aligned = nan(obj.n_cells, obj.n_interp_bins, 2*obj.window, last);
            for ii = 1:last % should go through each rep
                switch direction
                    case 'forward'
                        light_data = obj.expt.light_data(ii);
                        dark_data = obj.expt.dark_data(ii);
                    case 'backward'
                        light_data = obj.expt.light_data(ii+1);
                        dark_data = obj.expt.dark_data(ii);
                end
                
                light_spks = nan(obj.n_cells, obj.n_interp_bins, obj.window);
                dark_spks = nan(obj.n_cells, obj.n_interp_bins, obj.window);
                
                heading_final_l = nan(obj.n_interp_bins, obj.window);
                heading_final_d = nan(obj.n_interp_bins, obj.window);
                
                % For light
                heading = light_data.get('heading');
                spks = light_data.get('spikes');
                
                heading_reference = obj.findReferenceHeading(heading);
                heading = obj.cleanHeading(heading);
                segment_light = obj.segmentTrials(heading, heading_reference);
                if ~isempty(segment_light)
                    [spks_resampled_light, heading_resampled_light] = obj.resampleSpikes(spks, heading, segment_light);
                end
                % Fork dark
                heading = dark_data.get('heading');
                spks = dark_data.get('spikes');
                
                heading_reference = obj.findReferenceHeading(heading);
                heading = obj.cleanHeading(heading);
                segment_dark = obj.segmentTrials(heading, heading_reference);
                if ~isempty(segment_dark)
                    [spks_resampled_dark, heading_resampled_dark] = obj.resampleSpikes(spks, heading, segment_dark);                    
                end
                
                [light_tuning] = obj.expt.getTrialResponses(light_data); % manual bin here
                [dark_tuning] = obj.expt.getTrialResponses(dark_data); % manual bin here
                
                
                % insert properly
                switch direction
                    case 'forward'
                        if ~isempty(segment_light) % not super pretty, think about a better way here...
                        light_spks(:, :, end - size(spks_resampled_light, 3)+1:end) = spks_resampled_light;
                        heading_final_l(:, end-size(heading_resampled_light, 2) + 1 : end) = heading_resampled_light;
                        end
                        if ~isempty(segment_dark)
                        dark_spks(:, :, 1:size(spks_resampled_dark, 3)) = spks_resampled_dark;
                        heading_final_d(:, 1:size(heading_resampled_dark, 2)) = heading_resampled_dark;
                        end
                    case 'backward'
                        if ~isempty(segment_light)
                        light_spks(:, :,  1:size(spks_resampled_light, 3)) = spks_resampled_light;
                        heading_final_l(:, 1:size(heading_resampled_light, 2)) = heading_resampled_light;
                        end
                        if ~isempty(segment_dark)
                        dark_spks(:, :, end - size(spks_resampled_dark, 3) + 1 : end) = spks_resampled_dark;
                        heading_final_d(:, end - size(heading_resampled_dark, 2) + 1 : end) = heading_resampled_dark;
                        end
                end
                leave_one_tc(:, :, :, ii) = obj.getTuningCurves(light_tuning, dark_tuning);
                spks_aligned(:, :, :, ii) = cat(3, light_spks, dark_spks);
                heading_aligned(:, :, ii) = cat(2, heading_final_l, heading_final_d);
            end
        end
        
        
        function out = findReferenceHeading(obj, heading)
            slow_idx = find(diff(heading(end-100:end)) < 1); % find times when the speed has decreased, this is because the last trial isn't always exactly right
            slow_idx = slow_idx(diff(slow_idx) == 1); % consecutive only (tsries to get rid of issues)
            out = heading(end-(100-slow_idx(end))); % last slow trial is ideal, because it will get rid of junk
        end
        
        function heading = cleanHeading(obj, heading)
            is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350;
            ct = 1;
            while any(is_jump)
                if ct == 10
                    break
                end
                heading(is_jump) = heading(max(find(is_jump) - 1, 1));
                is_jump = abs(diff(heading)) > 20 & abs(diff(heading)) < 350;
                ct = ct + 1;
            end
        end
        
        function segment = segmentTrials(obj, heading, heading_reference)
            threshold = 0.5;
            breaks = find(abs(heading - heading_reference)<threshold);
            segment_length = diff(breaks);
            segment_length(segment_length == 1) = [];
            std_sl = std(segment_length);
            while any(diff(breaks) > 200) || std_sl > 20
                threshold = threshold + 0.5;
                breaks = find(abs(heading - heading_reference)<threshold);

                segment_length = diff(breaks);
                
                segment_length(segment_length == 1) = [];
                
                std_sl = std(segment_length);
            end
            
            segment = [];
            for ii = 1:length(breaks) - 1
                segment{ii} = breaks(ii) + 1:breaks(ii+1);
            end
            
            segment(cellfun(@length, segment) < 10) = []; % get rid of small chunks
        end
        
        function [spks_resampled, heading_resampled] = resampleSpikes(obj, spks, heading, segment)
            if isempty(segment)
                disp('Something not good right now...')
            else
                % circshift the data
                spks_resampled = nan(size(spks, 1), obj.n_interp_bins, numel(segment));
                heading_resampled = nan(obj.n_interp_bins, numel(segment));
                for ii = 1:numel(segment)
                    % differences
                    [~, min_idx] = min(abs(heading(segment{ii}) - 180));
                    shift = min_idx;
                    
                    spks_resampled(:, :, ii) = resample(circshift(spks(:, segment{ii}), 1 - shift, 2), obj.n_interp_bins, length(segment{ii}), 0, 'Dimension', 2);
                    heading_resampled(:, ii) = resample(circshift(heading(segment{ii}), 1 - shift), obj.n_interp_bins, length(segment{ii}), 0);
                end
            end
        end
        
        function tc = getTuningCurves(obj, light_tuning, dark_tuning, style)
            if nargin < 4 || isempty(style)
                style = 'global';
            end
            [light_tc, dark_tc] = deal(nan(obj.n_cells, 60, obj.window));
            switch style
                case 'local'
                    for ii = 1:size(light_tuning, 3)
                        light_tc(:, :, end - (ii - 1)) = nanmean(light_tuning(:, :, 1:end~=(end - (ii - 1))), 3);
                    end
                    for ii = 1:size(dark_tuning, 3)
                        dark_tc(:, :, ii) = nanmean(dark_tuning(:, :, 1:end~=ii), 3);
                    end  
                case 'global'
                    for ii = 1:size(light_tuning, 3)
                        light_tc(:, :, end - (ii - 1)) = nanmean(cat(3, light_tuning(:, :, 1:end~=(end - (ii - 1))), dark_tuning), 3);
                    end
                    
                    for ii = 1:size(dark_tuning, 3)
                        dark_tc(:, :, ii) = nanmean(cat(3, light_tuning, dark_tuning(:, :, 1:end~=ii)), 3);
                    end
            end
            tc = cat(3, light_tc, dark_tc);
        end
    end
end
