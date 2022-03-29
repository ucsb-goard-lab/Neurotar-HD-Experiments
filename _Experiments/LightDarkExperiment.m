classdef LightDarkExperiment < ForcedRotation & DualCueAnalyzer
    properties
        dark_data
        light_data
        n_repeats
        segment_length
        recording_length
        
        light_tuning
        dark_tuning
    end
    
    methods
        function obj = LightDarkExperiment(varargin)
            obj@DualCueAnalyzer(varargin{:});
            obj@ForcedRotation();
            
            obj.recording_length = length(obj.data.get('alpha'));
            
            obj.getRecordingParameters(); % hardcoded repeats and segment lengths, because there aren't associated stimfiles... oops
            
            fprintf('Repeat duration: %ds\n', obj.segment_length/10); % for confirmation
            
            s = Separator(obj.data, obj.segment_length, obj.n_repeats); % this uses the segment length and n_repeats to separate into individual datastructures
            data = s.run(); % separate the data
            obj.light_data = cat(2, data{:, 1}); % concatenate the light and dark repeats separately
            obj.dark_data = cat(2, data{:, 2});
            
            [obj.light_tuning, obj.dark_tuning] = obj.getConditionTuningCurves();
            obj.tuning_curves = obj.getTuningCurves(); % overall tuning
        end
        
        function getRecordingParameters(obj)
            error('LightDarkExperiment is an abstract class and shouldn''t be run on its own')
        end
        
        	function [tf, b, all_fits] = calculateTuningFits(obj)
			all_tuning = cat(2, obj.light_tuning, obj.dark_tuning);
			rescaled = rowrescale(all_tuning);
			all_tuning = reshape(rescaled, size(obj.light_tuning, 1), size(obj.light_tuning, 2), []);
			light_tuning = all_tuning(:, :, 1);
			dark_tuning = all_tuning(:, :, 2);
				for c = 1:size(obj.tuning_curves, 1)
				[fit_l] = obj.fitGaussians(light_tuning(c, :), false);
				[fit_d] = obj.fitGaussians(dark_tuning(c, :), false);

				x = linspace(0, 2*pi, size(obj.tuning_curves, 2));
				tf(c, :) = fit_l(x);
				b(c, :) = [fit_l.b1, fit_l.b2];
				all_fits{c} = {fit_l, fit_d};
			end

		end

        function [light_tuning, dark_tuning] = getConditionTuningCurves(obj)
            for r = 1:obj.n_repeats
                l_spikes = obj.light_data(r).get(obj.data_type);
                l_heading = obj.light_data(r).get('heading');
                d_spikes = obj.dark_data(r).get(obj.data_type);
                d_heading = obj.dark_data(r).get('heading');
                light_tuning(:, :, r) = obj.binData(l_spikes, l_heading);
                dark_tuning(:, :, r) = obj.binData(d_spikes, d_heading); % used to be 1:200, changed
            end
            light_tuning = nanmean(light_tuning, 3);
            dark_tuning = nanmean(dark_tuning, 3);
        end
        
        function [flip_score_l, flip_score_d] = getFlipScore(obj)
            flip_score_l = obj.getFlipScore@Analyzer(obj.light_tuning);
            flip_score_d = obj.getFlipScore@Analyzer(obj.dark_tuning);
        end
        function [out]  = calculateRayleighVector(obj, rescale_method)
            if nargin < 2 || isempty(rescale_method)
                rescale_method = 'coupled';
            end
            
            % First get the tuning curves from each condition and calculated folded tuning
            l_spikes = [];
            l_heading = [];
            d_spikes = [];
            d_heading = [];
            for r = 1:obj.n_repeats
                l_spikes = cat(2, l_spikes, obj.light_data(r).get(obj.data_type));
                l_heading = cat(2, l_heading, obj.light_data(r).get('heading'));
                d_spikes = cat(2, d_spikes, obj.dark_data(r).get(obj.data_type));
                d_heading = cat(2, d_heading, obj.dark_data(r).get('heading'));
                
            end
            light_tuning = obj.binData(l_spikes, l_heading, true);
            dark_tuning = obj.binData(d_spikes, d_heading, true);
            
            % Next, rescaling the tuning curves to each othr
            fprintf('%s rescaling applied.\n', rescale_method);
            switch rescale_method
                case 'coupled'
                    for c = 1:size(light_tuning, 1)
                        % min_val = min(light_tuning(c, :));
                        % max_val = max(light_tuning(c, :));
                        %
                        % %                 light_rescaled(c, :) = (light_tuning(c, :) - min_val) ./ (max_val - min_val);
                        % %                 dark_rescaled(c, :) = (dark_tuning(c, :) - min_val) ./ (max_val - min_val);
                        % light_rescaled(c, :) = (light_tuning(c, :));% ./ (max_val);
                        % dark_rescaled(c, :) = (dark_tuning(c, :));% ./ (max_val);
                        all_tuning = cat(3, light_tuning, dark_tuning);
                        combined = reshape(all_tuning, size(light_tuning, 1), []);
                        combined = rowrescale(combined);
                        combined = reshape(combined, size(all_tuning));
                    end
                    light_rescaled = combined(:, :, 1);
                    dark_rescaled = combined(:, :, 2);
                case 'independent'
                    light_rescaled = rowrescale(light_tuning);
                    dark_rescaled = rowrescale(dark_tuning);
                case 'light_max'
                    for c = 1:size(light_tuning)
                        min_val = min(light_tuning(c, :));
                        max_val = max(light_tuning(c, :));
                        light_rescaled(c, :) = (light_tuning(c, :)) ./ (max_val);
                        dark_rescaled(c, :) = (dark_tuning(c, :)) ./ (max_val);
                    end
            end
            
            out = cat(2, obj.calculateRayleighVector@DualCueAnalyzer(light_rescaled)', obj.calculateRayleighVector@DualCueAnalyzer(dark_rescaled)');
            
        end
    end
end
