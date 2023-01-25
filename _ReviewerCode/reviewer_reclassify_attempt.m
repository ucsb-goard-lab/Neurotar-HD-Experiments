load('D:\aggregate_analyzers\dual_cue.mat');

for d = dual_cue
    d.calculateHeadDirection();
end

double_ishd = cat(2, dual_cue.is_head_direction)';%reliability_double > 0.15; %cat(2, dual_cue.is_head_direction);

ct = 1;
for d = dual_cue
    disp(ct)
    trial_responses_light{ct} = d.getTrialResponses(d.light_data);
    trial_responses_dark{ct} = d.getTrialResponses(d.dark_data);
%     pref_dir{ct} = d.calculatePreferredDirection('fit', 'double');
    ct = ct + 1;
end

for r = 1:length(trial_responses_light)
    disp(r)
    [aligned_responses{r}] = leaveOneOutAligner({trial_responses_light{r}, trial_responses_dark{r}}); % the first one is the reference, so the alignment is performed on that one and applied to the rest
end

light_aligned_cell = cellfun(@(x) nanmean(x{1}, 3), aligned_responses, 'UniformOutput', false);
dark_aligned_cell = cellfun(@(x) nanmean(x{2}, 3), aligned_responses, 'UniformOutput', false);

light_aligned = cat(1, light_aligned_cell{:});
dark_aligned = cat(1, dark_aligned_cell{:});

% 

combined = cat(3, light_aligned, dark_aligned);
temp = reshape(combined, size(combined, 1), []);
temp = rowrescale(temp);
temp = reshape(temp, size(combined));
light_clean = (temp(:, :, 1));
dark_clean = (temp(:, :, 2));

% 
used_data_light = light_clean;
used_data_dark = dark_clean;

ct = 1;
for d = dual_cue
    disp(ct)
   
    [~, ~, all_fits{ct}] = d.calculateTuningFits();
	ct = ct + 1;
end

ishd = cat(2, dual_cue.is_head_direction)';
% fit stuff
a0_light = [];
a0_dark = [];
a1_light = [];
a1_dark = [];
a2_light = [];
a2_dark = [];
c1_light = [];
c1_dark = [];
c2_light = [];
c2_dark = [];

for c = 1:numel(all_fits)
    current = all_fits{c};
    a0_light = cat(2, a0_light, cellfun(@(x) x{1}.a0, current));
    a0_dark = cat(2, a0_dark, cellfun(@(x) x{2}.a0, current));    
    
    a1_light = cat(2, a1_light, cellfun(@(x) x{1}.a1, current));
    a1_dark = cat(2, a1_dark, cellfun(@(x) x{2}.a1, current));
    
    a2_light = cat(2, a2_light, cellfun(@(x) x{1}.a2, current));
    a2_dark = cat(2, a2_dark, cellfun(@(x) x{2}.a2, current));
    
    c1_light = cat(2, c1_light, cellfun(@(x) x{1}.c1, current));
    c1_dark = cat(2, c1_dark, cellfun(@(x) x{2}.c1, current));
    
    c2_light = cat(2, c2_light, cellfun(@(x) x{1}.c2, current));
    c2_dark = cat(2, c2_dark, cellfun(@(x) x{2}.c2, current));
end

% [~, clustering_data, ~, ~, expl] = pca(cat(2, a1_light', a2_light', a1_dark', a2_dark', c1_light', c2_light', c1_dark', c2_dark'));
clustering_data = cat(2, a1_light', a2_light', a1_dark', a2_dark', c1_light', c2_light', c1_dark', c2_dark');

options = statset('MaxIter', 1000);
gm = fitgmdist(clustering_data(ishd, :), 3, 'replicates', 20, 'Options', options);
cluster_id_hd = gm.cluster(clustering_data(ishd, :))';

too_many_zeroes = sum(light_clean == 0, 2) > 10 | sum(dark_clean == 0, 2) > 10;% was 10
too_many_nans = all(isnan(light_clean), 2) | all(isnan(dark_clean), 2);

cluster_id = -1 * ones(1, size(clustering_data, 1));
% cluster_id(~ishd | too_many_zeroes | too_many_nans) = -1;
cluster_id(ishd) = cluster_id_hd; 
tabulate(cluster_id)

% 
% % same as before, clean up the clusters one more time, since they are now
% % expanded
% u_clusters = unique(cluster_id);
% 
% for c = u_clusters(2:end)
% 
% %     m_cluster_resp = movmean(nanmean(tuning_curves(cluster_id == c, :)), 10);%cat(2, mean(rowrescale(used_data_light(cluster_id == c, :)), 1), mean(rowrescale(used_data_dark(cluster_id == c, :)), 1));
%     m_cluster_resp_l = nanmean(rowrescale(light_clean(cluster_id == c, :)), 1);
%     m_cluster_resp_d = nanmean(rowrescale(dark_clean(cluster_id == c, :)), 1);
% 
%     ct = 1;
%     test_corr = [];
%     for n = find(cluster_id == c)
% %         test_resp = tuning_curves(n, :); %cat(2, rescale(used_data_light(n, :)), rescale(used_data_dark(n, :)));
%         light_resp = light_clean(n, :);
%         dark_resp = dark_clean(n, :);
%         test_corr(ct, :) = [corr(light_resp', m_cluster_resp_l'), corr(dark_resp', m_cluster_resp_d')];
%         ct = ct + 1;
%     end
%     current_cluster = find(cluster_id == c);
%     
%     cluster_id(current_cluster(all(test_corr <= 0.5, 2))) = -1;
% end
% 
% % re-tabulate
% tabulate(cluster_id); 
% % 
% used_cid = cluster_id; 
% ct2 = 1;
% for c = u_clusters(2:end)
%     
%     %     m_cluster_resp = movmean(nanmean(tuning_curves(cluster_id == c, :)), 10);%cat(2, mean(rowrescale(used_data_light(cluster_id == c, :)), 1), mean(rowrescale(used_data_dark(cluster_id == c, :)), 1));
%     m_cluster_resp_l = nanmean(rowrescale(plotting_data_light(used_cid == c, :)), 1);
%     m_cluster_resp_d = nanmean(rowrescale(plotting_data_dark(used_cid == c, :)), 1);
%     
%     ct = 1;
%     test_corr = [];
%     for n = find(used_cid == c)
%         %         test_resp = tuning_curves(n, :); %cat(2, rescale(used_data_light(n, :)), rescale(used_data_dark(n, :)));
%         light_resp = plotting_data_light(n, :);
%         dark_resp = plotting_data_dark(n, :);
%         test_corr(ct, :) = [corr(light_resp', m_cluster_resp_l'), corr(dark_resp', m_cluster_resp_d')];
%         ct = ct + 1;
%     end
%     [~, sort_vec] = sort(mean(test_corr, 2));
%     
%     current_cluster = find(used_cid == c);
%     sorted_cluster_id{ct2} = current_cluster(sort_vec);
%     ct2 = ct2 + 1;
% end

ct = 1;
for d = expts
    flip_score_light_cell{ct} = d.calculateFlipScore(d.light_tuning);
    flip_score_dark_cell{ct} = d.calculateFlipScore(d.dark_tuning);
    ct =ct + 1;
end
% cat em
flip_score_light = cat(1, flip_score_light_cell{:});
flip_score_dark = cat(1, flip_score_dark_cell{:});


u_clusters = unique(cluster_id);
plotting_data_light = light_clean;
plotting_data_dark = dark_clean;

ct = 1;
for c = u_clusters(2:end)
    current_light = (plotting_data_light(cluster_id == ct, :)); %(plotting_data_light(sorted_cluster_id{ct}, :)) - min(mean((plotting_data_light(sorted_cluster_id{ct}, :))));
    current_dark = (plotting_data_dark(cluster_id == ct, :)); %(plotting_data_dark(sorted_cluster_id{ct}, :)) - min(mean((plotting_data_dark(sorted_cluster_id{ct}, :))));
    
    figure;
    subplot(3, 2, [1, 3])
    imagesc((current_light))
    hold on
    subplot(3, 2, [2, 4])
    imagesc((current_dark))
    xlabel('heading')
    ylabel('subtracted DFF')
    title(sprintf('Cluster #%d', c))
    subplot(3, 2, [5, 6])
    
    plot(nanmean((current_light)))
    hold on
    plot(nanmean((current_dark)));
    xlabel('heading')
    ylabel('subtracted DFF')
    title(sprintf('Cluster #%d', c))
    
    ct = ct + 1;
end


% ct = 1;
% for c = u_clusters(2:end)
%     current_light = (plotting_data_light(sorted_cluster_id{ct}, :)); %(plotting_data_light(sorted_cluster_id{ct}, :)) - min(mean((plotting_data_light(sorted_cluster_id{ct}, :))));
%     current_dark = (plotting_data_dark(sorted_cluster_id{ct}, :)); %(plotting_data_dark(sorted_cluster_id{ct}, :)) - min(mean((plotting_data_dark(sorted_cluster_id{ct}, :))));
%     
%     figure;
%     subplot(3, 2, [1, 3])
%     imagesc((current_light))
%     hold on
%     subplot(3, 2, [2, 4])
%     imagesc((current_dark))
%     xlabel('heading')
%     ylabel('subtracted DFF')
%     title(sprintf('Cluster #%d', c))
%     subplot(3, 2, [5, 6])
%     
%     plot(nanmean((current_light)))
%     hold on
%     plot(nanmean((current_dark)));
%     xlabel('heading')
%     ylabel('subtracted DFF')
%     title(sprintf('Cluster #%d', c))
%     
%     ct = ct + 1;
% end
