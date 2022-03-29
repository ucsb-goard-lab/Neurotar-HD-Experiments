function confusion_matrix = getConfusionMatrix(decoded_heading, true_heading, bin_width)
if nargin < 3 || isempty(bin_width)
    bin_width = 6;
end

bin_edges = [-180 : bin_width : 180];
bin_centers = bin_edges + bin_width/2;
bin_centers = bin_centers(1:end-1);

confusion_matrix = zeros(length(bin_centers),length(bin_centers));
decoded_bin = discretize(decoded_heading, bin_edges);
decoded_bin(isnan(decoded_bin)) = 0;
actual_bin = discretize(true_heading, bin_edges);
% confusion_matrix = confusionmat(actual_bin, decoded_bin); % this doens't work well because if bins aren't represented it shrinks the matrix
for actual_i = 1:length(bin_centers)
    for decoded_i = 1:length(bin_centers)
        confusion_matrix(actual_i, decoded_i) = sum(decoded_bin == decoded_i & actual_bin' == actual_i)./sum(actual_bin == actual_i);
    end
end
end
