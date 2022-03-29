function prediction_error = calculateDecoderError(decoded_heading, true_heading, bin_width)
if nargin < 3 || isempty(bin_width)
    bin_width = deg2rad(6);
end

bins = -pi:bin_width:pi;
bins = bins + bin_width/2;
bins = bins(1:end-1);
n_bins = numel(bins);

% Calculating error circularly, so that 360 -> 1 = 1
binned_heading = discretize(true_heading, rad2deg(bins));
decoded_heading = discretize(decoded_heading, rad2deg(bins));

is_double_nan = isnan(binned_heading) | isnan(decoded_heading)';
prediction_error_non_nan = angdiff(bins(binned_heading(~is_double_nan)), bins(decoded_heading(~is_double_nan)));
prediction_error = nan(1, length(binned_heading));
prediction_error(~is_double_nan) = prediction_error_non_nan;
prediction_error = rad2deg(prediction_error);
end
