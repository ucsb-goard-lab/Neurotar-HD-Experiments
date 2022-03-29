function out = calculatePercentCorrect(decoded_heading, true_heading, thresh)
    out = abs(angdiff(deg2rad(decoded_heading), deg2rad(true_heading'))) <= deg2rad(thresh);
end