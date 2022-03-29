function heading = cleanHeadingTrace(heading)
changes = diff(heading);
heading(abs(changes) > 1*std(changes)) = NaN;

end