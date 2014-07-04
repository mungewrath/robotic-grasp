% 6/30/14
% Determines a threshold value for a desired FPR.

function threshold = fprThreshold( labels,predictions,fpr )
    [perfx, ~, perft] = perfcurve(labels,predictions,1);
    i = 1;
    assert(fpr > 0 && fpr <= 1,'FPR must be 0 > and <= 1.');
    while perfx(i) < fpr
        i = i+1;
    end
    offset = (perfx(i-1) - fpr) / (perfx(i) - perfx(i-1));
    
    threshold = perft(i-1)+(perft(i-1)-perft(i))*offset;
end