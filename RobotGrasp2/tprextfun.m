function [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr )
%Pulls TPR values at given FPR cutoffs
%   Created 24OCT13 - RJC


%Pull out TPR Values (basic interpolation)
try  %Note that Try-Catch is used because often the first few values of an ROC curve form a virtical line (un-interpratable)
    tpr = interp1(ave(:,1)',ave(:,2)',fpr');
    tprerr = interp1(ave(:,1)',uppererr(:,2)',fpr');
    tprstd = interp1(ave(:,1)',upperstd(:,2)',fpr');
catch 
    try
        tpr = interp1(ave(4:end-4,1)',ave(4:end-4,2)',fpr');
        tprerr = interp1(ave(4:end-4,1)',uppererr(4:end-4,2)',fpr');
        tprstd = interp1(ave(4:end-4,1)',upperstd(4:end-4,2)',fpr');
    catch
        tpr = [NaN NaN NaN];
        tprerr = [NaN NaN NaN];
        tprstd = [NaN NaN NaN];
        disp('Note your ROC curve may be virtical at your lowest FPR value which results in infinite solutions.  You could add code in here to strip out the max value if you are looking for TPR@FPR = 0')
    end
end

tprerr = (tprerr-tpr)';
tprstd = (tprstd-tpr)';
tpr = tpr';
