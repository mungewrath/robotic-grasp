function [mfnew,thresh] = cutoffsearch(fpr,perfx,perft,mf)

%For troubleshooting
mfnew = zeros(length(mf),length(fpr));

for i = 1:length(fpr)
    for j = 1:length(perfx)
        if perfx(j)>=fpr(i)
            break
        end
    end
    thresh(i) = perft(j);
    mfnew(:,i) = mf>=thresh(i);
end
