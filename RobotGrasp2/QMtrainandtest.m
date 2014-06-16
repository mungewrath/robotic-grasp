function [aucave,aucerr,aucstd,ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = QMtrainandtest(x,y,index,groundtruth,loopn,leave,cutoff)
%Version 0 - Initial Creation 30AUG13 from remnents of code used in Humanoids paper

%This function pulls data in, randomly leaves out values and completes an
%ROC curve for it along with appropriate parameters

auc = zeros(1,loopn);
fpr = auc;
tpr = auc;

xdepth = length(x(1,:));


leave = 100-leave;

for l = 1:loopn

    %Define leave out percent
    [enx,eny,~,~, ~, ~,~] = leaverand2(x,y,index,leave);
    enx = normalizeer(enx);  
       
    [~,~,~,~,sumtemp] = truefalse([enx eny],.8,.8);



    try
        [perfx,perfy,~,auc(l)] = perfcurve(eny',enx',1);
        [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
    catch err
        auc(l) = NaN;
    end
    
    fpr(l) = sumtemp(2,2)/(sumtemp(2,2)+sumtemp(2,3));
    tpr(l) = sumtemp(2,4)/(sumtemp(2,1)+sumtemp(2,4));

end
 
%Remove 0 rows
for i = loopn:-1:1
    if sum(rocx(:,i)) ==0
        rocx(:,i) = [];
        rocy(:,i) = [];
    end
end

[ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,loopn);



%Average AUC
auc(isnan(auc(:,1)),:)=[];
aucave = mean(auc);
aucstd = std(auc);
aucerr = aucstd/(loopn)^.5;


end