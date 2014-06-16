function [aucave,aucerr,aucstd,ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd,mfout,indexout,track,cube] = gptrainandtestswitchingmd(x,y,y2,index,groundtruth,loopn,leave,cutoff,segment)
%Version 1 - Corrected error at 86
%Version 0 - Initial Creation 29AUG13 from remnents of code used in Humanoids paper
%This function pulls data in, optimizes and builds a GP on it, and exports
%evaluative properties.


% %%
% %For Troubleshooting
% x = data;
% y = results1;
% loopn = 100;
% index = (1:length(results1))';


%%
%Define zones of functions
[zoneind,cube] = binswitching(x,segment);
cube = unique([zoneind cube],'rows');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------Functions SELECTION------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;  %%CHANGING THIS WILL CHANGE OTHER THINGS, function lengths not computed for this

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterate and average hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

auc = zeros(1,loopn);
fpr = auc;
tpr = auc;

binshakeresults = bincutoff(groundtruth,cutoff);

xdepth = length(x(1,:));

%-----define input function lengths-----

%    ---cov function depths---
if strcmp('covRQard',char(covfunc))==1
    covdepth = xdepth+2;
elseif strcmp('covSEard2',char(covfunc))==1
    covdepth = xdepth+1;
elseif strcmp('covSEard',char(covfunc))==1
    covdepth = xdepth+1;
end

%    ---mean function depths---
if strcmp('meanConst',char(meanfunc))==1
    meandepth = 1;
end

%    ---likfuncdepths---
if strcmp('likGauss',char(likfunc))==1
    likdepth = 1;
end

covfunc = @covSEard;

mfout = [];
indexout = [];


track =[];

for l = 1:loopn

    hyp = [];

    %Train Hyperparameters

    %Define leave out percent
    [trainx, trainy, trainind, testx, testy, testind, leavevect] = leaverand2(x,y,index,leave);
    trainy2 = y2(trainind);
    
    hyp.mean = zeros(meandepth,1);
    hyp.cov(1:covdepth) = log(1);
    hyp.lik(1:likdepth) = -.2;
    hyp2 = hyp;

    %%%%%%%CHANGE ITER TO 1000!!!!%%%%%%%%%%
    hyp = minimize(hyp, @gp, -900, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);
    hyp2 = minimize(hyp2, @gp, -900, inffunc, meanfunc, covfunc, likfunc, trainx, trainy2);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%CALCULATE GP's
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [mf1, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);
    [mf2, s2f, fmu, fs2] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc,trainx,trainy2,testx);   
    res1 = y(testind);
    res2 = y2(testind);
    mf = zeros(length(mf1),1);
    zone1 = zoneind(testind);
    zoneunique = unique(zone1);
    
%     disp(size(testx))
%      disp(size(mf1))
%      disp(size(mf2))
%      disp('mf')
%      disp(size(mf))    
    
    for i = 1:length(zoneunique)
        tempind = zone1==zoneunique(i);

%         disp(size(tempind))
        mftemp1 = mf1(tempind);
        mftemp2 = mf2(tempind);
        
        restemp1 = res1(tempind);
%         disp(restemp1)
        restemp2 = res2(tempind);
        
        if mean(abs(mftemp1-restemp1))>=mean(abs(mftemp2-restemp1))
            %Crowd Soursing Data Prefferred
            mf(tempind) = mftemp2;
            track =[track;zoneunique(i) 0];
        else
            %Physical Testing Data Prefferred
            mf(tempind) = mftemp1;
            track =[track;zoneunique(i) 1];
        end
    
%         disp('mf')
        %disp([size(testx(:,1)) size(res1) size(mf1) size(mf2) size(mf)])
%         disp([testx(:,1) res1 mf1 mf2 mf])  
        

    end
    
%     disp('mf')
%     disp(mf)
    
    res = binshakeresults(testind);
    
    mfout = [mfout; mf];
    indexout = [indexout; testind];
    
    [~,~,~,~,sumtemp] = truefalse([mf res],.8,.8);

    mf = normalizeer(mf);

    try
        [perfx,perfy,perft,auc(l)] = perfcurve(res',mf',1);
        [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
    catch err
        auc(l) = NaN;
    end
    
    disp([l auc(l)])
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


% %AUC GP - trapezoidal method
% AUCGP = 0;
% for i = 1:length(ave(:,1))-1
%     AUCGP = AUC + (ave((i+1),1)-ave((i),1))*mean([ave((i),2) ave((i+1),2)]);
% end

%Average AUC

auc(isnan(auc))=[];
aucave = mean(auc);
aucstd = std(auc);
aucerr = aucstd/(loopn)^.5;


end