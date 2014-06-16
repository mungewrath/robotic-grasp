clc

%%%BEGIN COPY ---SETUP WORK FOR GPML---
disp(['executing gpml startup script...']);

OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path

addpath(mydir(1:end-1))
addpath([mydir,'cov'])
addpath([mydir,'doc'])
addpath([mydir,'inf'])
addpath([mydir,'lik'])
addpath([mydir,'mean'])
addpath([mydir,'util'])
addpath([mydir,'Mechanical Turk Data'])  %HAS ROC FUNCTION BUILT IN

clear me mydir
%%%%%END COPY


clear all, close all


%---Define parameters----
leave = 20;  %leaves out desired number in %
cutoff = .8;

%PULL IN IROS DATA EARLY FOR PROCESSING
shakeresults = xlsread('IROS_RESULTSp2.xls');
averesults = shakeresults;
shakedata = xlsread('IROS_DATA2p2.xls');

targetedbins = xlsread('top20bin.xls');

datacondition = 1;  %Select which dataset you want

%Select dataset
if datacondition ==1
    %Physical shake testing only
    [results, index, data] = dataprep(shakedata, shakeresults, 8);
    savestr = '2013shakeonly';
elseif datacondition ==2
    %Mechanical turk only
    mturresults = xlsread('mechturkrawdata.csv');
    results = mturresults(:,9);
    index = mturresults(:,1);
    dataindex = sort(unique(mturresults(:,1)));
    avemturresults = matave([index results]);
    data = dataprepmatcher(shakedata,dataindex,index);
    results = bincutoff(results,cutoff);
    savestr = '2013crowdtestingonly';
elseif datacondition ==3
    %Ravi's old data
    data = xlsread('ravisolddata.csv');
    results = xlsread('ravisoldresults.csv');
    [results, index, data] = dataprep(data, results, 5);
    savestr = '2009shaketestingonly';
elseif datacondition ==4
    %merge physical testing data - old and new
    [results, index, data] = dataprep(shakedata, shakeresults, 8);
    data2 = xlsread('ravisolddata.csv');
    results2 = xlsread('ravisoldresults.csv');
    averesults2 = results2;
    [results2, index2, data2] = dataprep(data2, results2, 5);
    index2 = index2+100;
    data = vertcat(data,data2);
    results = vertcat(results,results2);
    index = vertcat(index,index2);   
    averesults = vertcat(averesults,averesults2);
    savestr = 'combinedoldandnewdata';
elseif datacondition ==5
    %Test non-binary data
    data = shakedata;
    results = shakeresults;
    index = (1:length(data))';
    savestr = '2013nonexpanded';        
end


%Sphereize data
data = sphereize(data);

[~,data,latent] = pca(data);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------Functions SELECTION------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanfunc = @meanConst;
covfunc = @covSEard2;
likfunc = @likGauss;
inffunc = @infExact;  %%CHANGING THIS WILL CHANGE OTHER THINGS, function lengths not computed for this

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterate and average hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graspno = unique(index);
loopn = 300;

y = results;

disp('done with data')

auc = zeros(loopn,12);
fpr = auc;
tpr = auc;

binshakeresults = bincutoff(shakeresults,.8);

hyparch = zeros(length(targetedbins),12);

for m = 1:1
    
    %x = data;
    x = data(:,1);%For PC Reduction
    totalset = length(data(:,1));
    inputsize = round(totalset*(1-leave*.01)); %Calculate how many to read
    xdepth = length(x(1,:));
    
    mf = zeros(4,loopn); %zeros(totalset-inputsize,loopn)
    s2f = mf;
    fmu = mf;
    fs2 = mf;
    lp = mf;
    excl = mf;
    testindtot = [];

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

    
    for l = 1:loopn
        
        ellstore = [];
        hyp = [];
        hypcovtot = [];
        hypmeantot = [];
        hypliktot = [];

        %Train Hyperparameters

        %Define leave out percent
        [trainx, trainy, trainind, testx, testy, testind, leavevect] = leaverand2(x,y,index,leave);

        hyp.mean = zeros(meandepth,1);
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:likdepth) = -.2;

        %%%%%%%CHANGE ITER TO 1000!!!!%%%%%%%%%%
        hyp = minimize(hyp, @gp, -900, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);

        hyptemp = [hyp.cov 0 0 0 0 0 0 0 0 0 0 0 0];
        hyptemp = hyptemp(1:12);
        hyparch(m,:) = hyptemp;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%CALCULATE GP's
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);
        mf = normalizeer(mf);

        try
            res = binshakeresults(testind);
            [~,~,~,auc(l,m)] = perfcurve(res',mf',1);
        catch err
            auc(l,m) = NaN;
            %input('test')
        end
        
        disp([l auc(l,m)])
        
        [~,~,~,~,sumtemp] = truefalse([mf res],.8,.8);
        
        fpr(l,m) = sumtemp(2,2)/(sumtemp(2,2)+sumtemp(2,3));
        tpr(l,m) = sumtemp(2,4)/(sumtemp(2,1)+sumtemp(2,4));
        
        save('aucGP6heruronly.mat','auc','hyparch','tpr','fpr')

    end
 
    

end

disp(mean(fpr(:,1)))
disp(mean(tpr(:,1)))
disp(mean(auc(:,1)))
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %OUTPUTS
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disp('\nGP generated FP Rate: ')
% disp(fitnorm(statnorm,1))
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %ROC GRAPHs
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %Plot Setup 21Mar13
% hold on
% %Raw Data and fit curves
% plot(ccritnorm(:,6),ccritnorm(:,7),':k','LineWidth',1)
% plot(fitnorm(:,1),fitnorm(:,2),'k','LineWidth',3)
% %Random Gueass Line
% plot([0 1],[0 1],'color',[.2 .2 .2])
% legend('GP raw', 'GP Fitted Curve','Random Guess',4)
% hold off
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% title('ROC Space Normalized Grasp Data')
% axis([0 1 0 1])
% axis square
% 
% save(savestr);