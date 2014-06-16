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
leave = 5;  %leaves out desired number in #
cutoff = .8;

%PULL IN IROS DATA EARLY FOR PROCESSING
shakeresults = xlsread('IROS_RESULTS.csv');
averesults = shakeresults;
shakedata = xlsread('IROS_data.csv');

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

%data(:,[1 2 3 4 5 6 7 9]) = [];
results = results+1;

savestr = strcat(savestr,datestr(clock,'yyddHHMM'));

totalset = length(data(:,1));
inputsize = round(totalset*(1-leave*.01)); %Calculate how many to read
xdepth = length(data(1,:));



%Sphereize data
data = sphereize(data);

disp('done with data')

%COMPLETE DATA SET
x = data;
y = results;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------Functions SELECTION------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;  %%CHANGING THIS WILL CHANGE OTHER THINGS, function lengths not computed for this

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterate and average hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graspno = unique(index);
loopn = 100;
mf = zeros(leave,loopn); %zeros(totalset-inputsize,loopn)
s2f = mf;
fmu = mf;
fs2 = mf;
lp = mf;
excl = mf;
testindtot = [];


for l = 1:loopn
    ellstore = [];
    hyp = [];
    hypcovtot = [];
    hypmeantot = [];
    hypliktot = [];

    %Train Hyperparameters
    
    %Define leave out percent
    [trainx, trainy, trainind, testx, testy, testind, leavevect] = leavefiverand(x,y,index,leave);
    testindtot = cat(2,testindtot,testind);

    svm = svmtrain(trainx,trainy);    
    mf(:,l) = svmclassify(svm,testx);
  
    disp (l)
end

dataout = [reshape(testindtot,[],1) reshape(mf,[],1)];
%%Average Data
[dataout,~,~] = matave(dataout);
%Normalize
dataoutnorm(:,1) = dataout(:,1);
dataoutnorm(:,2) = normalizeer(dataout(:,2));

[ccritnorm,fitnorm,statnorm,auc] = roc(.8,.001,averesults,dataoutnorm(:,2));

disp('AUC')
disp(auc)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('\nGP generated FP Rate: ')
disp(fitnorm(statnorm,1))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ROC GRAPHs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot Setup 21Mar13
hold on
%Raw Data and fit curves
plot(ccritnorm(:,6),ccritnorm(:,7),':k','LineWidth',1)
plot(fitnorm(:,1),fitnorm(:,2),'k','LineWidth',3)
%Random Gueass Line
plot([0 1],[0 1],'color',[.2 .2 .2])
legend('GP raw', 'GP Fitted Curve','Random Guess',4)
hold off
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Space Normalized Grasp Data')
axis([0 1 0 1])
axis square

save(savestr);