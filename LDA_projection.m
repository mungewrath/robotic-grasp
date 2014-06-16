%%
%//////////////////////////////////////////////////////////////////////////
%------Start LOAD FILES---------------------------------------
%//////////////////////////////////////////////////////////////////////////
clc, clear all, close all

c = clock;

OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
%mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
% Hard-coded path for now
mydir = 'Z:\RobotGrasp2\';
savedir = 'Z:\Thesis\'
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
addpath([mydir,'results'])

addpath([mydir,'LDA'])
addpath([mydir,'roccurves'])

clear me mydir
%------------End Load Files----------------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Start Define parameters------------------------------------
%//////////////////////////////////////////////////////////////////////////

mySeed = 10;
rng(mySeed);             % Set the seed

% Number of features to read
dimensionCount = 11;

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter
groundtruth = [];

%Data Processing
runpca = 1;                 %Value of 1 initiates PCA analysis
pcakeep = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>=1;        %which PC's to keep converted to logical var
yset = 0;   %0 = train off ground truth, 1 = train off human rating
qmset = 3;  %1 how many of the top QM's to keep based on T-Tests, or decimal for p value cutoff
comptype = 1; %1 = PCA, 2 = FLD LCA, 3 = Fischer, 4 = no comp anal,5 = no PCA, 6 = phys and crowd;specify comp in compselect
gpiter = 100; %how many times to iterate GP
compselect = 3;
ldasel = [1 2 3 4 5 6];
fpr = [.05 .10 .15];  %Cutoff Values for FPR data

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
results1str = {};
results2str = {};
res2index = [];

%Shake data collected Jan13
dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'data_201301_QM.txt';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end

dzzf_autograsp = 1;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end
   
%------------End Defind Parameters-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Seed Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////

% Limited data with GraspIt and crowdsourcing
datatemp = importdata(datastr{1});
resultstemp1= importdata(results2str{1});
resultstemp2 = importdata(results1str{1});      

data = [data;datatemp.data(:,2:end-1)];
crowdsourceLength = size(data,1);
% Crowdsource estimate
results1 = resultstemp1.data(:,res2index(1));
% Physical testing
groundtruth = [groundtruth;resultstemp2.data(:,2)];
header = datatemp.colheaders(2:end-1);


% Extensive dataset with GraspIt, but no crowdsourcing
datatemp = importdata(datastr{2});  

data = [data;datatemp.data(:,1:end-3)];
% GraspIt estimate
results2 = datatemp.data(:,end-2);
% Physical testing
t = datatemp.data(:,end);
groundtruth = [groundtruth;t];
      
      

% for i = 1:length(datastr)
%       datatemp = importdata(datastr{1});
%       resultstemp1= importdata(results1str{1});
%       resultstemp2 = importdata(results2str{1});      
%       
%       % For these datasets we only want the first 11 metrics
%       data = [data;datatemp.data(:,2:dimensionCount)];
%       % Crowdsource estimate
%       results1 = [results1;resultstemp1.data(:,dimensionCount+1)];
%       % GraspIt estimate
%       results2 = [results2;resultstemp2.data(:,dimensionCount+3)];
%       % Physical testing
%       groundtruth = [groundtruth;resultstemp2.data(:,dimensionCount+3)];
%       if i ==1
%           header = datatemp.colheaders(2:end);
%       elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
%          error('headers do not match, check data files')   
%       end   
% end

%groundtruth = results2;

%------------End Seed Data-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Condition Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////




%Remove NAN columns
datatemp = isnan(mean(data));
data(:,datatemp) = [];
disp('!!!!below quality measures have been removed due to insufficient data!!!!!!!!!!!!')
disp( header(datatemp))
header(datatemp) = [];

%Figure out which QM's are significant with T-tests
[p,keep,delete] = ttester(data,groundtruth,qmset,cutoff);
disp('Keeping QMs:')
disp('name        p values')
for i = 1:length(keep)
    disp([header(keep(i)) num2str(p(keep(i)),3)] )
end


% Hold onto the non-reduced data for logistic
rawdata = data;

%Delete non-important QM's
disp('Deleting QMs:')
disp('name        p values')
for i = length(delete):-1:1
    disp([header(delete(i)) num2str(p(delete(i)),3)] )
end
header(delete) = [];
data(:,delete) = [];

%Sphereize data
data = sphereize(data);

if comptype ==1
    
    %Re-distribute using component analysis
    [~,data,latent] = pca(data);

    %Reduce components based on earlier specifications
    data = data(:,pcakeep(1:length(data(1,:))));
    
elseif comptype ==2
      [v,t] = fld2(data,bincutoff(results1,cutoff),length(data(1,:)));
      disp(t)
      data = data*v;
      data = data(:,ldasel);
      
elseif comptype ==3
      [w,t2,data] = fisher_training(data, bincutoff(results1,cutoff));

elseif comptype ==4
        %Reduce components based on earlier specifications
        data = data;
elseif comptype ==5
    data = data(:,compselect);    
elseif comptype ==6
    %Re-distribute using component analysis
    [~,data,latent] = pca(data);

    %Reduce components based on earlier specifications
    data = data(:,pcakeep(1:length(data(1,:))));
    data = [data;data];
    results1 = [results1;results2];
end


crowdsourceData = data(1:crowdsourceLength,:);
graspitData = data(crowdsourceLength+1:end,:);

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

% Compute GP

meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;

xdepth = length(data(1,:));
covdepth = xdepth+1;

loopn = 30;

auc = zeros(1,loopn);

% Keep with cutoff of 0.8
binshakeresults = bincutoff(groundtruth,0.8);

% Data to use as gating network training set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
gateTrainProportion = .2;

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .2;

coeff_linear = [];
coeff_const = [];
for l = 1:loopn
    hyp = [];
    hyp2 = [];
    
    % Train hyperparameters
   
    hyp.mean = 0;
    hyp2.mean = 0;
    hyp.cov(1:covdepth) = log(1);
    hyp2.cov(1:covdepth) = log(1);
    hyp.lik(1:1) = -.2;
    hyp2.lik(1:1) = -.2;
    disp(hyp)
    
    
    % Added in for comparative testing vs. hybrid
    if(testProportion >= 1)
        testx = zeros(testProportion,1);
    else
        testx = zeros(floor(crowdsourceLength * testProportion),1);
    end
    % Split the indices between the two datasets
    testIndices_crowdsource = zeros(length(testx),1);
    testIndices_graspit = zeros(length(testx),1);
    
    
    i = 1;
    % Pick test points randomly from crowdsource set
    while (i <= length(testIndices_crowdsource))
        newTest = randi(length(crowdsourceData));
        if ~any(testIndices_crowdsource==newTest)
            %testx(i,:) = crowdsourceData(newTest,:);
            testIndices_crowdsource(i) = newTest;
            i = i+1;
        end
    end
    i = 1;
    % Pick test points randomly from graspit set
    while (i <= length(testIndices_graspit))
        newTest = randi(length(graspitData));
        if ~any(testIndices_graspit==newTest)
            %testx(i,:) = graspitData(newTest,:);
            testIndices_graspit(i) = newTest;
            i = i+1;
        end
    end
    % Randomly remove points until graspit set is the same size as
    % crowdsource
    removed_grasp_indices = zeros(length(graspitData)-length(crowdsourceData),1);
    i = 1;
    while (i <= length(removed_grasp_indices))
        newTest = randi(length(graspitData));
        if ~(any(removed_grasp_indices==newTest) || any(testIndices_graspit==newTest))
            %testx(i,:) = graspitData(newTest,:);
            removed_grasp_indices(i) = newTest;
            i = i+1;
        end
    end
    
    % Separate training sets from culled data
    trainx_crowdsource = data(1:crowdsourceLength,:);
    trainx_crowdsource(testIndices_crowdsource,:) = [];
    trainx_graspit = data(crowdsourceLength+1:end,:);
    trainx_graspit([testIndices_graspit;removed_grasp_indices],:) = [];
    
    % Merge test sets into one block
    testIndices = [testIndices_crowdsource; testIndices_graspit+crowdsourceLength];
    testx = data(testIndices,:);
    
    
%    LDA_trainx = rawdata(testIndices_crowdsource,:);
    LDA_testx = rawdata(testIndices,:);
%     % Get rid of test data
%     trainx([testIndices; testIndices_final],:) = [];
    trainy_crowdsource = results1;
    trainy_crowdsource(testIndices_crowdsource) = [];
    % Squash GraspIt energy to 0:1
 	trainy_graspit = normalizeer(log(results2) ./ (1+log(results2)));
    trainy_graspit([testIndices_graspit; removed_grasp_indices]) = [];
    
    % Minimize hyperparameters
    hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainx_crowdsource, trainy_crowdsource);
	hyp2 = minimize(hyp2, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainx_graspit, trainy_graspit);
   
    % Run GP
    [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx_crowdsource,trainy_crowdsource,testx);
	[mf_2, s2f_2, fmu_2, fs2_2] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc,trainx_graspit,trainy_graspit,testx);
    
    
    % TODO: Run without binshakeresults
    % Get test ground truth
    res = binshakeresults(testIndices);
    mf = normalizeer(mf);
	mf_2 = normalizeer(mf_2);
    
    % Train LDA with results of GPs. The class label is given as whichever
    % GP is closer to ground truth.
    gp1_error = abs(mf - res);
    gp2_error = abs(mf_2 - res);
    LDA_y = (gp2_error < gp1_error);
    [class,err,POSTERIOR,logp,coeff] = classify(LDA_testx,LDA_testx,LDA_y);
    coeff_linear(l,:) = coeff(2,1).linear;
    coeff_const(l,:) = coeff(2,1).const;
    
    disp(['Completed iteration ' l])
end

save(strcat(savedir,'LDA_results.mat'),'coeff_linear','coeff_const');


%Calculate Computation time
disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)