%%
% LDA_success_lumped.m
% Same as LDA_projection, but only with the successful grasps.
% Compares crowdsourced vs. GraspIt predicted success.
%%

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
mydir = 'RobotGrasp2\';
savedir = 'Thesis\'
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
results1 = {};       %Y parameters for GP Classifier
results2 = {};      %Alternate Y parameter
groundtruth = {};

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
d_0113 = 1;% <<<TURN ON HERE<<<<<
if d_0113 > 0   
    datastr{end+1} = 'data_201301_QM.txt';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end

%Shake data collected Jan13
dzzf_human = 0;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'zhifei_human_data.csv';
end

dzzf_autograsp = 1;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
end
   
%------------End Defind Parameters-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Seed Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////

% Limited data with GraspIt and crowdsourcing
% datatemp = importdata(datastr{1});
% resultstemp1= importdata(results2str{1});
% resultstemp2 = importdata(results1str{1});      
% 
% data = [data;datatemp.data(:,2:end-1)];
% crowdsourceLength = size(data,1);
% % Crowdsource estimate
% results1 = resultstemp1.data(:,res2index(1));
% % Physical testing
% groundtruth = [groundtruth;resultstemp2.data(:,2)];
% header = datatemp.colheaders(2:end-1);
% 
% 
% % Extensive dataset with GraspIt, but no crowdsourcing
% datatemp = importdata(datastr{2});  
% 
% data = [data;datatemp.data(:,1:end-3)];
% % GraspIt estimate
% results2 = datatemp.data(:,end-2);
% % Physical testing
% t = datatemp.data(:,end);
% groundtruth = [groundtruth;t];
%       


%% Special handling for Jan dataset
datatemp = importdata(datastr{1});
resultstemp1= importdata(results1str{1});
resultstemp2 = importdata(results2str{1});      

% For these datasets we want to exclude energy as a feature
data = [data;datatemp.data(:,2:end-1)];
% GraspIt estimate
results1 = [datatemp.data(:,end)];
% Physical testing
groundtruth = [resultstemp1.data(:,2)];
% Crowdsource estimate
results2 = [resultstemp2.data(:,res2index(1))];
header = datatemp.colheaders(2:end);

%%

% for i = 2:length(datastr)
%       datatemp = importdata(datastr{i});
%       
%       % For these datasets we only want the first 11 metrics
%       data = [datatemp.data(:,1:dimensionCount)];
%       % GraspIt estimate
%       results1 = [datatemp.data(:,dimensionCount+1)];
%       results2 = ...
%       % Physical testing
%       groundtruth = [datatemp.data(:,dimensionCount+3)];
%       if i ==1
%           header = datatemp.colheaders(2:end);
%       elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
%          error('headers do not match, check data files')   
%       end   
% end

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


% Hold onto the non-reduced data for logistic
rawdata = data;


%Sphereize data
data = sphereize(data);

physicalCutoff = .8;
data = data((groundtruth > physicalCutoff),:);
groundtruth = groundtruth(groundtruth > physicalCutoff);
results1 = results1(groundtruth > physicalCutoff);
results2 = results2(groundtruth > physicalCutoff);

crowdsourceCutoff = .7;

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 30;

% Keep with cutoff of 0.8
%binshakeresults = bincutoff(groundtruth,0.8);

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .2;


% Amount to increase the energy cutoff by each iteration.
iterStep = 10;

energyCutoffCeiling = min(floor(max(results1)/iterStep)*iterStep,120);


for energycutoff = iterStep:iterStep:energyCutoffCeiling
    auc = zeros(1,loopn);
    
    coeff_linear = [];
    coeff_const = [];
    
    % Train LDA on points in data-dim space
    % Add points in class 1 if only graspIt thought success,
    % 2 if only crowdsource,
    % if both then duplicate the point for both classes.
    trainIndices_crowdsource = find(results2 >= crowdsourceCutoff);
    trainIndices_graspit = find(results1 < energycutoff);
    
    if (size(trainIndices_crowdsource,1)+size(trainIndices_graspit,1)) < 20
        disp(strcat({'Too few datapoints for cutoff '},num2str(energycutoff),'; skipping.'));
        continue;
    end

    for l = 1:loopn
        
        trainData = data([trainIndices_crowdsource; trainIndices_graspit],:);
        trainy = [zeros(length(trainIndices_crowdsource),1); ones(length(trainIndices_graspit),1)];
        
        % Make testx vector of appropriate size
        if(testProportion >= 1)
            testx = zeros(testProportion,1);
        else
            testx = zeros(floor(size(trainData,1) * testProportion),1);
        end

        testIndices = randomIndices(trainData,length(testx));
        testx = trainData(testIndices,:);
        trainData(testIndices,:) = [];
        trainy(testIndices) = [];
    

        dbstop if error
        [class,err,POSTERIOR,logp,coeff] = classify(testx,trainData,trainy);
        coeff_linear(l,:) = coeff(2,1).linear;
        coeff_const(l,:) = coeff(2,1).const;

        disp(strcat({'Completed iteration: '},num2str(l)));
    end

    save(strcat(savedir,'LDA_success_lumped_results_cutoff_',num2str(energycutoff),'.mat'),'coeff_const','coeff_linear');

end


%Calculate Computation time
disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)
