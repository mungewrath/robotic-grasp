%%
% LDA_GraspIt_predictions.m
% Looks at TPR, FPR, TNR, and FNR for a GraspIt energy-trained LDA
% classifier.
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


for i = 1:length(datastr)
      datatemp = importdata(datastr{i});
      
      % For these datasets we only want the first 11 metrics
      data = [data;datatemp.data(:,1:dimensionCount)];
      % GraspIt estimate
      results1 = [results1;datatemp.data(:,dimensionCount+1)];
      % No crowdsourced data in the sets
      %results2 = ...
      % Physical testing
      groundtruth = [groundtruth;datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
      end   
end

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

% Testing only on physical testing successes
physicalCutoff = .8;
% data = data((groundtruth > physicalCutoff),:);
% groundtruth = groundtruth(groundtruth > physicalCutoff);
% results1 = results1(groundtruth > physicalCutoff);
% results2 = results2(groundtruth > physicalCutoff);

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




summaryStats = [];

%For every cutoff:
    % For n iterations:
    % Train LDA on energy
    % Obtain predictions based on 80/20 split
    % Add 1 to each TPR,FPR,TNR,FNR based on physical testing
    % 
    % Aggregate all rates. Output each based on highest count occurrence for
    % each, and export
    % end iterations
%end cutoff
for energycutoff = iterStep:iterStep:energyCutoffCeiling
    auc = zeros(1,loopn);
    
    coeff_linear = [];
    coeff_const = [];
    
    TPR = zeros(size(data,1),1);
    FPR = zeros(size(data,1),1);
    TNR = zeros(size(data,1),1);
    FNR = zeros(size(data,1),1);
    
    if (sum(results1 <= energycutoff) < 20)
        disp(strcat({'Too few datapoints for cutoff '},num2str(energycutoff),'; skipping.'));
        continue;
    end

    for l = 1:loopn
        
        trainData = data;
        trainy = (results1 <= energycutoff);
        
        % Make testx vector of appropriate size
        if(testProportion >= 1)
            testx = zeros(testProportion,1);
        else
            testx = zeros(floor(size(trainData,1) * testProportion),1);
        end

        testIndices = randomIndices(trainData,length(testx));
        % Unlike most of the other experiments, we want predictions for
        % both test and training points.
        testx = trainData;
        trainData(testIndices,:) = [];
        trainy(testIndices) = [];
    

        dbstop if error
        [class,err,POSTERIOR,logp,coeff] = classify(testx,trainData,trainy);
        coeff_linear(l,:) = coeff(2,1).linear;
        coeff_const(l,:) = coeff(2,1).const;
        
        TPR = TPR + (class == 1 & (groundtruth >= cutoff) == 1);
        FPR = FPR + (class == 1 & (groundtruth >= cutoff) == 0);
        TNR = TNR + (class == 0 & (groundtruth >= cutoff) == 0);
        FNR = FNR + (class == 0 & (groundtruth >= cutoff) == 1);
        assert(sum([TPR; FPR; TNR; FNR]) == size(data,1)*l);
        
        
        res = +(groundtruth(testIndices) >= cutoff);
        mf = +class(testIndices);
        
        dbstop if error
        
        % Generate ROC curve
        try
            [perfx,perfy,perft,auc(l)] = perfcurve(res,mf,1);
            %rocResults = [rocResults; [perfx perfy]];
            [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
        catch err
            auc(l) = NaN;
        end

        %disp(strcat({'Completed iteration: '},num2str(l)));
    end
    
    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,loopn);
    
    %Display AUC Info
    disp(energycutoff);
    disp('AUC ave err std')
    disp(mean(auc))
    disp(std(auc)/(loopn)^.5)
    disp(std(auc))
    
    [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
    disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
    disp([fpr;tpr;tprerr;tprstd])
   
    
    rateTotal = (size(data,1)*loopn);
    summaryStats(end+1,:) = [sum(TPR)/rateTotal sum(FPR)/rateTotal sum(TNR)/rateTotal sum(FNR)/rateTotal energycutoff];
    
    TPR_sorted = sortrows([data(TPR > 0,:) TPR(TPR > 0)],-(size(data,2)+1));
    FPR_sorted = sortrows([data(FPR > 0,:) FPR(FPR > 0)],-(size(data,2)+1));
    TNR_sorted = sortrows([data(TNR > 0,:) TNR(TNR > 0)],-(size(data,2)+1));
    FNR_sorted = sortrows([data(FNR > 0,:) FNR(FNR > 0)],-(size(data,2)+1));
    
    save(strcat(savedir,'LDA_predictionRates_TPR',num2str(energycutoff),'.mat'),'TPR_sorted');
    save(strcat(savedir,'LDA_predictionRates_FPR',num2str(energycutoff),'.mat'),'FPR_sorted');
    save(strcat(savedir,'LDA_predictionRates_TNR',num2str(energycutoff),'.mat'),'TNR_sorted');
    save(strcat(savedir,'LDA_predictionRates_FNR',num2str(energycutoff),'.mat'),'FNR_sorted');

    save(strcat(savedir,'LDA_energyTrained_',num2str(energycutoff),'.mat'),'coeff_const','coeff_linear');

end

disp('    TPR       FPR       TNR       FNR       cutoff');
disp(summaryStats);

%Calculate Computation time
disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)
