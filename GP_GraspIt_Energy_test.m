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

%Data Processing
runpca = 1;                 %Value of 1 initiates PCA analysis
pcakeep = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>=1;        %which PC's to keep converted to logical var
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


for i = 1:length(datastr)
      datatemp = importdata(datastr{i});
      %resultstemp1= importdata(results1str{i});
      %resultstemp2 = importdata(results2str{i});      
      
      % For these datasets we only want the first 11 metrics
      data = [data;datatemp.data(:,1:dimensionCount)];
      % GraspIt estimate
      results1 = [results1;datatemp.data(:,dimensionCount+1)];
      % Physical testing
      results2 = [results2;datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
      end   
end

groundtruth = results2;

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
end


%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

% Compute GP

meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;

x = data;
y = results1;
xdepth = length(x(1,:));
covdepth = xdepth+1;

loopn = 30;

auc = zeros(1,loopn);

% Keep with cutoff of 0.8
binshakeresults = bincutoff(groundtruth,0.8);

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .2;

for l = 1:loopn
    hyp = [];
    
    % Train hyperparameters
   
    hyp.mean = 0;
    hyp.cov(1:covdepth) = log(1);
    hyp.lik(1:1) = -.2;
    disp(hyp)
    
    % Partition training & test data
    if(testProportion >= 1)
        testx = zeros(testProportion,2);
    else
        testx = zeros(floor(length(x) * testProportion),2);
    end
    testIndices = zeros(length(testx),1);
    
    i = 1;
    % Pick test points randomly from the data set
    while ((i <= length(x) * testProportion && testProportion < 1) || (i <= testProportion && testProportion >= 1))
        newTest = randi(length(x));
        if ~any(testIndices==newTest)
            testx(i,:) = x(newTest,:);
            testIndices(i) = newTest;
            i = i+1;
        end
    end
    
    trainx = x;
    logistic_trainx = rawdata(testIndices,:);
    % Get rid of test data
    trainx(testIndices,:) = [];
    % Squash into 0:1 range
    trainy = normalizeer(log(y) ./ (1+log(y)));
    trainy(testIndices) = [];
    
    % Minimize hyperparameters
    hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);
   
    % Run GP
    % TODO: Save gp hyperparameters
    [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);
    
    res = binshakeresults(testIndices);
    mf = normalizeer(mf);
    
    % Generate ROC curve
    try
        [perfx,perfy,perft,auc(l)] = perfcurve(res',mf',1);
        [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
    catch err
        auc(l) = NaN;
    end
    
    disp(strcat('Completed iteration ',l))
end

[ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,loopn);

% ----- Copied ROC code ----- %

close all
h = figure(1)
set(figure(1), 'units', 'inches', 'pos', [8 5 3.25 3])
set(gca,'FontSize',10)
fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
%fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
hold on
plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
% Done below
% plot([0 100],[0 100],'color',[.5 .5 .5])
% set(gca,'box','on','position',[.17,.15,.78,.8])
% axis square
% axis([0 100 0 100])
% ylabel({'TPR';'(%)'})
% xlabel('FPR (%)')
% set(get(gca,'YLabel'),'Rotation',0)
% plot([40 45],[52,60],'k','linewidth',1)
% set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
% text(10,84,'GP with PC1','FontSize',10)
% text(10,8,'Random Guess','FontSize',10)
% saveas(h,strcat(savedir,'GraspIt_energy_squashed_',date),'bmp');

%Display AUC Info
disp('AUC ave err std')
disp(mean(auc))
disp(std(auc)/(loopn)^.5)
disp(std(auc))

%Calculate TPR at Given FPR thresholds

[ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
disp([fpr;tpr;tprerr;tprstd])


energyPlotColors = ['b', 'g', 'r', 'c'];
% Test different cutoff levels
for energyCutoff = 1:4
    dbstop if error
    for l = 1:loopn
        dbstop if error
        hyp = [];

        % Train hyperparameters

        hyp.mean = 0;
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:1) = -.2;
        disp(hyp)

        % Partition training & test data
        if(testProportion >= 1)
            testx = zeros(testProportion,2);
        else
            testx = zeros(floor(length(x) * testProportion),2);
        end
        testIndices = zeros(length(testx),1);

        i = 1;
        % Pick test points randomly from the data set
        while ((i <= length(x) * testProportion && testProportion < 1) || (i <= testProportion && testProportion >= 1))
            newTest = randi(length(x));
            if ~any(testIndices==newTest)
                testx(i,:) = x(newTest,:);
                testIndices(i) = newTest;
                i = i+1;
            end
        end

        trainx = x;
        % Get rid of test data
        trainx(testIndices,:) = [];
        % Squash into 0:1 range
        trainy = (y <= energyCutoff*10);
        trainy(testIndices) = [];

        % Minimize hyperparameters
        hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);

        % Run GP
        % TODO: Save gp hyperparameters
        [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);

        res = binshakeresults(testIndices);
        mf = normalizeer(mf);

        % Generate ROC curve
        try
            [perfx,perfy,perft,auc(l)] = perfcurve(res',mf',1);
            [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
        catch err
            disp('ERROR')
            auc(l) = NaN;
        end

        disp(strcat('Completed iteration ',l))
    end

    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,loopn);

    % ----- Copied ROC code ----- %
    fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
    %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])   
    plot(ave(:,1).*100,ave(:,2).*100,energyPlotColors(energyCutoff),'linewidth',2)
    

    %Display AUC Info
    disp('AUC ave err std')
    disp(mean(auc))
    disp(std(auc)/(loopn)^.5)
    disp(std(auc))

    %Calculate TPR at Given FPR thresholds

    [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
    disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
    disp([fpr;tpr;tprerr;tprstd])
end

plot([0 100],[0 100],'color',[.5 .5 .5])
set(gca,'box','on','position',[.17,.15,.78,.8])
axis square
axis([0 100 0 100])
ylabel({'TPR';'(%)'})
xlabel('FPR (%)')
set(get(gca,'YLabel'),'Rotation',0)
plot([40 45],[52,60],'k','linewidth',1)
set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
text(10,84,'GP with PC1','FontSize',10)
text(10,8,'Random Guess','FontSize',10)
saveas(h,strcat(savedir,'GraspIt_energy_',date),'bmp');

%Calculate Computation time
disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)