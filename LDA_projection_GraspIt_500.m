%%
%  LDA_projection_GraspIt_500.m
%  Trains an LDA from all points based on physical testing as ground truth.
%

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
dimensionCount = 10;

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
datalabel = {};
figurelabel = {};
results1str = {};
results2str = {};
res2index = [];

%Shake data collected Jan13
d201301 = 0;% <<<TURN ON HERE<<<<<
%if d201301 > 0   
%(d201301 used below)
    datastr{end+1} = 'data_201301_QM.txt';
    datalabel{end+1} = 'January shake test data (72)';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
%end

dzzf_autograsp = 1;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
    datalabel{end+1} = 'zhifei autograsp data (220)';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end

dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'zhifei_human_data.csv';
    datalabel{end+1} = 'zhifei human data (500)';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end
   
%------------End Defind Parameters-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Seed Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////

if d201301 > 0
    % Limited data with GraspIt and crowdsourcing
    datatemp = importdata(datastr{1});
    resultstemp1= importdata(results2str{1});
    resultstemp2 = importdata(results1str{1});

    data{i} = datatemp.data(:,2:end-1);
    crowdsourceLength = size(data,1);
    % Crowdsource estimate
    results1{i} = resultstemp1.data(:,res2index(1));
    % Physical testing
    groundtruth{i} = resultstemp2.data(:,2);
    header = datatemp.colheaders(2:end-1);
end
 
for i = 2:length(datastr)
    % Extensive dataset with GraspIt, but no crowdsourcing
    datatemp = importdata(datastr{i});  

    data{end+1} = datatemp.data(:,1:dimensionCount);
    figurelabel{end+1} = datalabel{i};
    % GraspIt estimate
    results2{end+1} = datatemp.data(:,end-2);
    % Physical testing
    t = datatemp.data(:,end);
    groundtruth{end+1} = t;
    header = datatemp.colheaders(2:end-1);
end
      

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

close all

for i = 1:length(data)

    %Remove NAN columns
    datatemp = isnan(mean(data{i}));
    data{i}(:,datatemp) = [];
    disp('!!!!below quality measures have been removed due to insufficient data!!!!!!!!!!!!')
    disp( header(datatemp))
    header(datatemp) = [];


    %Sphereize data
    data{i} = sphereize(data{i});

    % Hold onto the non-reduced data for logistic
    rawdata = data{i};

    %------------End condition Data-----------------------------------------
    % ----- End copied code ----- %

    xdepth = length(data{i}(1,:));
    covdepth = xdepth+1;

    loopn = 100;

    auc = zeros(1,loopn);


    % Data to use as testing set. If < 1, it is interpreted as
    % a percentage; if >= 1, selects this many data points for the set.
    testProportion = .2;

    for l = 1:loopn
        % Keep with cutoff of 0.8
        binshakeresults = bincutoff(groundtruth{i},0.8);

        % Added in for comparative testing vs. hybrid
        if(testProportion >= 1)
            testx = zeros(testProportion,1);
        else
            testx = zeros(floor(size(data{i},1) * testProportion),1);
        end

        testIndices = randomIndices(data{i},length(testx));

        testx = data{i}(testIndices,:);
        trainx = data{i};
        trainx(testIndices,:) = [];
        trainy = (groundtruth{i} > cutoff);
        trainy(testIndices) = [];

        % TODO: Run without binshakeresults
        % Get test ground truth
        res = binshakeresults(testIndices);

        [class,err,POSTERIOR,logp,coeff] = classify(testx,trainx,trainy);
        %coeff_linear(l,:) = coeff(2,1).linear;
        %coeff_const(l,:) = coeff(2,1).const;

        % Generate ROC curve
        try
            [perfx,perfy,perft,auc(l)] = perfcurve(res',double(class)',1);
            %rocResults = [rocResults; [perfx perfy]];
            [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
        catch err
            auc(l) = NaN;
        end

        disp(['Completed iteration ' num2str(l)])

    end

    %save(strcat(savedir,'LDA_results.mat'),'coeff_linear','coeff_const');

    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,loopn);

    % ----- Copied ROC code ----- %

    figure(i)
    set(figure(i), 'units', 'inches', 'pos', [8 5 3.25 3])
    set(gca,'FontSize',10)
    fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
    %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
    hold on
    plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
    %plot(rocResults(:,1).*100,rocResults(:,2).*100,'k','linewidth',2)
    plot([0 100],[0 100],'color',[.5 .5 .5])
    set(gca,'box','on','position',[.17,.15,.78,.8])
    axis square
    axis([0 100 0 100])
    ylabel({'TPR';'(%)'})
    xlabel('FPR (%)')
    title(figurelabel{i})
    set(get(gca,'YLabel'),'Rotation',0)
    plot([40 45],[52,60],'k','linewidth',1)
    set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
    text(10,84,'GP with PC1','FontSize',10)
    text(10,8,'Random Guess','FontSize',10)

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

%Calculate Computation time
disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)