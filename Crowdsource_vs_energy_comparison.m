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

dimensionCount = 11;

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter
groundtruth = [];

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
datasetName = {};
results1str = {};
results2str = {};
res2index = [];

%Shake data collected Jan13

%Shake data collected Jan13
d201301 = 1;% <<<TURN ON HERE<<<<<
if d201301 > 0   
    datastr{end+1} = 'data_201301_QM.txt';
    datasetName{end+1} = 'Jan13 shake data';
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
      resultstemp1= importdata(results1str{i});
      resultstemp2 = importdata(results2str{i});      
      % For these datasets we only want the first 11 metrics
      data = [data;datatemp.data(:,2:end)];
      % Physical testing
      groundtruth = [groundtruth;resultstemp1.data(:,2)];
      % Crowdsource estimate
      results2 = [results2;resultstemp2.data(:,res2index(i))];
      % GraspIt estimate
      results1 = [results1;datatemp.data(:,dimensionCount+2)];
      if i ==1
          header = datatemp.colheaders(2:end);
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
      end   
end

threshold_ceiling = 100;

crowdsource_cutoff = .8;

% Restrict to successful grasps
graspitPredictions_all = results1(groundtruth >= cutoff);
crowdsourcePredictions_all = results2(groundtruth >= cutoff);
groundtruth_all = groundtruth(groundtruth >= cutoff);

close all

for energyCutoff = 10:10:threshold_ceiling
    % Separate into crowdsource, graspit, both, neither
    % graspit wants to minimize energy, while crowdsource wants to maximize
    % success estimate
    graspit_only = find(graspitPredictions_all < energyCutoff & crowdsourcePredictions_all < crowdsource_cutoff);
    crowdsource_only = find(graspitPredictions_all >= energyCutoff & crowdsourcePredictions_all >= crowdsource_cutoff);
    both = find(graspitPredictions_all < energyCutoff & crowdsourcePredictions_all >= crowdsource_cutoff);
    neither = find(graspitPredictions_all >= energyCutoff & crowdsourcePredictions_all < crowdsource_cutoff);
    
    figure(energyCutoff);
    scatter(graspitPredictions_all(crowdsource_only),crowdsourcePredictions_all(crowdsource_only),10,'b');
    title({'Grasp success vs. GraspIt energy'; datasetName{i}});
    hold on;
    scatter(graspitPredictions_all(graspit_only),crowdsourcePredictions_all(graspit_only),10,'r');
    scatter(graspitPredictions_all(both),crowdsourcePredictions_all(both),10,'m');
    scatter(graspitPredictions_all(neither),crowdsourcePredictions_all(neither),10,'k');

end