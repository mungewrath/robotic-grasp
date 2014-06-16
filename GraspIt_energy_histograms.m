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

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
datasetName = {};
results1str = {};
results2str = {};
res2index = [];

%Shake data collected Jan13

dzzf_autograsp = 1;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    %datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
    datastr{end+1} = 'zhifei_autograsp_data_post-correction.csv';
    datasetName{end+1} = 'zhifei autograsp data';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end

dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    %datastr{end+1} = 'zhifei_human_data.csv';
    datastr{end+1} = 'zhifei_human_data_post-correction.csv';
    datasetName{end+1} = '';
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
      data{i} = [datatemp.data(:,1:dimensionCount)];
      % GraspIt estimate
      results1{i} = [datatemp.data(:,dimensionCount+1)];
      % Physical testing
      results2{i} = [datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
      %elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         %error('headers do not match, check data files')   
      end   
end

% January data requires special handling
datastr{end+1} = 'data_201301_QM.txt';
datatemp = importdata(datastr{end});
resultstemp1= importdata('results_201301_shake.txt');
resultstemp2 = importdata('results_201301_human.txt');
datasetName{end+1} = 'January shake-tested data';
% Physical testing
results2{end+1} = resultstemp1.data(:,2);
% GraspIt estimate
results1{end+1} = datatemp.data(:,dimensionCount+2);



close all

for i = 1:length(datastr)
    % zhifei_autograsp_data
    groundtruth = results2{i};
    y = results1{i};

    figure(i*3-2);
    hist(y(y < 120));
    title({'Energy distribution'; datasetName{i}});
    xlabel('Energy');
    ylabel('Frequency');

    figure(i*3-1);
    % Success
    subplot(2,1,1);
    hist(y(groundtruth>=cutoff),0:5:ceil(max(y(y < 120))/5)*5);
    title({'Energy distribution separated by physical testing result'; datasetName{i}});
    axis([0 ceil(max(y(y < 120))/5)*5 0 length(y)/3]);
    xlabel('Energy');
    ylabel('Frequency (success)');
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','b');
    % Failure
    subplot(2,1,2);
    hist(y(groundtruth<cutoff),0:5:ceil(max(y(y < 120))/5)*5);
    axis([0 ceil(max(y(y < 120))/5)*5 0 length(y)/3]);
    xlabel('Energy');
    ylabel('Frequency (failure)');
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','r');

    figure(i*3);
    scatter(y(groundtruth>=cutoff),y(groundtruth>=cutoff),10,'b');
    title({'Grasp success vs. GraspIt energy'; datasetName{i}});
    hold on;
    scatter(y(groundtruth<cutoff),y(groundtruth<cutoff),10,'r');

end