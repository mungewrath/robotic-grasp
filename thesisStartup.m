%function thesisStartup(callingFile)
    %//////////////////////////////////////////////////////////////////////////
    %------Start LOAD FILES---------------------------------------
    %//////////////////////////////////////////////////////////////////////////
    clc, clearvars -except me energycutoff binThreshold
    close all

    c = clock;

    OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

    %me = callingFile;                                            % what is my filename
    %mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
    % Hard-coded path for now
    mydir = 'RobotGrasp2\';
    savedir = '.\';
   
    if exist('me','var')
        f = me;
    else
        f = mfilename;
    end
    
    if exist('use_timestamp','var') && use_timestamp == false
        diary(strcat(savedir,'logs/',f,'.log'));
    else
        diary(strcat(savedir,'logs/',f,'.log'));
    end
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
    addpath([savedir, 'toolbox_graph\toolbox_graph'])
    addpath([savedir, 'toolbox_graph\toolbox_graph\toolbox'])

    addpath([mydir,'LDA'])
    addpath([mydir,'roccurves'])

    clear mydir
    %------------End Load Files----------------------------------------------
    
    mySeed = 10;
    rng(mySeed);             % Set the seed
    
    %//////////////////////////////////////////////////////////////////////////
    %-------------Start Define parameters------------------------------------
    %//////////////////////////////////////////////////////////////////////////
    
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
%end