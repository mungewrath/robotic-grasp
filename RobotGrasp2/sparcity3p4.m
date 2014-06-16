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
shakedata = xlsread('IROS_data2p2.xls');

targetedbins = xlsread('top20bin.xls');

datacondition = 5;  %Select which dataset you want

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------Functions SELECTION------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;  %%CHANGING THIS WILL CHANGE OTHER THINGS, function lengths not computed for this

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterate and average hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graspno = unique(index);
loopn = 300;

y = results;

disp('done with data')



binshakeresults = bincutoff(shakeresults,.8);

hyparch = zeros(length(targetedbins),12);
indexarch = index;
sparceind = 8:8:72;
psparceind = (1-(sparceind./72))*100;
sample = 20; %%%%%

auc1 = zeros(loopn,sample,length(sparceind));
auc2 = auc1;
auc3 = auc1;
latent = zeros(6,sample,length(sparceind));

 

for k = 1:length(sparceind)
    for j = 1:sample
        [x, y, ind,~,~,~,~] = leaverand2( data,results,index,psparceind(k));
         x = sphereize(x);

        [~,x,latent(:,j,k)] = pca(x);
        xarch = x;

        for n = 1:3
            totalset = n;
            x = xarch(:,1:n);
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

            for l = 1:loopn

                ellstore = [];
                hyp = [];
                hypcovtot = [];
                hypmeantot = [];
                hypliktot = [];

                %Train Hyperparameters

                %Define leave out percent
                [trainx, trainy, trainind, testx, testy, testind, leavevect] = leaverand2(x,y,ind,leave);

                hyp.mean = zeros(meandepth,1);
                hyp.cov(1:covdepth) = log(1);
                hyp.lik(1:likdepth) = -.2;

                %%%%%%%CHANGE ITER TO 1000!!!!%%%%%%%%%%
                hyp = minimize(hyp, @gp, -600, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%CALCULATE GP's
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);
                mf = normalizeer(mf);

                try
                    res = binshakeresults(testind);
                    [~,~,~,auctemp] = perfcurve(res',mf',1);
                catch err
                    auctemp = NaN;
                end
                
                if n==1
                    auc1(l,j,k) = auctemp;
                elseif n==2
                    auc2(l,j,k) = auctemp;
                else
                    auc3(l,j,k) = auctemp;
                end

                disp([l auctemp])
            end
               
        end
        save('aucarchivesparcity3p4.mat','auc1','auc2','auc3','latent','psparceind')

    end
end

