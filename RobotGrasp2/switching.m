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
addpath([mydir,'results'])
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
shakedata = xlsread('IROS_DATA2p3.xls');
%shakedata = xlsread('IROS_DATA.xls');


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
    %Test non-binary data
    data = shakedata;
    results = shakeresults;
    index = (1:length(data))';
    savestr = '2013nonexpanded'; 
    %merge physical testing data - old and new
    data2 = xlsread('ravisolddatap3.xls');
    results2 = xlsread('ravisoldresults.xls');
    averesults2 = results2;
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
    mturresults = xlsread('mechturkrawdata.csv');
    mturindex = mturresults(:,1);
    mturresults = mturresults(:,9);
end


%Sphereize data
data = sphereize(data);

[~,data,latent] = pca(data);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------Functions SELECTION------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanfunc = @meanConst;
covfunc = @covSEard2;
likfunc = @likGauss;
inffunc = @infExact;  %%CHANGING THIS WILL CHANGE OTHER THINGS, function lengths not computed for this

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterate and average hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graspno = unique(index);
loopn = 300;

y = results;

disp('done with data')

auc = zeros(loopn,12);
fpr = auc;
tpr = auc;

binshakeresults = bincutoff(shakeresults,.8);

hyparch = zeros(length(targetedbins),12);

for m = 1:1
    
    %x = data;
    x = data(:,1);%For PC Reduction
    totalset = length(data(:,1));
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

    covfunc = @covSEard;

    perfx = zeros(15,loopn);
    perfy = perfx;
    perft = perfx;
    
    for l = 1:loopn
        
        ellstore = [];
        hyp = [];
        hypcovtot = [];
        hypmeantot = [];
        hypliktot = [];

        %Train Hyperparameters

        %Define leave out percent
        [trainx, trainy, trainind, testx, testy, testind, leavevect] = leaverand2(x,y,index,leave);

        hyp.mean = zeros(meandepth,1);
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:likdepth) = -.2;

        %%%%%%%CHANGE ITER TO 1000!!!!%%%%%%%%%%
        hyp = minimize(hyp, @gp, -900, inffunc, meanfunc, covfunc, likfunc, trainx, trainy);

        hyptemp = [hyp.cov 0 0 0 0 0 0 0 0 0 0 0 0];
        hyptemp = hyptemp(1:12);
        hyparch(m,:) = hyptemp;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%CALCULATE GP's
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainx,trainy,testx);
        res = binshakeresults(testind);
        [~,~,~,~,sumtemp] = truefalse([mf res],.8,.8);
        
        mf = normalizeer(mf);

        try

            [perfx(:,l),perfy(:,l),perft(:,l),auc(l,m)] = perfcurve(res',mf',1);
        catch err
            auc(l,m) = NaN;
        end
        
        disp([l auc(l,m)])
        

        
        fpr(l,m) = sumtemp(2,2)/(sumtemp(2,2)+sumtemp(2,3));
        tpr(l,m) = sumtemp(2,4)/(sumtemp(2,1)+sumtemp(2,4));
        
        save('aucGP6heruronly.mat','auc','hyparch','tpr','fpr')

    end
 
    

end

%Remove 0 rows
for i = loopn:-1:1
    if sum(perfx(:,i)) ==0
        perfx(:,i) = [];
        perfy(:,i) = [];
    end
end

[ave,upper,lower,kstat,kstatsterr] = rotandextrap(perfx,perfy);

enerx = shakedata(:,6);
enery = bincutoff(shakeresults,.8);
[rocenerx,rocenery,~,AUC] = perfcurve(enery,enerx,1);

[~,kstatener] = max(rocenerx*sind(-45)+rocenery*cosd(-45));


%AUC GP
AUCGP = 0;
for i = 1:length(ave(:,1))-1
    AUCGP = AUC + (ave((i+1),1)-ave((i),1))*mean([ave((i),2) ave((i+1),2)]);
end


%Energy Calculation
energyauc = zeros(1,loopn);
rocenergx = zeros(59,loopn);
rocenergy = rocenergx;

for i = 1:loopn
    [enx, eny, ~, ~, ~, ~, ~] = leaverand2(enerx,enery,(1:1:length(enerx))',leave);
    enx = normalizeer(enx);
    [rocenergx(:,i),rocenergy(:,i),~,energyauc(i)] = perfcurve(eny',enx',1);
end

[energyave,energyupper,energylower,energykstat,energykstatsterr] = rotandextrap(rocenergx,rocenergy);

close all
figure(1)
set(figure(1), 'units', 'inches', 'pos', [8 5 3.25 3])
set(gca,'FontSize',10)
fill(([upper(:,1)' rot90(lower(:,1),2)'].*100),([upper(:,2)' rot90(lower(:,2),2)'].*100),[.75 .75 .75])
hold on
fill(([energyupper(:,1)' rot90(energylower(:,1),2)'].*100),([energyupper(:,2)' rot90(energylower(:,2),2)'].*100),[.55 .55 .55])
plot(energyave(:,1).*100,energyave(:,2).*100,'k')
plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
%plot(rocenerx,rocenery,'--','color',[.25 .25 .25])
plot([0 100],[0 100],'color',[.5 .5 .5])
set(gca,'box','on','position',[.17,.15,.78,.8])
axis square
axis([0 100 0 100])
ylabel({'TPR';'(%)'})
%text(-25,47,'(Percent)','FontSize',10)
xlabel('FPR (%)')
set(get(gca,'YLabel'),'Rotation',0)
plot([40 45],[52,60],'k','linewidth',1)
set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
text(30,50,'Energy','FontSize',10)
text(10,84,'GP with PC1','FontSize',10)
text(10,8,'Random Guess','FontSize',10)
% plot(ave(kstat,1),ave(kstat,2),'kx')


disp('GP FPR TPR')
ave(kstat,:)

disp('Energy FPR TPR')
energyave(energykstat,:)

% for i = 1:300
%     plot(rocenergx(:,i),rocenergy(:,i))
%     hold on
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %OUTPUTS
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disp('\nGP generated FP Rate: ')
% disp(fitnorm(statnorm,1))
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %ROC GRAPHs
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %Plot Setup 21Mar13
% hold on
% %Raw Data and fit curves
% plot(ccritnorm(:,6),ccritnorm(:,7),':k','LineWidth',1)
% plot(fitnorm(:,1),fitnorm(:,2),'k','LineWidth',3)
% %Random Gueass Line
% plot([0 1],[0 1],'color',[.2 .2 .2])
% legend('GP raw', 'GP Fitted Curve','Random Guess',4)
% hold off
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% title('ROC Space Normalized Grasp Data')
% axis([0 1 0 1])
% axis square
% 
% save(savestr);