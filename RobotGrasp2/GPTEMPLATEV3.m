
%%
%//////////////////////////////////////////////////////////////////////////
%------Start LOAD FILES---------------------------------------
%//////////////////////////////////////////////////////////////////////////
clc, clear all, close all

c = clock;

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

addpath([mydir,'LDA'])
addpath([mydir,'roccurves'])

clear me mydir
%------------End Load Files----------------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Start Define parameters------------------------------------
%//////////////////////////////////////////////////////////////////////////

%Training/Testing Splits
leave = 20;         %leaves out desired number in %

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter

%Data Processing
runpca = 1;                 %Value of 1 initiates PCA analysis
pcakeep = [1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>=1;        %which PC's to keep converted to logical var
yset = 0;   %0 = train off ground truth, 1 = train off human rating
qmset = 3;  %1 how many of the top QM's to keep based on T-Tests, or decimal for p value cutoff
comptype = 6; %1 = PCA, 2 = FLD LCA, 3 = Fischer, 4 = no comp anal,5 = no PCA, 6 = phys and crowd;specify comp in compselect
gpiter = 100; %how many times to iterate GP
compselect = 3;
ldasel = [1 2 3 4 5 6];

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
results1str = {};
results2str = {};
res2index = [];

%Ravi's TRO data (human guided grasps)
d201208 = 0;%<<<TURN ON HERE<<<<<
if d201208 > 0   
    datastr{end+1} = 'data_201208_QM.txt';
    results1str{end+1} = 'results_201208_shake.txt';
    results2str{end+1} = 'results_201208_human.txt';
    res2index = [res2index 2];  %Pick which Y value to use (2 is default for most)
end

%Shake data collected Jan13
d201301 = 1;% <<<TURN ON HERE<<<<<
if d201301 > 0   
    datastr{end+1} = 'data_201301_QM.txt';
    results1str{end+1} = 'results_201301_shake.txt';
    results2str{end+1} = 'results_201301_human.txt';
    res2index = [res2index 3];  %Pick which Y value to use (2 is default for most) - 3 uses human perception of grasps instead of rating
end

%Interface Data collected 
d201308s = 0;          %<<<TURN ON HERE<<<<<
d201308i = 0;          %<<<TURN ON HERE<<<<<
d201308g = 0;          %<<<TURN ON HERE<<<<<

if d201308s > 0
    datastr{end+1} = 'data_201308_QMs.txt';
    results1str{end+1} = 'results_201308_shakes.txt';
    results2str{end+1} = 'results_201308_humans.txt';
    res2index = [res2index 2];  %Pick which Y value to use (2 is default for most)
end

if d201308i > 0
    datastr{end+1} = 'data_201308_QMi.txt';
    results1str{end+1} = 'results_201308_shakei.txt';
    results2str{end+1} = 'results_201308_humani.txt';
    res2index = [res2index 2];  %Pick which Y value to use (2 is default for most)
end

if d201308g > 0
    datastr{end+1} = 'data_201308_QMg.txt';
    results1str{end+1} = 'results_201308_shakeg.txt';
    results2str{end+1} = 'results_201308_humang.txt';
    res2index = [res2index 2];  %Pick which Y value to use (2 is default for most)
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
      data = [data;datatemp.data(:,2:end)];
      results1 = [results1;resultstemp1.data(:,2)];
      results2 = [results2;resultstemp2.data(:,res2index(i))];
      if i ==1
          header = datatemp.colheaders(2:end);
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
      end   
end

groundtruth = results1;
index = (1:length(results1))';

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


%------------End condition Data-----------------------------------------

%%
%Run GP

[aucave,aucerr,aucstd,ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd,mfout,indexout] = gptrainandtest2(data,results1,index,groundtruth,gpiter,leave,cutoff);

%End GP


close all
figure(1)
set(figure(1), 'units', 'inches', 'pos', [8 5 3.25 3])
set(gca,'FontSize',10)
fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
hold on
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
text(10,84,'GP with PC1','FontSize',10)
text(10,8,'Random Guess','FontSize',10)
% plot(ave(kstat,1),ave(kstat,2),'kx')


disp('FPR TPR sterr std')
ave(kstat,:)
fprtpr = ave(kstat,:);  
disp(kstatsterr)
disp(kstatstd)

disp('AUC ave err std')
disp(aucave)
disp(aucerr)
disp(aucstd)

disp('computation time (s)')
clk = clock-c;
disp(clk(6)+clk(5)*60 + clk(4)*3600)

%%
%Start Export Section
cl = clock;
writefile = input('Please enter desired filename ','s');
writefile = ['roccurves/' num2str(cl(1)) '_' num2str(cl(2),2) '_' num2str(cl(3),2) writefile '.mat'];
save(writefile,'clk','kstat','aucave','uppererr','upperstd','lowerstd','lowererr','ave','aucerr','aucstd','kstatsterr','kstatstd')

