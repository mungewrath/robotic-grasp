
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
%Import ROC Curves

rocload = dir('roccurves');
rocload = struct2cell(rocload);
rocload(2:end,:) = [];
rocload = rocload';

for i = length(rocload):-1:1
    if strcmp(rocload(i),'.') || strcmp(rocload(i),'..')
        rocload(i) = [];        
    end
end


avex = [];
avey = avex;


for i = 1:length(rocload)
    ind = rocload{i};
    temp = load (ind);
    avex = [avex temp.ave(:,1)];
    avey = [avey temp.ave(:,2)];
end

fig1head = {'2013_9_11Energythresh.mat';'2013_9_11Perpendicular Symmetrythresh.mat';'2013_9_106qm6pcpca.mat';'2013_9_11Finger Extensionthresh';'2013_9_11Finger Limitthresh'};
avexfig1 = [];
aveyfig1 = [];
upperxfig1 = [];
upperyfig1 = [];
lowerxfig1 = [];
loweryfig1 = [];

for i = 1:length(fig1head)
    ind = fig1head{i};
    temp = load (ind);
    avexfig1 = [avexfig1 temp.ave(:,1)];
    aveyfig1 = [aveyfig1 temp.ave(:,2)];
    upperxfig1 = [upperxfig1 temp.uppererr(:,1)];
    upperyfig1 = [upperyfig1 temp.uppererr(:,2)];
    lowerxfig1 = [lowerxfig1 temp.lowererr(:,1)];
    loweryfig1 = [loweryfig1 temp.lowererr(:,2)];
       
end

%Temporary Figure
figure (1)
plot(avex.*100,avey.*100)
hold on
axis square
axis([0 100 0 100])
ylabel({'TPR';'(%)'})
plot([0 100],[0 100],'color',[.5 .5 .5])
set(gca,'box','on','position',[.17,.15,.78,.8])
%text(-25,47,'(Percent)','FontSize',10)
xlabel('FPR (%)')
set(get(gca,'YLabel'),'Rotation',0)
% plot([40 45],[52,60],'k','linewidth',1)
set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
% text(10,84,'GP with PC1','FontSize',10)
% text(10,8,'Random Guess','FontSize',10)
% plot(ave(kstat,1),ave(kstat,2),'kx')
legend(rocload')


%Final Figure 1
figure (2)

hold on
col = .9;
set(figure(2), 'units', 'inches', 'pos', [10 3 3.25 3]) 
for i = 1:length(avexfig1(1,:))
    fill(([upperxfig1(:,i)' rot90(lowerxfig1(:,i),2)'].*100),([upperyfig1(:,i)' rot90(loweryfig1(:,i),2)'].*100),[col col col])
    plot(avexfig1(:,i).*100,aveyfig1(:,i).*100,'k')
    col = col-.1;
end

axis square
axis([0 100 0 100])
ylabel({'TPR';'(%)'})
plot([0 100],[0 100],'color',[.5 .5 .5])
set(gca,'box','on','position',[.17,.15,.78,.8])
%text(-25,47,'(Percent)','FontSize',10)
xlabel('FPR (%)')
set(get(gca,'YLabel'),'Rotation',0)
% plot([40 45],[52,60],'k','linewidth',1)
text(77,70,{'Random'; 'Guess'},'FontSize',10)
text(13,96,'GraspIt! Energy','FontSize',10)
text(25,8,{'Perpendicular'; 'Symmetry'},'FontSize',10)
text(2,79,{'GP on 6';'Quality';'Metrics'},'FontSize',10)
text(70,40,{'Finger';'Extension'},'FontSize',10)
text(50,25,{'Finger Limit'},'FontSize',10)
plot([68 45],[40 60],'k')
plot([48 34],[25 60],'k')
set(get(gca,'YLabel'),'Position',[-13 45. 1.001])