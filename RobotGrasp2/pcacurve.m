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


data = xlsread('IROS_DATA2p2.xls');
results = xlsread('IROS_RESULTSp2.xls');
good = find(results>=.8);
bad = find(results<.8);

data = sphereize(data);
[coef,score,latent,ts1,expl] = pca(data);



close all
figure(1)
set(gca,'FontSize',10)
set(figure(1), 'units', 'inches', 'pos', [10 5 3.25 2]) 
plot(1:6,latent,'k','linewidth',2)
xlabel('Principle Component')
ylabel({'Variance'; '(Eigenvalues)'})
set(gca,'XTick',[1:1:6])
set(gca,'box','on','position',[.33,.2,.62,.75])
set(get(gca,'YLabel'),'Rotation',0)
set(get(gca,'YLabel'),'Position',[-.5 1.05 1.001])
axis([1 6 0 2.5])



figure(3)
set(gca,'FontSize',10)
set(figure(3), 'units', 'inches', 'pos', [10 5 3.25 3]) 
plot(score(good,1),score(good,2),'x','color',[.5 .5 .5])
hold on
plot(score(bad,1),score(bad,2),'ok')
xlabel('PC1')
ylabel('PC2')
set(get(gca,'YLabel'),'Rotation',0)
legend('Successful Grasp','Failed Grasp',1)
set(gca,'box','on','position',[.15,.12,.83,.85])
set(get(gca,'YLabel'),'Position',[-4 1 1.001])
set(gca,'XTick',[-2:2:6])
set(gca,'YTick',[-2:2:6])
axis([-3 6 -3 6])
hold off
