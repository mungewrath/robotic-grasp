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

load aucarchivesparcity1


average = auc*0;
stand = zeros(1,length(auc(1,:)));

for i = 1:length(auc(1,:))
    temp = auc(:,i);
    temp = temp(isfinite(temp));
    tempave = zeros(1,length(temp));
        for j = 1:length(temp)
            tempave(j) = mean(temp(1:j));
        end
    stand(i) = std(temp);
    tempave = [tempave zeros(1,length(auc(1,:)))];
    average(:,i) = tempave(1:length(average(:,1)));
end

x = 1:length(average(:,1));

meanave = mean(average);

figure(1)
plot(21:72,meanave(21:72))
hold on
hold off

figure(2)
plot(1:length(latent),latent)
hold on
plot(1:length(latent),meanave(1:length(latent)),'g')
plot(1:length(latent),stand(1:length(latent)),'r')
legend('Variance','AUC','std')
hold off
