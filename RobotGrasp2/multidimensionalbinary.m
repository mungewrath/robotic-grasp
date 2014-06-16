clear
clc
close all

threshold = [.2 .4 .6 .8 1];

mturresults = xlsread('mechturkrawdata.csv');
mturresults = mturresults(:,[1 9]);
graspno = sort(unique(mturresults(:,1)));

%Shake Testing
data = xlsread('IROS_DATA.csv');
shakeresults = xlsread('IROS_RESULTS.csv');
aveshakeresults = shakeresults;
shakepr = zeros(length(shakeresults(:,1))*8,1);
shakedata = zeros(length(shakepr(:,1)),length(data(1,:)));
shakeindex = shakepr(:,1);

for i = 1:length(graspno)
    num = round(shakeresults(i)*8);
    if num>=0
        shakepr(((i-1)*8+1):((i-1)*8+num)) = 1;
    end
    shakedata((i*8-7):i*8,:) = repmat(data(i,:),8,1);
    shakeindex((i*8-7):i*8) = graspno(i);
end
results = shakepr;

%Mechanical Turk
%Read Data
mturResults = xlsread('mechturkrawdata.csv');
mturresults = mturResults(:,[1 9]);
mturresults = matave(mturresults);
mturresults(:,1) = [];

%Determine physical successes


X = sphereize(shakedata);
 
dissimilarities = pdist(X);

[Y,stress] = mdscale(dissimilarities,2,'criterion','metricstress');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Physical Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure (1)

for j = 1:length(threshold)
    sucind = 1;
    failind = 1;
    psuccess = [];
    pfailure = [];
    for i = 1:length(results)
        if results(i)>=threshold(j)
            psuccess(sucind) = i;
            sucind = sucind+1;  
        else
            pfailure(failind) = i;
            failind = failind+1;
        end
    end
    subplot(2,3,j)
    title(strcat('Physical Averaged Data Threshold = ',num2str(threshold(j))))
    hold on
    plot(Y(psuccess,1),Y(psuccess,2),'+','LineWidth',2);
    plot(Y(pfailure,1),Y(pfailure,2),'or','LineWidth',2);

end
legend('success','failure',1)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Mturk Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure (2)
% 
% for j = 1:length(threshold)
%     sucind = 1;
%     failind = 1;
%     msuccess = [];
%     mfailure = [];
%     for i = 1:length(mturresults)
%         if mturresults(i)>=threshold(j)
%             msuccess(sucind) = i;
%             sucind = sucind+1;  
%         else
%             mfailure(failind) = i;
%             failind = failind+1;
%         end
%     end
%     subplot(2,3,j)
%     title(strcat('Croudsourced Averaged Data Threshold = ',num2str(threshold(j))))
%     hold on
%     plot(Y(msuccess,1),Y(msuccess,2),'+','LineWidth',2);
%     plot(Y(mfailure,1),Y(mfailure,2),'or','LineWidth',2);
%     if j == 2
%         legend('success','failure',1)
%     end
% end
% hold off



