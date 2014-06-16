clear
clc
close all

%Shake Testing
data = xlsread('IROS_DATA2.csv');
results = xlsread('IROS_RESULTS.csv');
threshold = [.8];
psuccess = [];
pfailure = [];

data(:,[2 3 4]) = [];

%Determine physical successes

X = sphereize(data);
 
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
    title(strcat('Physical Averaged Data Threshold = ',num2str(threshold(j))))
    hold on
    plot(Y(psuccess,1),Y(psuccess,2),'+','LineWidth',2);
    plot(Y(pfailure,1),Y(pfailure,2),'or','LineWidth',2);

end
legend('success','failure',1)
hold off




