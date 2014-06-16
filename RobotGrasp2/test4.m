clear
clc
close all

%load cereal.mat

%X = [Calories Protein Fat Sodium Fiber Carbo Sugars Shelf Potass Vitamins];

% %Take a subset from a single manufacturer.
% mfg1 = strcmp('G',cellstr(Mfg));
% X = X(mfg1,:);

data = xlsread('IROS_DATA.csv');
X = sphereize(data);
 
%dissimilarities = pdist(zscore(X),'cityblock');
dissimilarities = pdist(X);


[Y,stress] = mdscale(dissimilarities,2,'criterion','metricstress');

[Y2,engen] = cmdscale(dissimilarities);

hold on
plot(Y(:,1),Y(:,2),'.','LineWidth',2);
plot(Y2(:,1),Y2(:,2),'.g','LineWidth',2);
hold off

% % Create a dissimilarity matrix.
% dissimilarities = pdist(X);
%
% % Use non-metric scaling to recreate the data in 2D,
% % and make a Shepard plot of the results.
% [Y,stress,disparities] = mdscale(dissimilarities,2);
% distances = pdist(Y);
% [dum,ord] = sortrows([disparities(:) dissimilarities(:)]);
% plot(dissimilarities,distances,'bo',dissimilarities(ord),disparities(ord),'r.-');
% xlabel('Dissimilarities'); ylabel('Distances/Disparities')
% legend({'Distances' 'Disparities'},'Location','NW');