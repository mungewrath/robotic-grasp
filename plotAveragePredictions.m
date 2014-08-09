% Plots data. Red corresponds to incorrect
% predictions, blue to correct.
% predictions: vector of all predictions. If averaging over multiple runs,
%     predictions can be made on the same point multiple times.
% testIndices: indices of predictions being made.
% groundtruth: vector of ground truth for all points.

function [pointsVisualized,x,y,accuracies] = plotAveragePredictions(data,predictions,testIndices,groundtruth,graphTitle,saveTitle)
    accuracies = zeros(length(groundtruth),1);
    
    for i = 1:length(accuracies)
        preds_i = predictions(testIndices == i);
        accuracies(i) = mean(preds_i == groundtruth(i));
    end
    %results = (predictions == groundtruth);
    h = figure;
    hold on;
    x = data(:,1);
    y = data(:,2);
    pointsVisualized = find(~isnan(accuracies));
    plotted_indices = find(~isnan(accuracies));
    
    %axis([-(max(abs(x))+1),(max(abs(x))+1),-(max(abs(y))+1),(max(abs(y))+1)]);
    
    results = (accuracies >= .5);
    results = results(plotted_indices);

    scatter(x(~results),y(~results),'r');
    scatter(x(results),y(results),'b');

    title({graphTitle,sprintf('Displaying %d / %d points',length(pointsVisualized),size(data,1))});

    if nargin > 5
        set(h, 'InvertHardCopy', 'off');
        print(h,'-dpng',strcat(saveTitle,'.png'));
    end
    
    accuracies = accuracies(plotted_indices);
end