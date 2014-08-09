% Plots data using isomap projection. Red corresponds to incorrect
% predictions, blue to correct.
% k: number of neighbors to use.
% predictions: vector of all predictions. If averaging over multiple runs,
%     predictions can be made on the same point multiple times.
% testIndices: indices of predictions being made.
% groundtruth: vector of ground truth for all points.

function [pointsVisualized,x,y,accuracies] = isomapAveragePredictions(data,predictions,testIndices,groundtruth,graphTitle,k,saveTitle,flipX,flipY)
    % L2_distance expects one datapoint per column, as opposed to row.
    distances = L2_distance(data',data',1);
    opts = [];
    opts.verbose = false;
    opts.dims = 2;
    opts.display = false;
    if ~exist('k','var')
        k = 3;
    end
    [Y, ~, ~] = Isomap(distances,'k',k,opts);
    
    accuracies = zeros(length(groundtruth),1);
    
    for i = 1:length(accuracies)
        preds_i = predictions(testIndices == i);
        accuracies(i) = mean(preds_i == groundtruth(i));
    end
    %results = (predictions == groundtruth);
    h = figure;
    hold on;
    base = subplot(1,1,1);
    x = (Y.coords{1}(1,:))';
    y = (Y.coords{1}(2,:))';
    pointsVisualized = (Y.index)';
    plotted_indices = (Y.index)';
    
    axis([-(max(abs(x))+1),(max(abs(x))+1),-(max(abs(y))+1),(max(abs(y))+1)]);
    
    results = (accuracies >= .5);
    results = results(plotted_indices);

    scatter(x(~results),y(~results),'r');
    scatter(x(results),y(results),'b');
    
    disp('Displaying high-accuracy prediction points (sorted by X value)');
    strongPts = (accuracies >= .8);
    strongPts = strongPts(plotted_indices);
    plottedGroundtruth = groundtruth(plotted_indices);
    strongIndices = find(strongPts);
    strongIndices = sortrows([strongIndices x(strongIndices)],2);
    strongIndices = strongIndices(:,1);
    for idx = strongIndices'
        fprintf('Point %d: (%f,%f), %f%% accuracy. Ground truth value=%d\n',plotted_indices(idx),x(idx),y(idx),100*accuracies(plotted_indices(idx)),plottedGroundtruth(idx));
    end

    title({graphTitle,sprintf('Displaying %d / %d points',length(pointsVisualized),size(data,1))});

    if nargin > 7 && flipX
        set(base,'XDir','reverse');
    end
    if nargin > 8 && flipY
        set(base,'YDir','reverse');
    end
    if nargin > 6
        set(h, 'InvertHardCopy', 'off');
        print(h,'-dpng',strcat(saveTitle,'.png'));
    end
    
    accuracies = accuracies(plotted_indices);
end