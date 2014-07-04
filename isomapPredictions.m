% Plots data using isomap projection. Red corresponds to incorrect
% predictions, blue to correct.
% k: number of neighbors to use.

function pointsVisualized = isomapPredictions(data,predictions,groundtruth,graphTitle,k)
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
    
    results = (predictions == groundtruth);
    plotted_indices = (Y.index)';
    results = results(plotted_indices);
    figure;
    hold on;
    x = (Y.coords{1}(1,:))';
    y = (Y.coords{1}(2,:))';
    pointsVisualized = length(x);
    scatter(x(results),y(results),'b');
    scatter(x(~results),y(~results),'r');
    title({graphTitle,sprintf('Displaying %d / %d points',pointsVisualized,size(data,1))});
end