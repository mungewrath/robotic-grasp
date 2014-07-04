% Takes in point mappings, predictions, and true values, and produces
% Pearson correlation coefficient values for every kd-tree node.
% pointMappings: kd-tree node index for every test point.
% nodeIndices: list of all kd-tree nodes indices.

function correlations = getNodeCorrelations(pointMappings, nodeIndices, predictions, groundtruth)
    % For every index in nodeIndices:
    % compute the confidence rating by looking at ground truth
    % create logical vector of same size as nodeIndices, saying whether
    % or not to predict in bin.
    correlations = zeros(size(nodeIndices));
    for i = 1:size(nodeIndices,1)
        idx = nodeIndices(i);
        preds = predictions(pointMappings == idx);
        actuals_i = groundtruth(pointMappings == idx);
        if isempty(preds) || isempty(actuals_i)
            correlations(i) = NaN;
        else
            correlations(i) = corr(preds,actuals_i);
        end
    end
end