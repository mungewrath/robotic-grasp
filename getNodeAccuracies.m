% Takes in point mappings, predictions, and true values, and produces
% conventional accuracy values for every kd-tree node.
% pointMappings: kd-tree node index for every test point.
% nodeIndices: list of all kd-tree nodes indices.

function accuracies = getNodeAccuracies(pointMappings, nodeIndices, predictions, groundtruth)
    % For every index in nodeIndices:
    % compute the confidence rating by looking at ground truth
    % create logical vector of same size as nodeIndices, saying whether
    % or not to predict in bin.
    accuracies = zeros(size(nodeIndices));
    for i = 1:size(nodeIndices,1)
        idx = nodeIndices(i);
        preds = predictions(pointMappings == idx);
        actuals_i = groundtruth(pointMappings == idx);
        accuracies(i) = sum(preds == actuals_i) / length(actuals_i);
    end
end