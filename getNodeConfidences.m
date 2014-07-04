% Takes in point mappings, predictions, and true values, and produces
% confidence values for every kd-tree node.
% pointMappings: kd-tree node index for every test point.
% nodeIndices: list of all kd-tree nodes indices.

function confidences = getNodeConfidences(pointMappings, nodeIndices, predictions, groundtruth)
    % For every index in nodeIndices:
    % compute the confidence rating by looking at ground truth
    % create logical vector of same size as nodeIndices, saying whether
    % or not to predict in bin.
    confidences = zeros(size(nodeIndices));
    for i = 1:size(nodeIndices,1)
        idx = nodeIndices(i);
        preds = predictions(pointMappings == idx);
        actuals_i = groundtruth(pointMappings == idx);
        % conf = TP / (TP + FP)
        TP = sum(preds & actuals_i);
        FP = sum(preds & ~actuals_i);
        confidences(i) = TP / (TP + FP);
    end
end