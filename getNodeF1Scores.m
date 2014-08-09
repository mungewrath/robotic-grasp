% Takes in point mappings, predictions, and true values, and produces
% confidence values for every kd-tree node.
% pointMappings: kd-tree node index for every test point.
% nodeIndices: list of all kd-tree nodes indices.

function f1scores = getNodeF1Scores(pointMappings, nodeIndices, predictions, groundtruth)
    % For every index in nodeIndices:
    % compute the confidence rating by looking at ground truth
    % create logical vector of same size as nodeIndices, saying whether
    % or not to predict in bin.
    f1scores = zeros(size(nodeIndices));
    for i = 1:size(nodeIndices,1)
        idx = nodeIndices(i);
        preds = predictions(pointMappings == idx);
        actuals_i = groundtruth(pointMappings == idx);
        % F1 = TP / (TP + FP)
        % Precision = TP / (TP + FP)
        % Recall = TP / (TP + FN)
        TP = sum(preds & actuals_i);
        FP = sum(preds & ~actuals_i);
        FN = sum(~preds & actuals_i);
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1scores(i) = 2*precision*recall/(precision+recall);
    end
end