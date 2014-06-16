% precisionAndRecall
% Accepts several vectors of predictions and ground truth values and
% calculates precision and recall with either macro or micro averaging
% method.
% If filtering out prediction points (as with most of the kd hybrid
% algorithms), pass in the unfiltered ground truth set as allTestPoints.

function [precision, recall, p_err, r_err] = precisionAndRecall(predictions, groundtruth, macroAveraging, allTestPoints)
    % Intentionally breaking backwards compatibility - Weng-Keen says
    % recall must be in terms of all test points
    %if nargin < 4
    %    allTestPoints = groundtruth;
    %end
    assert(iscell(predictions) && iscell(groundtruth) && iscell(allTestPoints),'Inputs must be cell arrays!')
    if(~macroAveraging)
        % Micro: Add sets together when calculating precision and recall, resulting
        % in proportional weighting.
        TP = 0;
        FP = 0;
        FN = 0;
        for i = 1:length(predictions)
            TP = TP + sum(predictions{i} == 1 & groundtruth{i} == 1);
            FP = FP + sum(predictions{i} == 1 & groundtruth{i} == 0);
            FN = FN + sum(predictions{i} == 0 & groundtruth{i} == 1);
        end
        precision = TP / (TP + FP);
        p_err = std(cellfun(@(a,b) sum(a==1 & b==1) / sum(a==1),predictions,groundtruth));
        if(isnan(precision))
            precision = 0;
        end
        recall = TP / sum(cellfun(@sum,allTestPoints));
        r_err = std(cellfun(@(a,b) sum(a==1 & b==1) / sum(b==1),predictions,groundtruth));
        %recall = TP / (TP + FN);
        if(isnan(recall))
            recall = 0;
        end
    else
        % Macro: Calculate rates for each separate set, then averaging the values.
        % Results in equal weighting even for different sized datasets.
        precision_iter = zeros(length(predictions),1);
        recall_iter = zeros(length(predictions),1);
        for i = 1:length(predictions)
            TP = sum(predictions{i} == 1 & groundtruth{i} == 1);
            FP = sum(predictions{i} == 1 & groundtruth{i} == 0);
            FN = sum(predictions{i} == 0 & groundtruth{i} == 1);
            precision_iter(i) = TP / (TP + FP);
            if(isnan(precision_iter(i)))
                precision_iter(i) = 0;
            end
            recall_iter(i) = TP / sum(allTestPoints{i});
            %recall_iter(i) = TP / (TP + FN);
            if(isnan(recall_iter(i)))
                recall_iter(i) = 0;
            end
        end
        precision = sum(precision_iter) / length(predictions);
        p_err = std(precision_iter);
        recall = sum(recall_iter) / length(predictions);
        r_err = std(recall_iter);
    end
end