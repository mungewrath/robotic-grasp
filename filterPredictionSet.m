% Filters test points based on confidence values for kd-tree nodes.

function [kept,eliminated,elimIndices] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences, tree, testx)
    elimIndices = false(size(testx,1),1);
    for i = 1:size(testx,1)
        idx = kdGetBinIndex(tree,testx(i,:));
        C = confidences(nodeIndices==idx);
        if(confidence_cutoff > 0 && (isnan(C) || C < confidence_cutoff))
            elimIndices(i) = true;
        end
    end
    kept = testx(~elimIndices);
    eliminated = testx(elimIndices);
end