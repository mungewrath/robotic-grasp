function [tp,fp,tn,fn] = getTestResults(predictions,groundtruth)
    tp = sum(predictions == 1 & groundtruth == 1);
    fp = sum(predictions == 1 & groundtruth == 0);
    tn = sum(predictions == 0 & groundtruth == 0);
    fn = sum(predictions == 0 & groundtruth == 1);
end