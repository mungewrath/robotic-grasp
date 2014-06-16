% splitTrainTest: Takes an input matrix and divides it into two partitions,
% one a testing set selected with testIndices and the remainder being
% training data.

function [train, test] = splitTrainTest(Input, testIndices)
    test = Input(testIndices,:);
    train = Input;
    train(testIndices,:) = [];
end