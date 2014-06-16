% Creates a KD tree, splitting on the median of the data. 
% threshold: the maximum number of elements allowed in a bin. If a bin has
% many identical members for a given dimension it will exceed this size,
% i.e. if the tree sorts on dimension 1, (9 2 9), (9 4 9), and (9 6 3) will
% all be returned in a single bin.
% maxDimension: The maximum dimension to split on. If not specified, the
% tree will split on all dimensions.
% dimension: the current dimension to split on.
% index_: used internally, do not pass as parameter.
%
% This is a modified version of kdCreateTree - the only difference is how
% it computes the pivot point for splitting bins. Trees produced by the
% function are compatible with all the other kd functions.

function tree = kdMedCreateTree(data,threshold,maxDimension,dimension,index_)
    if nargin < 5
        index_ = 0;
    end
    if nargin < 4
        if nargin < 3
            maxDimension = size(data,2);
        end
        dimension = 1;
    end

    % while tree not partitioned:
    if size(data,1) > threshold
        nextDim = 1 + mod(dimension,maxDimension);
        pivot = median(data(:,dimension));
        % partition along the current dimension
        if pivot == min(data(:,dimension))
            if data(1,:) == data(2,:)
                % There are many identical values in the current dimension, so
                % just give up and return the tree.
                tree = { data, index_ };
            else
                tree = kdMedCreateTree(data,threshold,maxDimension,nextDim,index_);
            end
        else
            partition1 = data(data(:,dimension) > pivot,:);
            partition2 = data(data(:,dimension) <= pivot,:);
            % divide data into two sets, and recursively create trees for each,
            % with different dimension
            
            % Only leaves actually hold the index, but pass it on to
            % children who might be that.
            tree = { pivot, kdMedCreateTree(partition1,threshold,maxDimension,nextDim,1+2*index_), kdMedCreateTree(partition2,threshold,maxDimension,nextDim,2+2*index_), maxDimension, size(data,2) };
        end
       
    else
        tree = { data, index_ };
    end
end