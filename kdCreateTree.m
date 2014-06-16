% Creates a KD tree. 
% threshold: the maximum number of elements allowed in a bin. If a bin has
% many identical members for a given dimension it will exceed this size,
% i.e. if the tree sorts on dimension 1, (9 2 9), (9 4 9), and (9 6 3) will
% all be returned in a single bin.
% maxDimension: The maximum dimension to split on. If not specified, the
% tree will split on all dimensions.
% dimension: the current dimension to split on.
% index_: used internally, do not pass as parameter.

function tree = kdCreateTree(data,threshold,maxDimension,dimension,index_)
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
        pivot = (min(data(:,dimension)) + max(data(:,dimension)))/2;
        % partition along the current dimension
        if pivot == min(data(:,dimension))
            % There are many identical values in the current dimension, so
            % just give up and return the tree.
            tree = { data, index_ };
        else
            partition1 = data(data(:,dimension) > pivot,:);
            partition2 = data(data(:,dimension) <= pivot,:);
            % divide data into two sets, and recursively create trees for each,
            % with different dimension
            nextDim = 1 + mod(dimension,maxDimension);
            
            % Only leaves actually hold the index, but pass it on to
            % children who might be that.
            tree = { pivot, kdCreateTree(partition1,threshold,maxDimension,nextDim,1+2*index_), kdCreateTree(partition2,threshold,maxDimension,nextDim,2+2*index_), maxDimension, size(data,2) };
        end
       
    else
        tree = { data, index_ };
    end
end