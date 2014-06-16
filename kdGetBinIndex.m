% Returns the index of the node containing x.
% x must match the dimensionality of the KD-tree's data, i.e. if the tree
% is composed of two-element vectors, x must be 2 elements as well.

function index = kdGetBinIndex(tree,x,dimension)
    if nargin < 3
        % Don't specify unless overridden in kdCreateTree
        dimension = 0;
    else
        dimension = dimension - 1;
    end
    % Check that the data is the same size as that contained in the tree
    assert(size(tree,2) == 2 || (size(x,2) == tree{5}),'Input data did not match tree data.');
    while size(tree,2) > 2
        pivot = tree{1,1};
        maxDimension = tree{1,4};
        dimension = 1 + mod(dimension,maxDimension);
        if x(dimension) > pivot
            tree = tree{1,2};
        else
            tree = tree{1,3};
        end
    end
    index = tree{1,2};
end