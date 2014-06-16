% Retrieves the index numbers of all leaf nodes in the tree.
% These may not be sequential, since the tree can be imbalanced due to
% uneven distribution of data.

function indices = kdGetAllNodeIndices(tree)
    if size(tree,2) > 2
        indices = [kdGetAllNodeIndices(tree{1,2}); kdGetAllNodeIndices(tree{1,3})];
    else
        indices = tree{1,2};
    end
end