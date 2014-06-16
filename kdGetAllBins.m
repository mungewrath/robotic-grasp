% Returns a Nx2 cell array of all bins in the kd-tree, followed by their
% depths.

function bins = kdGetAllBins(tree,depth)
    if nargin < 2
        depth = 0;
    end
    if size(tree,2) > 2
        bins = [kdGetAllBins(tree{1,2},depth+1); kdGetAllBins(tree{1,3},depth+1)];
    else
        bins = {tree{1,1} depth};
    end
end