% Returns whether the given child index is descended from the parent,
% following the rules of children of index i being 2i+1 and 2i+2.

function desc = isDescendant(childIndex, parentIndex)
    while(childIndex > parentIndex)
        childIndex = floor((childIndex-1)/2);
    end
    desc = (childIndex == parentIndex);
end