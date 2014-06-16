% Returns the range of values for this bin.

function extents = kdGetBinExtents(tree,binIndex)
    index = 0;
    dimension = 0;
    maxDimension = tree{1,4};
    % Store the max and min for as many dimensions
    % as are split on (the others are trivial anyway)
    extents = repmat([-Inf Inf],maxDimension,1);
    while(index ~= binIndex)
        dimension = 1 + mod(dimension,maxDimension);
        pivot = tree{1,1};
        if(isDescendant(binIndex,2*index+1))
            % Descendant of left side, meaning value must be greater than pivot
            extents(dimension,1) = max(extents(dimension,1),pivot);
            index = 2*index+1;
            tree = tree{1,2};
        else
            % Descendant of right side, meaning value must be LEQ pivot
            extents(dimension,2) = min(extents(dimension,2),pivot);
            index = 2*index+2;
            tree = tree{1,3};
        end
    end
end