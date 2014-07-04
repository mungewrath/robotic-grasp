% Partitions a matrix with size datalength into several splits. The
% returned values are the indices corresponding to each split.
% splits must be a vector of either integers or floating-point values.
% floats are interpreted as fractions of the data to use. Integers are
% interpreted as an absolute number of points to include in the split.

function varargout = partitionData(splits,datalength)
    for i = 1:length(splits)
        if(splits(i) < 1)
            splits(i) = floor(datalength * splits(i));
        end
    end
    assert(sum(splits) <= datalength,'Sum of split proportions adds up to more than the data length!');
    exclude = [];
    for i = 1:length(splits)
        varargout{i} = randomIndices((1:datalength)',splits(i),exclude);
        exclude = [exclude; varargout{i}];
    end
    varargout{end+1} = setxor(exclude,(1:datalength)');
end