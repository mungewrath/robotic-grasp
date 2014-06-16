function [ data ] = bincutoff( raw,cutoff )
%Converts data vector to binary form with everything above a given cutoff
%converted to 1 and everything below converted to 0 MUST BE ARRAY

    data = zeros(length(raw),1);

    for i = 1:length(raw)
        if raw(i) >=cutoff
            data(i) = 1;
        else
            data(i) = 0;
        end
    end

end

