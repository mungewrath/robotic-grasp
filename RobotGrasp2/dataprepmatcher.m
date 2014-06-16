function [data] = dataprepmatcher( rawdata, rawdataindex, rawresultsindex)
%Copies rows of rawdata into data matrix

data = zeros(length(rawresultsindex),length(rawdata(1,:)));

for i = 1:length(rawresultsindex)
    for j = 1:length(rawdataindex)
        if rawresultsindex(i) ==rawdataindex(j)
            data(i,:) = rawdata(j,:);
            break
        end
    end
end

end

