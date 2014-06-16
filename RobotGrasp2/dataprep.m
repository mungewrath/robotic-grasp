function [ results index data ] = dataprep( rawdata, rawresults, multiplier )
%Multiplies data by given number of multipliers and converts to a binary
%data set  - no intellegent seeding

data = zeros(length(rawresults)*multiplier,length(rawdata(1,:)));
results = data(:,1);
index = results;

for i = 1:length(rawresults)
    num = round(rawresults(i)*multiplier);
    if num>=0
        results(((i-1)*multiplier+1):((i-1)*multiplier+num)) = 1;
    end
    data(((i-1)*multiplier+1):i*multiplier,:) = repmat(rawdata(i,:),multiplier,1);
    index(((i-1)*multiplier+1):i*(multiplier)) = i;
end





end

