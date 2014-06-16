function [ index, cube ] = binswitching( data,resolution)
%BINSWITCHING runs through data and allocates data to a cell of given
%increment
%
%data is nxm matrix with n trials and m metrics
%resolution controls how many 

%Remove trailing zeros
resolution = resolution(1:find(resolution,1,'last'));

data = normalizeer(data);
binindex = zeros(length(data(:,1)),length(resolution));


for i = 1:length(resolution)
    binindex(:,i) = floor(data(:,i).*resolution(i));
    %Step down max value
    binindex(binindex(:,i)==resolution(i),i) = binindex(binindex(:,i)==resolution(i),i)-1;
end

%wipe out null columns
binindex(:,binindex(1,:)==-1) = [];

index = [];
for i = 1:length(binindex(1,:))
    countr= length(num2str(max(binindex(:,i))));
    tempind = [];
    for j = 1:length(binindex(:,1))
        temp = ['000000000000' num2str(binindex(j,i))];
        tempind =[tempind; temp((length(temp)-countr):end)];
    end
    index = [index tempind];
    cube(:,i) = (binindex(:,i)./resolution(i));
    cubetemp(:,i) = cube(:,i)+1/resolution(i);
end

cube = [cube cubetemp];


index = str2num(index);