function [data,stdev,sterr] = matave(raw)

%----------Document History----------------
%Version  Modified Date    By                      Notes    
%   A     4FEB13           Ryan Carpenter          Original
%
%


%returns an average of corresponding columns 2:length for each unique
%column 1 value
%

%raw = [2 2 6;1 3 5;1 3 6];

%Define unique grasp no's in first column
grasp = unique(raw(:,1));

%save length and width of matrix
wid = length(raw(1,:));
len = length(raw(:,1));

%Generate empty data matrix
data = zeros(length(grasp),wid);
stdev = data;
sterr = data;

for i = 1: length(grasp)
    temp = zeros(1,wid);
    step = 1;
    for j = 1:len
        if raw(j,1) == grasp(i)
            for k = 1:wid
                temp(step,k) = raw(j,k);
            end
            step = step + 1;
        end
    end
    if length(temp(:,1))==1
        data(i,:) = temp;
        stdev(i,:) = -1;
        sterr(i,:) = -1;
    else
        data(i,:) = mean(temp);
        stdev(i,:) = std(temp);
        sterr(i,:) = std(temp)./sqrt(length(temp(:,1)));
    end
end