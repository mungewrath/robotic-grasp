function [fn,fp,tn,tp,sumary] = truefalse(data,xcrit,ycrit)

%----------Document History----------------
%Version  Modified Date    By                      Notes    
%   A     3FEB13           Ryan Carpenter          Original
%
%

% %Debugging Data
% clear
% clc
% data = [.1 .1;.1 .9;.9 .1;.95 .15;.9 .9];
% xcrit = .8;
% ycrit = .8;


fn = [0 0];
fp = fn;
tn = fn;
tp = fn;

count = length(data(:,1));
step = [1 1 1 1];
for i = 1:count
    %false positive
    if data(i,1)< xcrit && data(i,2)>=ycrit
        fp(step(1),:) = [data(i,1) data(i,2)];
        step(1) = step(1)+1;
    %false negative
    elseif data(i,1)>= xcrit && data(i,2)<ycrit
        fn(step(2),:) = [data(i,1) data(i,2)];
        step(2) = step(2)+1;   
    %true positive
    elseif data(i,1)>= xcrit && data(i,2)>=ycrit
        tp(step(3),:) = [data(i,1) data(i,2)];
        step(3) = step(3)+1;        
    %true negative
    else
        tn(step(4),:) = [data(i,1) data(i,2)];
        step(4) = step(4)+1;
    end
end
sumary =[length(fn(:,1)) length(fp(:,1)) length(tn(:,1)) length(tp(:,1))];
sumary(2,:) = sumary./count;
%Determine TPR and FPR

