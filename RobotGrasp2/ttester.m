function [p, keep, delete] = ttester(data, groundtruth, keepnum, cutoff)
%Ttester selects keepnum quantity of quality measures for use in later
%calculations based on comparing each column to ground truth data via the
%t-test

% %For troubleshooting only
% cutoff = .8;
% ground_truth = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1]';
% data = [1 2 3 4 5 6 7 8 9 10;5 6 8 1 7 3 9 4 3 6; 10 9 8 7 6 5 4 3 2 1; 5 3 85 4 968 5 6 1 9 1]';
% keepnum = 2;
% %END TROUBLESHOOTING

%Sort out successes and failures
success = groundtruth>=cutoff;
failure = ((zeros(1,length(success))+1)'-success)>=1;
p = data(1,:)*0;

%Compare successes to failures
for i = 1:length(p);
    
    vartemp = vartest2(data(success,i),data(failure,i));
    % IF variances are the same, use 'equal' variance T-test. 
    % Otherwise, use 'unequal' T-test.
    if vartemp == 1
        [~,p(i)] = ttest2(data(success,i),data(failure,i),.05,'both','unequal');
    else
            [~,p(i)] = ttest2(data(success,i),data(failure,i),.05,'both','equal');
    end

end


%Sort out values
index = [p' (1:length(p))'];
index = sortrows(index,1);

if keepnum<1  %Case 1: only use p values above given significance value
    keep = index(:,1)<=keepnum;
    delete = index(:,1)>keepnum;
    keep = index(keep,2);
    delete = index(delete,2);  
else    %Case 2 = user specifies a given number of values to keep
    keep = index(1:keepnum,2);
    delete = index(keepnum+1:end,2);
end
end

