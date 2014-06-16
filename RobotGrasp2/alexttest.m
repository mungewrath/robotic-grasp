clear
clc

testnum = 475;  % The number of values from the file to train/test with
%data = xlsread('mydata_2.csv');
data = xlsread('alexdata.xls');

% Save GraspIt epsilon and autograsp quality to separate array
num = 1;
for k=2:526
   if data(k,11) > -100
       D(num, 1) = data(k,11);
       D(num, 2) = data(k,13);
       D(num, 3) = data(k,35);
       D(num, 4) = data(k,34);
       num = num + 1;
   else
       data(k,11) = NaN;
       data(k,13) = NaN;
   end
end

% Spherize data
data2 = data(2:526,1:13);
data(2:526,1:13) = data2;
newdata = data(2:526,1:10);
newdata(:,11) = data(2:526,12);
D(:,1:2) = (D(:,1:2));

% Split data into successful and unsuccessful matrices
s = 1;
f = 1;
for l = (2:526)
    if data(l,34) >= 8
        SucData(s,:) = data(l,1:13);
        s = s + 1;
    else
        FailData(f,:) = data(l,1:13);
        f = f + 1;
    end
end
s = 1;
f = 1;
for l = (1:length(D(:,1)) )
    if D(l,4) >= 8
        SucD(s,:) = D(l,1:2);
        s = s + 1;
    else
        FailD(f,:) = D(l,1:2);
        f = f + 1;
    end
end


for vars = (1:13) % DO F-tests to determine if variances are the same
    
    [h(1,vars)] = vartest2(SucData(:,vars),FailData(:,vars));
    
    % IF variances are the same, use 'equal' variance T-test. 
    % Otherwise, use 'unequal' T-test.
    if h(1,vars) == 1
        [Ttests(1,vars),Ttests(2,vars),ci,stats]  = ttest2(SucData(:,vars),FailData(:,vars),0.05,'both','unequal');
    else
        [Ttests(1,vars),Ttests(2,vars),ci,stats]  = ttest2(SucData(:,vars),FailData(:,vars),0.050,'both','equal');
    end
   
end

% Do F-test and T-test for GraspIt epsilon and autograsp quality
[h(1,11)] = vartest2(SucD(:,1),FailD(:,1));
    if h(1,11) == 1
        [Ttests(1,11),Ttests(2,11),ci,stats]  = ttest2(SucD(:,1),FailD(:,1),0.05,'both','unequal');
    else
        [Ttests(1,11),Ttests(2,11),ci,stats]  = ttest2(SucD(:,1),FailD(:,1),0.050,'both','equal');
    end
    
[h(1,13)] = vartest2(SucD(:,2),FailD(:,2));
    if h(1,vars) == 1
        [Ttests(1,vars),Ttests(2,vars),ci,stats]  = ttest2(SucD(:,2),FailD(:,2),0.05,'both','unequal');
    else
        [Ttests(1,vars),Ttests(2,vars),ci,stats]  = ttest2(SucD(:,2),FailD(:,2),0.050,'both','equal');
    end

