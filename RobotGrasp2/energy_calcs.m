clear
clc

data = xlsread('IROS_DATA2p2.xls');
results = xlsread('IROS_RESULTSp2.xls');
%data = sphereize(data);

%results = bincutoff(results,.8);

data = data(:,6);
data = normalizeer(data);

[~,~,~,~,sumtemp] = truefalse([data results],.8,.8);

fpr = sumtemp(2,2)/(sumtemp(2,2)+sumtemp(2,3))
tpr = sumtemp(2,4)/(sumtemp(2,1)+sumtemp(2,4))

