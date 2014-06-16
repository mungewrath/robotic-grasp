clc
clear all, close all

load aucarchivesparcity3p4   %1

auc1mean = zeros(length(auc1(1,:,1)),length(auc1(1,1,:)));
auc2mean = auc1mean;
auc3mean = auc1mean;
auc1std = auc1mean;
auc2std = auc1mean;
auc3std = auc1mean;

sparse = (100-psparceind)/100*72;

for j = 1:length(auc1(1,1,:))
    for i = 1:length(auc1(1,:,1))
        temp1 = auc1(:,i,j);
        temp2 = auc2(:,i,j);
        temp3 = auc3(:,i,j);
        
        temp1 = temp1(isfinite(temp1));
        temp2 = temp2(isfinite(temp2));        
        temp3 = temp3(isfinite(temp3));        
        
        auc1mean(i,j) = mean(temp1);
        auc2mean(i,j) = mean(temp2);
        auc3mean(i,j) = mean(temp3);
        
        auc1std(i,j) = std(temp1);
        auc2std(i,j) = std(temp2);
        auc3std(i,j) = std(temp3);
    end
end

sterr1 = std(auc1mean)/sqrt(length(auc1mean(:,1)));
sterr2 = std(auc2mean)/sqrt(length(auc2mean(:,1)));
sterr3 = std(auc3mean)/sqrt(length(auc3mean(:,1)));






