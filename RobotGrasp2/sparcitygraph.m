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

% figure(1)
% plot(sparse',auc1mean','k.')
% title('PC1 Only, Mean Value for each random combination')
% xlabel('Size of Dataset')
% ylabel('Mean AUC Value')

figure(2)
hold on
set(figure(2), 'units', 'inches', 'pos', [5 0 13 12]) 
set(gca,'FontSize',40)
% errorbar(sparse',mean(auc1mean)',(std(auc1mean)/sqrt(5))','k')
h1 = fill([sparse rot90(sparse,2)]',[(mean(auc3mean)-sterr3) rot90((mean(auc3mean)+sterr3),2)]',[.25 .25 .25]);
h2 = fill([sparse rot90(sparse,2)]',[(mean(auc2mean)-sterr2) rot90((mean(auc2mean)+sterr2),2)]',[.4 .4 .4],'FaceAlpha',.5);
h3 = fill([sparse rot90(sparse,2)]',[(mean(auc1mean)-sterr1) rot90((mean(auc1mean)+sterr1),2)]',[.75 .75 .75],'FaceAlpha',.3);
h4 = plot(sparse',mean(auc1mean)','--','color',[.75 .75 .75],'linewidth',2);
h5 = plot(sparse',mean(auc2mean)','--','color',[.5 .5 .5],'linewidth',2);
h6 = plot(sparse',mean(auc3mean)','--','color',[0 0 0],'linewidth',2);
set(gca,'box','on','position',[.2,.15,.78,.8])
%legend([h4 h5 h6 h3 h2 h1],'PC1    mean','PC1:2 mean','PC1:3 mean','PC1    STerr','PC1:2 STerr','PC1:3 STerr',4)
xlabel('Size of Dataset')
plot([45 50],[.74 .635],'k','linewidth',2) 
text(60,.77,'PC1','FontSize',40)
text(40,.77,'PC1:2','FontSize',40)
text(60,.59,'PC1:3','FontSize',40)
ylabel({'Mean'; 'AUC'; 'Value'})
set(gca,'FontSize',40)
set(get(gca,'YLabel'),'Rotation',0)
set(get(gca,'YLabel'),'Position',[-2.5 .49 1.001])
axis([8 72 0 1])
set(gca,'XTick',[10:10:70])
hold off




