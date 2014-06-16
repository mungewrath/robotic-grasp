%Sparsity Graphs

close all

%from excel:

energy = [32 0.78926 0.790811412259846 0.787708587740154;45 0.78758 0.788968351540497 0.786191648459503;72 0.77846 0.781156456934572 0.775763543065428;129 0.78986 0.793179590336171 0.786540409663829;245 0.8014 0.804613098193333 0.798186901806667;342 0.7715 0.775168569203382 0.767831430796618;417 0.79504 0.798804847938496 0.791275152061504;522 0.78884 0.790833108125517 0.786846891874483];

gp = [26 0.76958 0.800154007261071 0.739005992738929;45 0.73084 0.772456050749681 0.689223949250319;130 0.76992 0.786139134378875 0.753700865621125;261 0.78458 0.799664413147352 0.769495586852648;391 0.81336 0.822511201014075 0.804208798985925;522 0.82468 0.826329581765176 0.823030418234824];


figure(2)
hold on
set(figure(2), 'units', 'inches', 'pos', [5 0 3.25 3]) 
set(gca,'FontSize',10)
h1 = fill([gp(2:end,1)' rot90(gp(2:end,1),2)'],[gp(2:end,3)' rot90(gp(2:end,4),2)'],[.5 .5 .5]);
h2 = plot(gp(2:end,1),gp(2:end,2),'k','linewidth',2);
h3 = fill([energy(2:end,1)' rot90(energy(2:end,1),2)'],[energy(2:end,3)' rot90(energy(2:end,4),2)'],[.25 .25 .25]);
h4 = plot(energy(2:end,1),energy(2:end,2),'k','linewidth',2);
% h5 = fill([10 53 53 10],[.6 .6 .8 .8],[1 1 1]);
% set(h5,'EdgeColor','None')


set(gca,'box','on','position',[.2,.15,.78,.8])
xlabel('Size of Dataset')
text(60,.68,'GP on 3 Quality Metrics','FontSize',10)
text(400,.77,'Energy','FontSize',10)
ylabel({'Mean'; 'AUC'; 'Value'})
set(gca,'FontSize',10)
set(get(gca,'YLabel'),'Rotation',0)
set(get(gca,'YLabel'),'Position',[-82 .7 1.001])
axis([0 522 .5 1])  %Could use 26 here... but half the classifications were NAN and gave artifically high predictors
set(gca,'XTick',[0:100:500])
hold off