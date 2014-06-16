function fig = plotCompositeROC(files,labels,ptitle,colors)
    fig = figure;
    set(fig, 'units', 'inches', 'pos', [1 1 8 8]);
    set(gca,'FontSize',10);
    hold on;
    
    title(ptitle);
    
    if nargin < 4
        colors={'k','b','r','g','y','c'};
    end
    
    for i=1:length(files)
        load(files{i},'-mat');
        %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
        plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2,'color',colors{i});
    end
    
    legend(labels,'Location','SouthEast');
    % Plot random guess line
    plot([0 100],[0 100],'color',[.5 .5 .5])
    set(gca,'box','on','position',[.12,.10,.78,.8])
    axis square
    axis([0 100 0 100])
    ylabel({'TPR';'(%)'})
    xlabel('FPR (%)')
    set(get(gca,'YLabel'),'Rotation',0)
    set(get(gca,'YLabel'),'Position',[-10 45. 1.001])
    text(10,8,'Random Guess','FontSize',10)
end