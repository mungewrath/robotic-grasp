function generateROC(filename,uppererrs,lowererrs,aves,datasetNames,titleText)
    assert(length(uppererrs) == length(lowererrs) && length(lowererrs) == length(aves) && length(aves) == length(datasetNames),'Number of upper/lower err/average vectors must match!');
    fig = figure;
    set(fig, 'units', 'inches', 'pos', [8 5 3.25 3])
    set(gca,'FontSize',10)
    hold on
    
    % Plot random guess
    plot([0 100],[0 100],'color',[.5 .5 .5])
    
    % 
    plotColors = [0 0 0;
                  1 0 0;
                  0 0 1];
    
    lines = zeros(length(uppererrs),1);
    for i = 1:length(uppererrs)
        uppererr = uppererrs{i};
        lowererr = lowererrs{i};
        ave = aves{i};
        fillColor = min([plotColors(i,:)+.25; 1 1 1]);
        h = fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),fillColor);
        set(h,'facealpha',.4);
        set(h,'edgealpha',.5);
        %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
        lines(i) = plot(ave(:,1).*100,ave(:,2).*100,'color',plotColors(i,:),'linewidth',2);
        %plot(rocResults(:,1).*100,rocResults(:,2).*100,'k','linewidth',2)
    end
    
    legend(lines,datasetNames,'Location','SouthEast');
    set(gca,'box','on','position',[.17,.15,.78,.8])
    %text(10,84,'GP with PC1','FontSize',10)
    %text(10,8,'Random Guess','FontSize',10)
    if nargin > 5
        title(titleText);
    end
    axis square
    axis([0 100 0 100])
    ylabel({'TPR';'(%)'})
    xlabel('FPR (%)')
    set(get(gca,'YLabel'),'Rotation',0)
    set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
    
    print(fig,'-dpng',strcat(filename,'.png'));
end