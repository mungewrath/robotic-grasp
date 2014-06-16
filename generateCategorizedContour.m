% Derived from Ryan Carpenter's 3d contour plotter.
%
% data: 2D matrix of input values.
% indexout: column vector of test indices.
% mfout: column vector of GP predictions
% contour_file: file to export the contour to.
% g_title: title of the graph.
% contour_values: average values to draw contours at. Multiple
% monotonically increasing values can be passed to color different values.
% categorize: pass true to display positive and negative predictions
% differently (blue = positive, red = negative).
%
% Draws test points in two separate colors to indicate positive/negative
% prediction.

function generateCategorizedContour(data, groundtruth, indexout, mfout, contour_file, g_title, contour_values, categorize)

mfout(isnan(mfout)) = 0;

averages_pre = matave([indexout mfout]);
%averages = matave([averages(:,1) results1 averages(:,2:end)]);

% Put average data values into zero-default array
averages = [(1:size(data,1))' zeros(size(data,1),1)];
averages(averages_pre(:,1),:) = averages_pre;

x = data(:,1);
y = data(:,2);

[xq,yq] = meshgrid(min(x):.05:max(x),min(y):.05:max(y));
vq = griddata(x,y,averages(:,2),xq,yq);
if nargin < 7
    contour_values = .5;
end

fig = figure;

%subplot(2,1,1)
cutoffs = repmat(contour_values,[2 1]);
cutoffs = cutoffs(:)';

hold on
contourf(xq,yq,vq,cutoffs,'-k','LineWidth',2)
if length(contour_values) > 1
    colorbar;
end
colormap('summer');
%imagesc

TP = (groundtruth >= .8 & averages(:,2) >= contour_values(1));
FP = (groundtruth < .8 & averages(:,2) >= contour_values(1));
TN = (groundtruth >= .8 & ~(averages(:,2) >= contour_values(1)));
FN = (groundtruth < .8 & ~(averages(:,2) >= contour_values(1)));

% if(nargin >= 8 && categorize)
%     plot(x(TP),y(TP),'xb')
%     plot(x(FP),y(FP),'ob')
%     plot(x(TN),y(TN),'xr')
%     plot(x(FN),y(FN),'or')
% else
%     plot(x(TP | FN),y(TP | FN),'x')
%     plot(x(FP | TN),y(FP | TN),'o')
% end
xlabel('PC1')
ylabel('PC2')

title(g_title)
hold off

if nargin >= 5
    print(fig,'-dpng',contour_file);
    saveas(fig,contour_file,'fig');
end

end

