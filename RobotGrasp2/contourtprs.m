mfoutpr = normalizeer(mfout);
binshakeresults = bincutoff(results1,.8);
res = binshakeresults(indexout);

l = 1;

try
    [perfx,perfy,perft,auc(l)] = perfcurve(res',mfoutpr',1);
    [rocx(:,l),rocy(:,l)] = rotandproject(perfx,perfy,.01);
catch err
    auc(l) = NaN;
end

[~,thresh] = cutoffsearch([.05 .1 .15],perfx,perft,mfoutpr);

% averages = matave([indexout mfout abovecrit]);
averages =[indexout mfout];
averages = [averages(:,1) results1(indexout) averages(:,2:end)];

x = data(indexout,1);
y = data(indexout,2);

[xq,yq] = meshgrid(min(x):.05:max(x),min(y):.05:max(y));
vq = griddata(x,y,averages(:,2),xq,yq);
contourlevel = .8;


% surf(xq,yq,vq)

    % contour(xq,yq,vq,20)

close all
colormap('summer')
figure(1)
hold on
for i = length(thresh):-1:1
    
% contourlevel = .5;
    contourf(xq,yq,vq,[thresh(i),thresh(i)],'-k','LineWidth',.5)

end

good = averages(:,2)>=.8;
bad = averages(:,2)<.8;

% plot3((x(bad),y(bad),averages(bad,4)'o')
plot(x(good),y(good),'kx')
plot(x(bad),y(bad),'ko')
xlabel('PC1')
ylabel('PC2')

title('GP Predictions')
hold off

axis([-2 2 -3 2])
