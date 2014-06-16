function [ave,upper,lower,kstat,kstatsterr] = rotandextrap(x,y,loopn)

%Rotates X and Y 45 degrees, fits a curve, calculates AUC, Rotates fitted
%curve back and finds max index point (kstat)

%Best Fit line to polynomial data for ROC
xnew = 0:.01:1.41;
ynew = zeros(length(xnew),length(y(1,:)));


for i = 1:length(y(1,:))
    xprime = x(:,i)*cosd(-45)-y(:,i)*sind(-45);
    yprime = x(:,i)*sind(-45)+y(:,i)*cosd(-45);
    ynew(:,i) = interp1(xprime,yprime,xnew);
end

yave = mean(ynew');
yave = yave';
ystd = std(ynew');
ystd = ystd';
ysterr = ystd./(loopn^.5);
xnew = xnew';

yupper = yave+ysterr;
ylower = yave-ysterr;

[~,kstat] = max(yave);
kstatsterr = ysterr(kstat);

%For visualization
% for i = 1:300
%     plot(xnew,ynew(:,i))
%     hold on
% end

%Note each matrix contains x's and y's
ave = [xnew*cosd(45)-yave*sind(45) xnew*sind(45)+yave*cosd(45)];
upper = [xnew*cosd(45)-yupper*sind(45) xnew*sind(45)+yupper*cosd(45)];
lower = [xnew*cosd(45)-ylower*sind(45) xnew*sind(45)+ylower*cosd(45)];   



