function [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(x,y,loopn)

%Rotates X and Y 45 degrees, fits a curve, calculates AUC, Rotates fitted
%curve back and finds max index point (kstat)

%Best Fit line to polynomial data for ROC


for i = 1:length(y(1,:))
    xnew(:,i) = x(:,i)*cosd(-45)-y(:,i)*sind(-45);
    ynew(:,i) = x(:,i)*sind(-45)+y(:,i)*cosd(-45);
end

xnew = mean(xnew');
xnew = xnew';

yave = mean(ynew');
yave = yave';
ystd = std(ynew');
ystd = ystd';
ysterr = ystd./(loopn^.5);

yuppererr = yave+ysterr;
ylowererr = yave-ysterr;

yupperstd = yave + ystd;
ylowerstd = yave - ystd;

[~,kstat] = max(yave);
kstatstd = ((ystd(kstat)^2)/2)^.5;    %Find components for FPR TPR
kstatsterr = ((ysterr(kstat)^2)/2)^.5;  %Find components for FPR TPR

%For visualization
% for i = 1:300
%     plot(xnew,ynew(:,i))
%     hold on
% end

%Note each matrix contains x's and y's
ave = [xnew*cosd(45)-yave*sind(45) xnew*sind(45)+yave*cosd(45)];
uppererr = [xnew*cosd(45)-yuppererr*sind(45) xnew*sind(45)+yuppererr*cosd(45)];
lowererr = [xnew*cosd(45)-ylowererr*sind(45) xnew*sind(45)+ylowererr*cosd(45)];   
upperstd = [xnew*cosd(45)-yupperstd*sind(45) xnew*sind(45)+yupperstd*cosd(45)];
lowerstd = [xnew*cosd(45)-ylowerstd*sind(45) xnew*sind(45)+ylowerstd*cosd(45)];   



