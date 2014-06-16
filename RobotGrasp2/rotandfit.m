function [fitx,fity,kstat,AUC] = rotandfit(x,y)

%Rotates X and Y 45 degrees, fits a curve, calculates AUC, Rotates fitted
%curve back and finds max index point (kstat)

%Best Fit line to polynomial data for ROC
xout = 0:.01:1.41;

xprime = x*cosd(-45)-y*sind(-45);
yprime = x*sind(-45)+y*cosd(-45);
pol = polyfit(xprime,yprime,5);
fity = polyval(pol,xout);
[~,kstat] = max(fity);
fitx = xout*cosd(45)-fity*sind(45);
fity = xout*sind(45)+fity*cosd(45);

%AUC
AUC = 0;
for i = 1:length(fitx)-1
    AUC = AUC + (fitx(i+1)-fitx(i))*mean([fity(i) fity(i+1)]);
end

