function [xnew, ynew] = rotandproject(x,y,inc)

%Rotates X and Y 45 degrees, fits a curve, Rotates fitted
%curve back

%Best Fit line to polynomial data for ROC
xrot = 0:inc:1.41;

%Rotate 45 Deg CCW to avoid singularities on ROC curve
xprime = x*cosd(-45)-y*sind(-45);
yprime = x*sind(-45)+y*cosd(-45);

%Interpolate
yrot = interp1(xprime,yprime,xrot);

%Rotate back
xnew = xrot*cosd(45)-yrot*sind(45);
ynew = xrot*sind(45)+yrot*cosd(45);



