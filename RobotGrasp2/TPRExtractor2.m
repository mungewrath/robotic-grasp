avex = [];
avey = avex;
i = 1;


fpr = [.05 .10 .15]';
try
    tprtemp = interp1(ave(:,1)',ave(:,2)',fpr');
catch
    try
        tprtemp = interp1(ave(4:end-4,1)',ave(4:end-4,2)',fpr');
    catch
        tprtemp = [NaN NaN NaN];
    end
end
    tpr = tprtemp;
    disp(tpr)
    
% 
% %Rotate to find errors
% avenew(:,1) = fpr*cosd(-45)-tprtemp'*sind(-45);
% avenew(:,2) = fpr*sind(-45)+tprtemp'*cosd(-45);   
% 
% errornew(:,1) = errortemp(:,1)*cosd(-45)-errortemp(:,2)*sind(-45);
% errornew(:,2) = errortemp(:,1)*sind(-45)+errortemp(:,2)*cosd(-45);
% 
% stdnew(:,1) = stdtemp(:,1)*cosd(-45)-stdtemp(:,2)*sind(-45);
% stdnew(:,2) = stdtemp(:,1)*sind(-45)+stdtemp(:,2)*cosd(-45);
% 
% errorabs = interp1(errornew(:,1),errornew(:,2),avenew(:,1));
% errorabs = errorabs - avenew(:,2);
% error(i,:) = (((errorabs.^2)./2).^.5)';
% 
% devabs = interp1(stdnew(:,1),stdnew(:,2),avenew(:,1));
% devabs = devabs - avenew(:,2);
% dev(i,:) = (((devabs.^2)./2).^.5)';
    








