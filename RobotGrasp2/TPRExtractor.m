
%%
%//////////////////////////////////////////////////////////////////////////
%------Start LOAD FILES---------------------------------------
%//////////////////////////////////////////////////////////////////////////
clc, clear all, close all

c = clock;

OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path

addpath(mydir(1:end-1))
addpath([mydir,'cov'])
addpath([mydir,'doc'])
addpath([mydir,'inf'])
addpath([mydir,'lik'])
addpath([mydir,'mean'])
addpath([mydir,'util'])
addpath([mydir,'results'])
addpath([mydir,'LDA'])
addpath([mydir,'roccurves'])

clear me mydir
%------------End Load Files----------------------------------------------

%%
%Import ROC Curves

rocload = dir('roccurves');
rocload = struct2cell(rocload);
rocload(2:end,:) = [];
rocload = rocload';

for i = length(rocload):-1:1
    if strcmp(rocload(i),'.') || strcmp(rocload(i),'..')
        rocload(i) = [];        
    end
end


avex = [];
avey = avex;
fpr = [.05 .10 .15]';

for i = 1:length(rocload)
    ind = rocload{i};
    temp = load (ind);
    avetemp = temp.ave;
    errortemp = temp.uppererr;
    stdtemp = temp.upperstd;

    
    
    try
        tprtemp = interp1(avetemp(:,1)',avetemp(:,2)',fpr');
    catch
        try
            tprtemp = interp1(avetemp(4:end-4,1)',avetemp(4:end-4,2)',fpr');
        catch
            tprtemp(i,:) = [NaN NaN NaN];
        end
    end
        tpr(i,:) = tprtemp;
        
    %Rotate to find errors
    avenew(:,1) = fpr*cosd(-45)-tprtemp'*sind(-45);
    avenew(:,2) = fpr*sind(-45)+tprtemp'*cosd(-45);   

    errornew(:,1) = errortemp(:,1)*cosd(-45)-errortemp(:,2)*sind(-45);
    errornew(:,2) = errortemp(:,1)*sind(-45)+errortemp(:,2)*cosd(-45);
    
    stdnew(:,1) = stdtemp(:,1)*cosd(-45)-stdtemp(:,2)*sind(-45);
    stdnew(:,2) = stdtemp(:,1)*sind(-45)+stdtemp(:,2)*cosd(-45);
    
    errorabs = interp1(errornew(:,1),errornew(:,2),avenew(:,1));
    errorabs = errorabs - avenew(:,2);
    error(i,:) = (((errorabs.^2)./2).^.5)';
    
    devabs = interp1(stdnew(:,1),stdnew(:,2),avenew(:,1));
    devabs = devabs - avenew(:,2);
    dev(i,:) = (((devabs.^2)./2).^.5)';
    
end







