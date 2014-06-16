function loadContour(name,ctitle,dir)
addpath('Z:\Thesis\thesisStartup.m');
addpath('Z:\Thesis\thesisCleanup.m');

if nargin < 3
    dir = 'Z:\Thesis\contourdumps\';
end

load(strcat(dir,name),'-mat');

generateCategorizedContour(data_pca,groundtruth,(1:length(point_precisions))',point_precisions,contour_file,ctitle,[.5 .75 .9]);    

end