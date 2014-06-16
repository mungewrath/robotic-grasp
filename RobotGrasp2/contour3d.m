dataset = 1;



close all
clc



me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path



addpath(mydir(1:end-1))
if dataset ==1
    addpath([mydir,'images/ROBOT Grasping Images 9-11Jan13/Camera1'])
    graspno = [];
    for i = 1:9
        graspno= [graspno;([1; 2; 3; 4; 5; 6; 7; 8]+((i-1)*10))];
    end
end

averages = matave([indexout mfout]);
averages = matave([averages(:,1) results1 averages(:,2:end)]);

x = data(:,1);
y = data(:,2);

[xq,yq] = meshgrid(min(x):.05:max(x),min(y):.05:max(y));
vq = griddata(x,y,averages(:,2),xq,yq);
contourlevel = .5;

k=1;
fig = figure;
while k == 1

    subplot(2,1,1)
    hold on
    % contourlevel = .5;
    contourf(xq,yq,vq,[contourlevel,contourlevel],'-k','LineWidth',2)

    good = averages(:,2)>=.8;
    bad = averages(:,2)<.8;

    % plot3((x(bad),y(bad),averages(bad,4)'o')
    plot(x(good),y(good),'x')
    plot(x(bad),y(bad),'o')
    xlabel('PC1')
    ylabel('PC2')

    title(sprintf('GP Predictions greater than %f',contourlevel))
    hold off

    datacursormode on
    pause
    dcm_obj = datacursormode(fig);
    cinfo = getCursorInfo(dcm_obj);
    a = x==cinfo.Position(1,1);
    b = y==cinfo.Position(1,2);
    a = a.*b;
    if sum(a)>1
        error('multiple grasps at selected value')
    end

    photo = [index(a==1) data(a==1,:)];
    disp(photo)

    subplot(2,1,2)
    im1 = imread(sprintf('%.0f.jpg',graspno(photo(1,1))));
    image(im1)
    axis square
    
end
