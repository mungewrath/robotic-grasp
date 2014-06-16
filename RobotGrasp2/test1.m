clear
clc

y = (1:20)';
y(2,:) = 1;

index = y;
x = [y y y y];

[ trainx, trainy, trainind, testx, testy, testind, leavevect] = leaverand2(x,y,index,20);

disp([trainx trainy trainind])
disp([testx testind])