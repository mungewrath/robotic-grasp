function [ trainx trainy trainind testx testy testind leavevect] = leaverand( x,y,ind,leavepercent )
%RECIEVES FULL DATASET AND RANDOMLY PARSES INTO TRAINING DATA MATRICES AND
%TESTING MATRICES
%   !!leavepercent must be entered as a percentage!!

%Define Variables
totalset = length(x(:,1));
leave = round((leavepercent/100)*totalset);  %quantity of left items rounded to nearest integer

%SEED TEST MATRICES
trainx = x;
trainy = y;
trainind = ind;

%Randomly determine which rows to move to test matrix
leavevect = randperm(totalset);
leavevect = sort(leavevect(1:leave));

%Seed test matrices
testx = x(leavevect,:);
testy = y(leavevect,:);
testind = ind(leavevect,:);

%Remove test data from training matrices
trainx(leavevect,:) = [];
trainy(leavevect,:) = [];
trainind(leavevect,:) = [];

end

