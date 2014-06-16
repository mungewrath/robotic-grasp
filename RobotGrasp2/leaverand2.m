function [ trainx trainy trainind testx testy testind leavevect] = leaverand2( x,y,index,leavepercent )
%RECIEVES FULL DATASET AND RANDOMLY PARSES INTO TRAINING DATA MATRICES AND
%TESTING MATRICES
%   !!leavepercent must be entered as a percentage!!

%Define Variables
graspno = unique(index);
totalset = length(graspno);
leave = round((leavepercent/100)*totalset);  %quantity of left items rounded to nearest integer

%SEED TEST MATRICES
trainx = x;
trainy = y;
trainind = index;
testx = [];
testy = NaN;
   
%Randomly determine which rows to move to test matrix
leavevect = randperm(length(graspno));
leavevect = sort(leavevect(1:leave));
testind = graspno((leavevect));%this used to have a unique function in it
%now you know what grasp numbers you are looking for

%Split up
for i = 1:length(testind)
    tempind = find(trainind==testind(i));
    if length(tempind)>1
        tempx = mean(trainx(tempind,:));
    else
        tempx = trainx(tempind,:);
    end
    testx = vertcat(testx,tempx);
    trainx(tempind,:) = [];
    trainy(tempind,:) = [];
    trainind(tempind,:)= [];
end

