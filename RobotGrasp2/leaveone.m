function [ trainx, trainy, trainind, testx, testy, testind, leavevect] = leaveone( x,y,ind,leaveindex )
%RECIEVES FULL DATASET AND SPLITS OUT Grasps in leave index
%   !!leavepercent must correspond to an index number

trainx = x;
trainy = y;
trainind = ind;

tempind = find(ind==leaveindex);
testx = x(tempind,:);
testy = y(tempind,:);
testind = ind(tempind,:);


trainx(tempind,:) = [];
trainy(tempind,:) = [];
ind(tempind,:) = [];

leavevect = tempind;