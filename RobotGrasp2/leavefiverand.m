function [ trainx, trainy, trainind, testx, testy, testind, leavevect] = leavefiverand( x,y,index,leave)
%RECIEVES FULL DATASET AND PULLS OUT 5 Random grasps for testing

trainx = x;
trainy = y;
testx = [];
testy = [];
trainind = index;

graspno = unique(index);
leavetemp = randperm(length(graspno));
leavetemp = sort(leavetemp(1:leave));
testind = graspno(leavetemp);

for i = 1:length(leavetemp)
    tempind = find(trainind==leavetemp(i));
    temptestx = unique(x(tempind,:),'rows');
    testx = vertcat(testx,temptestx);
    testy = vertcat(testy,y(tempind,:));
    trainx(tempind,:) = [];
    trainy(tempind,:) = [];
    trainind(tempind,:) = [];
end


leavevect = tempind;