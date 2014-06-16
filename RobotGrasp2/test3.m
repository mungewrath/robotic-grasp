testx = (1:1:20)';
testy = [testx*5 testx*2];
leave = 20;

[ trainx trainy testx testy ]  = leaverand(testx,testy,leave);