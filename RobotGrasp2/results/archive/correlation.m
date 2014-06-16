
clear
clc

% resultsmtur = xlsread('mechturkrawdata.xls');
% resultsmtur = resultsmtur(:,[1 9]);
% resultsmtur = matave(resultsmtur);
% resultsmtur(:,1) = [];
% 
resultsphy = xlsread('IROS_RESULTS.xls');

data = xlsread('IROS_DATA.xls');
% 
% corrmtur = corr(resultsmtur,data);
% 
% corrphy = corr(resultsphy,data);
% disp('crowd    phys')
% disp([corrmtur' corrphy'])
% 
%cutoff = [.2 .4 .6 .8 1];
% 
cutoff = .8;
tphy = zeros(length(data(1,:)),length(cutoff));
% tmtur = tphy;

% data = data;
% resultsphy = groundtruth;

data = sphereize(data);

%Seed pass/fail matrices
for i = 1:length(cutoff)
    temp1 = 1;
    temp2 = 1;
    physucc = [];
    mtursucc = [];

    for j = 1:length(resultsphy)
        if resultsphy(j)>=cutoff(i)
            physucc(temp1) = j;
            temp1 = temp1+1;
        end
%         if resultsmtur(j)>=(cutoff(i))
%             mtursucc(temp2) = j;
%             temp2 = temp2+1;
%         end
    end
    
    phyfail = 1:length(resultsphy);
    phyfail(physucc) = [];
%     
%     mturfail = 1:length(resultsmtur);
%     mturfail(mtursucc) = [];
    
    for k = 1:length(data(1,:))
        [temp3 tphy(k,i)] = ttest2(data(phyfail,k),data(physucc,k));
%         [temp4 tmtur(k,i)] = ttest2(data(mturfail,k),data(mtursucc,k));
    end
    
end

disp('p values from ttest physical data')
disp(tphy)
% disp('p values from ttest crowd data')
% disp(tmtur)
