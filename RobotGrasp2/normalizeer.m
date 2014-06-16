function [mat] = normalizeer (raw)

%normalizes all columns of a given matrix and returns the result

mat = raw*0;

for i = 1:length(raw(1,:))
    mat(:,i) = raw(:,i)-min(raw(:,i));
    mat(:,i) = mat(:,i)./max(mat(:,i));
end