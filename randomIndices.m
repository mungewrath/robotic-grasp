% RandomIndices: generates n random indices in the size of the vector.
% None of the indices will be in avoid (optional parameter).

function Output = randomIndices(Input, nIndices, avoid)

if nargin < 3
    avoid = [];
end

i = 1;
Output = zeros(nIndices,1);
% Pick test points randomly from graspit set
while (i <= nIndices)
    newTest = randi(size(Input,1));
    if ~(any(Output==newTest) || any(avoid==newTest))
        %testx(i,:) = graspitData(newTest,:);
        Output(i) = newTest;
        i = i+1;
    end
end


% EOF