function [ ERMSvtest ] = test_cfs( M,L,sigma,attribute,relevance_label,W_cfs,MU)

weights_test = W_cfs;   

testSet = attribute(60001:end,:);
tsSize = length(testSet);
basisMatrixTest = zeros (1,M);
basisMatrixTest(:,1) = 1;
testTargetSet = relevance_label(60001:end,:);
testOutput = zeros(tsSize,1);

for k=1:tsSize
    for j=1:M-1
        basisMatrixTest(1,j+1) = exp(((testSet(k,:)' - MU(j,:)')'*(testSet(k,:)' - MU(j,:)'))*(-1/(2*(sigma^2))));
    end
    testOutput(k,1)=weights_test'*basisMatrixTest';
end

diff_test=testOutput - testTargetSet;
EDvtest = (diff_test'*diff_test)/2;
EDvtest = EDvtest + (1/2)*L*(weights_test'*weights_test);
ERMSvtest = sqrt(2*EDvtest/tsSize);


end

