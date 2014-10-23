function [ ERMSvtest_gd ] = test_gd( M,alpha,sigma,attribute,relevance_label,Wgd,MU)


testSet = attribute(60001:end,:);
tsSize = length(testSet);
basisMatrixTest_gd = zeros (1,M);
basisMatrixTest_gd(:,1) = 1;
testTargetSet = relevance_label(60001:end,:);
testOutput = zeros(tsSize,1);


for k=1:tsSize
    for j=1:M-1
        basisMatrixTest_gd(1,j+1) = exp(((testSet(k,:)' - MU(j,:)')'*(testSet(k,:)' - MU(j,:)'))*(-1/(2*(sigma^2))));
    end
    testOutput(k,1)=Wgd'*basisMatrixTest_gd';
end

diff_test_gd=testOutput - testTargetSet;
EDvtest_gd = (diff_test_gd'*diff_test_gd)/2;
ERMSvtest_gd = sqrt(2*EDvtest_gd/tsSize);



end

