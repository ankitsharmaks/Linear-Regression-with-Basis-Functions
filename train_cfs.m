function [ W_cfs, mu_cfs, s_cfs, ERMSt_cfs,ERMSv_cfs,M,L ] = train_cfs(attribute,relevance_label )

M = 90;
L = 0.00001;
trainingSet = attribute(1:50000,:);
trainingTargetSet = relevance_label(1:50000,:);
tsSize = size(trainingSet,1);
D = size(trainingSet,2); %No. of features
Mstart = M;
Mend = M;
Mint = 1;
Esize = int16((Mend - Mstart + 1)/Mint);
EDt = zeros(Esize,9);
ERMSt = zeros(Esize,9);
EDv_t = zeros(Esize,9);
ERMSv = zeros(Esize,9);

%for sigma = 0.7:0.1:0.7
    sigma = 0.9;
m=0;
for M=Mstart:Mint:Mend
    m = m+1;
    basisMatrix = zeros (tsSize,M);
    basisMatrix(:,1) = 1;
    weights = zeros(M,1);
    MU = zeros (M-1,D);
    for i=1:M-2
        MU(i,:) = mean(trainingSet((int32(tsSize/(M-1))*(i-1))+1:(int32(tsSize/(M-1))*(i)),:));
    end
    MU(M-1,:) = mean(trainingSet((int32(tsSize/(M-1))*(M-2))+1:tsSize,:));

    for i=1:tsSize
        for j=1:M-1
               basisMatrix(i,j+1) = exp(((trainingSet(i,:)' - MU(j,:)')'*(trainingSet(i,:)' - MU(j,:)'))*(-1/(2*(sigma^2))));
        end
    end
    
    weights = ((L*eye(M)) + (basisMatrix'*basisMatrix))\basisMatrix'*trainingTargetSet;   
    
    
    %%%%----Calculate ERMS for training Set-----%%%%%%%%%%%%%%%%%%%%
    
    
    trainingOutput=weights'*basisMatrix';
    trainingOutput = trainingOutput';
    Me = M - Mstart +1;
    diff_t=trainingOutput - trainingTargetSet;
    EDt(m,sigma*10) = (diff_t'*diff_t)/2;
    EDt(m,sigma*10) = EDt(m,sigma*10) + (1/2)*L*(weights'*weights);
    ERMSt(m,sigma*10) = sqrt((2*EDt(m,sigma*10))/tsSize);
   
    
    
    %%%%----Calculate ERMS for validation Set-----%%%%%%%%%%%%%%%%%%%%%
    validationSet = attribute(50001:60000,:);
    vsSize = length(validationSet);
    basisMatrixValidation = zeros (1,M);
    basisMatrixValidation(:,1) = 1;
    validationTargetSet = relevance_label(50001:60000,:);
    validationOutput = zeros(vsSize,1);
    for i=1:vsSize
        for j=1:M-1
            basisMatrixValidation(1,j+1) = exp(((validationSet(i,:)' - MU(j,:)')'*(validationSet(i,:)' - MU(j,:)'))*(-1/(2*(sigma^2))));
        end
        validationOutput(i,1)=weights'*basisMatrixValidation';
    end
    
    diff_v=validationOutput - validationTargetSet;
    EDv_t(m,sigma*10) = (diff_v'*diff_v)/2;
    EDv_t(m,sigma*10) = EDv_t(m,sigma*10) + (1/2)*L*(weights'*weights);
    ERMSv(m,sigma*10) = sqrt((2*EDv_t(m,sigma*10))/vsSize);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W_cfs = weights;
    mu_cfs = MU;
    s_cfs = sigma;
    ERMSt_cfs = ERMSt(m,sigma*10);
    ERMSv_cfs = ERMSv(m,sigma*10);
end
