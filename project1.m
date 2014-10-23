load('project1_data.mat');
%%%%%%%  CFS SOLUTION %%%%%%%%%
[W_cfs, mu_cfs, s_cfs, ERMSt_cfs,ERMSv_cfs,M_cfs,lambda_cfs] = train_cfs(attribute,relevance_label);
[rms_cfs] = test_cfs(M_cfs,lambda_cfs,s_cfs,attribute,relevance_label,W_cfs,mu_cfs);


%%%%%%%  GRADIENT DESCENT %%%%%%%%
[ W_gd, mu_gd, s_gd, ERMSt_gd,ERMSv_gd,M_gd,lambda_gd ] = train_gd(attribute,relevance_label);
[ rms_gd ] = test_gd( M_gd,lambda_gd,s_gd,attribute,relevance_label,W_gd,mu_gd);
    

ubitname = 'ankitkai';
Studentnumber = 50134053;
fprintf('My ubit name is %s\n',ubitname);
fprintf('My student number is %d \n',Studentnumber);
fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters lambda_gd is %4.2f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', rms_gd);