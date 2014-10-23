Following are the functions used in the program:


%%%%%%%  CFS SOLUTION %%%%%%%%%
	[W_cfs, mu_cfs, s_cfs, ERMSt_cfs,ERMSv_cfs,M_cfs,lambda_cfs] = train_cfs(attribute,relevance_label);

			Arguments:-
				attributes: Whole Dataset (69623x46)
				relevance_label: Target Values of Dataset (69623x1)

			Return Types:-
				W_cfs : Weight Matrix 
				mu_cfs: MU matrix
				s_cfs: Selected value of sigma
				ERMSt_cfs: Erms over training set
				ERMSv_cfs: Erms over validation set
				M_cfs: Selected Modular Complexity for CFS Solution
				lambda_cfs: Selected Value of Lambda for CFS Solution


	[rms_cfs] = test_cfs(M_cfs,lambda_cfs,s_cfs,attribute,relevance_label,W_cfs,mu_cfs);
			
			Arguments:-
				Arguments are the return types of train_cfs function.
			
			Return Type:-
				rms_cfs: Erms over test set




%%%%%%%  GRADIENT DESCENT %%%%%%%%
	[ W_gd, mu_gd, s_gd, ERMSt_gd,ERMSv_gd,M_gd,lambda_gd ] = train_gd(attribute,relevance_label);
			Arguments:-
				attributes: Whole Dataset (69623x46)
				relevance_label: Target Values of Dataset (69623x1)

			Return Types:-
				W_gd : Weight Matrix 
				mu_gd: MU matrix
				s_gd: Selected value of sigma
				ERMSt_gd: Erms over training set
				ERMSv_gd: Erms over validation set
				M_gd: Selected Modular Complexity for Gradient Descent Solution
				lambda_gd: Selected Value of Lambda for Gradient Descent Solution
				
				
	[ rms_gd ] = test_gd( M_gd,lambda_gd,s_gd,attribute,relevance_label,W_gd,mu_gd);
				
				Arguments:-
					Arguments are the return types of train_gd function.
			
				Return Type:-
					rms_gd: Erms over test set
