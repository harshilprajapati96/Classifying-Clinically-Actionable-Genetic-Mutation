function [svm_group, cv_ccr] = training_lib_SA_SVM(X_train,Y_train,alpha,tuning...
    ,mode,k_fold_bool)
m = length(unique(Y_train));
%% prepare index for parallel computing
pair_i = [];
pair_j = [];
for i = 1:m-1
    for j = i+1:m
        pair_i = [pair_i ; i];
        pair_j = [pair_j ; j];
    end
end
tot_m = m*(m-1)/2; % total time of itoration for OvO
end