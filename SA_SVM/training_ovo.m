%% SVM Classifier for Text Document Training
% MATLAB R2017b
% Bowen Song U04079758

%% ---------------------------------------------------%%
% Feature includes:
% Aware Sensing 2 kernel
% K-fold trianing: select the best from CV-CCR
% one vs. one for multi-class


%% ---------------------------------------------------%%
% X_train, Y_train, and vocab_len should be raw data 
% (docID,WordNum,Occurrance),(Class label for each unique doc)
% tuning should be array of cadidates




function svms = training_ovo(X_train,Y_train,Y_train_expand,vocab_len,tuning)

%% K-fold for selectring tuning parameter
k = 5; % K-fold parameter
m = unique(Y_train);

%% prepare index for parallel computing
pair_i = [];
pair_j = [];
for i = 1:m-1
    for j = i+1:m
    pair_i = [pair_i ; i];
    pair_j = [pair_j ; j];
    end
end
tot_m = m(m-1)/2; % total time of itoration for OvO
%% preallocate memory
svm_ccr = zeros(k,length(N));

%% start training
for i = 1:tot_m 
[X_train_1_2,~,Y_train_1_2,~] ...
    = ovo(pair_i(i),pair_j(j),X_train,Y_train,Y_train_expand,vocab_len);
kfold = cvpartition(n_1_2_train,'KFold',k);  
    for j = 1:length(c)
        for i_fold = 1:k
        svms = svmtrain(X_train_1_2(training(kfold,i_fold),:),Y_train_1_2(training(kfold,i_fold),:),...
             'autoscale','false');
        svm_ccr(i_fold,j) = mean(svmclassify(svms,X_train_1_2(test(kfold,i_fold),:))...
            ==Y_train_1_2(test(kfold,i_fold),:));
        end
    end
    % elect the best N parameter
    cv_ccr = mean(svm_ccr);
    [~,star_ind] = max(cv_ccr);
    star = tuning(star_ind);
    
    svms(i) = svmtrain(X_train_1_2,Y_train_1_2,...
                'autoscale','false'...
                 ,'tuning_parameter',star);
end