%% SVM Classifier for Text Document Training
% MATLAB R2017b
% Bowen Song U04079758

%% ---------------------------------------------------%%
% Feature includes:
% Aware Sensing 2 kernel
% K-fold trianing: select the best from CV-CCR
% one vs. one for multi-class


%% ---------------------------------------------------%%
% Input: X_train, Y_train,alpha,tuning and mode
% X_train: (uniquedoc) X (N number of words randomly picked) of the word
% occurernace
% Y_train: (uniquedoc) X 1 of class label
% alpha: a constant term of Kernal in log
% tuning: N 1X1
% mode: 'ovo' and 'ova'


%% ---------------------------------------------------%%
% Output: svm_group an array of structs of svm classifyer



function [svm_group, cv_ccr] = training_SA_SVM(X_train,Y_train,alpha,tuning...
    ,mode,k_fold_bool)

%% K-fold for selectring tuning parameter
k = 5; % K-fold parameter
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
%% preallocate memory


%% start training
warning('off','all')
warning

switch mode
    case 'ovo'
        
        cv_ccr = zeros(tot_m,length(tuning));
        for i = 1:tot_m % interate through all pairs of classes
            [X_train_1_2, Y_train_1_2] ...
                = ovo(pair_i(i),pair_j(i),X_train,Y_train); % prepare 2 Class X and Y_train
            if(k_fold_bool)
                kfold = cvpartition(length(Y_train_1_2),'KFold',k);
                svm_ccr = zeros(k,1);
                for i_fold = 1:k
                    svm_group(i_fold) = svmtrain(X_train_1_2(training(kfold,i_fold),:),...
                        Y_train_1_2(training(kfold,i_fold),:),...
                        'kernel_function',@(u,v) sensing2kernal(u,v,alpha),'autoscale','false');
                    
                    %                 temp = svmtrain(X_train_1_2(training(kfold,i_fold),:),...
                    %                     Y_train_1_2(training(kfold,i_fold),:),...
                    %                     'autoscale','false');
                    % svm_group = fitcsvm(X_train_1_2(training(kfold,i_fold),:),...
                    %                     Y_train_1_2(training(kfold,i_fold),:),...
                    %                     'KernelFunction','sensing2kernal');
                    
                    svm_ccr(i_fold) = mean(svmclassify(svm_group(i_fold),X_train_1_2(test(kfold,i_fold),:))...
                        ==Y_train_1_2(test(kfold,i_fold),:));
                end
                cv_ccr(i,:) = mean(svm_ccr);
            else % not usint k-fold
                svm_group = svmtrain(X_train_1_2,...
                    Y_train_1_2,...
                    'kernel_function',@(u,v) sensing2kernal(u,v,alpha),'autoscale','false');
                cv_ccr(i,:) = mean(svmclassify(svm_group,X_train_1_2)...
                    ==Y_train_1_2);
            end
        end

    case 'ova'
        svm_group = zeros(m,1);
        parfor i = 1:m
            [X_train_1_2, Y_train_1_2] ...
                = ova(pair_i(i),X_train,Y_train);
            kfold = cvpartition(length(Y_train_1_2),'KFold',k);
            svm_ccr = zeros(k,length(tuning));
            for j = 1:length(tuning)
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
            svm_group(i) = svmtrain(X_train_1_2,Y_train_1_2,...
                'autoscale','false');
            
        end
        
end
