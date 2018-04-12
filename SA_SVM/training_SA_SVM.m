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
options.MaxIter = 1e5;
switch mode
    case 'ovo'
        if(k_fold_bool)
            cv_ccr = zeros(tot_m,length(tuning));
            for i = 1:tot_m % interate through all pairs of classes
                [X_train_1_2, Y_train_1_2] ...
                    = ovo(pair_i(i),pair_j(i),X_train,Y_train); % prepare 2 Class X and Y_train
                
                kfold = cvpartition(length(Y_train_1_2),'KFold',k);
                svm_ccr = zeros(k,1);
                for i_fold = 1:k
                    svm_group(i_fold) = svmtrain(X_train_1_2(training(kfold,i_fold),:),...
                        Y_train_1_2(training(kfold,i_fold),:),...
                        'kernel_function',@(u,v) sensing2kernal(u,v,alpha)...
                        ,'Options',options,'kernelcachelimit',Inf);
                    
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
            end
        else % not usint k-fold performing final taining
            % train on entire training data
            for i = 1:tot_m % interate through all pairs of classes % parfor
                [X_train_1_2, Y_train_1_2] ...
                    = ovo(pair_i(i),pair_j(i),X_train,Y_train);
                warning('off','all')
                svm_group(i) = svmtrain(X_train_1_2,Y_train_1_2,...
                    'kernel_function',@(u,v) sensing2kernal(u,v,alpha),...
                    'Options',options,'kernelcachelimit',Inf); %'autoscale','false',
            end
            cv_ccr = -1;
        end
        
    case 'ova'
        if(k_fold_bool)
            for i = 1:m % interate through all pairs of classes
                [X_train_1_2, Y_train_1_2] ...
                    = ova(pair_i(i),X_train,Y_train); 
                svm_ccr = zeros(k,length(tuning));
                kfold = cvpartition(length(Y_train_1_2),'KFold',k);
                for i_fold = 1:k
                    svm_group(i_fold) = svmtrain(X_train_1_2(training(kfold,i_fold),:),...
                        Y_train_1_2(training(kfold,i_fold),:),...
                        'kernel_function',@(u,v) sensing2kernal(u,v,alpha),...
                        'kernelcachelimit',Inf,'Options',options);
                    
                    svm_ccr(i_fold) = mean(svmclassify(svm_group(i_fold),X_train_1_2(test(kfold,i_fold),:))...
                        ==Y_train_1_2(test(kfold,i_fold),:));
                end
                cv_ccr(i,:) = mean(svm_ccr);
            end
        else % not usint k-fold performing final taining
            % train on entire training data
            for i = 1:tot_m % interate through all pairs of classes % parfor
                [X_train_1_2, Y_train_1_2] ...
                    = ova(pair_i(i),X_train,Y_train);
                warning('off','all')
                svm_group(i) = svmtrain(X_train_1_2,...
                    Y_train_1_2,...
                    'kernel_function',@(u,v) sensing2kernal(u,v,alpha)...
                    ,'Options',options,'kernelcachelimit',Inf); %'autoscale','false',
            end
            cv_ccr = -1;
        end
       
        
end
