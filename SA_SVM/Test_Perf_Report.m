%% %% SVM Classifier for Text Document
% MATLAB R2017b
% Bowen Song U04079758

%% News 20 group
% preprocessing
tic
Preprocessing_new20;
disp("Preprocessing is done:")
toc
% use data with filtered stoped words
%% Training woth OVO
tic
tuning = [150]; % tuning prarmeter for US new is suggested to be 150 at best
for i_tuning = 1:length(tuning)
    disp("X_train preperation time:")
    tic
    [X_train_processed,alpha] = RRN_preprocessing(X_train_woSTOP,tuning(i_tuning),length(vocab));
    toc
    disp("Training time:")
    tic
    [svm_group_ovo, ccrs] = training_SA_SVM(X_train_processed,Y_train,...
        alpha,tuning(i_tuning),'ovo',true);
    toc
end
disp("Training for OVO is done:")
toc
% 10 parameters
% without parfor
% with most inner parfor
% with outter parfor
% %% Training woth OVA
% tic
% svm_group_ova = training_SA_SVM(X_train_woSTOP,Y_train,Y_train_expand,length(vocab),tuning,'ova');
% disp("Training for OVA is done:")
% toc
% %% prepare for test
% [~,~,docIDreorder_test] = unique(X_test(:,1));
% X_test = sparse(docIDreorder_test,X_test(:,2),...
%     X_test(:,3),length(Y_test),length(vocab));
% %% Evaluation
% tic
% 
% prediction_ovo = zeros(length(Y_test),length(svm_group_ovo));
% parfor i = 1:length(svm_group_ovo)
%     prediction_ovo(:,i) = svmclassify(svm_group_ovo(i),X_test);
% end
% 
% prediction_ovo = mode(prediction_ovo,2);
% ccr_ovo = mean(prediction_ovo==Y_test);
% 
% toc
% display(ccr_ovo)
% PreXtruth = confusionmat(prediction_ovo,Y_test);
% display(PreXtruth);
% disp("Evaluation for OVO is done:")
% toc
% 
% tic
% prediction_ova = zeros(length(Y_test),length(svm_group_ova));
% parfor i = 1:length(svm_group_ova)
%     prediction_ova(:,i) = svmclassify(svm_group_ova(i),X_test);
% end
% 
% prediction_ova = mode(prediction_ova,2);
% ccr_ova = mean(prediction_ova==Y_test);
% 
% toc
% display(ccr_ova)
% PreXtruth = confusionmat(prediction_ova,Y_test);
% display(PreXtruth);
% disp("Evaluation for OVA is done:")
% toc
% %% Report CCR
% 
% % Confustion matrix based on best CCR
% 
% %% Report F-score and recall
% 
% % Confustion matrix based on best F-score and recall
