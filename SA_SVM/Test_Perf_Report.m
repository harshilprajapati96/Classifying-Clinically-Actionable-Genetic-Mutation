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
tuning = [1 1 1 1 1 11 1 1 1 1];
svm_group_ovo = training_ovo(X_train_woSTOP,Y_train,Y_train_expand,length(vocab),tuning,'ova');
disp("Training for OVO is done:")
toc
% 10 parameters 
% without parfor
% with most inner parfor
% with outter parfor
%% Training woth OVA
tic
tuning = [1 1 1 1 1 11 1 1 1 1];
svm_group_ova = training_SA_SVM(X_train_woSTOP,Y_train,Y_train_expand,length(vocab),tuning,'ova');
disp("Training for OVA is done:")
toc

%% Evaluation
tic

disp("Evaluation for OVO is done:")
toc

tic

disp("Evaluation for OVA is done:")
toc
%% Report CCR

% Confustion matrix based on best CCR

%% Report F-score and recall

% Confustion matrix based on best F-score and recall
