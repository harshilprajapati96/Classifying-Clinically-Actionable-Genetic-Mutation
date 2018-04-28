%% %% SVM Classifier for Text Document News 20 group
% MATLAB R2017b
% Bowen Song U04079758
% OVO

%% preprocessing
tic
Preprocessing_new20;
disp("Preprocessing is done:")
toc
%% use data with filtered stoped words
% preprocess Further with stop wrods technique

%% Training with OVO
tic
tuning = 150; % tuning prarmeter for US new is suggested to be 150 at best

disp("X_train preperation time:")
global alphaCust
tic
[X_train_processed,alphaCust] = RRN_preprocessing(X_train_woSTOP,tuning,...
    length(vocab));
%X_train_processed = full(Norm_preprocessing(X_train_woSTOP,length(vocab)));
toc
classNames = unique(Y_train);
numClasses = length(classNames);
tot_iter = numClasses*(numClasses-1)/2;
inds = cell(tot_iter,1); % Preallocation
SVMModel = cell(tot_iter,1);
rng(1); % For reproducibility
pair_i = [];
pair_j = [];
for i = 1:numClasses-1
    for j = i+1:numClasses
        pair_i = [pair_i ; i];
        pair_j = [pair_j ; j];
    end
end
for j = 1:tot_iter % parfor in the end
    [X_train_1_2, Y_train_1_2] ...
        = ovo(pair_i(j),pair_j(j),X_train_processed,Y_train);
    SVMModel{j} = fitcsvm(X_train_1_2,Y_train_1_2,...
        'KernelFunction','sensing2kernal');
end

disp("Total Training time:")
toc
%% Testing with star_tuning OVA
star_tuning = 1; % should be set to the best cv-CCR
[X_test_processed,alphaCust] = RRN_preprocessing(X_test_woSTOP,tuning(star_tuning),length(vocab));
% X_test_processed = full(Norm_preprocessing(X_test_woSTOP,length(vocab)));

disp("Decising time:")
n = size(X_test_processed,1);
decision = zeros(n,tot_iter);
tic
for j = 1:tot_iter
    decision(:,j) = predict(SVMModel{j},X_test_processed);
end
finaldecision = mode(decision,2);

toc
disp("Training for OVO is done:")
OV0ccr = mean(finaldecision==Y_test);

toc
save('result_OVO.mat')
