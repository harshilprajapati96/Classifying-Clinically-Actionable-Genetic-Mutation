%% SVM Classifier for Text Document News 20 group
% MATLAB R2017b
% Bowen Song U04079758
% OVA
%% change to sparse matrix and parfor
addpath('libsvm-3.22/matlab');
%% preprocessing
tic
Preprocessing_new20;
disp("Preprocessing is done:")
toc
% toy size
% howmanytoys = 2;
% X_train_woSTOP = X_train_woSTOP(1:find(Y_train_expand<howmanytoys+1,1,'last'),:);
% X_test_woSTOP = X_test_woSTOP(1:find(Y_test_expand<howmanytoys+1,1,'last'),:);
% Y_train = Y_train(1:find(Y_train<howmanytoys+1,1,'last'));
% Y_test = Y_test(1:find(Y_test<howmanytoys+1,1,'last'));
%% use data with filtered stoped words
% preprocess Further with stop wrods technique

%% Training with OVA
tuning = 150; % tuning prarmeter for US new is suggested to be 150 at best
disp("X_train preperation time:")
global SA_n
tic
[X_train_processed,SA_n] = SA1_preprocessing(X_train_woSTOP,tuning,...
    length(vocab));
%X_train_processed = full(Norm_preprocessing(X_train_woSTOP,length(vocab)));
toc
classNames = unique(Y_train);
numClasses = length(classNames);
inds = cell(numClasses,1); % Preallocation
SVMModel = cell(numClasses,1);
numTrain = size(X_train_processed,1);

%% the kernel
disp("Training time:")
tic
K =  [ (1:numTrain)' , sensing1kernal(X_train_processed,X_train_processed) ];
rng(1); % For reproducibility
for j = 1:numClasses
    fprintf("Training class %d",j);
    tic
%     SVMModel{j} = fitcsvm(X_train_processed,(Y_train==classNames(j)),...
%         'ClassNames',[0 1],'Standardize',true,'KernelFunction','sensing2kernal');
    SVMModel{j} = svmtrain(double(Y_train==classNames(j)),K,'-t 4 -b 1');
    toc
end
trainTime = toc;

%% Testing with star_tuning OVA
star_tuning = 1; % should be set to the best cv-CCR
[X_test_processed,SA_n] = SA1_preprocessing(X_test_woSTOP,tuning(star_tuning),length(vocab));
% X_test_processed = full(Norm_preprocessing(X_test_woSTOP,length(vocab)));

disp("Testing time:")
tic
n = size(X_test_processed,1);
posterior = zeros(n,numClasses);
KK =  [ (1:size(X_test_processed,1))' , sensing1kernal(X_test_processed,X_test_processed) ];
for j = 1:numClasses
    [~,~,postt] = svmpredict(double(Y_train==classNames(j)),K,SVMModel{j}, '-b 1');
    posteriort(:,j) = postt(:,SVMModel{j}.Label==1);  
    [~,~,post] = svmpredict(double(Y_test==classNames(j)),KK,SVMModel{j}, '-b 1');
     posterior(:,j) = post(:,SVMModel{j}.Label==1);    %# probability of class==k
end
[confidencet,decisiont] = max(posteriort,[],2);

[confidence,decision] = max(posterior,[],2);

testTime = toc;
disp("Testing for SASVM_OVA_fitc kernal 1 is done:")
OVAccr = mean(decision==Y_test);
disp(OVAccr)

save('SASVM_OVA_fitc.mat')
rmpath('libsvm-3.22/matlab');
