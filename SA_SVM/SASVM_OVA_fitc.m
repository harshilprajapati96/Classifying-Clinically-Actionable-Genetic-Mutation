%% SVM Classifier for Text Document News 20 group
% MATLAB R2017b
% Bowen Song U04079758
% OVA
addpath('../libsvm-3.12/matlab');
%% preprocessing
tic
Preprocessing_new20;
disp("Preprocessing is done:")
toc
%% toy size
howmanytoys = 2;
X_train_woSTOP = X_train_woSTOP(1:find(Y_train_expand<howmanytoys+1,1,'last'),:);
X_test_woSTOP = X_test_woSTOP(1:find(Y_test_expand<howmanytoys+1,1,'last'),:);
Y_train = Y_train(1:find(Y_train<howmanytoys+1,1,'last'));
Y_test = Y_test(1:find(Y_test<howmanytoys+1,1,'last'));
%% use data with filtered stoped words
% preprocess Further with stop wrods technique

%% Training with OVA
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
inds = cell(numClasses,1); % Preallocation
SVMModel = cell(numClasses,1);
rng(1); % For reproducibility
for j = 1:numClasses
    fprintf("Training class %d",j);
    K =  [ X_train_processed , rbfKernel(trainData,trainData) ];
    tic
    SVMModel{j} = fitcsvm(X_train_processed,(Y_train==classNames(j)),...
        'ClassNames',[0 1],'Standardize',true,'KernelFunction','sensing2kernal');
    toc
end
% for j = 1:numClasses
%     fprintf("Getting Posterior for class %d",j);
%     tic
%     SVMModel{j} = fitPosterior(SVMModel{j});
%     toc
% end

%% Testing with star_tuning OVA
star_tuning = 1; % should be set to the best cv-CCR
[X_test_processed,alphaCust] = RRN_preprocessing(X_test_woSTOP,tuning(star_tuning),length(vocab));
% X_test_processed = full(Norm_preprocessing(X_test_woSTOP,length(vocab)));

disp("Decising time:")
n = size(X_test_processed,1);
posterior = zeros(n,numClasses);
tic
for j = 1:numClasses
    [~,post] = predict(SVMModel{j},X_test_processed);
    posterior(:,j) = post(:,2);
end
[confidence,decision] = max(posterior,[],2);

toc
disp("Training for OVA is done:")
OVAccr = mean(decision==Y_test);
disp(OVAccr)
toc
save('result_OVA.mat')
