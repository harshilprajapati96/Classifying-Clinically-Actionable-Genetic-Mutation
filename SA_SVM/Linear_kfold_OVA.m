%% Linear Choosing best box constant
% MATLAB R2017b
% Bowen Song U04079758

%% preprocessing
tic
Preprocessing_new20;
X_train_processed = Norm_preprocessing(X_train_woSTOP,length(vocab));
disp("Preprocessing is done:")
toc

addpath('libsvm-3.22/matlab');
rmpath('libsvm-3.22/matlab');
%% Finding Best sigma and Box constant
classNames = unique(Y_train);
numClasses = length(classNames);
inds = cell(numClasses,1); % Preallocation
SVMModel = cell(numClasses,1);


boxcon_power = -7:13;
boxcon = 2.^boxcon_power;

svm_ccr = zeros(length(boxcon),5);
warning('off','all')
warning
for t_i = 1:numClasses
    [X_train_1_2, Y_train_1_2] ...
        = ova(t_i,X_train_processed,Y_train);
    for j = 1:length(boxcon)
        kfold = cvpartition(length(Y_train_1_2),'KFold',5);
        parfor i = 1:5
            svms = svmtrain(X_train_1_2(training(kfold,i),:),Y_train_1_2(training(kfold,i),:),...
                'boxconstraint',boxcon(j)*ones(kfold.TrainSize(i),1),...
                'autoscale','false','kernel_function','linear','kernelcachelimit',inf);
            svm_ccr(j,i) = mean(svmclassify(svms,X_train_1_2(test(kfold,i),:))...
                ==Y_train_1_2(test(kfold,i),:));
        end
    end
    cv_ccr = mean(svm_ccr,2);
    %% Selecting Best sigma and Box constant
    [~,boxcon_star_ind] = max(cv_ccr);
    boxcon_star(t_i) = boxcon(boxcon_star_ind);
end
%% actual Training with Star sigma and boxconstant
tic
addpath('libsvm-3.22/matlab');
warning('off','all')
warning
svms = cell(numClasses,1);
parfor j = 1:numClasses
    [X_train_1_2, Y_train_1_2] ...
        = ova(j,X_train_processed,Y_train);
%     svms{j} = svmtrain(X_train_1_2,Y_train_1_2,'boxconstraint',boxcon_star(j),...
%         'autoscale','false','kernel_function','Linear','kernelcachelimit',inf);
    svms{j} = svmtrain(Y_train_1_2,X_train_1_2,sprintf('-c %f -t 0 -b 1 -m inf',boxcon_star(j)));
end
train_time = toc;
[~,~,docIDreorder_test] = unique(X_test(:,1));
X_test = sparse(docIDreorder_test,X_test(:,2),...
    X_test(:,3),length(Y_test),length(vocab));
tic
prediction_prob = zeros(length(Y_test),length(svms));
parfor i = 1:length(svms)
%     prediction(:,i) = svmclassify(svms{i},X_test);
[~,~,p] = svmpredict(Y_test, X_test, svms{i}, '-b 1');
    prediction_prob(:,i) = p(:,svms{i}.Label==1);    %# probability of class==k
end

[~,prediction] = max(prediction_prob,[],2);
ccr = mean(prediction==Y_test);

test_time = toc;
disp('Linear_kfold_OVA')
display(ccr)
% PreXtruth = confusionmat(prediction,Y_test);
% display(PreXtruth);
save('Linear_kfold_OVA.mat')
rmpath('libsvm-3.22/matlab');

