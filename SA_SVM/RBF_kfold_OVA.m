%% RBF Choosing best box constant and sigma
% MATLAB R2017b
% Bowen Song U04079758
%% OVA
%% preprocessing
addpath('libsvm-3.22/matlab');
tic
Preprocessing_new20;
X_train_processed = Norm_preprocessing(X_train_woSTOP,length(vocab));
disp("Preprocessing is done:")
toc
%% toy size
howmanytoys = 2;
X_train_woSTOP = X_train_woSTOP(1:find(Y_train_expand<howmanytoys+1,1,'last'),:);
X_test_woSTOP = X_test_woSTOP(1:find(Y_test_expand<howmanytoys+1,1,'last'),:);
Y_train = Y_train(1:find(Y_train<howmanytoys+1,1,'last'));
Y_test = Y_test(1:find(Y_test<howmanytoys+1,1,'last'));
%% Finding Best sigma and Box constant
classNames = unique(Y_train);
numClasses = length(classNames);
tot_iter = numClasses;
inds = cell(tot_iter,1); % Preallocation
SVMModel = cell(tot_iter,1);

boxcon_power = -7:13;
boxcon = 2.^boxcon_power;

rbf_power = -7:9;
rbf_sig = 2.^rbf_power;
svm_ccr = zeros(length(boxcon),length(rbf_sig),5);
warning('off','all')
warning
for t_i = 1:tot_iter
        [X_train_1_2, Y_train_1_2] ...
            = ova(j,X_train_processed,Y_train);
        tic
for k = 1:length(rbf_sig)
    for j = 1:length(boxcon)
    kfold = cvpartition(length(Y_train_1_2),'KFold',5);
        parfor i = 1:5
%             svms = svmtrain(X_train_1_2(training(kfold,i),:),Y_train_1_2(training(kfold,i),:),...
%                  'boxconstraint',boxcon(j)*ones(kfold.TrainSize(i),1),...
%                  'autoscale','false','kernel_function','rbf'...
%                  ,'rbf_sigma',rbf_sig(k));

            rbfKernel = @(X,Y) exp(-rbf_sig(k) .* pdist2(X,Y,'euclidean').^2);

             K =  [ (1:size(X_train_1_2(training(kfold,i),:),1))' ,...
                 rbfKernel(X_train_1_2(training(kfold,i),:),X_train_1_2(training(kfold,i),:)) ];
             
             model = svmtrain(double(Y_train_1_2(training(kfold,i),:)), K, sprintf('-t 4 -c %f -m inf -q',boxcon(j)));

             svm_ccr(j,k,i) = mean(Y_train_1_2(test(kfold,i),:)==...
                svmpredict(Y_train_1_2(test(kfold,i),:),X_train_1_2(test(kfold,i),:),model,'-q'));
%             svm_ccr(j,k,i) = mean(svmclassify(svms,X_train_1_2(test(kfold,i),:))...
%                 ==Y_train_1_2(test(kfold,i),:));
        end
    end
end
fprintf('class %d out of %d is done',t_i,tot_iter);
toc

cv_ccr = mean(svm_ccr,3);
%% Selecting Best sigma and Box constant
[boxcon_star_perf(t_i),boxcon_star_ind] = max(max(cv_ccr,[],2));
[rbf_sig_star_perf(t_i),rbf_sig_star_ind] = max(max(cv_ccr));
boxcon_star(t_i) = boxcon(boxcon_star_ind);
rbf_sig_star(t_i) = rbf_sig(rbf_sig_star_ind);
end
%% actual Training with Star sigma and boxconstant


svms = cell(tot_iter,1);d
parfor j = 1:tot_iter
[X_train_1_2, Y_train_1_2] ...
        = ova(j,X_train_processed,Y_train);
%     svms{j} = svmtrain(X_train_1_2,Y_train_1_2,'boxconstraint',boxcon_star(j),...
%     'autoscale','false','kernel_function','rbf'...
%                  ,'rbf_sigma',rbf_sig_star(j));
             
               rbfKernel = @(X,Y) exp(-rbf_sig_star(j) .* pdist2(X,Y,'euclidean').^2);
            
             K =  [ (1:size(X_train_1_2,1))' ,...
                 rbfKernel(X_train_1_2,X_train_1_2) ];
             
             svms(j) = svmtrain(double(Y_train_1_2), K, sprintf('-t 4 -c %f -m inf',boxcon_star(j)));
     
end

[~,~,docIDreorder_test] = unique(X_test(:,1));
X_test = sparse(docIDreorder_test,X_test(:,2),...
    X_test(:,3),length(Y_test),length(vocab));

prediction_prob = zeros(length(Y_test),length(svms));
parfor i = 1:length(svms)
% prediction(:,i) = svmclassify(svms{i},X_test);

[~,~,p] = svmpredict(Y_test, X_test, svms{i}, '-b 1');
    prediction_prob(:,i) = p(:,svms{i}.Label==1);    %# probability of class==k
end


[~,prediction] = max(prediction_prob,[],2);
ccr = mean(prediction==Y_test);

display(ccr)
PreXtruth = confusionmat(prediction,Y_test);

save('RBF_kfold_OVA.mat')
rmpath('libsvm-3.22/matlab');