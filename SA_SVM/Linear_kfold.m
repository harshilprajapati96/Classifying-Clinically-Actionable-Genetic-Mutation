%% Linear Choosing best box constant
% MATLAB R2017b
% Bowen Song U04079758

%% preprocessing
tic
Preprocessing_new20;
X_train_processed = Norm_preprocessing(X_train_woSTOP,length(vocab));
disp("Preprocessing is done:")
toc
%% Finding Best sigma and Box constant
classNames = unique(Y_train);
numClasses = length(classNames);
tot_iter = numClasses*(numClasses-1)/2;
inds = cell(tot_iter,1); % Preallocation
SVMModel = cell(tot_iter,1);
pair_i = [];
pair_j = [];
for i = 1:numClasses-1
    for j = i+1:numClasses
        pair_i = [pair_i ; i];
        pair_j = [pair_j ; j];
    end
end

boxcon_power = -7:13;
boxcon = 2.^boxcon_power;

svm_ccr = zeros(length(boxcon),5);
warning('off','all')
warning
for t_i = 1:tot_iter
    [X_train_1_2, Y_train_1_2] ...
        = ovo(pair_i(t_i),pair_j(t_i),X_train_processed,Y_train);
    for j = 1:length(boxcon)
        kfold = cvpartition(length(Y_train_1_2),'KFold',5);
        parfor i = 1:5
            svms = svmtrain(X_train_1_2(training(kfold,i),:),Y_train_1_2(training(kfold,i),:),...
                'boxconstraint',boxcon(j)*ones(kfold.TrainSize(i),1),...
                'autoscale','false','kernel_function','linear');
            svm_ccr(j,i) = mean(svmclassify(svms,X_train_1_2(test(kfold,i),:))...
                ==Y_train_1_2(test(kfold,i),:));
        end
        
    end
    cv_ccr = mean(svm_ccr,3);
    %% Selecting Best sigma and Box constant
    [~,boxcon_star_ind] = max(max(cv_ccr,[],2));
    boxcon_star(t_i) = boxcon(boxcon_star_ind);
end
%% actual Training with Star sigma and boxconstant
tic

warning('off','all')
warning
svms = cell(tot_iter,1);
parfor j = 1:tot_iter
    [X_train_1_2, Y_train_1_2] ...
        = ovo(pair_i(j),pair_j(j),X_train_processed,Y_train);
    svms{j} = svmtrain(X_train_1_2,Y_train_1_2,'boxconstraint',boxcon_star(j),...
        'autoscale','false','kernel_function','Linear');
    
end

[~,~,docIDreorder_test] = unique(X_test(:,1));
X_test = sparse(docIDreorder_test,X_test(:,2),...
    X_test(:,3),length(Y_test),length(vocab));

prediction = zeros(length(Y_test),length(svms));
parfor i = 1:length(svms)
    prediction(:,i) = svmclassify(svms{i},X_test);
end

prediction = mode(prediction,2);
ccr = mean(prediction==Y_test);

toc
display(ccr)
% PreXtruth = confusionmat(prediction,Y_test);
% display(PreXtruth);
