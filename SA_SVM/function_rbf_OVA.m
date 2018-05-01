%% RBF Choosing best box constant and sigma
% MATLAB R2017b
% Bowen Song U04079758
%% OVA
%% preprocessing
function function_rbf_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,rbf_sig,filename)

X_train_processed = Norm_preprocessing(X_train_woSTOP,length(vocab));

addpath('libsvm-320/matlab');

X_train_processed = Norm_preprocessing(X_train_woSTOP,length(vocab));
disp("Preprocessing is done:")


%% Finding Best sigma and Box constant
classNames = unique(Y_train);
numClasses = length(classNames);
tot_iter = numClasses;


svm_ccr = zeros(length(boxcon),length(rbf_sig),5);
warning('off','all')
warning
for t_i = 1:tot_iter
    [X_train_1_2, Y_train_1_2] ...
        = ova(t_i,X_train_processed,Y_train);
    tic
    for k = 1:length(rbf_sig)
        for j = 1:length(boxcon)
            kfold = cvpartition(length(Y_train_1_2),'KFold',5);
            parfor i = 1:5
                %             svms = svmtrain(X_train_1_2(training(kfold,i),:),Y_train_1_2(training(kfold,i),:),...
                %                  'boxconstraint',boxcon(j)*ones(kfold.TrainSize(i),1),...
                %                  'autoscale','false','kernel_function','rbf'...
                %                  ,'rbf_sigma',rbf_sig(k));
                
                svms = svmtrain(Y_train_1_2(training(kfold,i),:),X_train_1_2(training(kfold,i),:),...
                    sprintf('-c %f -t 2 -m inf -h 0 -g %f -q',boxcon(j),rbf_sig(k)));
                
                svm_ccr(j,k,i) = mean(Y_train_1_2(test(kfold,i),:)==...
                    svmpredict(Y_train_1_2(test(kfold,i),:),X_train_1_2(test(kfold,i),:),svms,'-q'));
                
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
disp("start training")
tic
svms = cell(tot_iter,1);
parfor j = 1:tot_iter
    [X_train_1_2, Y_train_1_2] ...
        = ova(j,X_train_processed,Y_train);
    %     svms{j} = svmtrain(X_train_1_2,Y_train_1_2,'boxconstraint',boxcon_star(j),...
    %     'autoscale','false','kernel_function','rbf'...
    %                  ,'rbf_sigma',rbf_sig_star(j));
    
    svms{j} = svmtrain(double(Y_train_1_2),X_train_1_2,sprintf('-c %f -t 2 -g %f -m inf -h 0 -b 1 -q',boxcon_star(j),rbf_sig_star(j)));
    
end

traintime = toc;

X_test = Norm_preprocessing(X_test_woSTOP,length(vocab));
disp("predicting")
tic
prediction_prob = zeros(length(Y_test),length(svms));
parfor i = 1:length(svms)
    % prediction(:,i) = svmclassify(svms{i},X_test);
    
    [~,~,p] = svmpredict(Y_test, X_test, svms{i}, '-b 1');
    prediction_prob(:,i) = p(:,svms{i}.Label==1);    %# probability of class==k
end
testtime=toc;


[~,prediction] = max(prediction_prob,[],2);
ccr = mean(prediction==Y_test);

display(ccr)
PreXtruth = confusionmat(prediction,Y_test);

save(sprintf('%s.mat',filename))
rmpath('libsvm-320/matlab');