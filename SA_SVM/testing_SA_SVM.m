%% SVM Classifier for Text Document Testing
% MATLAB R2017b
% Bowen Song U04079758

%% ---------------------------------------------------%%
% Feature includes:
% one vs. one for multi-class


%% ---------------------------------------------------%%
% Input: X_train, Y_train, svm_group, and mode
% X_train: (uniquedoc) X (N number of words randomly picked) of the word
% occurernace
% Y_train: (uniquedoc) X 1 of class label
% svm_group: m svmClassifyer (case ova) or m(m-1)/2 svmClassifyer (case ovo)
% mode: 'ovo' and 'ova'


%% ---------------------------------------------------%%
% Output: Prediction for each document

function prediction = testing_SA_SVM(X_test,m,svm_group,mode)


switch mode
    case 'ovo'
        
        prediction = zeros(size(X_test,1),length(svm_group));
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
        parfor i = 1:length(svm_group)
            % give result
            prediction(:,i) = svmclassify(svm_group(i),X_test);
        end
        
    case 'ova'
        
end