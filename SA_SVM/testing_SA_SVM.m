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

function prediction = testing_SA_SVM(X_test,svm_group)


prediction = zeros(size(X_test,1),length(svm_group));

for i = 1:length(svm_group)
    % give result
    tic
    disp("Prediction Time per svm_group")
%     prediction(:,i) = svmclassify(svm_group(i),X_test);
    prediction(:,i) = predict(svm_group(i),X_test);
    toc
end

end