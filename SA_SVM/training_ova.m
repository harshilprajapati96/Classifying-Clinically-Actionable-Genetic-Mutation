%% SVM Classifier for Text Document Training
% MATLAB R2017b
% Bowen Song U04079758

% evaluation should not rely on CCR
% Evaluate based on F-score and recall

function svms = training_ovo(X_train,Y_train,X_test...
    ,Y_test,Y_train_expand,Y_test_expand,vocab)