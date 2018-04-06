%% SVM Classifier for Text Document OVO fucntion
% MATLAB R2017b
% Bowen Song U04079758

%% Input Parameters
% Two class numbers: class1,class2
% X_train: doc X N, doc= unique doc_id N = tuning prameter
% Y_train: no change doc X 1
%% Output Paramters
% X_train_1_2: doc(1y= 1&2) X N
% Y_train_1_2: doc(1y= 1&2) X 1

function [X_train_1_2,Y_train_1_2]...
    = ovo(class1,class2,X_train,Y_train)
% for Training documents
Y_train_1_2 = Y_train.*(Y_train==class2)+ Y_train.*(Y_train==class1);
X_train_1_2 = X_train(Y_train_1_2~=0,:);
X_train_1_2 = X_train_1_2(any(X_train_1_2,2),:);
Y_train_1_2 = Y_train_1_2(any(Y_train_1_2,2),:);
if (nnz(X_train_1_2)/numel(X_train_1_2) ~= 1)
    error("OVO X_train_1_2: class %d and %d contains zero elements"...
        ,class1,class2)
end

end