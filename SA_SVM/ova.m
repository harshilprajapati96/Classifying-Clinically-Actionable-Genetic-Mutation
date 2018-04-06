%% SVM Classifier for Text Document OVA fucntion
% MATLAB R2017b
% Bowen Song U04079758

%% Input Parameters
% Two class numbers: class1,class2
% X_train: doc X N, doc= unique doc_id N = tuning prameter
% Y_train: no change doc X 1
%% Output Paramters
% X_train: doc X N
% Y_train: doc X 1 (class labeld +1 not class is -1)

function [X_train,Y_train]...
    = ova(class,X_train,Y_train)
% for Training documents
Y_train(Y_train~=class)=-1;
Y_train(Y_train==class)=1;
% make sure there is no zero element in the matrix
if (nnz(X_train)/numel(X_train) ~= 1)
    error("OVA X_train_1_2: class %d and %d contains zero elements"...
        ,class1,class2)
end

end