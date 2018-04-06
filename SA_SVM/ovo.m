%% SVM Classifier for Text Document OVO fucntion
% MATLAB R2017b
% Bowen Song U04079758
function [X_train_1_2,Y_train_1_2]...
    = ovo(class1,class2,X_train,Y_train,Y_train_expand,vocab_length)
% for Training documents
Y_train_1_2_expand = Y_train_expand.*(Y_train_expand==class2)...
    + Y_train_expand.*(Y_train_expand==class1);
X_train_1_2_expand = X_train(Y_train_1_2_expand~=0,:);
Y_train_1_2 = Y_train.*(Y_train==class2) + Y_train.*(Y_train==class1);
% Clearup the zeros
Y_train_1_2 = Y_train_1_2(any(Y_train_1_2,2),:);
X_train_1_2_expand = X_train_1_2_expand(any(X_train_1_2_expand,2),:);

%% Make X train and X test based on Y
[~,~,docIDreorder_train] = unique(X_train_1_2_expand(:,1));
X_train_1_2 = sparse(docIDreorder_train,X_train_1_2_expand(:,2),...
    X_train_1_2_expand(:,3),length(Y_train_1_2),vocab_length);

end