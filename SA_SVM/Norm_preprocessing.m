%% SVM Classifier for Text Document Preprocessing for sensen aware Kernel
% Prepare for 
% MATLAB R2017b
% Bowen Song U04079758

% Randomly resampling with replacment for N-times per document. 
% forcing every document to have smae length
% EMP occrrance based on resampling

%% Input Parameters
% X_train: doc_id, word_id, word_occr
% vocab_len: length of the Vocabulary list
%% Output Paramters
% X_N: doc X N of word occurrance


function docXvocab = Norm_preprocessing(X_train,vocab_len)
% Set a seed for randsample
% seed = RandStream('mlfg6331_64');

[~,~,Doc_Index] = unique(X_train(:,1));
docXvocab = sparse(Doc_Index,X_train(:,2),X_train(:,3),...
    Doc_Index(length(Doc_Index)),vocab_len);

%% should add normalizing part, right now this only workds for regular svm withoutkernel
end


