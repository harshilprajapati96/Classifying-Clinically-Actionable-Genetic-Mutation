%% SVM Classifier for Text Document Preprocessing for sensen aware Kernel
% Prepare for 
% MATLAB R2017b
% Bowen Song U04079758

% Randomly resampling with replacment for N-times per document. 
% forcing every document to have smae length
% EMP occrrance based on resampling

%% Input Parameters
% X_train: doc_id, word_id, word_occr
% N: turnign paramter , fixing the number of words in a docuemnt 
% vocab_len: length of the Vocabulary list
%% Output Paramters
% X_N: doc X N of word occurrance
% alpha: fixed value for a pair of N and Lenght of Vocab

function [X_train_processed,alpha] = RRN_preprocessing(X_train,N,vocab_len)
% Set a seed for randsample
% seed = RandStream('mlfg6331_64');

[~,~,Doc_Index] = unique(X_train(:,1));
docXvocab = sparse(Doc_Index,X_train(:,2),X_train(:,3),...
    Doc_Index(length(Doc_Index)),vocab_len);
% Choosing N words randomly and with replacement from the sequence ACGT, 
% according to the specified probabilities.
WordProb = full(docXvocab./sum(docXvocab,2));
X_train_processed = zeros(size(docXvocab,1),N);
parfor i = 1:size(docXvocab,1)
X_train_processed(i,:) = full(randsample(docXvocab(i,:),N,true,WordProb(i,:))); 
end
alpha = 2*gammaln(N+1) - gammaln(2*N+vocab_len);
end


