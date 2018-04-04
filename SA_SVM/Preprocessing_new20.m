%% SVM Classifier for Text Document preprocessing
% MATLAB R2017b
% Bowen Song U04079758

%%----------------------------------------------%%
% output packaged Dataset 'sbowen_20news_essential.mat'
% output Dataset without stop words 'sbowen_woStop.mat'
clear
close all

% exempt reload, comment out for first use
%% Load and Parse the data
%% load data 
if (~exist('sbowen_20news_essential.mat'))
vocab = textread('vocabulary.txt','%s');
Y_test = load('test.label');
Y_train = load('train.label');
X_test = load('test.data');
X_train = load('train.data'); 
stopwords = textread('stoplist.txt','%s');
Y_newsgroup = textread('newsgrouplabels.txt','%s');
save('sbowen_20news_essential.mat')
else
load ('sbowen_20news_essential.mat')
end

%% removew stopwords from Y_test Y_train vocab X_test X_train

% get words ID from vocab
StopwordID = find(ismember(vocab,stopwords));

% process vocab
nonStopwordID = find(~ismember(vocab,stopwords));
vocab_wo_Stop = vocab(nonStopwordID);

% process train data
train_stop_i = (~ismember(X_train(:,2),StopwordID)); % get logic array of if its stopwords
X_train_woSTOP = train_stop_i.*X_train;
X_train_woSTOP = X_train_woSTOP(any(X_train_woSTOP,2),:);
% make sure there is no doc made entirely stopwords and got deleted
noDocRm_train = (length(unique(X_train_woSTOP(:,1))) == length(Y_train));
if (~noDocRm_train)
     missingdoc_train = setdiff(unique(X_train(:,1)),unique(X_train_woSTOP(:,1)));
end
X_train = X_train_woSTOP;
Y_train(missingdoc_train)=0;
Y_train = Y_train(any(Y_train,2),:);


train_class_occur = accumarray(X_train_woSTOP(:,1),1); % we can do consecutive num
train_class_occur = train_class_occur(any(train_class_occur,2),:);
% from above, we get docuemnt unique word size
Y_train_expand = repelem(Y_train,train_class_occur); 

% process test data
test_stop_i = (~ismember(X_test(:,2),StopwordID)); % get logic array of if its stopwords
X_test_woSTOP = test_stop_i.*X_test;
X_test_woSTOP = X_test_woSTOP(any(X_test_woSTOP,2),:);
% make sure there is no doc made entirely stopwords and got deleted
noDocRm_test = (length(unique(X_test_woSTOP(:,1))) == length(Y_test));
if (~noDocRm_test)
     missingdoc_test = setdiff(unique(X_test(:,1)),unique(X_test_woSTOP(:,1)));
end
X_test = X_test_woSTOP;
Y_test(missingdoc_test)=0;
Y_test = Y_test(any(Y_test,2),:);

test_class_occur = accumarray(X_test_woSTOP(:,1),1); % we can do consecutive num
test_class_occur = test_class_occur(any(test_class_occur,2),:);
% from above, we get docuemnt unique word size
Y_test_expand = repelem(Y_test,test_class_occur); 
% save the work space for later use
save('sbowen_woStop.mat')


