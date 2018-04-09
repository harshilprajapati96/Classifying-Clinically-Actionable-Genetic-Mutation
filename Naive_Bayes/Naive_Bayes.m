clear; close all; clc;
tic;

%Loads data

train_data=load('train.data');
train_label=load('train.label');
test_data=load('test.data');
test_label=load('test.label');
vocab=importdata('vocabulary.txt');
stopwords=importdata('stoplist.txt');
%sets alpha value
alpha=(1/length(vocab));

train_label_length=length(train_label);
prior_prob_train=zeros(20,1);

%removing the unique words of test from test
% train_data_unique=unique(train_data(:,2));
% test_data_unique=unique(test_data(:,2));
% unique_test_words=setdiff(test_data_unique,train_data_unique);
% uniqueindex=find(ismember(test_data(:,2),unique_test_words)==1);
% test_data(uniqueindex,:)=[];


% [~,stopwordindex]=ismember(stopwords,vocab);
% 
% test_stopworddata=ismember(test_data(:,2),stopwordindex);
% test_data(test_stopworddata,:)=[];
% 
% train_stopworddata=ismember(train_data(:,2),stopwordindex);
% train_data(train_stopworddata,:)=[];


for i=1:max(train_label)
    
train_total_doc=length(find(train_label==i));
prior_prob_train(i)=train_total_doc/train_label_length;

end

%preallocates and adds alpha to beta.
train_word_est=zeros(length(vocab),max(train_label))+alpha;

train_wordsperlabel=zeros(1,max(train_label));

%gets the total words per doc
train_wordsperdoc = accumarray(train_data(:,1),(train_data(:,3)));

%loops through each training doc and stores the number of times a word
%appears in the label of that doc. But adds what is already there to
%account for the previous docs in that label
for k=1:max(train_data(:,1))

  doclabel=train_label(k);
  docindex=find(train_data(:,1)==k);
  vocabindex=train_data(docindex,2);
  train_word_est(vocabindex,doclabel) = train_data(docindex,3)+train_word_est(vocabindex,doclabel);

end


%Get the probability of a word being in a certain label.

train_word_est=train_word_est./sum(train_word_est);


%Preallocating
test_word_est=zeros(length(vocab),max(test_data(:,1)));


 y=zeros(max(train_label),max(test_data(:,1)));

%Predicting the label for each test doc. Taking the amount of times a vocab
%appears in each word and storing it in a matrix
for j=1:max(test_data(:,1))

  
  testdocindex=find(test_data(:,1)==j);
  
  test_vocabindex=test_data(testdocindex,2);
  test_word_est(test_vocabindex,j) = test_data(testdocindex,3);%+alpha;
  
  
end



y=(log(train_word_est)'*test_word_est)+log(prior_prob_train);


[~,test_label_predict]=max(y);
cm_test=confusionmat(test_label,test_label_predict);
CCR_test=sum(diag(cm_test))/sum(sum(cm_test)); 


toc

