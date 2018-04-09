train_data=load('train.data');
train_label=load('train.label');

vocab=importdata('vocabulary.txt');

train_words_doc=ones(1,length(train_data(:,3)));
train_tfidf=zeros(length(train_label),length(vocab));
train_tfidf_ind=sub2ind(size(train_tfidf),train_data(:,1),train_data(:,2));
train_tfidf(train_tfidf_ind)=train_words_doc;

tf=zeros(1,length(vocab));
df=zeros(1,length(vocab));
tic
for i=1:length(vocab)
    
    wordindex=ismember(train_data(:,2),i);
    tf(i)=sum(train_data(wordindex,3));
    df(i)=sum(wordindex);
    
    
end

idf=1./df;
tf_idf=tf.*idf;


tf_idf_mat=train_tfidf.*tf_idf;



toc