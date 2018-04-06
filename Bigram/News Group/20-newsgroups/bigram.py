import re
import nltk
import os
import enchant
import json
import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

def get_data(fl):
    data = open(fl, 'r').read()
    data_arr = data.split("\n")
    op_arr = []
    prevcl = ""
    i = 0
    for each in tqdm(data_arr):
        if each != '':
            cl, dat = each.split("\t")
            if cl != prevcl:
                i = i + 1
                prevcl = cl
            op_arr.append([dat, int(i)])
    return op_arr

def write_json(data, fname):
    f = open(fname, 'w')
    f.write(json.dumps(data, indent=4))
    f.close()

def cleandata(dataset,len_dat):
    corpus = []
    for i in range(len_dat):
        review = re.sub('[^a-zA-Z]', ' ',dataset[i] )
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        review = remove_noneng(review)
        corpus.append(review)
    return corpus

def remove_noneng(string):
    d = enchant.Dict("en_US")
    english_words = []
    for word in string.split():
        if d.check(word):
            english_words.append(word)
    return " ".join(english_words)

def makeclassvec(X,y):
    class_data = [''] * 20
    for i, each in enumerate(X):
        index = y[i] - 1 
        class_data[index] += each
    return class_data

def createBigram(X_bigram):
    bigrams = ngrams(X_bigram.split(), 2)
    return collections.Counter(bigrams)

if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    len_dat = len(newsgroups_train.data)
    X = cleandata(newsgroups_train.data,len_dat)
    write_json(X, "X_new.json")
    y = newsgroups_train.target
    X_bigram = makeclassvec(X,y)
    frequencies = []
    for i in range(len(X_bigram)):
        frequencies.append(createBigram(X_bigram[i]))

        

    # write_json(y, "y.json")
    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=0)
    # X_bigram = makeclassvec(X,y)
    # write_json(X_bigram, "X_bigram.json")
    # X = json.loads(open("X.json",'r').read())
    # y = json.loads(open("y.json",'r').read())
    # X_bigram = json.loads(open("X_bigram.json",'r').read())
    # bigram, frequencies = createBigram(X_bigram)


    # # Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = 1500)``
# X = cv.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values

# # Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
