from sklearn import svm
from sklearn.datasets import load_files
import numpy as np
import scipy.io as spio
from scipy.sparse import csr_matrix
from scipy.special import gammaln
frommat = spio.loadmat('sbowen_woStop.mat', squeeze_me=True)

xTrain = frommat['X_train_woSTOP'] # Xtest Mtx
yTrain = frommat['X_test_woSTOP'] # Xtest Mtx
xTest = frommat['Y_train'] # Xtest Mtx
yTest = frommat['Y_test'] # Xtest Mtx
vocab = frommat['vocab'] # Xtest Mtx

print(len(vocab))

def RRNpreprocessing(M,tune,vocablen,special=False):
    u, doc_indices = np.unique(M[:,0],return_inverse=True)
    xProcessed = csr_matrix((M[:,2], (doc_indices,M[:,1])), shape=(len(doc_indices), vocablen))
    print(xProcessed)
    alpha = 2*gammaln(tune+1) - gammaln(2*tune+vocablen)
    print(xProcessed.sum(axis=1))
    rowsum = xProcessed.sum(axis=1)
    WordProb = xProcessed.multiply(rowsum.power(-1))
    # print(WordProb)
    # if (special):
    #     for idx, val in enumerate(u):
    #         print(idx, val)
    #         xProcessed[idx,:] = np.random.choice(xProcessed[idx,:],tune,WordProb[idx,:])
    #     pass
    return [xProcessed,alpha]

#tuning = [150,200]
itune = 150
print("tuning at :"+str(itune))
[xTrainProce,alpha] = RRNpreprocessing(xTrain,itune,len(vocab))
print("Alpha is: "+ str(alpha))
print("Xtrain is :"+str(xTrainProce))

