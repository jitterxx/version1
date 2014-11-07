# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:35:58 2014

@author: sergey
"""
import re
import pymorphy2
from time import time
import pandas as pd


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import tree
from sklearn import svm
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#split for words
def tokenizer_1(data):
    #print 'data tokenizer: ', data
    morph = pymorphy2.MorphAnalyzer()
    
    splitter = re.compile('\W',re.I|re.U)
    
    clear = splitter.split(data)
    f = []
    for i in clear:
        #print i
        m = morph.parse(i)
        if len(m) > 2 and len(m)<20: 
            word = m[0]
            if word.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                  f.append(word.normal_form)
                  #print word.tag.POS
        
    
    return f



debug = True

category = {0:'Personal',1:'Notification',2:'Promo'}

pd.set_option('display.max_colwidth',1000)

f2 = open("email_df", 'r')
email_df = pd.read_json(f2)
f2.close()

text = email_df.loc['Text']
index = text.index
print 'Text samples count: ',text.shape
print 'Test samples count: ',index.shape

f = open("email_df.train", 'r')
train = pd.read_json(f)
f.close()

train_text = train.loc['Text']
target = train.loc['Target']
labels = map(int,target)
print 'Train samples count: ',train_text.shape
print 'Train target vector: ',target.shape

    

#raw_input('...')


vectorizer = TfidfVectorizer(tokenizer=tokenizer_1,use_idf=True,\
                            #max_df=0.95, min_df=2,\
                            max_features=200)


print("Vectorization...")
t0 = time()

train_v = vectorizer.fit_transform(train_text)
v = vectorizer.transform(text)

print 'Test vector length: ', v.shape
print 'Train vector length: ', train_v.shape

print('Done in %fs' % (time() - t0))


n_cat = len(category)
print 'Category number: ', n_cat

#raw_input('Classify?')

print('Do classification...')
C = 1.0  # SVM regularization parameter
clf = svm.NuSVC(kernel='linear',probability=True)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
lin_svc = svm.LinearSVC(C=C)



#Обучаем модель
print('Fit model...')

t0 = time()

X = train_v.todense()
V = v.todense()

clf.fit(X,labels)

#print("Clustering sparse data with %s" % km)
print('Predicting...')
t0 = time()

Z = clf.predict(V)
P = clf.predict_proba(V)

print("done in %0.3fs \n" % (time() - t0))

#print 'Predicted category: ',Z
print 'Predicted samples count :',Z.shape


print("\n")

terms = vectorizer.get_feature_names()
docs = {}

for i in category.keys():
    print('Category "%s":' % category[i])
    docs[i]=list()

    for j in range(len(Z)):
        if i == Z[j]:
            docs[i].append(index[j])
            prob = P[j]*100
            print('%s:  %s' % (prob,email_df.loc['From',index[j]]))
    print '\n\n'



def draw_clusters(data,real,target):
    
    
    X = PCA(n_components=2).fit_transform(data)

    V = PCA(n_components=2).fit_transform(real)
    y = target

    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y)
    
    h = .02  # step size in the mesh
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\
                         np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.scatter(V[:, 0], V[:, 1], c='black', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Email SVM')

    plt.show()        
    
    return 

draw_clusters(X,V,labels)