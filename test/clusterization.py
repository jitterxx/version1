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

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


debug = True


def training():
    svd = TruncatedSVD(n_components=100)
    f = open("email_df.train", 'r')
    train = pd.read_json(f)
    f.close()
    pd.set_option('display.max_colwidth',1000)
    
    
    text = train.loc['Text']
    #print text
    index = text.index
    print 'Train samples count: ',len(index)    
    
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_1,use_idf=True)
    print("Training vectorization...")
    t0 = time()
    
    v = vectorizer.fit_transform(text)
    print 'Train vector length: ', v.shape
    print('Done in %fs' % (time() - t0))
    
    lsa = make_pipeline(svd, Normalizer(copy=False))

    tv = lsa.fit_transform(v)
    
    print 'Train truncated matrix: ', tv.shape
    print('Done in %fs' % (time() - t0))


    n_clusters=8
    
    print('Do train clustering...')
    km = KMeans(init='k-means++', max_iter=100, n_init=1,n_clusters=n_clusters)
    
    t0 = time()
    km.fit(tv,train.loc['Target'])
    print("done in %0.3fs \n" % (time() - t0))
    
    
    return km




f2 = open("email_df", 'r')
email_df = pd.read_json(f2)
f2.close()

pd.set_option('display.max_colwidth',1000)


text = email_df.loc['Text']
#print text
index = text.index
print 'Samples count: ',len(text.index)


#raw_input('...')

#split for wordsfrom sklearn.pipeline import make_pipeline
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

vectorizer = TfidfVectorizer(tokenizer=tokenizer_1,use_idf=True)


print("Vectorization...")
t0 = time()

v = vectorizer.fit_transform(text)

print 'Vector length: ', v.shape
#print v
#print type(v1)

print('Done in %fs' % (time() - t0))
print('SVD transforming...')
t0 = time()
svd = TruncatedSVD(n_components=100)


lsa = make_pipeline(svd, Normalizer(copy=False))

tv = lsa.fit_transform(v)
#tv1 = lsa.fit_transform(v1)

print 'Truncated matrix: ', tv.shape
print('Done in %fs' % (time() - t0))


#print tv,tv.shape
#print tv1,tv1.shape

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    


n_clusters=8

print('Do clustering...')
#km = KMeans(init='k-means++', max_iter=100, n_init=1,n_clusters=n_clusters)
#Обучаем модель
km = training()

#print("Clustering sparse data with %s" % km)
print('Predicting...')
t0 = time()

#km.fit(tv)
km.predict(tv)
print("done in %0.3fs \n" % (time() - t0))

print km.cluster_centers_.shape
print km.labels_
#print vector.get_feature_names()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
docs = {}
for i in range(n_clusters):
    print("Cluster %d:" % i)
    docs[i]=list()
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        pass
    print '\n'

    #Make samples sort by cluster
    for j in range(len(km.labels_)):
        if i == km.labels_[j]:
            docs[i].append(index[j])
            print email_df.loc['From',index[j]]

   


def draw_clusters(km):
    svd = TruncatedSVD(n_components=2)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    tv = lsa.fit_transform(km.cluster_centers_)
    
    fig, ax = plt.subplots()
    ax.set_title('Clusters')
    j = 0
    for i in tv:
        ax.scatter(i[0],i[1],alpha=0.5,marker='o',color='green')
        ax.annotate('Cluster '+str(j), xy=(i[0],i[1]),  xycoords='data', \
                    xytext=(-50, j*5), textcoords='offset points',\
                    arrowprops=dict(arrowstyle="->"))
        j = j + 1
        
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    return 

draw_clusters(km)