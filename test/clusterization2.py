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
labels = target
print 'Train samples count: ',train_text.shape
print 'Train target vector: ',target.shape

    

raw_input('...')


vectorizer = TfidfVectorizer(tokenizer=tokenizer_1,use_idf=True,\
                            max_df=0.95, min_df=2,max_features=100)


print("Vectorization...")
t0 = time()

train_v = vectorizer.fit_transform(train_text)
v = vectorizer.transform(text)

print 'Test vector length: ', v.shape
print 'Train vector length: ', train_v.shape

print('Done in %fs' % (time() - t0))


n_clusters=len(category)
print 'Cluster number: ', n_clusters

raw_input('Clustering?')

print('Do clustering...')
km = KMeans(init='k-means++', max_iter=100, n_init=10,n_clusters=n_clusters)


#Обучаем модель
print('Fit model...')

t0 = time()
km.fit(train_v,target)
print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      % ('Train', (time() - t0), km.inertia_,
         metrics.homogeneity_score(labels, km.labels_),
         metrics.completeness_score(labels, km.labels_),
         metrics.v_measure_score(labels, km.labels_),
         metrics.adjusted_rand_score(labels, km.labels_),
         metrics.adjusted_mutual_info_score(labels,  km.labels_),
         metrics.silhouette_score(train_v, km.labels_,
                                  metric='euclidean',
                                  sample_size=300)))


#print("Clustering sparse data with %s" % km)
print('Predicting...')
t0 = time()

Z = km.predict(v)
print("done in %0.3fs \n" % (time() - t0))

print 'Predicted category: ',Z
print Z.shape

raw_input('Draw?')

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
    for j in range(len(Z)):
        if i == Z[j]:
            docs[i].append(index[j])
            print email_df.loc['From',index[j]]


"""    #Make samples sort by cluster
    for j in range(len(km.labels_)):
        if i == km.labels_[j]:
            docs[i].append(index[j])
            print email_df.loc['From',index[j]]

   """


def draw_clusters(km,data):
    
    tv = PCA(n_components=2).fit_transform(km.cluster_centers_)
    #rd = PCA(n_components=2).fit_transform(data)    
    
    fig, ax = plt.subplots()
    ax.set_title('Clusters')
    j = 0
    for i in tv:
        ax.scatter(i[0],i[1],alpha=0.5,marker='o',color='green')
        ax.annotate('Cluster '+str(j), xy=(i[0],i[1]),  xycoords='data', \
                    xytext=(-50, j*5), textcoords='offset points',\
                    arrowprops=dict(arrowstyle="->"))
        j = j + 1

    #for i in rd:
    #    ax.scatter(i[0],i[1],alpha=0.5,marker='x',color='red')
        
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    return 

draw_clusters(km,v)