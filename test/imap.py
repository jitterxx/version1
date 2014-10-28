#!/usr/bin/python -t
# coding: utf8

import imaplib
import email
import pandas as pd
from bs4 import BeautifulSoup
import chardet
import re
import pymorphy2
from time import time



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

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


debug = True

def get_text(data):
    #Извлекаем из html сообщения только текст
    soup = BeautifulSoup(data,from_encoding="utf8")

    # Содержимое ссылок заменяем на LINK
    tag = soup.new_tag(soup,"b")
    tag.string = 'LINK'
    for link in soup.find_all('a'):
        link.replaceWith(tag)
    for link in soup.find_all('script'):
        link.replaceWith(tag)
    for link in soup.find_all(''):
        link.replaceWith(tag)
    
    text = soup.get_text()
    text = strip_text(text)
    
    return text

def strip_text(data):
    #Delete spec \t\n\r\f\v
    #text = re.sub(u'[\r|\n|\t]+',u' ',data,re.I|re.U|re.M)
    
    #Multiple spaces in one
    #text = re.sub(r'\s{2:}',r' ',data,re.I|re.U|re.M)
    #text = re.sub(u'\s+',u' ', text,re.I|re.U|re.M)
    #text = data.replace('\\n',' ')
    #text = text.replace('\\t',' ')    
    text = data.replace('\\W',' ')
    
    text = ' '.join(text.split())
    
    return text


def get_emails():
    server = "imap.gmail.com"
    port = "993"
    login = "sergey@reshim.com"
    password = ""
    date_after =  "21-oct-2014"
    date_before = "21-oct-2014"
    
    try:
        M = imaplib.IMAP4_SSL(server)
    except: 
        print 'Connection problem!', sys.exc_info()[0]
        
            
    
    print M.PROTOCOL_VERSION
    password = raw_input('Password:')
    
    M.login(login,password)
    M.select()
    
    
    typ, data = M.search('UTF8','SINCE',date_after)
    print typ

    s = {}
    
    
    for num in data[0].split():
        typ, data = M.fetch(num, '(RFC822)')
    
        msg = email.message_from_string(data[0][1])
        
        
        msg_data = {}
        for n,m in msg.items():
            k = ''
    
            if debug:
                #print m
                pass
            m = m.replace('?=<','?= <')
            m = strip_text(m)
            if debug and num == '6297':
                #print m
                pass
            
            for h in email.header.decode_header(m):
                
                k = ' '.join((k,h[0]))            
                if not (h[1] == None):            
                    #Делаем перекодировку в UTF8
                    k = k.decode(h[1]).encode('utf8')
                else:
                    #Проверяем что строка корректно перекодирована
                    if not (chardet.detect(k)['encoding'] == 'UTF-8'):
                        k = k.decode(chardet.detect(k)['encoding']).encode('utf8')
                
            if n in msg_data.keys():
                msg_data[n] = msg_data[n] + k
            else:
                msg_data[n] = k
        
        if debug:
            #print msg_data['From']
            pass
    
        
        msg_data['Text'] = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    html = unicode(part.get_payload(decode=True),
                                   part.get_content_charset(),
                                   'replace').encode('utf8','replace')
                    msg_data['Text'] +=  get_text(html)
                elif part.get_content_type() == "text/html":
                    html = unicode(part.get_payload(decode=True),
                                   part.get_content_charset(),
                                   'replace').encode('utf8','replace')
                    msg_data['Text'] +=  get_text(html)
    
        else:
            if msg.get_content_type() == "text/plain":
                html = unicode(msg.get_payload(decode=True),
                               msg.get_content_charset(),
                               'replace').encode('utf8','replace')
                msg_data['Text'] +=  get_text(html)
            elif msg.get_content_type() == "text/html":
                html = unicode(msg.get_payload(decode=True),
                               msg.get_content_charset(),
                               'replace').encode('utf8','replace')
                msg_data['Text'] +=  get_text(html)
            #print 'HTML:',html
            #print 'TEXT:',msg_data['Text']       
        
        
        s[num] = pd.Series(msg_data.values(),msg_data.keys())

    M.close()
    M.logout()

    return pd.DataFrame(s)


email_df = get_emails()

f2 = open("email_df", 'w')
email_df.to_json(f2,orient='columns')
f2.close()
"""


f2 = open("email_df.old", 'r')
email_df = pd.read_json(f2)
f2.close()
"""
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
vectorizer1 = HashingVectorizer(tokenizer=tokenizer_1)

print("Vectorization...")
t0 = time()

v = vectorizer.fit_transform(text)
#v1 = vectorizer1.fit_transform(text)
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

    



import matplotlib.pyplot as plt

"""
p = pd.DataFrame(tv,index=index,columns=list('XY'))
#p1 = pd.DataFrame(tv1,index=index,columns=list('XY'))


for i in p.index:
    dot = p.loc[i]
    plt.scatter(dot.X,dot.Y,alpha=0.5,marker='o',color='red')
    plt.annotate(str(i), xy=dot,  xycoords='data', \
                xytext=(50*dot.Y, 30*dot.X), textcoords='offset points',\
                arrowprops=dict(arrowstyle="->"))
"""
from sklearn.cluster import KMeans
n_clusters=8

print('Do clustering...')
km = KMeans(init='k-means++', max_iter=100, n_init=1,n_clusters=n_clusters)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(v)
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

print docs

    


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