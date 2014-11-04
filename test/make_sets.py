# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:35:05 2014

@author: sergey
"""

import re
import pymorphy2
from time import time
import pandas as pd




import sys
reload(sys)
sys.setdefaultencoding("utf-8")


debug = True


f2 = open("email_df.to_train", 'r')
email_df = pd.read_json(f2)
f2.close()

category = {0:'Personal',1:'Notification',2:'Promo'}

#Создаем вектор с принадлежностью каждого письма к кластеру
msg_num = email_df.columns
print email_df.shape

target = pd.DataFrame(columns=msg_num,index=['Target'])

for num in msg_num:
    print email_df[num]['Text']
    print category
    str = raw_input('Категория :')
    print '\n'
    target[num]['Target'] = str


train = email_df.append(target.iloc[0])

f2 = open("email_df.train", 'w')
train.to_json(f2)
f2.close()
