#!/usr/bin/python -t
# coding: utf8

import imaplib
import email
import pandas as pd
from bs4 import BeautifulSoup
import chardet
import re


import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def get_text(data):
    #Извлекаем из html сообщения только текст
    soup = BeautifulSoup(data)
    return soup.get_text()


server = "imap.gmail.com"
port = "993"
login = "sergey@reshim.com"
password = ""
date_after =  "21-oct-2014"
date_before = "21-oct-2014"

M = imaplib.IMAP4_SSL(server)

print M.PROTOCOL_VERSION
password = raw_input('Password:')

M.login(login,password)
M.select()


typ, data = M.search('UTF8','SINCE',date_after)
print typ
#print data


#typ1, data1 = M.thread('REFERENCES','UTF-8','ALL')
#print typ1
#print data1

s = {}


for num in data[0].split():
    typ, data = M.fetch(num, '(RFC822)')

    #print data[0][0]
    #print data[0][1]

    msg = email.message_from_string(data[0][1])
    
    msg_data = {}
    for n,m in msg.items():
        k = ''
        m = re.sub(u'[\n\r\t]',' ',m)
        for h in email.header.decode_header(m):
            k = ' '.join((k,h[0]))            
            if not h[1] == None:            
                #Делаем перекодировку в UTF8
                k = k.decode(h[1]).encode('utf8')
            else:
                #Проверяем что строка корректно перекодирована
                if not chardet.detect(k)['encoding'] == 'UTF-8':
                    k = k.decode(chardet.detect(k)['encoding']).encode('utf8')
            typ, data = M.fetch(num, '(RFC822)')

        if n in msg_data.keys():
            msg_data[n] = msg_data[n] + k
        else:
            msg_data[n] = k
    
    
    if msg.is_multipart():
        msg_data['Text'] = msg.get_payload(0)
        
    else:
        msg_data['Text'] = msg.get_payload()
        #print type(body_data),' is NOT multipart'
        #print header_data['Content-Transfer-Encoding']
        #print get_text(body_data)
    
    
    s[num] = pd.Series(msg_data.values(),msg_data.keys())
    d = pd.Series(msg_data.values(),msg_data.keys())


email_df = pd.DataFrame(s)

print email_df.loc['Subject']
print email_df.loc['From']
print email_df.loc['To']
#print email_df.loc['References']
#st = email_df['6286']['From']
#print st

#typ, data = M.fetch('6286', '(RFC822)')
#msg = email.message_from_string(data[0][1])

#print msg['From']


#st = re.sub(u'[\n\r\t]',' ',st)
#print email.header.decode_header(st)


#print email_df.columns
    
    #print msg['From']
    #print msg['Date']    
    #print msg['Subject']
    #print msg['References']
    #print msg.keys()

    
    
    
    #print 'Message %s\n%s\n' % (num, data[0][1])
M.close()
M.logout()



