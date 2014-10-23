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

    # Содержимое ссылок заменяем на LINK
    tag = soup.new_tag(soup,"b")
    tag.string = 'LINK'
    for link in soup.find_all('a'):
        link.replaceWith(tag)
        #print link
        
    text = soup.get_text()
    text = strip_text(text)

    return text

def strip_text(data):
    #Delete spec \t\n\r\f\v
    #text = re.sub(r'[\r|\n|\t]',r' ',data,re.I|re.U|re.M)
    
    #Multiple spaces in one
    #text = re.sub(r'\s{2:}',r' ',data,re.I|re.U|re.M)
    text = re.sub('\s+',' ', data,re.I|re.U|re.M).strip()
    return text

debug = True

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

    msg = email.message_from_string(data[0][1])
    
    if debug:
        print num
    
    msg_data = {}
    for n,m in msg.items():
        k = ''
        if debug:
            #print m
            pass

        m = re.sub(u'[\n|\r|\t]',' ',m)
        if debug:
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
    
    
    s[num] = pd.Series(msg_data.values(),msg_data.keys())


email_df = pd.DataFrame(s)

#pd.set_option('display.max_colwidth',1000)

#print email_df.loc['To']
#print email_df.loc['From']
#print email_df.loc['Subject']
#print email_df.loc['Text']



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

f2 = open("email_df", 'w')

email_df.to_json(f2,orient='columns')


f2.close()

