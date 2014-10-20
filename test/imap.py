#!/usr/bin/python -t
# coding: utf8

import imaplib
import sys
reload(sys)
sys.setdefaultencoding("utf-8")



server = "imap.gmail.com"
port = "993"
login = "sergey@reshim.com"
password = ""
date_after =  "18-oct-2014"
date_before = "21-oct-2014"

M = imaplib.IMAP4_SSL(server)

print M.PROTOCOL_VERSION
password = raw_input('Password:')

M.login(login,password)
M.select()


typ, data = M.search('UTF8','SINCE',date_after, 'BEFORE',date_before)
print typ
print data


#typ1, data1 = M.thread('REFERENCES','UTF-8','ALL')
#print typ1
#print data1



for num in data[0].split():
    typ, data = M.fetch(num, '(RFC822)')
    print data[0][0]
    print "--------------------------------------------------"
    print data[0][1]
    
    
    #print 'Message %s\n%s\n' % (num, data[0][1])
M.close()
M.logout()


