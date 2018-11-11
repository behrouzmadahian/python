from snakebite.client import Client
client=Client('localhost',8020) #port is the RPC port of the namenode.
for i in client.ls(['/user/cloudera/behrouz']): #takes a list of paths!!
     print i
#get this parameters from /etc/hadoop/conf/core-site.xml under the fs.defaults
#many of the methods in snake bite return generators

#creating a directory:
#create two directories behrouz, behrouz1/b1 on HDFS:
print '*'*40
for p in client.mkdir(['/behrouz','behrouz1/b1'],create_parent=True):
    print p
print '*'*40
#deleting files and directories: deletes any subdirectories and files a directory contains
#recursively deleting the directories!
for p in client.delete(['/behrouz','behrouz1/b1'],recurse=True):
    print p
print '*'*40
# retrieving data from hdfs:
#copying files from HDFS to Local file system:
for f in client.copyToLocal(['/user/cloudera/wordCount.out'],'/home/cloudera/'):
     print f
print '*'*40
#######
#reading contents of a file 
for l in client.text(['/user/cloudera/testfile.txt']):
    print l
#the text method automatically decompress and display gzip and bzip2 files.


