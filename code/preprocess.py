import csv
import os
import urllib.parse
from pprint import pprint
import pandas as pd
from sklearn.utils import shuffle



def loadFile(name):

    directory = os.getcwd()
    filepath = directory + "/" + name
    data = open(filepath,'r').readlines()

    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))
        result.append(d)
    return result



def main():

    malicious = loadFile('malicious.txt')
    benign = loadFile('benign.txt')


    thefile = open('malicious.csv', 'w')
    for item in malicious:
            thefile.write(item)

    thefile = open('benign.csv', 'w')
    for item in benign:
        thefile.write(item)

    maliciousDF = pd.read_csv('malicious.csv', names=['code'], header=None, error_bad_lines=False)
    maliciousDF['status'] = '1'
    benignDF = pd.read_csv('benign.csv', names=['code'], header=None, error_bad_lines=False)
    benignDF['status'] = '0'

    data = maliciousDF.append(benignDF, ignore_index=True)

    data = shuffle(data)

    data.to_csv('data.csv', index=False)


    # pprint(benign)
    #
    # maliciousDF = pd.DataFrame(malicious, columns=['code'])
    # maliciousDF['status'] = 1
    # benignDF = pd.DataFrame(benign, columns=['code'])
    # benignDF['status'] = 0


    # data = maliciousDF.append(benignDF, ignore_index=True)

    # print(data)
    #
    # data = shuffle(data)
    # data.to_csv('data.csv', index=False)


main()