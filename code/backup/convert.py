import csv
import os
import urllib.parse
from pprint import pprint


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

    data = loadFile('benign.txt')

    thefile = open('new.txt', 'w')
    for item in data:
            thefile.write(item)

main()