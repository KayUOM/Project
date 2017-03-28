import pandas as pd
from sklearn.utils import shuffle

maliciousDF = pd.read_csv("malicious.csv", names=['code'], header=None, error_bad_lines=False)
maliciousDF['status']='1'
benignDF = pd.read_csv("benign.csv", names=['code'], header=None, error_bad_lines=False)
benignDF['status']='0'

data = maliciousDF.append(benignDF, ignore_index=True)

data = shuffle(data)


print(data)

data.to_csv('combine.csv',  index=False)

