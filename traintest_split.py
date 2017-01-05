
import numpy as np
import pdb
from sklearn.cross_validation import train_test_split


R = []
C = []
D = []

f = open('Trust.txt','r')
f1 = open('train_80_80', 'w')
f2 = open('test_80_80.txt', 'w')

for x in f.readlines():
    x = x.strip()
    r,c,d = map(lambda y:float(y), x.split())
    R.append(r)
    C.append(c)
    D.append(d)

R = np.array(R)
C = np.array(C)
D = np.array(D)
data = np.vstack((R, C, D)).astype(np.int32).T

train, test = train_test_split(data, test_size=0.8, random_state=42)

for row in train:
    line = str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n'
    f1.write(line)

for row in test:
    line = str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n'
    f2.write(line)

print "training testing split done"
pdb.set_trace()
