

from __future__ import print_function
import numpy as np
import time

x= np.random.rand(22166, 22166)
n = 22166
t = time.time()

for i in xrange(n):
    for j in xrange(n):
        temp = np.dot(x[i, :], x[j, :])

print("time taken is")
print(time.time() - t)
