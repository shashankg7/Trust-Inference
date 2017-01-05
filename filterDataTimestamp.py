import numpy as np
import scipy.io as sio
import pdb
# tData = sio.loadmat("epinions/trustnetwork.mat")
# rData = sio.loadmat("epinions/rating.mat")

tData = sio.loadmat("./trust_with_timestamp.mat")
rData = sio.loadmat("./rating_with_timestamp.mat")

pdb.set_trace()

# ff = open('t.txt', 'w')

# for i in rData['rating']:
# 	print>>ff, i

# ff.close()
# fTemp = open("trustDumy",'w')

fo = open("trustDataTimestamp.txt",'w')

# fname = "./trustData/trust.txt"
# rname = "./trustData/rating.txt"
trustorThreshold = 3
itemThreshold = 2
uRThreshold = 2

# with open(fname) as f:
#     content = f.read().splitlines()

# with open(rname) as fr:
#     icontent = fr.read().splitlines()

itemDict = {}
uIDict = {}
uRCountDict = {}

for values in rData['rating']:
	# values = line.split("\t")

	if (values[0], values[1]) not in uIDict:

		if values[1] not in itemDict:
			itemDict[values[1]] =  1
		else:
			itemDict[values[1]] =  itemDict[values[1]] + 1

	uIDict[(values[0], values[1])] = values[3]/5.0
print "1"  


# print "ItemDict"
# for key, val in itemDict.items():
# 	print (key, " : ", val	)
# print "---------"

# print "UIDict"
# for key, val in uIDict.items():
# 	print (key, " : ", val	)
# print "---------"

for key, val in uIDict.items():
	if itemDict[key[1]] < itemThreshold:
		del uIDict[key]
	else:
		if key[0] not in uRCountDict:
			uRCountDict[key[0]] =  1
		else:
			uRCountDict[key[0]] =  uRCountDict[key[0]] + 1

print "2"  

# print >> fo, "URCountDict"
# for key, val in uRCountDict.items():
# 	print >> fo, (key, " : ", val)
# print >> fo, "---------"

# print "UIDict"
# for key, val in uIDict.items():
# 	print (key, " : ", val	)
# print "---------"


trustorDict = {}
tDict = {}

for values in tData['trust']:
	# values = line.split("\t")
	# fTemp.write("%s %s\n" % (values[0],values[1]))
	if values[0] == values[1]:
		continue

	if (values[0], values[1]) in tDict:
		tDict[(values[0], values[1])] = values[2]
		continue

	if values[1] not in trustorDict:
		trustorDict[values[1]] =  1
	else:
		trustorDict[values[1]] =  trustorDict[values[1]] + 1

	tDict[(values[0], values[1])] = values[2]

# print tList
print "3"  

# print "indegree filtered"
# for key, val in trustorDict.items():
# 	if val < trustorThreshold:
# 		print (key, " : ", val	)
# print "---------"

count = 0
indexDict = {}
tTuple = []

edgeCount = 0;
for x, y in tDict.items():
	# print  >> fo, x[0], x[1]
	if ((x[0] not in uRCountDict) or (uRCountDict[x[0]] < uRThreshold) or (x[1] not in uRCountDict) or (uRCountDict[x[1]] < uRThreshold)):
		continue

	if ((x[0] not in trustorDict) or (trustorDict[x[0]] < trustorThreshold) or (x[1] not in trustorDict) or (trustorDict[x[1]] < trustorThreshold)):
		continue

	if not x[0] in indexDict:
		indexDict[x[0]] =  count
		count = count + 1

	if not x[1] in indexDict:
		indexDict[x[1]] =  count
		count = count + 1	

	fo.write("%s %s %s\n" % (indexDict[x[0]],indexDict[x[1]],y))

	tTuple.append([indexDict[x[0]],indexDict[x[1]]])
	edgeCount = edgeCount + 1

print "4"  

# tMat = np.zeros((count, count))

# for x in tTuple:
#     tMat[x[0]][x[1]] = 1

# fo.write("%s %s\n" % (count, edgeCount))
print("%s %s\n" % (count, edgeCount))

fo.close()
# fTemp.close()
