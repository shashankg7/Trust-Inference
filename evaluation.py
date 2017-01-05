# import igraph as ig
from __future__ import print_function
import numpy as np
from copy import deepcopy
import random
from data_handler1 import gen_data
import pdb

# from data_handler import data_handler
from model1 import user2vec
import sys
# n = data.shape[0]
# data = data_handler("rating_with_timestamp.mat", "trust.mat",
# "rating_with_timestamp.mat")
from test_social2vec import load_model, test


def trustPredict(G, U, batch_size, n):
	i = 100
	h = 64
	d = 32

	n_epochs = 10
	lr = 0.4
	print(n)
	u2v = user2vec(n, h, d, i)
	# u2v.model1()
	u2v.model_batch_uu()
	ind = 0
	f = open('Data.txt','r')
	batch = []
	res = []
	for epoch in xrange(n_epochs):
		batch = []
		print("Initiating epoch %d"%epoch)
		if epoch % 10 == 0:
			lr = lr/(1 + epoch * 0.001)
		with open('Data.txt', 'r') as f:
			for line in f:
				data1 = line.strip()
				data1 = map(lambda x:int(x), data1.split())
				if len(data1) == 3:
					batch.append(data1)
				else:
					print(":(((((((((((((((((((((((((((((((((((((((")
					#pdb.set_trace()
					continue
				if (ind + 1) % batch_size == 0:
					#print(batch)
					print(len(batch))
					try:
						batch = np.array(batch).astype(np.int32)
					except Exception as E:
						print("in batch exception")
						#pdb.set_trace()
					try:
						cost = u2v.uu_batch(batch[:, 0:2], batch[:, 2], lr)
					   # cost1 = u2v.debug(batch[:, :2])
					   # cost2 = u2v.debug1(batch[:, :2], batch[:, 2])
						print(cost)
					   # pdb.set_trace()
					   # if max(batch[:, 0]) >= n or max(batch[:, 1]) >= n:
					   #    print " in buggy region"
					   #    pdb.set_trace()
					   # assert max(batch[:, 0]) > n and max(batch[:, 1]) > n
						batch = []
					except Exception as e:
						print(str(e))
						print("in exception, check batch")
					   # pdb.set_trace()
				ind += 1
		#pdb.set_trace()
		u2v.get_params(epoch)
		load_model(epoch)
		l = test(U)
		print(epoch)
		#print(l)
		#pdb.set_trace()
		res.append(l)
	#print "UU training completed"
	return res


def evalu(edges, n, x, factor):
	index = int(round(len(edges)*x))
	O = edges[0:index, 0:2]
	N = edges[index:len(edges), 0:2]

	G = np.zeros((n, n))
	for e in edges:
		G[e[0], e[1]] = 1.0

	B = []
	i = 0
	while i < factor*len(N):
	# for i in range(0, factor*len(N)):
		l = random.randint(0,n-1)	
		r = random.randint(0,n-1)	
		if G[l,r] == 1:
			continue
		B.append([l, r])
		i = i + 1

	for e in N:
		G[e[0], e[1]] = 0.0

	gen_data(G)
	
	U = np.vstack((N, B))
	#print(trustPredict(G, U, 64, n))
	#pdb.set_trace()
	TValuesList = trustPredict(G, U, 64, n)
	pdb.set_trace()
	# TValues = G[U[:,0],U[:,1]]
	for TValues in TValuesList:
		UTrust = np.c_[U, TValues]

		UTrust = UTrust[UTrust[:,2].argsort()[::-1]]

		topN = UTrust[0:len(N),0:2]

		intersec = np.array([x for x in set(tuple(x) for x in N) & set(tuple(x) for x in topN)])

		acc = len(intersec)*100.0/len(N)

		print(acc)

fname  = "trustDataTimestamp.txt"

f = open(fname, 'r')

n  = 0
nodeHash = {}
nEdges = 0
edgesList = []

for line in f:
	components = line.split()
	if components[0] not in nodeHash:
		nodeHash[components[0]] = n
		n = n + 1

	if components[1] not in nodeHash:
		nodeHash[components[1]] = n
		n = n + 1

	edgesList.append([nodeHash[components[0]], nodeHash[components[1]], int(components[2])])

	nEdges = nEdges + 1


print(n, nEdges)

# sEdges = sorted(edges, key=getkey)

edges = np.array(edgesList)
del edgesList

edges = edges[edges[:,2].argsort()]
#[::-1]]

evalu(edges, n, 0.4, 4)

f.close()

# def getkey(item):
#   return item[2]

