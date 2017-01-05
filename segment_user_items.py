import numpy as np
from scipy.io import loadmat
import collections
import math
from collections import OrderedDict
import pdb
from scipy.sparse import coo_matrix
from networkx import read_edgelist



class data_handler():

    def __init__(self,rating_path,trust_path,time_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.time_path = time_path
        self.n = 0
        self.k = 0
        self.d = 0
        #BRING G BACK BEFORE RUNNING MAIN--------------------

    def load_matrices(self, test=0.2, i_thres=50, d_eps=10):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        f3 = open(self.time_path)

        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
        G_raw = G_raw - 1
        f = open('edge_list_trust.txt', 'r')
        #for row in G_raw:
        #    line = str(row[0]) + '\t' + str(row[1]) + '\n'
        #    f.write(line)
        # Number of users
        graph = read_edgelist('./edge_list_trust.txt')
        degree = graph.degree()
        pdb.set_trace()
        self.n = G_raw.max() + 1
        # number of items
        self.i = max(P_initial[:,1])
        # user and item and rating vectors from P matrix
        U = P_initial[:, 0]
        I = P_initial[:, 1]
        U = U-1
        I = I-1
        R = P_initial[:, 3]
        R = R/float(5)
        self.T1 = np.vstack((U, I, R)).T.astype(np.int32)
        self.T1 = self.T1.astype(np.int32)
        users = np.unique(self.T1[:, 0])
        segmented_users_i = []
        segmented_users_d = []
        for user in users:
            l = np.where(self.T1[:, 0] == user)
            L = l[0].shape[0]
            if L > i_thres:
                segmented_users_i.append(user)
        
        for k, v in degree.items():
            if v > d_eps:
                segmented_users_d.append(int(k))

        #self.UI = coo_matrix((R, (U, I)))
        pdb.set_trace()

data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
data.load_matrices()








