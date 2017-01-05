import numpy as np
from scipy.io import loadmat
import collections
import math
import os
from collections import OrderedDict
import pdb
from scipy.sparse import coo_matrix
from subprocess import call

def gen_data(G):
    #Loading matrices from data
    G_raw = np.where(G == 1)

    #Writing all positive samples to the file
    f = open('trust.txt', 'w')
    for ind in xrange(len(G_raw[0])):
        row = G_raw[0][ind]
        col = G_raw[1][ind]
        data = 1
        f.write(str(row) + "\t" + str(col) + "\t" + str(data) + '\n')
    n = len(G)
    print "data generated till now, now generating negative samples"
    #pdb.set_trace()
    # List of users in training data (G_raw)
    user_list = np.unique(np.sort(G_raw[0][:]))
    # Generate negative samples for training
    for i in user_list:
        # list of all indices in the row
        ind = np.arange(n)
        try:
            nonzero = np.where(G[i, :] > 0)[0]
            # Check if a user has no trustee
            assert len(nonzero) > 0
            #pdb.set_trace()
            # number of non-zero elements
            m = len(nonzero)
            # remove all positive samples (nonneg.) from the list
            zeros = np.setdiff1d(ind, nonzero)
            # Randomly select m negative samples from the leftout list
            #print "pre-processing done, now randomly sample data"
            #pdb.set_trace()
            neg_ind = np.random.randint(len(zeros),size=(m))
            neg_samples = zeros[neg_ind]
            #pdb.set_trace()
            # writing negative samples to file
            for neg in neg_samples:
                f.write(str(i) + "\t" + str(neg) + "\t" + str(0) + '\n')
        except Exception as e:
            print str(e)
    os.system("shuf trust.txt > Data.txt")


# data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
# data.load_matrices()








