
import numpy as np
import sys
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from networkx import read_edgelist
import networkx as nx

path = ''
Wu = []
Wm1 = []
Wp1 = []
B11 = []
B21 = []
U1 = []


def load_model(epoch_no):
    global Wu, Wm1, Wp1, B11, B21, U1
    path = './model/model' + str(epoch_no)
    Wu = np.load(path + '/Wu.npy')
    Wm1 = np.load(path + '/Wm1.npy')
    Wp1 = np.load(path + '/Wp1.npy')
    B11 = np.load(path + '/B11.npy')
    B21 = np.load(path + '/B21.npy')
    U1 = np.load(path + '/U1.npy')



def test_segment():
    # Loading testing data
    f = open('test.txt', 'r')
    graph = read_edgelist('./edge_Data', create_using=nx.DiGraph())
    indegree = graph.in_degree()
    outdegree = graph.out_degree()
    indegree_counts = indegree.values()
    outdegree_counts = outdegree.values()
    pdb.set_trace()
    eps1 = int(np.mean(indegree_counts))
    eps2 = int(np.mean(outdegree_counts))


    print("median for indegree is %d"%eps1)
    print("media for outdegree is %d"%eps2)


    segmented_users_i = [int(k) for k, v in indegree.iteritems() if v >= eps1]
    segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v >= eps2]

    print("number of users with high indegree are %d"%len(segmented_users_i))
    print("number of users with high outdegree are %d"%len(segmented_users_o))


    segmented_users_i = [int(k) for k, v in indegree.iteritems() if v < eps1]
    segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v < eps2]
    print("number of users with low indegree are %d"%len(segmented_users_i))
    print("number of users with low outdegree are %d"%len(segmented_users_o))


    pdb.set_trace()


def test(epochs, k):
    # Load saved params for prediction
    test_segment()


if __name__ == "__main__":
    epochs = sys.argv[1]
    k = int( sys.argv[2])
    test(int(epochs), k)

