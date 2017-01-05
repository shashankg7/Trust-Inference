
import numpy as np
# Import logistic regrs
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

Wu = 0
Wm1 = 0
Wp1 = 0
B11 = 0
B21 = 0
U1 = 0

def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e)
        return dist

def load_model(epoch_no):
    # Load saved params for prediction
    global Wu, Wm1, Wp1, B11, B21, U1
    path = './model/model' + str(epoch_no)
    Wu = np.load(path + '/Wu.npy')
    Wm1 = np.load(path + '/Wm1.npy')
    Wp1 = np.load(path + '/Wp1.npy')
    B11 = np.load(path + '/B11.npy')
    B21 = np.load(path + '/B21.npy')
    U1 = np.load(path + '/U1.npy')
    # Loading testing data
    f = open('test.txt', 'r')
    batch = []

    
def test(T):
    # Running the data through the model
    l = []
    for row in T:
        u = row[0]
        v = row[1]
        U = Wu[u, :]
        V = Wu[v, :]
        U = U.reshape((1,len(U)))
        V = V.reshape((1,len(V)))
        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        #pdb.set_trace()
        l.append(softmax(x)[1])
    pdb.set_trace()
    return np.array(l)
    # yp = np.argmax(l, axis=0)
    # #print "accuracy"
    # return accuracy_score(Y, yp)

    #print "precision"
    #print precision_score(Y, yp)

    #print "recall"
    #print recall_score(Y, yp)

    #print "f1 score"
    #print f1_score(Y, yp)

    ##pdb.set_trace()

if __name__ == "__main__":
    print(test())

