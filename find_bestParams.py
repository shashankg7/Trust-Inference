
import numpy as np
import sys
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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


def inference():
    # Loading testing data
    f = open('test.txt', 'r')
    batch = []

    def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e)
        return dist

    for line in f:
        data = line.strip()
        data = map(lambda x:float(x), data.split())
        batch.append(data)

    batch = np.array(batch).astype(np.int32)
    X = batch[:, :2]
    Y = batch[:, 2]


    # Running the data through the model
    U = Wu[X[:, 0], :]
    V = Wu[X[:, 1], :]

    hLm = U * V
    hLp = abs(U - V)

    hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
    x = np.dot(U1, hL) + B21
    l = softmax(x)

    yp = np.argmax(l, axis=0)
    #print "accuracy"

    print("Accuracy is %f"%(accuracy_score(Y, yp)))
    return accuracy_score(Y, yp)

    #print "precision"
    #print precision_score(Y, yp)

    #print "recall"
    #print recall_score(Y, yp)

    #print "f1 score"
    #print f1_score(Y, yp)

    ##pdb.set_trace()



def test(epochs):
    # Load saved params for prediction
    accs = []
    for epoch in range(epochs):
        load_model(epoch)
        accs.append(inference())

    accs = np.array(accs)
    best_model = np.argmax(accs)
    print(best_model)
    print("Inference done on all models")
    load_model(best_model)
    acc = inference()
    print("Acc is %f"%(acc))
    return best_model


if __name__ == "__main__":
    epochs = sys.argv[1]
    test(int(epochs))

