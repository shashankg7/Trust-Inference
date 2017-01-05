from theano import tensor as T
import theano
import numpy as np
#from preprocess import data
from theano.compile.nanguardmode import NanGuardMode
import pdb

class user2vec(object):
    def __init__(self, n_user, d, h, n_item):
        self.n_user = n_user 
        self.d = d
        self.h = h
        self.n_item = n_item 
        # Shared parameter (user embedding vector)
        self.Wu = theano.shared(np.random.uniform(low = - np.sqrt(6.0/float(n_user + d)),\
                                   high =  np.sqrt(6.0/float(n_user + d)),\
                                   size=(n_user,d)).astype(theano.config.floatX))
        # Item embedding matrix
        self.Wi = theano.shared(np.random.uniform(low = - np.sqrt(6.0/float(n_item + d)),\
                                   high =  np.sqrt(6.0/float(n_item + d)),\
                                   size=(n_item ,d)).astype(theano.config.floatX))

        self.W1 = self.Wu
        self.W2 = self.Wi
        self.W3 = self.Wu
        # Paramters for user-user model
        self.Wm1 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.Wp1 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        # Param for single example model
        self.b11 = theano.shared(np.zeros((h), dtype=theano.config.floatX))
        # Param for batch model
        self.B11 = theano.shared(np.zeros((h,1), dtype=theano.config.floatX), broadcastable=(False, True))

        # Param for single example model
        self.b21 = theano.shared(np.zeros((2), dtype=theano.config.floatX))
        # Param for batch model
        self.B21 = theano.shared(np.zeros((2,1), dtype=theano.config.floatX), broadcastable=(False, True))

        self.U1 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
                                              high = np.sqrt(6.0/float(2 + h)),
                                              size=(2,h)).astype(theano.config.floatX))

        # Parameters for user-item model
        self.Wm2 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.Wp2 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.b12 = theano.shared(np.zeros((h), dtype=theano.config.floatX))
        # Mini batch model param
        self.B12 = theano.shared(np.zeros((h,1), dtype=theano.config.floatX), broadcastable=(False, True))


        #elf.b22 = theano.shared(np.zeros((2), dtype=theano.config.floatX))
        # Mini batch model param
        #elf.B22 = theano.shared(np.zeros((2), dtype=theano.config.floatX), broadcastable=(False, True))

        self.U2 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
                                              high = np.sqrt(6.0/float(2 + h)),
                                              size=(1,h)).astype(theano.config.floatX))

        self.params1 = [self.Wm1, self.Wp1, self.b11, self.b21, self.U1]
        self.Params1 = [self.Wm1, self.Wp1, self.B11, self.B21, self.U1]

        self.params2 = [self.Wm2, self.Wp2, self.b12, self.U2]
        self.Params2 = [self.Wm2, self.Wp2, self.B12, self.U2]

    def model_batch_ui(self, lr=0.01, reg_coef=0.1):
        # U-I model
        ui = T.imatrix()
        yi = T.vector()

        U1 = self.Wu[ui[:, 0], :]
        I = self.Wi[ui[:, 1], :]

        hLm1 = U1 * I
        hLp1 = abs(U1 - I)

        hL1 = T.tanh(T.dot(self.Wm2, hLm1.T) + T.dot(self.Wp2, hLp1.T) + self.B12)
        l1 = T.dot(self.U2, hL1)

        self.debug1 = theano.function([ui], l1, allow_input_downcast=True)
        cost1 = T.sum((l1 - yi) ** 2) + reg_coef * ( T.sum(self.Wm2 ** 2) + T.sum(self.Wp2 ** 2) \
                + T.sum(self.U2 ** 2))
        grad2 = T.grad(cost1, [U1, I])
        grads1 = T.grad(cost1, self.Params2)
        #print grads1

        # NOTE : THIS UPDATE WOULD REWRITE UPDATE FROM THE OTHER MODEL BECAUSE IT WILL UPDATE WU WITH CURRENT MODEL'S UPDATE
        self.W3 = T.set_subtensor(self.W3[ui[:, 0], :], self.W3[ui[:, 0], :] - lr * grad2[0])
        self.W2 = T.set_subtensor(self.W2[ui[:, 1], :], self.W2[ui[:, 1], :] - lr * grad2[1])

        updates21 = [(self.Wu, self.W3)]
        updates22 = [(self.Wi, self.W2)]
        #updates23 = [(self.W1, self.W3)]
        updates24 = [(param, param - lr * grad) for (param, grad) in zip(self.Params2, grads1)]
        #pdb.set_trace()
        updates2 = updates21 + updates22 + updates24# + updates24
        param_norm = T.sum(self.Wu ** 2)
        self.debug1 = theano.function([], param_norm, allow_input_downcast=True)
        
        self.ui_batch = theano.function([ui, yi], cost1, updates=updates2, allow_input_downcast=True)


    def model_batch_uu(self, lr=10.0001):
        # U-U model
        # theano matrix storing node embeddings
        uu = T.imatrix()
        # Target labels for input
        yu = T.ivector()
        # Extract the word vectors corresponding to inputs
        U = self.Wu[uu[:, 0], :]
        V = self.Wu[uu[:, 1], :]
        hLm = U * V
        hLp = abs(U - V)
        hL = T.tanh(T.dot(self.Wm1, hLm.T) + T.dot(self.Wp1, hLp.T) + self.B11)
        # Likelihood
        l = T.nnet.softmax(T.dot(self.U1, hL) + self.B21)

        #elf.debug = theano.function([uu], l)
        #cost = T.mean(T.nnet.binary_crossentropy(l, yu))

        cost = -T.mean(T.log(l[yu, :]))
        #self.debug1 = theano.function([X,y], l)
        grad1 = T.grad(cost, [U,V])
        gradU = grad1[0]
        gradV = grad1[1]
        # Check norm of gradient, if it is moving or not
        self.debug_grad = theano.function([uu, yu], T.sum(gradV ** 2))

        #grad1[0].eval(np.random.randint(1000, size=(32,2)), np.random.randint(1, size=(32,)))
        grads = T.grad(cost, self.Params1)
        #updates1 = [(self.W1, T.inc_subtensor(self.W[X[:, 0]], grads[0]))]
        #updates2 = [(self.W, T.inc_subtensor(self.W1[X[:, 1]], grads[1]))]
        self.W1 = T.inc_subtensor(self.W1[uu[:,0], :], - lr * grad1[0])
        self.W1 = T.set_subtensor(self.W1[uu[:,1], :], self.W1[uu[:,1], :] - lr * grad1[1])
        updates11 = [(self.Wu, self.W1)]
        updates31 = [(param, param - lr * grad) for (param, grad) in zip(self.Params1, grads)]
        updates1 = updates11  + updates31
        param_norm = T.sum(self.Wu ** 2)
        self.debug = theano.function([], param_norm, allow_input_downcast=True)
        self.uu_batch = theano.function([uu,yu], cost, updates=updates1, allow_input_downcast=True) #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

    def model(self, lr=0.01):
        # Tuple for user-user-item
        uu = T.ivector()
        # Tuple for user-item
        #ui = T.ivector()
        yu = T.vector()
        #yi = T.scalar()

        u = self.Wu[uu[0], :]
        v = self.Wu[uu[1], :]
        i = self.Wi[uu[2], :]

        # Model for uu
        hm1 = u * v
        hp1 = abs(u - v)
        # Function to debug dimensions
        #self.debug = theano.function([uu], hm1)
        ## paramter for numerical stablility of log
        ##eps = T.scalar()
        h1 = T.tanh(T.dot(self.Wm1, hm1) + T.dot(self.Wp1, hp1) + self.b11)
        l = T.nnet.softmax(T.dot(self.U1, h1) + self.b21)
        #self.debug2 = theano.function([uu], l)
        # cost 1
        J1 = T.switch(T.eq(yu[1], 0), l[0], l[1])

        # Model for ui
        hm2 = u * i
        hp2 = abs(u - i)
        h2 = T.tanh(T.dot(self.Wm1, hm2) + T.dot(self.Wp2, hp2) + self.b12)
        l1 = T.dot(self.U2, h2)
        self.debug2 = theano.function([uu], l1)
        J2 = (l1 - yu[1]) ** 2

        cost = J1 + J2
        #cost = -T.log(l[0][y] + eps)
        grad1 = T.grad(cost, [u, v, i])
        grads = T.grad(cost, self.params)
        self.W1 = T.set_subtensor(self.W1[uu[0],:], self.W1[uu[0],:] - lr * grad1[0])
        self.W1 = T.set_subtensor(self.W1[uu[1], :], self.W1[uu[1],:] - lr * grad1[1])
        self.W2 = T.set_subtensor(self.W2[uu[2], :] - self.W2[uu[2], :] - lr * grad1[2])
        updates1 = [(self.Wu, self.W1)]
        updates2 = [(self.Wi, self.W2)]
        updates3 = [(param, param - lr * grad) for (param, grad) in zip(self.params, grads)]
        updates = updates1 + updates2 + updates3
        self.gd = theano.function([uu, yu], cost, updates=updates, mode='DebugMode')
        #self.gd = theano.function([uu,yu], cost, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))


    def model1(self, lr=0.01):
        # Tuple for user-user-item
        uu = T.ivector()
        ui = T.ivector()
        # Tuple for user-item
        #ui = T.ivector()
        yu = T.iscalar()
        yi = T.scalar()
        #yi = T.scalar()

        u1 = self.Wu[uu[0], :]
        v = self.Wu[uu[1], :]

        u2 = self.Wu[ui[0], :]
        i = self.Wi[ui[1], :]

       # Model for uu
        hm1 = u1 * v
        hp1 = abs(u1 - v)
        # Function to debug dimensions
        #self.debug = theano.function([uu], hm1)
        ## paramter for numerical stablility of log
        ##eps = T.scalar()
        h1 = T.tanh(T.dot(self.Wm1, hm1) + T.dot(self.Wp1, hp1) + self.b11)
        l = T.nnet.softmax(T.dot(self.U1, h1) + self.b21)
        # cost 1
        #J1 = T.switch(T.eq(yu, 0), -T.log(l[0][0]), -T.log(l[0][1]))
        J1 = -T.log(l[0][yu])
        #J1 = T.nnet.binary_crossentropy(l, yu)
        #self.debug1 = theano.function([uu, yu], J1)

        grad1 = T.grad(J1, [u1, v])
        grads1 = T.grad(J1, self.params1)
        self.W1 = T.set_subtensor(self.W1[uu[0],:], self.W1[uu[0],:] - lr * grad1[0])
        self.W1 = T.set_subtensor(self.W1[uu[1], :], self.W1[uu[1],:] - lr * grad1[1])



        # Model for ui
        hm2 = u2 * i
        hp2 = abs(u2 - i)
        h2 = T.tanh(T.dot(self.Wm2, hm2) + T.dot(self.Wp2, hp2) + self.b12)
        l1 = T.dot(self.U2, h2)
        J2 = (l1 - yi) ** 2
        J3 = T.sum(J2)
        #print type(J2)
        #self.debug2 = theano.function([ui, yi], [J3, J2], allow_input_downcast=True)
        grad2 = T.grad(J3, [u2, i])
        self.W2 = T.set_subtensor(self.W2[ui[0],:], self.W2[ui[0],:] - lr * grad2[0])
        self.W2 = T.set_subtensor(self.W2[ui[1], :], self.W2[ui[1],:] - lr * grad2[1])


        ##cost = -T.log(l[0][y] + eps)
        #grads1 = T.grad(J1, self.params1)
        #print zip(self.params1,grads1)
        grads2 = T.grad(J3, self.params2)
        #grads2 = T.grad(J3, self.params2)

        updates11 = [(self.Wu, self.W1)]
        updates21 = [(self.Wi, self.W2)]

        #print [(x[0], x[1]) for x in zip(self.params1, grads1)]
        #pdb.set_trace()
        updates1 = [(param, param - lr * grad) for (param, grad) in zip(self.params1, grads1)]
        updates2 = [(param, param - lr * grad) for (param, grad) in zip(self.params2, grads2)]
        #update12 = [(self.params1[0], self.params1[0] - lr * grads1[0])]
        #update13 = [(self.params1[1], self.params1[1] - lr * grads1[1])]
        #update14 = [(self.params1[2], self.params1[2] - lr * grads1[2])]
        #update15 = [(self.params1[3], self.params1[3] - lr * grads1[3])]
        #update16 = [(self.params1[4], self.params1[4] - lr * grads1[4])]

        #update22 = [(self.params2[0], self.params2[0] - lr * grads2[0])]
        #update23 = [(self.params2[1], self.params2[1] - lr * grads2[1])]
        #update24 = [(self.params2[2], self.params2[2] - lr * grads2[2])]
        #update25 = [(self.params2[3], self.params2[3] - lr * grads2[3])]
        #update26 = [(self.params2[4], self.params2[4] - lr * grads2[4])]


        #update1 = updates11 + updates12 + updates13 + updates14 + updates15 + updates16
        #update2 = updates21 + updates22 + updates23 + updates24 + updates25 + updates26

        update1 = updates11 + updates1
        update2 = updates21 + updates2

        self.sgd_uu = theano.function([uu, yu], J1, updates=updates1, mode='DebugMode', allow_input_downcast=True)
        self.sgd_ui = theano.function([ui, yi], J2, updates=updates2, mode='DebugMode', allow_input_downcast=True)
        #self.gd = theano.function([uu,yu], cost, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))



#if __name__ == "__main__":
#       u2v = user2vec(22166, 100, 100, 200)
#       u2v.model()
#       pdb.set_trace()

