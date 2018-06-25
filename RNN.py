import theano
import theano.tensor as T
from theano import function
import numpy as np
import random

MINL = 50
MAXL = 55 

def step(x_t,a_tm1,y_tm1):
    a_t = T.nnet.sigmoid(T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh)
    y_t = T.dot(a_t,Wo) + bo
    return a_t, y_t

def gen_data(minl = MINL,maxl = MAXL):
    length = np.random.randint(minl, maxl)
    x_seq = np.concatenate([np.random.uniform(size = (length, 1)),
                            np.zeros((length, 1))], axis = -1)
    x_seq[np.random.randint(length/10), 1] = 1
    x_seq[np.random.randint(length/2,length), 1] = 1
    y_hat = np.sum(x_seq[:,0] * x_seq[:,1])
    return x_seq, y_hat

def MyUpdate(parameters,gradients):
    mu = 0.001
    parameters_updates = [(p,p - mu * g) for p,g in zip(parameters,gradients)]
    return parameters_updates

x_seq = T.matrix('input')
y_hat_seq = T.scalar('target')

Wi = theano.shared(np.asarray(np.random.uniform(size = (2,5)),dtype = theano.config.floatX))
Wh = theano.shared(np.asarray(np.random.uniform(size = (5,5)),dtype = theano.config.floatX))
Wo = theano.shared(np.asarray(np.random.uniform(size = (5,2)),dtype = theano.config.floatX))
bh = theano.shared(np.zeros((5),dtype = theano.config.floatX))
bo = theano.shared(np.zeros((2),dtype = theano.config.floatX))

parameters = [Wi,bh,Wo,bo,Wh]

a_0 = theano.shared(np.zeros(5))
y_0 = theano.shared(np.zeros(2))

[a_seq,y_seq],_ = theano.scan(step, 
                              sequences = x_seq,
                              outputs_info = [a_0, y_0],
                              truncate_gradient = -1
                             )

y_hat_result = y_seq[-1][0]
cost = T.sum((y_hat_result - y_hat_seq)**2)
gradients = T.grad(cost,parameters)

test = theano.function(
						inputs = [x_seq],
						outputs = y_hat_result
						)
train = theano.function(inputs = [x_seq,y_hat_seq], 
                        outputs = cost, 
                        updates = MyUpdate(parameters,gradients)
                       )
for i in range(1000000):
	x_seq, y_hat_seq = gen_data()
	print("第",i+1,"次train","cost = ",train(x_seq,y_hat_seq))

for i in range(10):
    x_seq, y_hat_seq = gen_data()
    print("y_hat = ",test(x_seq))


