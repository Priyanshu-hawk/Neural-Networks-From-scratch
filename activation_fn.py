import numpy as np
from Genrate_test_data import Create_Data
np.random.seed(0) # mantin random state consistancy

# inputs remain the same throughout. input batching
gen_data = Create_Data()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))
    def feed_f(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.bias

class Activation:
    """
    1. ReLU ensure only possitive or zero..
    2. Softmax is probaility distribution with exponential vals btwn (-inf,0)
    """
    def apply_relu(self, input):
        self.output_relu = np.maximum(0, input)
    def apply_softmax(self, input):
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=1))
        proba = exp_vals / np.sum(exp_vals, axis=1, keepdims=1)
        self.output_softmax = proba 

class Loss:
    def cross_entropy_loss(self, y_preds, y_true):
        samples = len(y_preds)
        y_preds_clipped = np.clip(y_preds, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # it means a scaler values like [1,0,2,1,0] represent index
            correct_confidence = y_preds_clipped[range(samples),y_true]
        
        else: # it means it is a one hot encoded value like [[0,1,0],[1,,0,0],[0,0,1],[0,1,0],[1,0,0]] 
            correct_confidence = np.sum(y_preds_clipped*y_true,axis=1)

        # print(correct_confidence)

        neg_log = -np.log(correct_confidence)

        # print(neg_log)
        return np.mean(neg_log)


X, y = gen_data.create_data(100,3) # spiral data genration

dense1 = Layer_Dense(2,3)
activation1 = Activation()

dense2 = Layer_Dense(3,3)

dense1.feed_f(X)
activation1.apply_relu(dense1.output)

dense2.feed_f(activation1.output_relu)
activation1.apply_softmax(dense2.output)

print(activation1.output_softmax[:5])

loss_fn = Loss()
loss = loss_fn.cross_entropy_loss(activation1.output_softmax, y)

print('Loss:',loss)