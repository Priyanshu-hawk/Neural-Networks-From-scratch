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

class Activation: # ReLU ensure only possitive or zero..
    
    def apply_relu(self, input):
        self.output_relu = np.maximum(0, input)
    def apply_softmax(self, input):
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=1))
        proba = exp_vals / np.sum(exp_vals, axis=1, keepdims=1)
        self.output_softmax = proba 

X, y = gen_data.create_data(100,3) # spiral data genration

dense1 = Layer_Dense(2,3)
activation1 = Activation()

dense2 = Layer_Dense(3,3)

dense1.feed_f(X)
activation1.apply_relu(dense1.output)

dense2.feed_f(activation1.output_relu)
activation1.apply_softmax(dense2.output)

print(activation1.output_softmax)