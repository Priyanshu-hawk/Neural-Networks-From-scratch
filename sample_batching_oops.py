import numpy as np

np.random.seed(0) # mantin random state consistancy

# inputs remain the same throughout. input batching
X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))
    def feed_f(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.bias


layer1 = Layer_Dense(4,3) # 4 is neuron parllel and 3 is numbers of hidden layer
layer2 = Layer_Dense(3,2) # 2 is output layer


layer1.feed_f(X)
print(layer1.output)

layer2.feed_f(layer1.output) # layer 2 output is layer 2 input
print(layer2.output)