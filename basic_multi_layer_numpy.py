import numpy as np

input_l_vals = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

bias = [2,3,0.5]

out_mult = np.dot(weights, input_l_vals)

print('without bias: ', out_mult)
print('with bias: ', out_mult + bias)