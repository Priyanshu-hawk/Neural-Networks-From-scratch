'''
Using Catagorical Cross Entropy

Genral = log(base e)*x == ln(x) = e ** x = b , here x means log of that number 'b'
base e = 2.71828 == euler's number


One hot encoding/vector = its kind of vector map = [1,0,0,0] = 4 class and 1st lable is on
'''

'''Example Code'''

import math

softmax_output = [0.7,0.2,0.1]

target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0]+
         math.log(softmax_output[1])*target_output[1]+
         math.log(softmax_output[2])*target_output[2])


print(loss)