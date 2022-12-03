#3 input layer maps to 1 output layer

input_l_vals = [1,2,3]
weights = [0.2,0.8,-0.5]
bias = 2

output = 0
for i in range(len(input_l_vals)):
    output += (input_l_vals[i] * weights[i])

print(output+bias)