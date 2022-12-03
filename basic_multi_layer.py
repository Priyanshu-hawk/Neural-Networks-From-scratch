#4 inputs layers and 3 outputs layers

input_l_vals = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

bias = [2,3,0.5]

def dot_product(input_l_vals, weights):
    output = 0
    for i in range(len(input_l_vals)):
        output += (input_l_vals[i] * weights[i])
    return output

def output_layers(input_l_vals, weights, bias):
    output = []
    for i in range(len(weights)):
        output.append(dot_product(input_l_vals, weights[i]) + bias[i])
    return output

print(output_layers(input_l_vals, weights, bias))



