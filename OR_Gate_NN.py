"""
Basic And Gate

1 | 1 = 1
1 | 0 = 1
0 | 1 = 1
0 | 0 = 0

Expected = [1,1,1,0]
"""

input_layer = [[1,1],
               [1,0],
               [0,1],
               [0,0]]

weights = [1,1]

bias = -1

def step_fn(n_out):
    if n_out >= 0:
        return 1
    return 0

output_set = []
for i in input_layer:
    out = i[0]*weights[0]+i[1]*weights[1]+bias
    apply_activation = step_fn(out)
    output_set.append(apply_activation)

print(output_set)