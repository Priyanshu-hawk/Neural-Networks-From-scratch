import numpy as np
import matplotlib.pyplot as plt

a = [
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    1,1,1,1,1,1,
    1,0,0,0,0,1,
    1,0,0,0,0,1,
]

b = [
    0,1,1,1,0,0,
    0,1,0,0,1,0,
    0,1,1,1,0,0,
    0,1,0,0,1,0,
    0,1,1,1,0,0,
]

c = [
    0,1,1,1,1,0,
    0,1,0,0,0,0,
    0,1,0,0,0,0,
    0,1,0,0,0,0,
    0,1,1,1,1,0,
]

y = [[1,0,0],
    [0,1,0],
    [0,0,1]]

X = [np.array(a).reshape(1,30),np.array(b).reshape(1,30),np.array(c).reshape(1,30)]
Y = np.array(y)

# plt.imshow(np.array(a).reshape(5,6))
# plt.show()

print(X)

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)
def f_forward(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    return a2

def genrate_weights(x,y):
    np.random.seed(20)
    l = []
    for i in range(x*y):
        l.append(np.random.randn())
    return np.array(l).reshape(x,y)

def loss(out, Y):
    # impli MSE mean squred error
    s = np.square(out-Y)
    return(np.sum(s)/len(Y))

def back_prop(x,y, w1,w2, alpha):
    z1 = x.dot(w1) # layer 1 inp
    a1 = sigmoid(z1) # layer 1 out

    z2 = a1.dot(w2) # layer 2 inp
    a2 = sigmoid(z2) # layer 2 out

    # error in output layer
    d2 = (a2-y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), (np.multiply(a1, 1-a1)))

    #Gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    #update with learning rate
    w1 = w1-(alpha*(w1_adj)) 
    w2 = w2-(alpha*(w2_adj))

    return (w1,w2)

def train(x,Y, w1, w2, alpha=0.2,epoch=10):
    acc = []
    losses = []

    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append(loss(out, Y[i]))
            w1, w2 = back_prop(x[i],y[i],w1,w2,alpha)
        print("Epoch:",j+1,"=========== acc:", (1-(sum(l)/len(x)))*100)
        acc.append((1-(sum(l)/len(x)))*100)
        losses.append(sum(l)/len(x))
    return(acc, losses, w1, w2)

def predict(x,w1,w2):
    out = f_forward(x,w1,w2)
    maxi = 0
    k=0

    for i in range(len(out[0])):
        if(maxi<out[0][i]):
            maxi = out[0][i]
            k = i
    print(k)
    if k==0: print("Its 'A'")
    if k==1: print("Its 'B'")
    if k==2: print("Its 'C'")
    plt.imshow(x.reshape(5, 6))
    plt.show()


w1 = genrate_weights(30,5) # 30 input to 5 output
w2 = genrate_weights(5,3) # 5 input to 3 output for A,B,C

acc, losses, w1 , w2 = train(X,Y,w1,w2,0.005,10000)

# # ploting accuracy
# plt.plot(acc)
# plt.ylabel('Accuracy')
# plt.xlabel("Epochs:")
# plt.show()
# # plotting Loss
# plt.plot(losses)
# plt.ylabel('Loss')
# plt.xlabel("Epochs:")
# plt.show()

predict(X[1], w1, w2)
predict(X[0], w1, w2)
predict(X[2], w1, w2)
