import numpy as np
import matplotlib.pyplot as plt


class Create_Data:
    def create_data(self, points, classes):
        
        X = np.zeros((points*classes,2)) # data matrix (each row = single example)
        y = np.zeros(points*classes, dtype='uint8') # class labels
        for j in range(classes):
            ix = range(points*j,points*(j+1))
            r = np.linspace(0.0,1,points) # radius
            t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2 # theta
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = j
        return X, y
        

# cd = Create_Data()
# X,y = cd.create_data(100,3)
# # lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()