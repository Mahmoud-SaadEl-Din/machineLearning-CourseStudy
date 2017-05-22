from operator import inv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotData(data):
    X = data[:, :2]
    x1 = []
    x2 = []
    for point in data:
        if point[2] == 1:
            x1.append(point[0])
            x2.append(point[1])
    plt.plot(x1,x2,'ro')

    x1 = []
    x2 = []
    for point in data:
        if point[2] == 0:
            x1.append(point[0])
            x2.append(point[1])
    plt.plot(x1,x2,'go')
    plt.show()
def plotClassProbability(data,p):
    X = data[:, :2]
    x1 = []
    x2 = []
    prob = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,data[:,0].shape[0]):
        if p[i] >= 0.5:
            x1.append(X[i][0])
            x2.append(X[i][1])
            prob.append(p[i])
    ax.scatter(x1,x2,prob, c='r', marker='o')

    x1 = []
    x2 = []
    prob = []
    for i in range(0,data[:,0].shape[0]):
        if p[i] < 0.5:
            x1.append(X[i][0])
            x2.append(X[i][1])
            prob.append(p[i])
    ax.scatter(x1,x2,prob, c='g', marker='v')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Probability')

    plt.show()


def prepend_one(X):
	"""prepend a one vector to X."""
	return np.column_stack([np.ones(X.shape[0]), X])

def linearFeature(data):
    X= data[:, :2]
    X = prepend_one(X)
    return X
def quadraticFeature(data):
    X= data[:, :2]
    X = prepend_one(X)
    X = np.column_stack([X, data[:, 0] * data[:, 0], data[:, 0] * data[:, 1], data[:, 1] * data[:, 1]])
    return X
def sigmoid(X,beta):
    f = np.dot(X, beta)
    p = np.exp(f) / (1+ np.exp(f))
    return p
    # p size n x 1

def gradient(X,y,beta,lamda):
     I = np.identity(X.shape[1])
     p = sigmoid(X,beta)
     a = np.zeros((200, 1))
     for i in range(0, 200):
         a[i] = p[i] - y[i]
     b = 2*lamda*np.dot(I,beta)
     gra = np.dot(X.T, a)
     gra = gra + b
     return gra #size k x 1
def hessian(X,beta,lamda):

    I = np.identity(X.shape[1])
    p = sigmoid(X,beta)
    W = np.multiply(p,(1-p)) * np.identity(p.shape[0])
    H = X.T.dot(W).dot(X) + 2*lamda*I
    return H #size k x k
def newtonMedthod(X,y,beta,lamda,step):
    for i in range(step):
        H = hessian(X, beta, lamda)
        G = gradient(X, y, beta, lamda)
        denta = np.dot(np.linalg.inv(H), G)
        beta = beta - denta
    return beta #size k x 1


lamda = 0.1**5;
data = np.loadtxt("data2Class.txt")
# X = linearFeature(data) #size n x k
X = quadraticFeature(data)
y = data[:, 2] #size n x 1
beta = np.zeros((X.shape[1],1)) #size k x 1
stepSize = 10
beta = newtonMedthod(X,y,beta,lamda,stepSize)
print "Optimim beta\n", beta

p = sigmoid(X,beta)
plotClassProbability(data,p)

