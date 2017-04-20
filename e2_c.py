#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D plotting
###############################################################################
# Helper functions
def mdot(*args):
	"""Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
	return reduce(np.dot, args)
def prepend_one(X):
	"""prepend a one vector to X."""
	return np.column_stack([np.ones(X.shape[0]), X])
def grid2d(start, end, num=50):
	"""Create an 2D array where each row is a 2D coordinate.
	np.meshgrid is pretty annoying!
	"""
	dom = np.linspace(start, end, num)
	X0, X1 = np.meshgrid(dom, dom)
	return np.column_stack([X0.flatten(), X1.flatten()])
###############################################################################
# load the data
data = np.loadtxt("dataQuadReg2D.txt")
#print "data.shape:", data.shape
#np.savetxt("tmp.txt", data) # save data if you want to
myLamda = [0.0001,0.0005, 0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 5000.0, 10000.0, 100000.0, 1000000.0]
bestLamda = myLamda[0] #init best Lamda
minMeanSquaredError = 10000000000; #init smallest Mean Squared Error
for lamda in myLamda:
	print "lamda: ", lamda

	#Start cross-validation:

	#Step 1:Partition data D in k equal sized subsets
	k = 5 #number of subsets
	subsets = np.array_split(data,k) #split data
	sumError = 0
	#Step 2: Training with a part of data, test with the rest
	for i in range(k):
		#print "subsets:", subsets[i];
		trainingData = np.empty([1,3],dtype=float) #init 3-d array with 1 random element
		for j in range(k): #create training data without i_th subdata
			if j != i:
				trainingData = np.append(trainingData,subsets[j], axis = 0)
		trainingData = np.delete(trainingData, 0, 0) #remove 1st (random) element
		#print "training:", trainingData
		X, y = trainingData[:, :2], trainingData[:, 2]
		#print "X.shape:", X.shape
		#print "y.shape:", y.shape
		X = prepend_one(X)
		X = np.column_stack([ X, trainingData[:, 0]*trainingData[:, 0], trainingData[:, 0]*trainingData[:, 1], trainingData[:, 1]*trainingData[:, 1]])
		
		identityMatrix = np.identity(6) #quadratic
		identityMatrix[0,0] = 0
		beta_ = mdot(inv(dot(X.T, X) + lamda*identityMatrix), X.T, y)
		#print "Optimal beta:", beta_

		X_test, y_test = subsets[i][:, :2], subsets[i][:, 2] #use i_th subdata for testing
		X_test = prepend_one(X_test)
		X_test = np.column_stack([ X_test, X_test[:, 0]*X_test[:, 0], X_test[:, 0]*X_test[:, 1], X_test[:, 1]*X_test[:, 1]])
		#compute squared Error when testing with i_th subdata
		squaredError = np.linalg.norm(y_test - mdot(X_test, beta_)) * np.linalg.norm(y_test - mdot(X_test, beta_))
		sumError += squaredError;
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d') # the projection part is important
	# ax.scatter(X_test[:, 1], X_test[:, 2], y_test) # don't use the 1 infront
	# ax.scatter(X[:, 1], X[:, 2], y, color="red") # also show the real data
	# ax.set_title("predicted data")
	# plt.show()

	#Step 3:report mean error
	meanSquaredError = sumError/k;
	print "meanSquaredError", meanSquaredError
	print " "
	if meanSquaredError < minMeanSquaredError:
		minMeanSquaredError = meanSquaredError
		bestLamda = lamda

print "minMeanSquaredError:", minMeanSquaredError
print "bestLamda:", bestLamda









