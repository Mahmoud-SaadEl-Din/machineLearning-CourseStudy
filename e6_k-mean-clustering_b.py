import numpy as np
import matplotlib.pyplot as plt
import os
import random

def loadImage(imgPath):
    listPath = os.listdir(imgPath)
    listPath.sort()
    X = []
    count = 0
    for img in listPath:
        if img == "Readme.txt":
            continue
        else:
            im = plt.imread(imgPath+img)
            im2 = im[:, :, 0]

            X.append(im2.flatten())
            count += 1
    return np.asarray(X,dtype=np.float64 ), count



def assignClusterCenter(X, imgNumber, clusterNumber, clusterCenters):
    r = np.zeros((imgNumber, clusterNumber))
    for n in range(0,imgNumber):
        normSet = []
        for k in range(0, clusterNumber):
            norm = np.linalg.norm(X[n] - clusterCenters[k]) ** 2
            normSet.append(norm)
        minIndex = np.argmin(normSet)
        r[n][minIndex] = 1
    return r
def checkFirstCluster(X, imgNumber, clusterNumber, r):
    minIndex = []
    for k in range(0, clusterNumber):
        count = 0
        for n in range(0, imgNumber):

            if r[n][k] == 1:
                count+=1
        minIndex.append(count)
    minIndex = np.min(minIndex)
    return minIndex # if minIndex = 0 -> exist a cluster with no element
def newCenter(X, imgNumber, clusterNumber, r):
    clusterCenters = []
    for k in range(0, clusterNumber):
        sum = 0
        count = 0
        for n in range(0, imgNumber):

            if r[n][k] == 1:
                count+=1
                sum += X[n]
        center = sum/count
        clusterCenters.append(center)

    return np.asarray(clusterCenters)

def calError(X , r, imgNumber, clusterNumber, clusterCenters):
    for n in range(0,imgNumber):
        error = 0
        for k in range(0, clusterNumber):
            norm = np.linalg.norm(X[n] - clusterCenters[k]) ** 2
            error += norm*r[n][k]
    return error

def k_meanClustering(step, X, imgNumber, k):
    clusterCenters = np.random.uniform(0, 255, (k, X.shape[1]))

    r = assignClusterCenter(X, imgNumber, k, clusterCenters)
    minElementClusterNo = checkFirstCluster(X, imgNumber, k, r)
    while minElementClusterNo == 0: # make sure data was assign to k random cluster
        clusterCenters = np.random.uniform(0, 255, (k, X.shape[1]))
        r = assignClusterCenter(X, imgNumber, k, clusterCenters)
        minElementClusterNo = checkFirstCluster(X, imgNumber, k, r)

    minError =  calError(X, r, imgNumber, k, clusterCenters) #assign first error to be min
    for i in range(0,step):
        clusterCenters = newCenter(X, imgNumber, k, r)
        r = assignClusterCenter(X, imgNumber, k, clusterCenters)
        currentError =  calError(X , r, imgNumber, k, clusterCenters)
        if currentError < minError:
            bestClusterCenters = clusterCenters
            minError = currentError

    print "Min Error = ", minError
    return r, np.asarray(bestClusterCenters), minError

imgPath = "yalefaces_cropBackground/"
X, imgNumber= loadImage(imgPath)




step = 10
k_array = [2,3,4,5,6]
# k_array = [4]
error = []
for  k in k_array:
    print " k = ", k
    r ,clusterCenters, minError = k_meanClustering(step, X, imgNumber, k)
    error.append(minError)

    # for centerFace in clusterCenters:
    #     centerFace = centerFace.reshape((243,160))
    #     plt.figure() #important
    #     imgplot = plt.imshow(centerFace, cmap='gray')
    #     debug = 1
    # plt.show()

plt.plot(k_array, error)
plt.show()
print"1"