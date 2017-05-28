import numpy as np
import matplotlib.pyplot as plt
import os

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

            X.append(im.flatten())
            count += 1
    return np.asarray(X,dtype=np.float64 ), count

def meanAndCenter(Data):
    X_centered = np.copy(Data)
    mean = np.mean(Data, axis=0)
    for img in X_centered:
        img -= mean
    return mean, X_centered
def svd(Data):
    # u, s, vt = sla.svds(Data, k=imgNumber)
    u, s, vt = np.linalg.svd(Data, full_matrices=False)
    return u, s, vt
def reconstruct(X,mean,imgNumber,Z,Vp):
    X_new = np.dot(Z,Vp.T)
    error = 0
    for i in range(0,imgNumber):
        X_new[i] += mean
        error += np.linalg.norm(X[i] - X_new[i])**2

    return X_new,error

imgPath = "yalefaces/"
X, imgNumber= loadImage(imgPath) #part a
mean, X_centered=meanAndCenter(X) #part b
u,s,vt = svd(X_centered) #part c
V = vt.T#get V
p = 100 #number of smaller dimension
Vp = V[:,0:p] #part d
Z = np.dot(X_centered,Vp)
#end part d

X_new,error = reconstruct(X,mean,imgNumber,Z,Vp) #part e
print "Error = ", error

chooseImage = X_new[0]
chooseImage = chooseImage.reshape((243,320))
imgplot = plt.imshow(chooseImage, cmap='gray')
debug = 1
plt.show()