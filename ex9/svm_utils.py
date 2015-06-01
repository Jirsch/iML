import numpy as np
import matplotlib.pyplot as plt

def show_SVM_linear(X,Y,w):
    # Given data (X,Y) and a classifier w, display the result.
    plt.plot(X[Y==1,0],X[Y==1,1],"+",markeredgecolor="b",markersize=10,linestyle="none",linewidth=3)
    plt.plot(X[Y==-1,0],X[Y==-1,1],"o",markeredgecolor="r",markersize=10,linestyle="none",linewidth=3)
    minx = np.min(X[:,0])
    maxx = np.max(X[:,0])
    miny = np.min(X[:,1])
    maxy = np.max(X[:,1])
    A = np.arange(minx,maxx,0.01)
    plt.plot(A,-(w[0]/w[1])*A,"k")
    plt.axis([minx, maxx, miny, maxy])

def show_SVM_gaussian(X,Y,alphas,sigma2):
    # Given data (X,Y) and a kernel classifier alpha, display the result.
    plt.plot(X[Y==1,0],X[Y==1,1],"+",markeredgecolor="b",markersize=10,linestyle="none",linewidth=3)
    plt.plot(X[Y==-1,0],X[Y==-1,1],"o",markeredgecolor="r",markersize=10,linestyle="none",linewidth=3)
    minx = np.min(X[:,0])
    maxx = np.max(X[:,0])
    miny = np.min(X[:,1])
    maxy = np.max(X[:,1])
    X_range = np.arange(minx,maxx,0.05)
    Y_range = np.arange(miny,maxy,0.05)
    Z = np.zeros((len(X_range),len(Y_range)))
    plt.axis([minx, maxx, miny, maxy])
    m = X.shape[0]
    for i in range(len(X_range)):
        for j in range(len(Y_range)):
            Z[i,j] = alphas.dot(np.exp(-np.sum((np.tile([X_range[i], Y_range[j]],(m,1))-X)**2,1)/sigma2))
    [A, B] = np.meshgrid(Y_range,X_range)
    plt.contour(B,A,Z,[0, 0],colors="k")