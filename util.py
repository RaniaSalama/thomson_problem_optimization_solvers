from cvxpy import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.decomposition as deco
import time

alpha = 1

def compute_norms(x, k, n):
    norms = np.zeros(n)
    for i in range(n):
        norms[i] = LA.norm(x[:,i])
    return norms

def f_x(x, k, n):
    fx = 0
    for i in range(n):
        for j in range(i):
            if i != j:
                fx = fx + (1.0/LA.norm(x[:,i] - x[:,j])**2)
    return fx
    

def f_s(s, n):
    s = reshape(s, n, n)
    fx = 0.0
    for i in range(n):
        ei = np.zeros((1, n))
        ei[0, i] = 1
        for j in range(i):
            ej = np.zeros(n)
            ej.shape = (n, 1)
            ej[j, 0] = 1
            fx +=  (ei*s*np.transpose(ei))[0,0] - 2 * (ei * s * ej)[0,0] + (np.transpose(ej) * s* ej)[0,0]
    return fx

# PCA code is taken from http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
    data_recovered = NP.dot(eigenvectors, m).T
    data_recovered += data_recovered.mean(axis=0)
    assert NP.allclose(data, data_recovered)    