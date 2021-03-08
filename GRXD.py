# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:20:32 2021

@author: fatmanur
"""

import numpy as np
from numpy.linalg import inv

##################
## GRX DETECTOR ##
##################
## D = transpose(x-m)(Cov-1)(x-m)
def GRXD(data):
    M = data.shape[0]
    N = data.shape[1]
    L = data.shape[2]
    
    # Global Mean
    n_sample = M*N
    m = np.sum(np.sum(data,axis=0),axis=0)/n_sample

    # Covariance Matrix
    Cov = np.zeros((L,L))
    for i in range(0,M):
        for j in range(0,N):
            x = data[i,j]
            Cov += np.matmul((x-m),(np.transpose(x-m)))
    Cov = Cov/(n_sample-1)
    
    # Mahalanobis Distance
    D = np.zeros((M,N))
    # to prevent singular matrix error:
    c = 0.01
    I = c*np.identity(L)
    for i in range(0,M):
        for j in range(0,N):
            x = data[i,j]
            D[i,j] = np.matmul(np.matmul(np.transpose(x-m),inv(Cov+I)),(x-m))
    return D