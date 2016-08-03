"""This is the Python port of the simulateOneRun.m Matlab code
provided by the causality challenge of Biomag2014.
"""
import numpy as np
from numpy.random import rand, randn

def mymvfilter(ar, x):
    """Filter function.
    """
    nChan, nOrder = ar.shape
    nOrder = nOrder / float(nChan)
    # the following three lines should be equivalent to: arReshape = ar.reshape(nChan, nChan, int(nOrder))
    arReshape = np.zeros((nChan, nChan, int(nOrder))) # in the original code nOrder is left as a real number: matlab just recast to int while numpy throws an error
    for i in range(int(nOrder)): # in the original code nOrder is left as a real number: matlab just recast to int while numpy throws an error
        arReshape[:,:,i] = ar[:, i*nChan:(i+1)*nChan]

    N = x.shape[1]
    y = x.copy() # in orignal code it is an assignment, but in matlab an assignment is a copy
    for i in range(1, N):
        nOrderLoc = np.min([i, nOrder])
        for j in range(int(nOrderLoc)):
            y[:,i] = y[:,i] + np.dot(arReshape[:,:,j], y[:,i-j-1]) # NOTE: both here and in the original code the first column is not modified. This occurs because here 'i' is always different from 0 and in the original code it is always different from 1.

    return y


def simulate_one_run(configuration, N=6000, P=10, gamma=None):
    """Generates one sample for the causality challenge
    
    Input:
    N:     number of time points
    P:     order of AR-system
    gamma: parameter controlling the relative strength of noise
    and signal, gamma=0 means only signal, gamma=1 means only noise
    conf:  3x3 matrix with binary elements defining (the presence of) causal
    interactions. A causal influence from channel i to j exists
    if conf(i,j)==1. Note that diagonal elements will be ignored! 
    Note alo that AR matrices have a 'transpose meaning', 
    i.e. interaction from i to j are represented in matrix
    elements A(j,i) with A an AR matrix at some delay). 
    
    Output:
    data:  Nx3 matrix of data
    
    Every sample of the challenge was generated with the commands:
    N=6000; 
    P=10;
    gamma=rand;
    conf=randn(3)>0;
    data=simuldata_biomag(N,P,gamma,conf)
    """
    if gamma is None:
        gamma = rand()

    M = configuration.shape[0] # in the original code: the largest dimension 
    np.fill_diagonal(configuration, 1)

    sSignal = 1 - gamma
    sNoise = gamma

    done = False

    while not done:
        arSig=np.zeros((M, M * P))
        for k in range(P):
            aloc = (rand(M, M) - 0.5) / 2.2
            arSig[:, k * M + np.arange(M)] = aloc * configuration.T

        E = np.eye(M * P)
        A = np.vstack([arSig, E[:-M, :]]) # the original code is E(1:end-M,:), i.e. all elements but the last M. Example in matlab: t=[1 2 3 4], t(1:end-2) gives '1 2'. In Python: t=[1,2,3,4], t[:-2] gives [1, 2].
        lambdaMax = np.max(np.abs(np.linalg.eigvals(A)))

        arNoise = np.zeros((M, M*P))
        for k in range(P):
            aloc = np.diag(np.diag((rand(M, M) - 0.5) / 2.2))
            arNoise[:, k * M + np.arange(M)] = aloc

        E = np.eye(M * P)
        Anoise = np.vstack([arNoise, E[:-M, :]])
        lambdaMaxNoise = np.max(np.abs(np.linalg.eigvals(Anoise)))

        if (lambdaMax < 0.95) and (lambdaMaxNoise < 0.95):
            x = randn(M, N)
            dataSignal = mymvfilter(arSig, x).T
            sigLevel = np.linalg.norm(dataSignal, ord='fro')
            
            x = randn(M, N)
            dataNoise = np.dot(mymvfilter(arNoise, x).T, randn(M, M))
            noiseLevel = np.linalg.norm(dataNoise, ord='fro')

            data = sSignal * (dataSignal / sigLevel) + sNoise * (dataNoise / noiseLevel)

            done = True

    return data
