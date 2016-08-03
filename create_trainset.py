""" Main file for generating the MAR dataset

    INPUT
    iterations: n. of trials per configuration;
    configurations: n. of causal configurations among 3 time series; 
    N: n. if time points;
    P: order of MAR model.
    
    OUTPUT
    one file for each configuration that contains:
    c: configuration number from 0 to 63;   
    conf: 3x3 binary matrix, 1 in position (i,j) means connection from signal i to j;
    data: generated signals, shape [iterations, N, 3];
    gamma: noise level. 
"""

import numpy as np
from simulator import simulate_one_run
import pickle
from sys import stdout
import functools

# From https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
# note that this decorator ignores **kwargs
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

@memoize
def class_to_configuration(c, verbose=True):
    c_bin = np.binary_repr(c, 6)
    c_bin_array = np.array([int(v) for v in c_bin]) 
    configuration = np.eye(3)
    configuration[np.triu_indices(3, k=1)] = c_bin_array[:3]
    configuration[np.tril_indices(3, k=-1)] = c_bin_array[3:]
    if verbose: print "Configuration:", c, c_bin, c_bin_array
    if verbose: print configuration
    return configuration
    
def configuration_to_class(configuration, verbose=True):
    n = configuration.shape[0]
    c_bin_array = np.zeros((n*(n-1)))
    c_bin_array[:len(c_bin_array)/2] = configuration[np.triu_indices(n, k=1)]
    c_bin_array[len(c_bin_array)/2:] = configuration[np.tril_indices(n, k=-1)]
    c = np.dot(c_bin_array, 2**np.arange(n*(n-1)-1,-1,-1))
    if verbose: print "Configuration:", configuration, c_bin_array
    if verbose: print c
    return c


if __name__ == '__main__':

    np.random.seed(0)
    iterations = 1000 #trials per configuration
    configurations = 64 #configurations of 3 time series  
    N = 6000 #time points per trial
    P = 10 #MAR order

    for c in range(configurations):
        data = []
        configuration = class_to_configuration(c)
        gamma = np.random.rand(iterations) #gamma is uniformly distributed [0,1) 
        print "Creating trials:",
        for i in range(iterations):
            print i,
            stdout.flush()
            data.append(simulate_one_run(configuration, N=N, P=P, gamma=gamma[i]).astype(np.float32))
            
        print
        data = np.array(data)
        filename = 'data/simulated_data_class_'+str(c)+'.pickle'
        print "Saving to", filename
        pickle.dump({'c': c,
                     'conf': configuration,
                     'data': data,
                     'gamma': gamma
                     },
                    open(filename, 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print
