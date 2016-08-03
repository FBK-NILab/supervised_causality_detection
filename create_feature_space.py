""" Compute the features (r2,mse,granger) for a given set of trials

    INPUT:
    time_window_size: time points considered in the regression;
    time_lag: time lag from which the selection of time_windows_size time points starts, if None it takes points from (-1)*time_window_size to -1;
    N: time points considered in the mapping, if None it takes the entire time serie;
    n_folds: number of folds in the regression problem, default=5;
    order_granger: order of the MAR model for computing granger features;
    n_jobs: CPU cores;
    filedata: source file- pwd_source + filedata_structure + i_file.
    
    OUTPUT:
    3 pickle files, one for each type of feature r2, mse and granger.
"""

import numpy as np
import pickle
from scipy.misc import comb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sys import stdout
from joblib import Parallel, delayed

from score_function import compute_score_matrix, best_decision
from create_level2_dataset import regression_scores, granger_scores, feature_engineering, feature_normalisation
from create_trainset import configuration_to_class
   
causality_structures = [((0,),0), ((0,),1), ((0,),2),
                        ((1,),0), ((1,),1), ((1,),2),
                        ((2,),0), ((2,),1), ((2,),2),
                        ((0,1),0), ((0,1),1), ((0,1),2),
                        ((0,2),0), ((0,2),1), ((0,2),2),
                        ((1,2),0), ((1,2),1), ((1,2),2),
                        ((0,1,2),0), ((0,1,2),1), ((0,1,2),2)]

def compute_lev2_regression_general_case(filedata, time_window_size=10, time_lag=None, N=None, reg=None, cv=5, scoring='r2', n_jobs=-1):
    """Compute the portion of a second level dataset related to a set of trial.
    """
    print "Computing regression-based features."    
    data = pickle.load(open(filedata))
    data_timeseries = data['data']
    [nTrial, nTime, nCh] = data_timeseries.shape
    
    #for reducing the number of time points
    if not(N is None):
        data_timeseries = data_timeseries[:,:N,:]
        nTime = N

    ch = np.arange(nCh)
    y_level2_conf = data['conf']  
    y_level2_conf = np.reshape(np.repeat(y_level2_conf[None,:,:], nTrial, axis=0),[nTrial,nCh,nCh])
    gamma_test_level2 = data['gamma'] #None in case we don't know
    
    print "Data set shape:", data_timeseries.shape
    n_comb = comb(nCh,3,exact=1)
    X_lev2_regression = np.zeros((nTrial, n_comb, len(causality_structures), 2), dtype=np.float64) #added 2 dimensions to compute r2 and mse
    y_level2 = np.zeros((nTrial, n_comb), dtype=np.int32)
    order_combinations = np.zeros((n_comb, 3), dtype=np.int32)
    i_comb = 0
    for i in range(0,nCh):
        for j in range(i+1,nCh):
            for z in range(j+1,nCh):
                print "combination", i_comb
                #conditionCh=np.delete(ch,[i,j,z])
                result = Parallel(n_jobs=n_jobs)(delayed(regression_scores)(data_timeseries[trial_i,:,[i,j,z]].T, time_window_size=time_window_size, time_lag=time_lag, reg=reg, cv=n_folds, scoring=scoring, timeseriesZ=None) for trial_i in range(nTrial))#timeseriesZ=data_timeseries[trial_i,:,conditionCh].T
                X_lev2_tmp = zip(*result) # See http://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
                #X_lev2_regression[:,i_comb,:] = np.vstack(X_lev2_tmp).T
                X_lev2_regression[:,i_comb,:,0] = np.squeeze(np.array(X_lev2_tmp)[:,:,0]).T #r2score
                X_lev2_regression[:,i_comb,:,1] = np.squeeze(np.array(X_lev2_tmp)[:,:,1]).T #mse               
                order_combinations[i_comb] = np.array([i,j,z], dtype=int)
                tmp = []
                tmp += [(y_level2_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]]) for i_trial in range(nTrial)]                 
                tmp = np.array(tmp)
                tmp_res = []
                tmp_res += [(configuration_to_class(tmp[i_trial], verbose=False)) for i_trial in range(nTrial)]
                y_level2[:,i_comb] = np.array(tmp_res) 
                i_comb += 1
    
    y_level2_conf = np.array(y_level2_conf, dtype=np.int32)
    gamma_test_level2 = np.array(gamma_test_level2, dtype=np.float32)
    return X_lev2_regression, y_level2_conf, y_level2, gamma_test_level2, order_combinations
    

def compute_lev2_granger_general_case(filedata, order=10, N=None, n_jobs=-1):
    """Compute the granger causality coefficients for each triad in the entire set
    """
    print "Computing Granger causality coefficients"
    data = pickle.load(open(filedata))
    data_timeseries = data['data']
    [nTrial, nTime, nCh] = data_timeseries.shape

    if not(N is None):
        data_timeseries = data_timeseries[:,:N,:]
        nTime = N
    
    y_level2_conf = data['conf'] 
    y_level2_conf = np.reshape(np.repeat(y_level2_conf[None,:,:], nTrial, axis=0),[nTrial,nCh,nCh])
    gamma_test_level2 =  data['gamma'] #data['synpaticEfficacies'] or None in case we don't know
    
    print "Data set shape:", data_timeseries.shape
    n_comb = comb(nCh,3,exact=1)
    X_lev2_granger = np.zeros((nTrial, n_comb, 6))
    order_combinations = np.zeros((n_comb, 3), dtype=int)
    i_comb = 0
    for i in range(0,nCh):
        for j in range(i+1,nCh):
            for z in range(j+1,nCh):
                print "combination", i_comb
                result = Parallel(n_jobs=n_jobs)(delayed(granger_scores)(data_timeseries[trial_i,:,[i,j,z]], order) for trial_i in range(nTrial))
                X_lev2_tmp = zip(*result) # See http://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
                X_lev2_granger[:,i_comb,:] = np.vstack(X_lev2_tmp).T
                order_combinations[i_comb] = np.array([i,j,z], dtype=int)
                i_comb += 1
    
    return X_lev2_granger, y_level2_conf, gamma_test_level2, order_combinations
        

if __name__ == '__main__':

    time_window_size = 10 #time points considered in the regression
    time_lag = None #time lag from which the selection of time_windows_size time points starts, if None it takes points from (-1)time_window_size to -1
    N = None #time points to consider in the mapping, if None it takes the entire time serie

    reg = LinearRegression(fit_intercept=True, normalize=True)
    #reg = SVR(C=1.0, epsilon=0.2)
    # reg = BayesianRidge()
    n_folds = 5
    #scoring = 'residual_tests'
    scoring = 'r2'#'mean_squared_error'
    order_granger = 10
    n_jobs = -1 # '-1' = use all available CPU cores

    ## File in which features are saved, one file for each type of feature (r2, mse and granger)
    pwd = 'data/'
    filename_level2_r2 = '%sdataset_level2_tws%d_cv%d_r2_shift_window.pickle' % (pwd, time_window_size, n_folds)       
    filename_level2_mse = '%sdataset_level2_tws%d_cv%d_mse_shift_window.pickle' % (pwd, time_window_size, n_folds)    
    
    # Source file name
    configurations = 64 #number of files to map, one file for each configuration
    pwd_source = 'data/'
    filedata_structure = 'simulated_data_class_'
     
    #r2 and mse features
    filename_level2 = filename_level2_r2
    #filename_level2 = filename_level2_mse
    try:
        print "Loading", filename_level2
        level2 = pickle.load(open(filename_level2))
        X_test_level2 = level2['X_level2']
        y_test_level2 = level2['y_level2']
        gamma_test_level2 = level2['gamma_level2']
        order_combinations = level2['order_combinations']
        
    except IOError:
        print "Not found!"
        print
        # The following is the parallel (multicore) loop over all classes calling compute_lev2_regression():
        X_test_level2 = []
        y_test_level2 = []
        y_level2 = []
        gamma_test_level2 = []
        
        for i_file in range(configurations):
        
            print "n. file:", i_file
            filedata = '%s%s%d%s' % (pwd_source,filedata_structure,i_file,'.pickle')   
            print "Loading file data", filedata
            
            tmp_X_test_level2, tmp_y_test_level2, tmp_y_level2, tmp_gamma_test_level2, order_combinations = compute_lev2_regression_general_case(filedata, time_window_size, time_lag, N, reg, cv=n_folds, scoring=scoring, n_jobs=n_jobs)          
            X_test_level2.append(tmp_X_test_level2) 
            y_test_level2.append(tmp_y_test_level2)
            y_level2.append(tmp_y_level2)
            gamma_test_level2.append(tmp_gamma_test_level2)
            
        
        X_test_level2 = np.vstack(X_test_level2)
        y_test_level2 = np.vstack(y_test_level2)
        y_level2 = np.vstack(y_level2)
        #gamma_test_level2 = np.hstack(gamma_test_level2)
        print
        print "Saving level2 dataset in", filename_level2
        pickle.dump({'time_window_size': time_window_size,
                     'reg': reg,
                     'cv': n_folds,
                     'X_level2': np.squeeze(X_test_level2[:,:,:,0]),
                     'y_level2_conf': y_test_level2,
                     'y_level2': y_level2,
                     'gamma_level2': gamma_test_level2,
                     'order_combinations': order_combinations,
                     },
                    open(filename_level2, 'w'),
                    protocol = pickle.HIGHEST_PROTOCOL)
        
        filename_level2 = filename_level2_mse
        print
        print "Saving level2 dataset in", filename_level2
        pickle.dump({'time_window_size': time_window_size,
                     'reg': reg,
                     'cv': n_folds,
                     'X_level2': np.squeeze(X_test_level2[:,:,:,1]),
                     'y_level2_conf': y_test_level2,
                     'y_level2': y_level2,
                     'gamma_level2': gamma_test_level2,
                     'order_combinations': order_combinations,
                     },
                    open(filename_level2, 'w'),
                    protocol = pickle.HIGHEST_PROTOCOL)    
    
    #Granger features     
    filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger.pickle' % (pwd, order_granger, n_folds)        
    try:
        print "Loading", filename_level2_granger
        level2 = pickle.load(open(filename_level2_granger))
        X_test_level2_granger = level2['X_level2']
        
    except IOError:
        print "Not found!"
        print
        # The following is the parallel (multicore) loop over all classes calling compute_lev2_granger_general_case():
        X_test_level2_granger = []
        y_test_level2 = []
        gamma_test_level2 = []
        
        
        for i_file in range(configurations):
        
            print "n. file:", i_file
            filedata = '%s%s%d%s' % (pwd_source,filedata_structure,i_file,'.pickle')                 
            print "Loading file data", filedata
            
            tmp_X_test_level2_granger, tmp_y_test_level2, tmp_gamma_test_level2, order_combinations = compute_lev2_granger_general_case(filedata, order_granger, N, n_jobs)
            X_test_level2_granger.append(tmp_X_test_level2_granger) 
            y_test_level2.append(tmp_y_test_level2)
            gamma_test_level2.append(tmp_gamma_test_level2) 
            
        X_test_level2_granger = np.vstack(X_test_level2_granger)
        y_test_level2 = np.vstack(y_test_level2)    
        print
        print "Saving level2 dataset in", filename_level2_granger
        pickle.dump({'time_window_size': time_window_size,
                     'order_granger': order_granger,
                     'X_level2': np.squeeze(X_test_level2_granger),
                     'y_level2_conf': y_test_level2,
                     'gamma_level2': gamma_test_level2,
                     'order_combinations': order_combinations,
                     },
                    open(filename_level2_granger, 'w'),
                    protocol = pickle.HIGHEST_PROTOCOL)
        