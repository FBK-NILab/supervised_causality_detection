import numpy as np
import pickle
from scipy.io import loadmat
from statsmodels.stats.stattools import durbin_watson, omni_normtest, jarque_bera
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sys import stdout
from joblib import Parallel, delayed
import nitime.analysis as nta
from nitime.timeseries import TimeSeries
#from load_challenge_data import load_challenge_data


causality_structures = [((0,),0), ((0,),1), ((0,),2),
                        ((1,),0), ((1,),1), ((1,),2),
                        ((2,),0), ((2,),1), ((2,),2),
                        ((0,1),0), ((0,1),1), ((0,1),2),
                        ((0,2),0), ((0,2),1), ((0,2),2),
                        ((1,2),0), ((1,2),1), ((1,2),2),
                        ((0,1,2),0), ((0,1,2),1), ((0,1,2),2)]
    
def regression_scores(timeseries, time_window_size, time_lag, reg, cv, scoring, timeseriesZ=None):
    """Compute regression scores for a given set of 3 timeseries
    according to the causality structures.
    """
    global causality_structures
    if scoring == 'residual_tests':
        features_regression = np.zeros([len(causality_structures),7])
    else:
        features_regression = np.zeros([len(causality_structures),2]) #added 2 dimensions to compute r2 and mse
    for j, (cs_train, cs_test) in enumerate(causality_structures):
        ts_train = timeseries[:,cs_train]
        if not(timeseriesZ is None):
            ts_train = np.hstack([ts_train, timeseriesZ])
        
        if time_lag is None:
            time_lag=time_window_size
        ts_test = timeseries[:,cs_test]
        tmp_score = np.zeros([time_window_size,2]) #added 2 dimensions to compute r2 and mse
        residuals = np.zeros(timeseries.shape[0]-time_window_size)
        for i_reg in range(time_window_size):
            idx_example = np.arange(i_reg, timeseries.shape[0]-time_lag, time_window_size)[:-1]
            X = np.zeros((idx_example.size, time_window_size, ts_train.shape[1]))#len(cs_train)))
            for k in range(time_window_size):
                X[:,k] = ts_train[idx_example+k]
    
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            y = ts_test[idx_example + time_lag]
            if scoring == 'residual_tests':
                y_pred_i_reg = np.zeros(y.size)
                kfold = KFold(n=y.size, n_folds=cv)
                for train, test in kfold:
                    reg.fit(X[train], y[train])
                    y_pred_i_reg[test] = reg.predict(X[test])
                
                residuals[idx_example] = y - y_pred_i_reg #residuals
            else:
                tmp_predict = cross_val_predict(reg, X, y, cv=cv)
                tmp_score[i_reg,0] = r2_score(y,tmp_predict).mean()
                tmp_score[i_reg,1] = mean_squared_error(y,tmp_predict).mean()
                #tmp_score[i_reg] = cross_val_score(reg, X, y, cv=cv, scoring=scoring).mean()
        
        if scoring == 'residual_tests':
            features_regression[j,0] = durbin_watson(residuals)
            features_regression[j,[1,2]] = omni_normtest(residuals) 
            features_regression[j,3:] = jarque_bera(residuals)
        else:
            features_regression[j] = tmp_score.mean(0)

    return features_regression
    
    
def regression_scores_different_domain(timeseries_causes, timeseries_effect, time_window_size, reg, cv, scoring, timeseriesZ=None):
    """Compute regression scores for a given set of 3 timeseries as causes and 3 as effects
    according to the causality structures.
    """
    global causality_structures
    if scoring == 'residual_tests':
        features_regression = np.zeros([len(causality_structures),7])
    else:
        features_regression = np.zeros([len(causality_structures),2]) #added 2 dimensions to compute r2 and mse
    for j, (cs_train, cs_test) in enumerate(causality_structures):
        ts_train = timeseries_causes[:,cs_train]
        if not(timeseriesZ is None):
            ts_train = np.hstack([ts_train, timeseriesZ])
        
        ts_test = timeseries_effect[:,cs_test]
        tmp_score = np.zeros([time_window_size,2]) #added 2 dimensions to compute r2 and mse
        residuals = np.zeros(timeseries_causes.shape[0]-time_window_size)
        for i_reg in range(time_window_size):
            idx_example = np.arange(i_reg, timeseries_causes.shape[0], time_window_size)[:-1]
            X = np.zeros((idx_example.size, time_window_size, ts_train.shape[1]))#len(cs_train)))
            for k in range(time_window_size):
                X[:,k] = ts_train[idx_example+k]
    
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            y = ts_test[idx_example + time_window_size]
            if scoring == 'residual_tests':
                y_pred_i_reg = np.zeros(y.size)
                kfold = KFold(n=y.size, n_folds=cv)
                for train, test in kfold:
                    reg.fit(X[train], y[train])
                    y_pred_i_reg[test] = reg.predict(X[test])
                
                residuals[idx_example] = y - y_pred_i_reg #residuals
            else:
                tmp_predict = cross_val_predict(reg, X, y, cv=cv)
                tmp_score[i_reg,0] = r2_score(y,tmp_predict).mean()
                tmp_score[i_reg,1] = mean_squared_error(y,tmp_predict).mean()
                #tmp_score[i_reg] = cross_val_score(reg, X, y, cv=cv, scoring=scoring).mean()
        
        if scoring == 'residual_tests':
            features_regression[j,0] = durbin_watson(residuals)
            features_regression[j,[1,2]] = omni_normtest(residuals) 
            features_regression[j,3:] = jarque_bera(residuals)
        else:
            features_regression[j] = tmp_score.mean(0)

    return features_regression    


def granger_scores(timeseries, order):
    timeseries = TimeSeries(timeseries, sampling_interval=1)
    g = nta.GrangerAnalyzer(timeseries, order=order)
    g_xy_mat = np.mean(g.causality_xy, axis=-1)
    g_yx_mat = np.mean(g.causality_yx, axis=-1)
    return np.concatenate([g_xy_mat[np.tril_indices(3,-1)], g_yx_mat.T[np.triu_indices(3,1)]])


def feature_engineering(Xs, block_normalisation=False):
    print "Feature Engineering."
    feature_space = []
    for X in Xs:
        if block_normalisation :
            print "Block-normalization r2, mse, granger"
            X = row_normalise(X)#grand_normalise(X)#feature_scaling(X)
        feature_space += [X, np.power(X, 2), np.power(X, 3), np.sign(X) * np.sqrt(np.abs(X))]
        # Feature engineering: all possible products between the original feature values:
        feature_space.append(np.array([np.multiply.outer(X[i], X[i])[np.triu_indices(X.shape[1], 1)] for i in range(X.shape[0])]))
    return feature_space
        
def feature_engineering2(X, X_granger):
    # Feature engineering: all possible products between the original feature values:
    X_pairwise = np.array([np.multiply.outer(X[i], X[i])[np.triu_indices(X.shape[1], 1)] for i in range(X.shape[0])])
    X_granger_pairwise = np.array([np.multiply.outer(X_granger[i], X_granger[i])[np.triu_indices(X_granger.shape[1], 1)] for i in range(X_granger.shape[0])])
    # Add new features to the original ones:
    feature_space = [X, np.power(X, 2), np.power(X, 3), np.sign(X) * np.sqrt(np.abs(X)), X_pairwise, X_granger, np.power(X_granger, 2), np.power(X_granger, 3), np.sign(X_granger) * np.sqrt(np.abs(X_granger)), X_granger_pairwise]
    return feature_space


def feature_scaling(A):
    """Feature scaling according to wikipedia x-x_min / x_max-x_min 
    """  
    A = (A - A.min()) / (A.max() - A.min())
    return A

def grand_normalise(A):
    """Normalise (z-scoring) array A.
    """
    A = A - A.mean()
    A = np.nan_to_num(A / A.std())
    return A

def row_normalise(A):
    """Normalize along row array A
    """    
    A = column_normalise(A.T) 
    return A.T

def column_normalise(A):
    """NOrmalise along column array A
    """
    A = A - A.mean(0)
    A = np.nan_to_num(A / A.std(0))    
    return A
    
def feature_normalisation(feature_space_train, feature_space_test=None, block_normalisation=False):
    print "Normalisation."
    if feature_space_test is None:
        feature_space = feature_space_train
    else:
        size_train = feature_space_train[0].shape[0]
        feature_space = [np.vstack([A_train, A_test]) for A_train, A_test in zip(feature_space_train, feature_space_test)]
        
    if block_normalisation:
        print "Block-normalisation."
        X = np.hstack([grand_normalise(A) for A in feature_space])
    else:
        print "Per-feature Normalisation."
        X = np.hstack(feature_space)

    if feature_space_test is None:
        return X
    else:
        X_train = X[:size_train,:]
        X_test = X[size_train:,:]
        return X_train, X_test