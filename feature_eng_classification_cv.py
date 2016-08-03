""" Feature engineering and classification
    
    INPUT
    configurations: n. of causal configurations among 3 signals; 
    n_folds:  n. of folds in the classification, default=5;
    n_jobs: CPU cores;
    test_set: bool variable if False the loaded dataset is cross-validated, if True two datasets are loaded the train and test sets;
    time_window_size: time points considered in the regression;
    order_granger: order of the MAR model;
    gamma_threshold: for filtering noisy trials, if gamma_thresholding=True;
    feat_sel: selection of a subset of feautes, if reduce_feat=True;
    
    OUTPUT:
    y_test_pred: predicted causal configurations;
    y_test_true: ground truth;
    predicted_probability: predicted probability for each connections;
    score_matrix: cost assigned in case of wrong predictions.   

"""
import numpy as np
import pickle
from scipy.misc import comb
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sys import stdout
from joblib import Parallel, delayed

from score_function import compute_score_matrix, best_decision
from create_level2_dataset import feature_engineering, feature_normalisation

        
if __name__=='__main__':

    configurations = 64 # n. of causal configurations among 3 time series
    classes = range(configurations)
    n_folds = 5 # folds in which train and test are divided, if only one dataset folds of the cross-validation procedure
    n_jobs = -1 # '-1' = use all available CPU cores
    test_set = False # True: after training the classifier will be applied on this dataset, False: train set wiil be cross-validated    
    
    ###############################################   
    print "Train set:"
    time_window_size_train = 10
    order_granger_train = 10    
    pwd_source_train = 'data/'
    
    filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)
    print "Loading", filename_level2_granger
    level2 = pickle.load(open(filename_level2_granger))
    X_train_level2_granger = level2['X_level2']
    
    # Trainset R2 
    filename_level2 = '%sdataset_level2_tws%d_cv%d_r2_shift_window_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)        
    print "Loading", filename_level2
    level2 = pickle.load(open(filename_level2))
    X_train_level2 = level2['X_level2']
    y_train_level2 = np.squeeze(level2['y_level2'])
    y_train_level2_conf = np.squeeze(level2['y_level2_conf'])    
    gamma_train_level2 = level2['gamma_level2']
    order_combinations = level2['order_combinations']

    # Trainset MSE
    filename_level2_mse = '%sdataset_level2_tws%d_cv%d_mse_shift_window_norm.pickle' % (pwd_source_train, time_window_size_train, n_folds)
    print "Loading", filename_level2_mse
    level2_mse = pickle.load(open(filename_level2_mse))
    X_train_level2_mse = level2_mse['X_level2']
    y_train_level2_mse = np.squeeze(level2_mse['y_level2'])
    y_train_level2_conf_mse = np.squeeze(level2_mse['y_level2_conf'])
    gamma_train_level2_mse = level2_mse['gamma_level2']
    assert((y_train_level2_mse == y_train_level2).all())
    assert((y_train_level2_conf_mse == y_train_level2_conf).all())
    assert((gamma_train_level2_mse == gamma_train_level2).all())
    
    #############################################
    if test_set:
        print "Test set"
        time_window_size_test=10
        order_granger_test=10  
        pwd_source = 'data/'
    
        filename_level2_r2 = '%sdataset_level2_tws%d_cv%d_r2_shift_window_test.pickle' % (pwd_source, time_window_size_test, n_folds)     
        filename_level2 = filename_level2_r2
        print "Loading", filename_level2
        level2 = pickle.load(open(filename_level2))
        X_level2 = level2['X_level2']
        y_level2_conf = level2['y_level2_conf'] #the entire configuration matrix nCh x nCh
        y_level2 = np.squeeze(level2['y_level2'])
        gamma_level2 = level2['gamma_level2']
            
        filename_level2_mse = '%sdataset_level2_tws%d_cv%d_mse_shift_window_test.pickle' % (pwd_source, time_window_size_test, n_folds)     
        filename_level2 = filename_level2_mse
        print "Loading", filename_level2
        level2 = pickle.load(open(filename_level2))
        X_level2_mse = level2['X_level2']
        y_level2_conf_mse = level2['y_level2_conf']
        y_level2_mse = np.squeeze(level2['y_level2'])
        gamma_level2_mse = level2['gamma_level2']
        
        assert((y_level2_conf == y_level2_conf_mse).all())    
        assert((y_level2 == y_level2_mse).all())    
        assert((gamma_level2_mse == gamma_level2).all())
        
        filename_level2_granger = '%sdataset_level2_tws%d_cv%d_granger_test.pickle' % (pwd_source, order_granger_test, n_folds)    
        print "Loading", filename_level2_granger
        level2 = pickle.load(open(filename_level2_granger))
        X_level2_granger = level2['X_level2']
    

    print
    print "Doing classification."
    from create_trainset import class_to_configuration
    from sklearn.linear_model import LogisticRegression
    from itertools import izip
        
    clf = LogisticRegression(C=1.0, penalty='l2', random_state=0)    
    print clf

    gamma_thresholding = False # Filtering trials according to the noise level
    gamma_threshold = 1.0
    print "Keeping all training examples with gamma < %s" % gamma_threshold
    nTrial_train = y_train_level2.shape[0]    
    if gamma_thresholding:
        idx = gamma_train_level2 < gamma_threshold        
    else:
        idx = np.ones(nTrial_train,dtype=np.bool)
    
    print "Reducing features" # Experiments related to the role of the features
    reduce_feat = False
    if reduce_feat:
        feat_sel=np.array([9,10,12,14,16,17,18,19,20]) #pairWise conditional (c-pw) feature space
                #np.array([0,4,8,9,10,12,14,16,17]) #pairWise (pw) feature space
        X_train_level2 = X_train_level2[:,feat_sel]
        X_train_level2_mse = X_train_level2_mse[:,feat_sel]
        
    X_train = X_train_level2[idx]
    y_train = y_train_level2[idx]
    y_train_conf = y_train_level2_conf[idx]
    X_train_granger = X_train_level2_granger[idx] 
    X_train_mse = X_train_level2_mse[idx]
    del X_train_level2, y_train_level2, y_train_level2_mse,  y_train_level2_conf, y_train_level2_conf_mse, X_train_level2_granger, X_train_level2_mse

    print "Preprocessing Train set"
    block_normalisation = True
    feature_space_train = feature_engineering([X_train, X_train_granger, X_train_mse], block_normalisation=block_normalisation)
    del X_train, X_train_granger, X_train_mse

    block_normalisation = False
    X_train = feature_normalisation(feature_space_train, block_normalisation=block_normalisation)
    
    # Same procedure for the test set, if present
    if test_set:    
        gamma_threshold = -1
        print "Keeping all testing examples with gamma < %s" % gamma_threshold
        nTrial_test = y_level2.shape[0]
        if gamma_thresholding:
            idx = gamma_level2 < gamma_threshold        
        else:
            idx = np.ones(nTrial_test,dtype=np.bool)
        
        print "Reducing test data features"
        if reduce_feat:
            X_level2 = X_level2[:,feat_sel]
            X_level2_mse = X_level2_mse[:,feat_sel]
        
        X = X_level2[idx]
        y = y_level2[idx]
        y_level2_conf = y_level2_conf[idx]
        X_granger = X_level2_granger[idx] 
        X_mse = X_level2_mse[idx]       
        del X_level2, X_level2_mse, X_level2_granger, gamma_level2, gamma_level2_mse, y_level2, y_level2_mse, y_level2_conf_mse
        
        print "Preprocessing Test set"
        block_normalisation = True
        feature_space = feature_engineering([X, X_granger, X_mse], block_normalisation=block_normalisation)    
        del X, X_granger, X_mse    
        
        block_normalisation = False
        X = feature_normalisation(feature_space, block_normalisation=block_normalisation)
    
    else:
        X = X_train
        y = y_train
        y_level2_conf = y_train_conf
    
    
    print "X train:", X_train.shape
    print "X test", X.shape
    print "Learning and prediction."
    cv_train = StratifiedKFold(y_train, n_folds=n_folds)
    cv_test = StratifiedKFold(y, n_folds=n_folds)
    cv = izip(cv_train, cv_test)
    
    binary_class = True
    nComb = order_combinations.shape[0]
    nTrial, nCh = y_level2_conf.shape[:2]

    if binary_class:
            
        score_matrix = np.array([[1,0],[0,1]], dtype=int) #cost assigned to [true_pos, false_neg, false_pos, true_neg]
        y_new = []
        y_new += [(class_to_configuration(y_i, verbose=False)) for y_i in np.squeeze(y_train)] 
        y_new = np.array(y_new, dtype=int)
        y_pred = np.zeros([y.shape[0],nCh,nCh], dtype=int)
        predicted_probability = np.zeros((X.shape[0], 6, 2))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i
            train = train[0] #train part of the train set
            test = test[1] #test part of the test set
            
            index_x = np.append(np.triu_indices(nCh,1)[0], np.tril_indices(nCh,-1)[0])
            index_y = np.append(np.triu_indices(nCh,1)[1], np.tril_indices(nCh,-1)[1])
                        
            for j in range(len(index_x)):
                print "Train."   
                clf.fit(X_train[train], y_new[train,index_x[j],index_y[j]])
                
                print "Predict."
                y_pred_proba = clf.predict_proba(X[test])
                predicted_probability[test,j] = y_pred_proba.copy()                        
                y_pred[test,index_x[j],index_y[j]] = np.array([best_decision(prob_configuration, score_matrix=score_matrix)[0] for prob_configuration in y_pred_proba])
        
        y_pred_conf = np.zeros((nTrial, nCh, nCh))
        for i_trial in range(nTrial):
            for i_comb in range(order_combinations.shape[0]):
                y_pred_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]] += y_pred[i_trial*nComb+i_comb]

    else: 
        
        y_pred = np.zeros(y.shape, dtype=int)
        score_matrix = compute_score_matrix(n=configurations, binary_score=[1,0,0,1])
        predicted_probability = np.zeros((X.shape[0], configurations))
        
        for i, (train, test) in enumerate(cv):
            print "Fold %d" % i            
            train = train[0] #train part of the train set 
            test = test[1] #test part of the test set
            
            print "Train."        
            clf.fit(X_train[train], y_train[train])

            print "Predict."
            y_pred_proba = clf.predict_proba(X[test])
            predicted_probability[test] = y_pred_proba.copy()                    
            y_pred[test] = np.array([best_decision(prob_configuration, score_matrix=score_matrix)[0] for prob_configuration in y_pred_proba])

        y_pred_conf = np.zeros((nTrial, nCh, nCh))
        for i_trial in range(nTrial):
            for i_comb in range(order_combinations.shape[0]):
                y_pred_conf[i_trial][order_combinations[i_comb][:,None], order_combinations[i_comb]] += class_to_configuration(y_pred[i_trial*nComb+i_comb])
        
    
    print "Set zero the diagonal"
    for i_trial in range(nTrial):
        y_pred_conf[i_trial][np.diag_indices(nCh)] = 0
        y_level2_conf[i_trial][np.diag_indices(nCh)] = 0
    
    pwd_dest = 'data/'     
    filename_save = '%ssimulated_Ldataset_tws%d_r2_mse_granger_binary_class_rowNorm_fEng_cv.pickle' % (pwd_dest, time_window_size_test)
    print "Saving %s" % filename_save
    pickle.dump({'y_test_pred': y_pred_conf,
                 'y_test_true': y_level2_conf,
                 'predicted_probability': predicted_probability,
                 'score_matrix': score_matrix,                     
                 },
                open(filename_save, 'w'),
                protocol = pickle.HIGHEST_PROTOCOL)