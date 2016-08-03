"""
Compute the ROC curve and AUC for CBC and MBC (in this latter case by defining different cost matrices)
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from score_function import best_decision, compute_score_matrix
from create_trainset import class_to_configuration
import itertools

def compute_roc_auc(y_test_pred, y_test_level2, predicted_probability, nTrial, nCh, mvgc_flag):
    
    print "Assigning label"
    index_x = np.append(np.triu_indices(nCh,1)[0], np.tril_indices(nCh,-1)[0])
    index_y = np.append(np.triu_indices(nCh,1)[1], np.tril_indices(nCh,-1)[1])
    #################################
    print "In case of mvgc"
    if mvgc_flag:
        pred_prob_tmp = []
        pred_prob_tmp += [predicted_probability[i][index_x[None,:],index_y] for i in range(nTrial)]
        pred_prob_tmp = np.vstack(pred_prob_tmp)
        predicted_probability = np.zeros([nTrial, len(index_x), 2])
        predicted_probability[:,:,0] = pred_prob_tmp
        predicted_probability[:,:,1] = 1-pred_prob_tmp
        del pred_prob_tmp
    
    print "Stacking for using sklearn roc_curve, for MVGC and SL"
    y_true = []
    y_true += [y_test_level2[i][index_x[None,:],index_y] for i in range(nTrial)]
    y_true = np.hstack(np.vstack(y_true))
    predicted_probability_class1 = np.hstack(np.squeeze(predicted_probability[:,:,1]))
    
    print "Roc curve computed by sklearn"
    from sklearn import metrics
    idx_not_nan=np.logical_not(np.isnan(predicted_probability_class1))
    fpr, tpr, thresholds = metrics.roc_curve(y_true[idx_not_nan], predicted_probability_class1[idx_not_nan], pos_label=1)
    auc = metrics.roc_auc_score(y_true[idx_not_nan], predicted_probability_class1[idx_not_nan])
    return fpr,tpr,auc


if __name__ == '__main__':

    cbc = True #CBC or MBC
    pwd = 'data/'
    
    if cbc:
        
        print "CBC cell based classifier"
        filename_open = '%ssimulated_Ldataset_tws10_r2_mse_granger_binary_class_rowNorm_fEng_cv.pickle' % (pwd)
        print "Opening %s" % filename_open            
        data = pickle.load(open(filename_open))
        y_test_pred = data['y_test_pred']
        y_test_level2 = data['y_test_true']   
        predicted_probability = data['predicted_probability']    
        nTrial, nCh = y_test_level2.shape[:2]
        
        mvgc_flag=0        
        fpr,tpr,auc_score = compute_roc_auc(y_test_pred, y_test_level2, predicted_probability, nTrial, nCh, mvgc_flag)
        plot_label = 'CBC'

    else:
        
        print "MBC matrix based classifier"
        filename_open = '%ssimulated_Ldataset_tws10_r2_mse_granger_notBinary_class_rowNorm_fEng_cv.pickle' % (pwd)
        print "Opening %s" % filename_open            
        data = pickle.load(open(filename_open))
        y_test_pred = data['y_test_pred']
        y_test_level2 = data['y_test_true']   
        predicted_probability = data['predicted_probability']    
        nTrial, nCh = y_test_level2.shape[:2]
    
        n_iter=50
        fpr = np.zeros(n_iter)
        tpr = np.zeros(n_iter)    
        for i_iter, iter_i in enumerate(itertools.product(np.arange(-3,0,0.3),np.arange(0,1,0.2))):
            print "Building score matrix"
            print iter_i
            binary_score = [1,0,iter_i[0],iter_i[1]]
            score_matrix = compute_score_matrix(n=64, binary_score=binary_score)
            
            print "Compute prediction according to the score matrix"
            y_pred = np.array([best_decision(prob_configuration, score_matrix=score_matrix)[0] for prob_configuration in predicted_probability])
           
            y_pred_conf = []
            y_pred_conf += [class_to_configuration(y_pred[i_trial], verbose=False) for i_trial in range(nTrial)]
            y_pred_conf = np.array(y_pred_conf)
            
            print "Confusion matrices"
            conf_mat = np.zeros([2,2])
            n_conect = np.array(y_test_level2.sum(-1).sum(-1), dtype=np.float)   
            n_noconect = np.repeat(nCh*(nCh-1), nTrial) - n_conect
            true_pos = np.zeros(nTrial)
            false_pos = np.zeros(nTrial)
            false_neg = np.zeros(nTrial)
            true_neg = np.zeros(nTrial)
            
            for i_trial in range(nTrial):
                true_pos[i_trial] = np.logical_and(y_test_level2[i_trial], y_pred_conf[i_trial]).sum() 
                false_pos[i_trial] = np.logical_and( np.logical_xor(y_test_level2[i_trial], y_pred_conf[i_trial]), y_pred_conf[i_trial]).sum() - nCh #to remove the diagonal
                false_neg[i_trial] = np.logical_and( np.logical_xor(y_test_level2[i_trial], y_pred_conf[i_trial]), y_test_level2[i_trial]).sum()
                true_neg[i_trial] = np.logical_and(np.logical_not(y_test_level2[i_trial]), np.logical_not(y_pred_conf[i_trial])).sum() 
              
            conf_mat[0,0] = np.sum(true_pos)/np.sum(n_conect)#true_pos[i_bin_th].mean()
            conf_mat[0,1] = np.sum(false_neg)/np.sum(n_conect)#false_neg[i_bin_th].mean()
            conf_mat[1,0] = np.sum(false_pos)/np.sum(n_noconect)#false_pos[i_bin_th].mean()
            conf_mat[1,1] = np.sum(true_neg)/np.sum(n_noconect)#true_neg[i_bin_th].mean()#1 - conf_mat[i_bin_th,1,0]
            
            fpr[i_iter] = conf_mat[1,0]
            tpr[i_iter] = conf_mat[0,0]
        
        print "Compute auc"
        from sklearn import metrics
        x_point = np.append(np.insert(fpr,0,0),1)
        y_point = np.append(np.insert(tpr,0,0),1)
        auc_score = metrics.auc(x_point[np.argsort(x_point)], y_point[np.argsort(x_point)])
        plot_label = 'MBC'
    
    print "Roc curve"
    plt.plot(fpr, tpr,'.k', label=plot_label)
    plt.legend(loc=4, numpoints=1, scatterpoints=1,fontsize='x-large')
    fontsize=15    
    plt.plot([0,0,1], [0,1,1], '-.k')
    plt.plot([0,1], [0,1], '-.k')
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.show()  