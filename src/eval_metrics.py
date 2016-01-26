# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:23:30 2015

@author: alumbreras
"""
import numpy as np
from scipy.stats import norm

def loglike(y_true, X, coeff_samples, std_samples):
    """
    Compute average log likelihood of the true values
    according to the posterior distribution of the parameters:
    p(y_test | y_train) = 1/M \int p(y_test | b, s_y)p(b, s_y | y_train)
    
    Parameters:
    y_true (nummpy.array): true values
    X (numpy.ndarray): n by d input values
    coeff_samples (numpy.ndarray): n samples of the d regression coefficients
    std_samples (numpy.array): n samples of the standard deviation of y
    """
    n_dim = y_true.shape[0]
    n_sample = coeff_samples.shape[0]
    n_coeff = coeff_samples.shape[1]
    assert(n_coeff == X.shape[0]), "Dims do not match"
    assert(n_dim == X.shape[1]), "Dims do not match"
    
    loglike = 0
    locs = np.dot(X.T, coeff_samples.T)
    for j in xrange(n_sample):
        loglike += norm.logpdf(y_true, 
                           loc = locs[:,j], #np.dot(X[:,i].T, coeff_samples[j]), 
                           scale = std_samples[j]).sum() 
    return 1.0*loglike/(n_sample*n_dim)
  
  
def pairwise_clustering(label_samples):
    """
    Compute the pairwise posterior clustering
    Parameters:
    ===========
    label_samples (numpy.ndarray): n samples of the d-labels array
    
    Returns:
    ==========
    square pairwise coincidence matrix with counts of how many times two users
    have been assigned to the same cluster
    """
    n_sample = label_samples.shape[0]
    n_dim = label_samples.shape[1]
    pairwise_posterior = np.zeros((n_dim,n_dim))
    for labels in label_samples:
        pairwise_posterior += (labels[np.newaxis,:] == 
                                    labels[:,np.newaxis]).astype(int)   
    return pairwise_posterior/n_sample
    

def least_squares_clustering(z_samples, pairwise_posterior):
    """
    Least Squares Model-based Clustering
    http://dahl.byu.edu/papers/dahl-2006.pdf page 208
    Search, amongst the sample, the candidate clustering closest to the 
    posterior pairwise
    """        
    best_labels = z_samples[0]
    ls_min = 1000000000
    for labels in z_samples:
        ls = (((labels[np.newaxis,:] == labels[:,np.newaxis]).astype(int) - \
                pairwise_posterior)**2).sum()
        if ls < ls_min:
            ls_min = ls
            best_labels = labels
    return best_labels     