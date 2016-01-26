# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:09:57 2014
Test supervided clustering with toy data

@author: Alberto Lumbreras
"""
from __future__ import division

import sys
sys.path.append('../')

import numpy as np
from numpy import dot, isnan
from matplotlib import pylab as plt
from sklearn.metrics import r2_score, mean_squared_error


from supervised_clustering import GibbsSampler
import gendata        

from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
coda = importr('coda')

############################################
############################################
############################################


if __name__ == "__main__":
    R = 5 # number of roles 
    U = 50  # number of users
    T = 500 # number of threads 
    F = 2 # number of attributes

    ######################################
    ## DATA GENERATION
    ######################################
    # Assign roles uniformly
    z_true = np.zeros(U, dtype=np.int8)
    binsize = int(U/R)
    for i in xrange(R):
        z_true[binsize*i:] = i
        
    # Generation of participations
    ###############################
    
    # The higher the role id, the higher its coefficient
    mu_br = np.linspace(-200,200,R)
    s_br = np.ones(R)
    s_y = 1/np.sqrt(20) # noise to threads lengths 

    b = gendata.gen_b(z_true, mu_br, s_br)
    P = gendata.gen_participations(z_true, U=U, T=T, density=0.5)
    y = gendata.gen_lengths(P, b, s_y)
    P_test = gendata.gen_participations(z_true, U=U, T=T, density=0.5)
    y_test = gendata.gen_lengths(P_test, b, s_y)    
      
    plt.title("Coefficients")
    plt.yticks([])
    plt.xlim(-200,200 )
    plt.ylim=(-30,30)
    plt.scatter(b, [0]*len(b))
    plt.show()
      
    assert(P.shape[0] == U+1)
    assert(P.shape[1] == T)
    print "z:", z_true, len(z_true)
    print "Participations P:", P.shape
   
    # Generation of attributes
    ###############################
    plt.figure()
    A = gendata.gen_line(F=F, U=U, z=z_true)    
    plt.xticks([])
    plt.yticks([])
    plt.scatter(A[0,:], A[1,:])
    plt.title("Features")
    plt.show()
    
    ######################################################
    # Gibbs sampler ( = model fitting = get samples from the posterior)
    ######################################################  
    nsamples = 100
    animated = False
    
    # Non-parametric model   
    #model = GibbsSampler(P = P, y = y, A = A, R = 1, DP = True, z = z_true)
    model = GibbsSampler(P = P, y = y, A = A, R = 1, DP = True, z = np.zeros(U))                    
    #model = GibbsSampler(P = P, y = y, A = A, R = 1, DP = False, z = np.zeros(U))    
    model.fit(nsamples, animated=animated,  title="roles")    

if False:
    # Careful: if animated option, this code will execute before animation/Gibbs ends
    # and raises exception since it does not have the results yet.

    fig, ax = plt.subplots(1,2, sharey=True, sharex=True, figsize=(16,8))
    n = model.last_iter
    
    T_test = P_test.shape[1]
    burning = int(0.5*n)
    samples_idx = np.arange(burning, n)             
    
    
    # Plot truth in both axes
    ax[0].plot(np.arange(T_test), y_test[np.argsort(y_test)], c='k')
    ax[1].plot(np.arange(T_test), y_test[np.argsort(y_test)], c='k')    
    
    # Plot draws from predictive posterior
    y_preds = model.sample_predictive_posterior_y(P_test, samples_idx)
    for i in xrange(5):
        ax[0].plot(np.arange(T_test), y_preds[i,np.argsort(y_test)], c='k', ls=":")        
        
    # Plot predictions from MAP estimators
    y_preds = model.predict_y_MAP(P_test, samples_idx)
    ax[1].plot(np.arange(T_test), y_preds[np.argsort(y_test)], c='k', ls=":")

    # Save into files
    plt.margins(0.05, 0.01)
    s_y_str = str(s_y).split('.')[0] + "_" + str(s_y).split('.')[1][:2]
    plt.savefig("./img/test_%s_U=%d_T=%d_sigma_y=%s_samples=%d.eps" % ("toy", U, T, s_y_str, n), format='eps')
    plt.savefig("./img/test_%s_U=%d_T=%d_sigma_y=%s_samples=%d.png" % ("toy", U, T, s_y_str, n), format='png')
    ax[0].set_title("./img/test_%s_U=%d_T=%d_sigma_y=%s_samples=%d.png" % ("toy", U, T, s_y_str, n))
    plt.show()
    r2 = r2_score(y_test, y_preds)
    mse = mean_squared_error(y_test, y_preds)

    y_pred = dot(P_test.T, b)
    plt.plot(np.arange(T_test), y_pred[np.argsort(y_test)])
    plt.title("Prediction with true coefficients")
    plt.show()  

if False:
#################################
# Convergence checks
#################################
    tr = model.traces_beta_b0[:nsamples]
    trmcmc = coda.mcmc(tr)
    coda.effectiveSize(trmcmc)
    coda.autocorr_plot(trmcmc)
    coda.geweke_plot(trmcmc)
    coda.geweke_diag(trmcmc)
    coda.heidel_diag(trmcmc)
