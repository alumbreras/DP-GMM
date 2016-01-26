# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 – Thomson Licensing, SAS

The source code form of this open source project is subject to the terms of the
Clear BSD license.

You can redistribute it and/or modify it under the terms of the Clear BSD
License (See LICENSE file).

Description: Set of experiments for the paper
Author: Alberto Lumbreras
"""

# Warnings: 
# P_test_ includes intercept
# P_test does not include it
# Be careful with unintentional modifications of the original objects in numpy !

from __future__ import division

import argparse
import cPickle as pck 
import csv
import random
import time, os
import traceback 
from functools import partial 

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pylab as plt
from numpy.linalg import LinAlgError
from rpy2.rinterface import RRuntimeError
from sklearn.metrics import adjusted_rand_score

seaborn.set_style("white")

from eval_metrics import loglike, pairwise_clustering, least_squares_clustering
from plot_posterior import plot_posterior
from supervised_clustering import GibbsSampler

#import gendata # delete once I am sure I do not use that fiel anymore

def benchmark_models(U=100, T=100, density=0.5, nsamples=100, 
                     datapath="clear", outpath='./out/', 
                     worker=None, experiment=None):
    """
    Compare DP, fixed, and no-roles in a setting of 
    U users and T threads where participations matrix has a given density.
    """
    R = 5
    F = 2 #TODO: autodetect from A later
    
    # Iris dataset
    R = 3
    F = 3
    ######################################################
    # Data
    ######################################################  
    datafile = os.path.join(datapath, "data_users_" + str(U) + ".csv")
    df = pd.read_csv(datafile, delimiter='\t')
    z_true = np.asarray(df['z'])
    b_true = np.asarray(df['b'])
    #b = np.asarray(df['b'])
    A = np.asarray(df[['f1', 'f2']]).T      
    
    # iris dataset
    A = np.asarray(df[['f1', 'f2', 'f3']]).T   
      
    # Participations
    # (training change at every experiment, to simulate a Cross Validation)
    ####################################################################
    # Read from synthetic thread, select a random subset of T
    datafile = os.path.join(datapath, "train_participations_" + str(U) + ".csv")
    df = pd.read_csv(datafile, delimiter='\t')
    P_all = df.as_matrix()

    datafile = os.path.join(datapath, "train_lengths_" + str(U) + ".csv")
    df = pd.read_csv(datafile, delimiter='\t')
    y_all = df.as_matrix().flatten()

    # Randomly select training set of T threads
    # features are fixed. The variance in the results will be caused by the
    # different traininig participation matrices
    T_all = P_all.shape[1]    
    idx = np.random.permutation(T_all)
    idx_training = idx[:T] # T threads  for training
    P = P_all[:,idx_training]
    y = y_all[idx_training]

    # Test set
    datafile = os.path.join(datapath, "test_participations_" + str(U) + ".csv")
    df = pd.read_csv(datafile, delimiter='\t')
    P_test = df.as_matrix()

    datafile = os.path.join(datapath, "test_lengths_" + str(U) + ".csv")
    df = pd.read_csv(datafile, delimiter='\t')
    y_test = df.as_matrix().flatten()

    
    #P = gendata.gen_participations(U=U, T=T, density=density)    
    #P_test = gendata.gen_participations(U=U, T=100, density=density)
    
    P_test_ = np.vstack((np.ones(P_test.shape[1]), P_test))  #includes intercept
    if (P[:,:].sum(axis=1)==0).sum() > 0:
        raise ValueError("Some users in 0 threads")
    if (P_test[:,:].sum(axis=1)==0).sum() > 0:
        raise ValueError("Some users in 0 threads")
    
    # Generate lengths
    s_y = 1/np.sqrt(20) # noise to threads lengths 
    #y = gendata.gen_lengths(P, b, s_y)    
    #y_test = gendata.gen_lengths(P_test, b, s_y)        
    

 
    ######################################################
    # Gibbs sampler 
    # ( = Training = model fitting = get samples from the posterior)
    ######################################################    
    # Non-parametric model   
    worker_exp = worker + "-DP-" + experiment  
    Gibbs_DP = GibbsSampler(P = P, y = y, A = A, R = 1, DP = True, 
                            z = z_true, b =np.hstack((0, b_true)),
                            outpath = outpath,
                            worker_exp = worker_exp)
                            

    # Model with R roles (knows the true number of clusters)
    worker_exp = worker + "-fixed-" + experiment 
    Gibbs_fixed_roles = GibbsSampler(P = P, y = y, A = A, R = R, DP = False, 
                                     z = z_true, b =np.hstack((0, b_true)),
                                     outpath = outpath,
                                     worker_exp = worker_exp)
        
    # Model with no roles
    worker_exp = worker + "-norole-" + experiment 
    Gibbs_norole = GibbsSampler(P = P, y = y, A = A, R = 1, DP = False, 
                                b =np.hstack((0, b_true)),
                                outpath = outpath,
                                worker_exp = worker_exp)
      
    models = [Gibbs_DP, Gibbs_fixed_roles, Gibbs_norole]               
    models_names = ["DP", "fixed", "norole"]
    animated=False
            
    for m_idx, model in enumerate(models):         
        ###############
        # Training
        ###############
        model.fit(nsamples, animated=animated,  title=models_names[m_idx])        
        n = model.last_iter
        
        T_test = P_test.shape[1]
        burning = int(0.5*n)
        samples_idx = np.arange(burning, n)             
        
        ###############
        # Test
        ###############  
        # (predictions) Negative loglikelihood     
        b_samples = model.traces['b'][samples_idx]
        s_y_samples = model.traces['s_y'][samples_idx]   
        negloglike = - loglike(y_test, P_test_, b_samples, 1/np.sqrt(s_y_samples))

        # (clustering) Least Squares Model-based Clustering
        z_samples = model.traces['z'][samples_idx]
        pairwise_posterior = pairwise_clustering(z_samples)
        ls_z = least_squares_clustering(z_samples, pairwise_posterior)
        
        # (clustering) Adjusted Rand Index
        ari = adjusted_rand_score(z_true, ls_z)
        
        ###############
        # Plots
        ###############
        fig, ax = plt.subplots(1,2, figsize=(16,8), subplot_kw={'autoscale_on':'True'})
        fig.canvas.set_window_title(models_names[m_idx])
        # Use R plot_posteriors.r instead
        if True:
            # TODO: if 1000 threads, very slow...
            # Plot predictive posterior distribution (mean and credible intervals)
            y_preds = model.sample_predictive_posterior_y(P_test_, samples_idx)
            print y_preds[:,np.argsort(y_test)]
            plot_posterior(y_preds[:,np.argsort(y_test)], ax=ax[0])          
            seaborn.despine(ax=ax[0])
    
            # Plot truth
            ax[0].plot(np.arange(T_test), y_test[np.argsort(y_test)], c='r')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            
            # Plot pairwise clustering matrix
            seaborn.heatmap(pairwise_posterior, vmin=0, vmax=1, ax=ax[1], cbar=False, 
                            xticklabels=False, yticklabels=False) 
            ax[1].set_xticks([])
            ax[1].set_yticks([])  
        
        ###############
        # Save into files
        ###############  
        # TODO: this is a mess. Get this path once on top of the loop
        # and use it also when constructing the model
        worker_exp = worker + "-" + models_names[m_idx] + "-" + experiment
        # All plots are stored here        
        plots_dir = os.path.join(outpath, worker_exp, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        model.save_traces()
        
        s_y_str = str(s_y).split('.')[0] + "_" + str(s_y).split('.')[1][:2]
        outfile = os.path.join(plots_dir,
                               "U_%d_T_%d_sigma_y_%s_samples_%d_%s" % 
                               (U, T, s_y_str, n, models_names[m_idx]))

        plt.savefig(outfile + ".eps", format='eps')
        plt.savefig(outfile + ".png", format='png')
        plt.close(plt.gcf())
        
        filename = 'results.csv'
    
        outfile = os.path.join(outpath, filename)
        with open(outfile,'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([models_names[m_idx], U, T, s_y_str, n, negloglike, ari])

             
    return models


def repetition(u, t, nsamples, outpath, datapath, i):
    # Every repetition is given to a new worker
    random.seed(i) # to avoid two processes give the very same result!
    datadir = os.path.basename(datapath)
    
    print("\n rep:%d %s" % (i, time.strftime("%c")))
    worker_name = mp.current_process().name
    print "Worker:", worker_name
    print "-------------------------------------------"
    
    finished = False
    while not finished:
        try:                            
            benchmark_models(U=u, T=t, density=0.5, 
                             nsamples=nsamples, 
                             outpath=outpath, 
                             datapath=datapath, 
                             worker = worker_name,
                             experiment = str(datadir) + "-" + 
                                                str(u) + "-" + 
                                                str(t) + "~" + 
                                                str(i))
            finished = True
        except (RRuntimeError, ValueError, LinAlgError, AssertionError, IndexError) as e:
            # LinAlgError: - when matrix is not sdp. Usually if T=10
            #              - when SVD did not converge (LAPACK issues)
            # RRuntimeError: - when exception in ARS. 
            # ValueError: - when users in 0 threads ?
            print e
            traceback.print_exc()
            #print "RETRY..."
    

if __name__ == "__main__":
    
    DEBUG=False
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Execute experiments')
    parser.add_argument('--reps', type=int, nargs='?', default=5)
    parser.add_argument('--samples', type=int, nargs='?', default=20000)
    parser.add_argument('--datapath', nargs='?', default='./data/real') # do not write clear/
    args = parser.parse_args()
    print "Arguments:", args
    nsamples = args.samples # 30000
    nreps = args.reps # 7
    datapath = args.datapath
    datadir = os.path.basename(datapath)

    # Set up output folder with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M")
    outpath = os.path.join(os.path.dirname(__file__), 'out', datadir, timestamp)
    try:
        os.makedirs(outpath)    
    except OSError as exception:
        print "Directory already exists" 
        
    # Set up directory to monitor workers
    if not os.path.exists(".workers/"):
        os.makedirs(".workers/")
    else:
        filelist = os.listdir(".workers/")
        for filename in filelist:
            filepath = os.path.join('.workers', filename) 
            os.remove(filepath) 
    
    # Go!        
    # Every setting U, T, is repeated nrep times in parallel to average results.      
    for u in [50]: 
        for t in [10, 50, 100, 500]:
            print "\n\n", "*"*20,
            print("%s U=%d T=%d" % (time.strftime("%c"), u, t))

            with open(".experiments.log", "a") as fexp:
                fexp.write("%s U=%d T=%d\n" % (time.strftime("%c"), u, t))
                        
            if not DEBUG:
                p = mp.Pool(nreps)
                common_args = [u,t,nsamples,outpath,datapath]
                p.map(partial(repetition, *common_args), range(nreps))
            
            # Use this for debugging
            if DEBUG:
                repetition(u,t,nsamples,outpath,datapath, 1)            
            
            print "###########################################"
            print " Pool processed. Next!"
            print "###########################################"