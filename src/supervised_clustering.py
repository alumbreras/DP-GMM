# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 – Thomson Licensing, SAS

The source code form of this open source project is subject to the terms of the
Clear BSD license.

You can redistribute it and/or modify it under the terms of the Clear BSD
License (See LICENSE file).

Author: Alberto Lumbreras
"""

# See also: "Implementing the Infinite GMM":
# http://mr-pc.org/work/cs4771igmm.pdf 

# Frank Wood introduction:
# http://www.robots.ox.ac.uk/~fwood/talks/Wood-IGMM-Intro-2006.pdf

# Frank Wood's derivation of Gibbs sampler for finite GMM:
# http://www.robots.ox.ac.uk/~fwood/teaching/C19_hilary_2013_2014/gmm.pdf

# Edwin Chen's post and code R and Ruby:
# http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/

# Future work: variational version:
# for CV SBM (poterior P(Z) is a clique see http://math.univ-lille1.fr/~tran/Exposesgraphesaleatoires/Robin.pdf)

# type %matplotlib qt in the ipython console 
# to get interactive plots instead of inliners

# TODO: perplexity empty sometimes ? 

# Issues:
# - sometimes the comoponent precisions s_br are very similar, 
# - and this draws beta_b0 very high
# (or viceversa)
# Possible cause: numercial issues in ARS
# Update: it might be nothing.

from __future__ import division

import cPickle as pickle
import math
import os
import operator
import pdb
import sys
import traceback

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import mlab
from matplotlib import pylab as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import numpy as np
from numpy import dot
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mnorm

from scipy.stats import norm
from scipy.stats import multivariate_normal, gamma, wishart
from scipy.special import multigammaln

import pandas as pd

from rpy2 import robjects
from rpy2.robjects.packages import importr, STAP
from rpy2.rinterface import RRuntimeError
import rpy2.robjects.numpy2ri

rpy2.rinterface.initr() # it might be need for correct picking
rpy2.robjects.numpy2ri.activate()
MCMCpack = importr('MCMCpack')
ARS = importr('ars')
MASS = importr('MASS')



class GibbsSampler():
    
    def __init__(self, P=None,
                 y=None, s_y=None,  
                 z = None, R=1, alpha=10,
                 b=None, mu_br=None, s_br=None, # component parameters
                 A=None, mu_ar=None, S_ar=None, # component parameters
                 mu_b0 = None, r_b0 = None, w_b0 = None, beta_b0 = None,
                 mu_a0 = None, R_a0 = None, W_a0 = None, beta_a0 = None,
                 DP = True,
                 outpath = ".",
                 worker_exp = "worker_exp"):
                 #worker = "worker", experiment = "exp"):
    
        # TODO: merge worker and experiment argument  
        np.random.seed(seed=None)

        # Execution has a unique name. Useful to analyse results when multiprocessing
        self.worker_exp = worker_exp #worker + "-" + experiment

        # All traces are stored here        
        self.traces_dir = os.path.join(outpath, self.worker_exp, "traces")
        if not os.path.exists(self.traces_dir):
            os.makedirs(self.traces_dir)
        
        # TODO: If R>R-true animation keeps drawing covariances (?)
        if (P[:,:].sum(axis=1)==0).sum() > 0:
            raise ValueError("Some users in 0 threads")
                
        self.DP = DP
        
        # Observed data
        #self.P = np.vstack((np.ones(T), P)) # intercept included
        self.P = np.vstack((np.ones(P.shape[1]), P)) # add intercept        
        self.A = A # matrix of users features (attributes)
        self.y = np.array(y, dtype=np.int32)

  
        # Dimensions
        self.F = A.shape[0]
        self.U = A.shape[1]
        self.T = P.shape[1]

        # Data statistics to be used for hyper-priors       
        self.mu_a = np.mean(A, axis=1)
        self.Sigma_a = np.cov(A)
        self.Lambda_a = inv(self.Sigma_a)
        
        # For b, we use their MLE since they are not fixed observations
        alphaI = np.zeros((self.U+1,self.U+1))
        np.fill_diagonal(alphaI, 0.01) # regularization factor. i.e: 0.01
        self.b_MLE = dot(dot(inv(dot(self.P,self.P.T)+alphaI),self.P), self.y)  
        self.sigma_b = self.b_MLE.var()
        self.mu_b = self.b_MLE.mean()
        self.sigma_y = np.cov(self.y) #this was causing the strange bug of increasing negloglikes?
        #self.sigma_y = 25 # flat prior
        #self.sigma_y = 1000 #TODO do not sample s_y #TODO separar mas los beta
        # a mas rango de y mejor
        #self.sigma_y = np.var(self.y)
        
        ###############################################################
        # INIT STATE
        ###############################################################
        # Init z state
        if z is not None:
            if (len(z) != A.shape[1]):
                raise ValueError("Invalid length of z")
            if (max(z)+1>len(set(z))):
                raise ValueError("Invalid z labels. Must go from 0 to R-1 with no gaps")
            self.z = np.zeros(len(z)+1, dtype=np.int16) # add intercept
            self.z[1:] = z.copy()
            self.z[1:] +=1 # role 0 is kept for the intercept
            self.z[0] = 0 # intercept has its own role
        else:
            self.z = np.ones(self.U+1, dtype=np.int16)
            self.z[0] = 0
            
        # Number of classes. If Dirichlet Process, sync with initial z
        self.R = R
        if DP:
            self.R = len(set(self.z[1:]))
        else:
            if self.R < len(set(self.z[1:])):
                print self.R, len(set(self.z[1:]))
                raise ValueError("Clusters assigned > clusters available")
                
                
        # Hyper-parameters (common to all components).
        ########################################################  
        # Hyper-parameters shared by akk components
        # http://www.michaelchughes.com/blog/probability-basics/inverse-wishart-distribution/
        if alpha is not  None:
            self.alpha = alpha
        else:
            self.alpha = 2

        # Feature branch
        if mu_a0 is not None: 
            self.mu_a0 = mu_a0.copy()
        else: 
            self.mu_a0 = A.mean(axis=1)
            
        if R_a0 is not None:     
            self.R_a0 = R_a0.copy()
        else:
            self.R_a0 = inv(np.cov(A))
        self.R_a0_inv = inv(self.R_a0)
        
        if W_a0 is not None: 
            self.W_a0 = W_a0.copy()
        else:
            self.W_a0 =  np.cov(A)
        self.W_a0_inv = inv(self.W_a0)
        
        if beta_a0 is not None: 
            self.beta_a0 = beta_a0
        else:
            self.beta_a0 = self.F
            
        #TODO: asserts   
        assert(self.beta_a0>self.F-1) # in N-D wishart: beta_a0 > D-1
        
        # Prediction branch
        if mu_b0 is not None: 
            self.mu_b0 = mu_b0
        else:
            self.mu_b0 = 0
    
        if r_b0 is not None: 
            self.r_b0 = r_b0
        else:
            self.r_b0 = 1/100
        
        if w_b0 is not None: 
            self.w_b0 = w_b0
        else:
            self.w_b0 = 100
        
        if beta_b0 is not None: 
            self.beta_b0 = beta_b0 
        else:
            self.beta_b0 = 1       
        
        assert(np.abs(self.mu_b0) < np.inf)        
        assert(self.r_b0>0)
        assert(self.w_b0>0)
        assert(self.beta_b0>0) # in 1-D wishart: beta_b0 > 0
        
        # Component parameters.
        ###################################################
        # Feature-based clustering side
        if mu_ar is not None:
            self.mu_ar = mu_ar.copy()
        else:
            self.mu_ar = np.zeros((self.F, self.R), dtype=np.float32) 
        
        if S_ar is not None:
            self.S_ar = S_ar.copy() # Careful, this is a RxFxF matrix
        else:
            S_ar_base = inv(np.identity(self.F, dtype = np.float32)*0.2)
            self.S_ar = np.tile(S_ar_base, (self.R,1,1)) # one covariance matrix for every cluster
        self.S_ar_inv = inv(self.S_ar)    
        
        # Prediction-based clustering side
        if mu_br is not None:
            self.mu_br = mu_br.copy()
        else:
            self.mu_br = np.zeros(self.R+1, dtype=np.float32)
            
        if s_br is not None:
            self.s_br = s_br.copy()
        else:
            self.s_br = np.ones(R+1, dtype = np.float32)/1
            
        # Leave some space for new clusters
        self.padding = 100 
        self.mu_ar = np.hstack((self.mu_ar, np.zeros((self.mu_ar.shape[0], self.padding))))
        I = np.identity(self.F, dtype = np.float32)
        II = S_ar = np.tile(I, (self.padding,1,1))
        self.S_ar = np.concatenate((self.S_ar, II), axis=0)
        self.S_ar_inv = inv(self.S_ar)
        self.mu_br = np.concatenate((self.mu_br, np.zeros(self.padding)))
        self.s_br = np.concatenate((self.s_br, np.ones(self.padding)))
        
        # Users parameters.
        ###################################################
        if b is not None: 
            self.b = b.copy()
        else:
            self.b = np.zeros(self.U+1)

        # Threads parameters.
        ###################################################
        if s_y is not None:
            self.s_y = s_y
        else:
            self.s_y = 1/self.sigma_y #1/sigma_y where sigma_y is the noise on the thread lengths

        #######
        # Utils
        #######
        # Initial Z. Includes intercept 
        self.Z = np.zeros((self.R+1+self.padding,self.U+1), dtype=np.int16)
        for group in range(0,self.R+1):
            self.Z[group,:] = [1 if z_== group else 0 for z_ in self.z]
             
        # MAP Z
        self.Z_MAP = self.Z.copy()
        
        # Counter
        self.n = self.Z.sum(axis=1)

        # A dictionary to fast queries on where does a user participate 
        self.user_threads = {}
        for u in xrange(self.U+1):
            threads = []
            for t in xrange(self.T):
                if self.P[u,t]>0: threads.append(t)
            self.user_threads[u] = threads 

        
        ###################
        # Traces storage
        ###################
        self.max_iters = 50000
        self.last_iter = 0
        self.traces = {}
        
        self.traces['alpha'] = np.zeros(self.max_iters)
        self.traces['z'] = np.zeros((self.max_iters, self.U), dtype=np.int8) # do not trace auxilary role of intercept
        self.traces['b'] = np.zeros((self.max_iters, self.U+1), dtype=np.float64) # trace intercept
        self.traces['mu_br'] = np.zeros((self.max_iters, self.R+1+self.padding), dtype=np.float64) # trace intercept
        self.traces['s_br'] = np.zeros((self.max_iters, self.R+1+self.padding), dtype=np.float64) # trace intercept
        self.traces['mu_b0'] = np.zeros(self.max_iters, dtype=np.float64) 
        self.traces['r_b0'] = np.zeros(self.max_iters) 
        self.traces['w_b0'] = np.zeros(self.max_iters)
        self.traces['beta_b0'] = np.zeros(self.max_iters)   
        self.traces['s_y'] = np.zeros(self.max_iters)

        self.traces['mu_ar'] = np.zeros((self.max_iters, self.F, self.R+self.padding), dtype=np.float64)          
        self.traces['beta_a0'] = np.zeros(self.max_iters)
        
        self.ncomponents = np.zeros(self.max_iters,  dtype=np.int8)
        self.confusion_matrix = np.zeros((self.U, self.U))
        self.traces['perplexity'] = np.zeros(self.max_iters) 
        self.traces['y_predicted'] = np.zeros((self.max_iters, self.T))
 
     
        ###################################################################
        #### R functions for ARS sampling of beta and alpha parameters ####
        ###################################################################
        __path__ = os.path.dirname(__file__)
        with open(os.path.join(__path__, 'ars_alpha.r'), 'r') as f:
            string = f.read()
            self.r_ars_alpha = STAP(string, "r_ars_alpha")

        with open(os.path.join(__path__, 'ars_beta_a0.r'), 'r') as f:
            string = f.read()
            self.r_ars_beta_a0 = STAP(string, "r_ars_beta_a0")
     
        with open(os.path.join(__path__, 'ars_beta_b0.r'), 'r') as f:
            string = f.read()
            self.r_ars_beta_b0 = STAP(string, "r_ars_beta_b0")
            
    def perplexity(self):
        """
        Compute perplexity pf current state given the observations data
        """
        perplexity = 0
        for u in range(self.U):
            perplexity -= self.likelihood_a(u)
        for u in range(self.U+1):
            perplexity -= self.likelihood_b(u)   
        return perplexity

        
    def z_update(self, u, z_new):
        """
        Updates Z matrix, z vector and counts n only for user u
        """        
        z_old = self.z[u]
        self.Z[z_old,u] = 0
        self.Z[z_new,u] = 1
        self.z[u] = z_new
        self.n[z_old]-=1
        self.n[z_new]+=1
     
     
    def likelihood_b(self, u):
        """
        Computes likelihood of coefficient of user u given the 
        current cluster assignment and the parameters  (mean and variance) 
        of that cluster.
        """
        try:
            # TODO: sometimes upervised_clustering.py:373: RuntimeWarning: divide by zero encountered in double_scalars
            return norm.logpdf(self.b[u], loc=self.mu_br[self.z[u]], scale=np.sqrt(1/self.s_br[self.z[u]]))
        except Exception as e:
            print "Exception on likelihood_b"
            print e
            self.pickle_state()
            raise e
                            
    def likelihood_a(self, u):
        """
        Computes likelihood of attributes of user u given the
        current cluster assignment and the parameters (mean and covariance) 
        of that cluster
        """ 
        r = self.z[u+1]-1 # take into account intercept
        mean = self.mu_ar[:,r]
        cov = self.S_ar_inv[r] 
        # TODO: sometimes non - Semidefinite positive
        return multivariate_normal.logpdf(self.A[:,u], mean=mean, cov=cov)


        
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
  
    def sample_alpha(self):
        """ 
        Sample alpha
        by Adaptive Rejection Sampling
        https://www1.maths.leeds.ac.uk/~wally.gilks/adaptive.rejection/web_page/Welcome.html
        http://cran.r-project.org/web/packages/ars/ars.pdf
        """
        y = self.alpha
        try:
            y = self.r_ars_alpha.sample_alpha(K = self.R, U = self.U)
            y = y[0]
        except RRuntimeError as e:
            print "Exception in ARS for alpha:"
            print e
            print "R:", self.R
            print "U:", self.U
            print "Skipping sample"
            sys.exit()
        return y    
          
    def sample_z(self, u):
        """ 
        Samples from conditional probability of z
        Conditional probability is proportional to the prior on z
        times the likelihoods.
        """
        # If we only use p(z)likelihood(a | z): GMM clustering
        # If we only use p(z)likelihood(b | z): GMM regression
        logprobs = []
        logp = 0
        
        # chose a role but not role 0 (intercept)
        for x in xrange(1,self.R+1):
            self.z_update(u, x)
            logp = np.log(self.alpha/self.R + self.n[x]- 1) - np.log(self.alpha + self.U - 1)
            logp += self.likelihood_b(u)
            logp += self.likelihood_a(u-1) # no intercept for attributes
            logprobs.append(logp)
        
        # Normalize probs
        logprobs = logprobs-max(logprobs) # to avoid numerical underflow
        probs = np.exp(logprobs)
        probs = np.array(probs)/sum(probs)
        
        # Choose assignment
        return np.random.choice(np.arange(1,self.R+1), p=probs)
         
         
    def sample_z_CRP(self, u, m=3):
        """ 
        Samples a cluster assignment for user u from its
        conditional probability under Dirichlet Process prior (Chinese Restaurant)
        Conditional probability is proportional to the prior on z
        times the likelihoods.
        
        Implemented following Neal's alg. 8
        (http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf)
        for non-conjugate priors
        
        m: number of auxiliary components
        """
        # If we only use p(z)likelihood(a | z): GMM clustering
        # If we only use p(z)likelihood(b | z): GMM regression
        # Do not chose role 0 (intercept)
        z_past = self.z[u]

        logprobs = []
                
        # Set up auxiliary tables
        ###############################
        
        # pointers to ative and auxiliary tables
        n_without = self.n.copy()
        n_without[self.z[u]] -= 1
        k_minus_idx = np.where(n_without>0)[0][1:] # skip intercept
        k_idx = np.where(self.n>0)[0][1:] # skip intercept
        aux_idx = np.arange(k_idx.max()+1, k_idx.max()+1+m)
        
        # if current table becomes empty use it as auxiliary
        if n_without[self.z[u]] == 0:
            aux_idx[0] = self.z[u]

        # Generate parameters for the new auxiliary tables 
        # if they are not the just abandoned one
        mask = (self.n[aux_idx]==0)
        for i in aux_idx[mask]:
            self.mu_ar[:,i-1] = mnorm(mean=self.mu_a0, cov=self.R_a0_inv)    
            try:
                df = max(self.beta_a0, self.F) 
                self.S_ar[i-1] = wishart.rvs(df=df, scale=self.W_a0_inv/self.beta_a0)
            except Exception as e:
                print e
                print "Exception generating auxiliary tables"
                print "beta_a0:", self.beta_a0
                print "W_a0:\n", self.W_a0

                type, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
              
            # +1 offset because the intercept has its own cluster 
            self.mu_br[i] = norm.rvs(loc = self.mu_b0, scale = np.sqrt(1/self.r_b0))
            try:
                self.s_br[i] = gamma.rvs(self.beta_b0/2, scale=2/(self.beta_b0*self.w_b0))  
            except IndexError as e:
                print "---------------------"
                print "R:", self.R
                print e
                print self.s_br
                traceback.print_exc()
                
        # Compute probability of every table    
        #####################################
        logprobs = []
        for i in k_minus_idx:
            self.z_update(u, i)
            logp = np.log(self.n[i]-1)
            logp += self.likelihood_b(u)
            logp += self.likelihood_a(u-1) # no intercept for attributes
            logprobs.append(logp) 
            
        for i in aux_idx:
            self.z_update(u, i)
            logp = np.log(self.alpha/m)                
            logp += self.likelihood_b(u)
            logp += self.likelihood_a(u-1) # no intercept for attributes
            logprobs.append(logp)      
                              
        # Normalize probs
        logprobs = logprobs-max(logprobs) # avoids numerical underflow
        probs = np.exp(logprobs)
        probs = np.array(probs)/sum(probs)
        
        # Chose table
        #####################################
        z_chosen = np.random.choice(np.hstack((k_minus_idx, aux_idx)), p=probs)        
        self.z_update(u, z_past)
        
        
        #####################################
        # Relabelings
        
        if n_without[z_past]==0:
            self.R -= 1
            
        # if auxiliar label as the first auxiliar
        # no empty tables between occupied ones
        if z_chosen in aux_idx:
            first_aux = aux_idx[0]
            self.z_update(u, first_aux)
            self.mu_ar[:,first_aux-1] = self.mu_ar[:,z_chosen-1]
            self.S_ar[first_aux-1] =  self.S_ar[z_chosen-1]
            self.mu_br[first_aux] = self.mu_br[z_chosen]
            self.s_br[first_aux] = self.s_br[z_chosen]            
            self.R += 1 # there is one new table
            z_chosen = first_aux
            
        # if chose existing table and the former table is now empty
        # shift -1 the label of the active tables on the right
        elif ((z_chosen in k_minus_idx) and (n_without[z_past] == 0)):
            self.z_update(u, z_chosen)
            
            # Move parameters
            for i in xrange(z_past+1, np.max(self.z)+1):
                self.mu_ar[:,i-1] = self.mu_ar[:,i]
                self.S_ar[i-1] =  self.S_ar[i]
                self.mu_br[i] = self.mu_br[i+1]
                self.s_br[i] = self.s_br[i+1]
            
            # Move z (-1 shift in the positions at the right of empty table)
            mask = np.where(self.z>z_past)
            self.z[mask] -= 1
            self.Z[z_past:-1,:] = self.Z[z_past+1:,:]   
            self.n = self.Z.sum(axis=1)
            z_chosen = self.z[u]

        return z_chosen
        
#############################################################################
#############################################################################
#############################################################################

        
    def sample_mu_ar(self, r):
        """
        Samples a mean for cluster r 
        from its conditional probability
        """ 
        mask = (self.Z[r+1,:]==1)[1:]
        
        # If nobody in the cluster sample from prior
        if mask.sum() == 0:
            return mnorm(mean = self.mu_a0, cov = self.R_a0_inv)
                    
        ar_mean = self.A[:,mask].mean(axis=1)
        N = self.n[r+1]
        Lambda_post = self.R_a0 + N*self.S_ar[r]
        Sigma_post = inv(Lambda_post)
        mu_post = dot(Sigma_post, dot(self.R_a0, self.mu_a0) + 
                                  N*dot(self.S_ar[r], ar_mean))  
                                  
        return mnorm(mean = mu_post, cov = Sigma_post)  
   

    def sample_S_ar(self, r):
        """
        Sample attributes covariance matrix of cluster r
        from its conditional probability
        """
        mask = (self.Z[r+1,:]==1)[1:]

        # If nobody in the cluster sample from prior
        if mask.sum() == 0:
            try:
                # unfortunately wishart implementation does not accept dof > F-1
                # but dof >= F
                # This will only affect when the cluster is empty.
                df = max(self.beta_a0, self.F) 
                v = wishart.rvs(df=df, scale=self.W_a0_inv/self.beta_a0)
                return v 
            except ValueError as e:
                print "Exception at S_ar"
                print e
                print self.beta_a0
                print self.W_a0
            
             
        A_r = self.A[:,mask]
        N = A_r.shape[1]
        scatter_matrix = np.zeros((self.F,self.F))                        
        for u in range(N):
            scatter_matrix += dot(A_r[:,u].reshape(self.F,1) - self.mu_ar[:,r].reshape(self.F,1), 
                                 (A_r[:,u].reshape(self.F,1) - self.mu_ar[:,r].reshape(self.F,1)).T)
                
        wishart_dof_post = self.beta_a0 + self.n[r+1]
        #before        
        wishart_S_post = inv(self.beta_a0*self.W_a0 + scatter_matrix)
        #wishart_S_post_inv = self.beta_a0*self.W_a0 + scatter_matrix
        try:
            #before
            v = wishart.rvs(df=wishart_dof_post, scale=wishart_S_post)
            #v = invwishart.rvs(df=wishart_dof_post, scale=wishart_S_post_inv)
            return v
        except ValueError as e:
            print "Exception at S_ar"
            print e
            print wishart_dof_post
            print wishart_S_post
             
        

    ################################################################
    # COMMON HYPERPARAMETERS FOR CLUSTERS CENTROIDS
    ################################################################
    def sample_mu_a0(self):
        """
        Samples mean of gaussian hyperprior placed over clusters centroids
        """
        mean_ar = self.mu_ar.mean(axis=1)
        Lambda_post = self.Lambda_a + self.R*self.R_a0
        Sigma_post = inv(Lambda_post)
        mu_post = dot(Sigma_post, dot(self.Lambda_a, self.mu_a) + 
                                  self.R*dot(self.R_a0, mean_ar))     
        return mnorm(mean = mu_post, cov = Sigma_post)  
        
        
    def sample_R_a0(self):
        """
        Samples precision of gaussian hyperprior placed over clusters centroids
        """
        scatter_matrix = np.zeros((self.F,self.F))
        for r in range(self.R):
            #TODO: check this scatter matrix # same than with reshqpe? check
            scatter_matrix += dot(self.mu_ar[:,r][:,np.newaxis] - self.mu_a0[:,np.newaxis], 
                                 (self.mu_ar[:,r][:,np.newaxis] - self.mu_a0[:,np.newaxis]).T)
                                 
        wishart_dof_post = self.F + self.R
        
        # before
        wishart_S_post = inv(self.F*self.Sigma_a + scatter_matrix)
        v = wishart.rvs(df=wishart_dof_post, scale=wishart_S_post)
        #wishart_S_post_inv = self.F*self.Sigma_a + scatter_matrix
        #v = invwishart.rvs(df=wishart_dof_post, scale=wishart_S_post_inv)
        return np.array(v)     
               
               
    ###############################################################
    # COMMON HYPERPARAMETERS FOR CLUSTERS COVARIANCES
    ###############################################################   
    def sample_W_a0(self):
        """
        Sample base covariance of attributes (hyperparameter)
        """
        scatter_matrix = np.zeros((self.F,self.F))
        for r in range(self.R):
            scatter_matrix += self.S_ar[r]
          
        wishart_dof_post = self.F + self.R*self.beta_a0
        #before
        wishart_S_post = inv(self.F * self.Lambda_a + self.beta_a0*scatter_matrix)
        v = wishart.rvs(df=wishart_dof_post, scale=wishart_S_post)    
        
        #wishart_S_post_inv = self.F * self.Lambda_a + self.beta_a0*scatter_matrix
        #v = invwishart.rvs(df=wishart_dof_post, scale=wishart_S_post_inv)    
        return v


    def sample_beta_a0(self):
        """ 
        Sample degrees of freedom (hyperparameter)
        by Adaptive Rejection Sampling
        https://www1.maths.leeds.ac.uk/~wally.gilks/adaptive.rejection/web_page/Welcome.html
        http://cran.r-project.org/web/packages/ars/ars.pdf
        https://code.google.com/p/pmtk3/source/browse/trunk/toolbox/Algorithms/mcmc/ars.m?r=2678
        """
        # Note: do not use [0] to get first sample, it complains 'dims' cannot be of length 0 (?)
        # TODO: if only one cluster use f_uni instead
        # xlb=0 makes rwishart to generate singular matrices if dof<p.
        # I don't know how to deal with this, but at the moment I'll bound
        # dof>=p
        S_ar = self.S_ar[:self.R]         

        y = self.beta_a0
        try:
            #y = self.r_ars_beta_a0.sample_beta_a0(S=S_ar, W=self.W_a0)
            y = self.r_ars_beta_a0.sample_beta_a0(S=S_ar, W=self.W_a0, init=self.beta_a0)
            y = y[0]
        except RRuntimeError as e:
            print "Exception in ARS for beta_a0:"
            print e
            print "S:", S_ar
            print "W:", self.W_a0
            print "Skipping sample"
            #sys.exit()
        
        if (y>100000):
            print "\nS_ar:\n", S_ar
            print "W_a0:\n", self.W_a0
            
        #assert(y>=0)
        assert(y>self.F-1)
        try:
            assert(y<10000000), y
        except AssertionError as e:
            self.pickle_state()                   
        return y   
    
#############################################################
#############################################################


    def sample_b(self):
        """
        Samples from conditional probability of b
        """
        # TODO: sampling b's individually, if possible,
        # would be much faster and less buggy.
        
        Z = self.Z[:self.R+1,:]
        mu_br = self.mu_br[:self.R+1]
        s_br = self.s_br[:self.R+1]

        mu_b = dot(Z.T, mu_br)
        Lambda_b = np.diag(dot(Z.T, s_br))
        Lambda_y = np.identity(self.T)*self.s_y
        Lambda_post = Lambda_b + dot(dot(self.P, Lambda_y), self.P.T)
        
        try:
            Sigma_post = inv(Lambda_post)
            np.linalg.cholesky(inv(Lambda_b))
            np.linalg.cholesky(inv(Lambda_y))
            Sigma_post = inv(Lambda_post)
            np.linalg.cholesky(Sigma_post)
        except Exception as e:
            print "Exception in sample_b"
            print e
            with open("Lambda_b.pck", "wb") as pckfile:
                np.save(pckfile, Lambda_b)
                print Z
            with open("Z.pck", "wb") as pckfile:
                np.save(pckfile, Z)
            with open("s_br.pck", "wb") as pckfile:
                np.save(pckfile, s_br)
            with open("z.pck", "wb") as pckfile:
                np.save(pckfile, self.z)                
            self.pickle_state()
            
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
            
        mu_post = dot(Sigma_post, dot(Lambda_b, mu_b) + dot(dot(self.P, Lambda_y), self.y))
        res = None
        try:        
            res = multivariate_normal.rvs(mean = mu_post, cov = Sigma_post)
        except Exception as e:        
            res = MASS.mvrnorm(mu=robjects.FloatVector(mu_post), Sigma=robjects.Matrix(Sigma_post))
            res = np.asarray(mu_post)
            print "Using R function MASS:mvnorm instead of Python multivariate_normal"
            print "R mvnorm: ", res[:5]
        return res
        
       
    
    def sample_mu_br(self, r):
        """ 
        Samples means of b for every cluster 
        """
        mask = self.Z[r,:]==1
        if mask.sum() == 0:
            return norm.rvs(loc = self.mu_b0, scale = np.sqrt(1/self.r_b0))
        
        b_mean = self.b[mask].mean()        
        lambda_post = self.r_b0 + self.n[r]*self.s_br[r]
        sigma_post = 1/lambda_post
        mu_post = sigma_post*(self.r_b0*self.mu_b0 + self.n[r]*self.s_br[r] * b_mean)
        
        res = norm.rvs(loc = mu_post, scale = np.sqrt(sigma_post))
        if np.isnan(res):
            print "NaN value sampled for mu_br"
            print "mu_b0:", self.mu_b0
            self.pickle_state()
            pdb.set_trace()
            
        return res
        
        
    def sample_s_br(self, r):
        """
        Sample variance of b for every cluster.
        Note that python (or R) parametrization is different to that proposed
        by Rasmussen. 
        Rasmussen's G(a,b) = Python G(a/2, b/(a/2))
        """
        mask = self.Z[r,:]==1
        
        # if empty component, draw from prior
        if mask.sum() == 0:
             try:
                 # Wishart implementation does not allow beta < F (real beta)
                 #return wishart.rvs(self.beta_b0, scale=1/(self.beta_b0*self.w_b0))
                 return gamma.rvs(self.beta_b0/2, scale=2/(self.beta_b0*self.w_b0))
             except ValueError as e:
                 print e
                 print "***beta_b0:", self.beta_b0
                 sys.exit()
        sumSq = np.sum((self.b[mask]-self.mu_br[r])**2)
        assert(not np.isnan(sumSq)), "sumSq is NaN"
 
        if True:
            gamma_dof_post = self.beta_b0 + self.n[r]
            gamma_s_post = 1/(self.beta_b0*self.w_b0 + sumSq)
            try:
               #return wishart.rvs(gamma_dof_post, gamma_s_post)
                return gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post) # use "scale"!
            except ValueError as e:
                 print e
                 print "*beta_b0:", self.beta_b0
                 sys.exit()

    def sample_mu_b0(self):
        """
        Samples hyperparameter mu_b0 
        """
        mu_br = self.mu_br[:self.R+1]
        
        lambda_post = 1/self.sigma_b + (self.R+1)*self.r_b0
        sigma_post = 1/lambda_post
        
        mu_br_mean = mu_br.mean() # good
        mu_post = sigma_post*(self.mu_b/self.sigma_b + (self.R+1)*self.r_b0*mu_br_mean)
        return norm.rvs(loc=mu_post, scale=np.sqrt(sigma_post))
        
        
    def sample_r_b0(self):
        """
        Samples hyperparameter sample_Lambda_b0 
        """
        mu_br = self.mu_br[:self.R+1]        
        sumSq = ((mu_br - self.mu_b0)**2).sum()
        gamma_dof_post = 1 + (self.R+1)
        gamma_s_post = 1/(self.sigma_b + sumSq)
        try:
            return gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post) #use scale
        except ValueError as e:
            print e
            return self.r_b0 #TODO gestionar esta exception en lugar de pasarla por alto
    
    
    def sample_w_b0(self):
        """
        Samples hyperparameter w_b0
        """   
        s_br = self.s_br[:self.R+1]        
        sumS = np.sum(s_br)
        gamma_dof_post = 1 + (self.R+1)*self.beta_b0
        gamma_s_post = 1/(1/self.sigma_b + self.beta_b0*sumS)
        #return wishart.rvs(gamma_dof_post, gamma_s_post)
        return gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post)
    
    
    def sample_beta_b0(self):
        """ 
        Sample degrees of freedom (hyperparameter)
        by Adaptive Rejection Sampling
        https://www1.maths.leeds.ac.uk/~wally.gilks/adaptive.rejection/web_page/Welcome.html
        http://cran.r-project.org/web/packages/ars/ars.pdf
        """
        # TODO: auto-adjustinitial points x specially for the norole case
        s_br = self.s_br[:self.R+1]         
        y = self.beta_b0
        try:
            #y = self.r_ars_beta_b0.sample_beta_b0(s=s_br, w=self.w_b0)
            y = self.r_ars_beta_a0.sample_beta_a0(S=s_br, W=self.w_b0, init=self.beta_b0)
            y = y[0]
        except RRuntimeError as e:
            print "Exception in ARS for beta_b0:"
            print e
            print "s:", s_br
            print "w:", self.w_b0
            print "Skipping sample"
            #sys.exit()
            
        # Debug # data is irrelevant for this variable, given s_br and w_b0
        #if y>200:
        if y>2000:
            print "Suspicious beta_b0", y
            print "*******************************"
            print "sample #", self.iteration
            print "***********"
            print "s_br:", s_br
            print "w_b0:", self.w_b0
            y_ = self.r_ars_beta_b0.sample_beta_b0(s=s_br, w=self.w_b0)
            y_ = y_[0]        
            print "multidimensional sampler:", y
            print "unidimensional sampler:", y_
            self.pickle_state(fname="high_beta_b0.pck",
                              fnamedata="high_beta_b0_data.pck")
            
        return y   


    def sample_s_y(self):
        """
        Sample the noise inherent in threads length (y_t = p^t b + N(0, 1/s_y)) 
        """
        gamma_dof_post = self.T+1
        gamma_s_post = 1/(self.sigma_y + np.sum((self.y - dot(self.P.T, self.b))**2))
        #print "\nREGRESSION ERROR:", np.sum((self.y - dot(self.P.T, self.b))**2)
        #print "Variance", np.mean(1/gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post,  size=1000))
        #print "Variance 2", np.mean(1/wishart.rvs(gamma_dof_post, gamma_s_post,  size=1000))
        #return wishart.rvs(gamma_dof_post, gamma_s_post)
        
        return gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post) #buggy. Must explicit "scale"
        
        # In version 1.0
        #gamma_dof_post = self.T+1
        #gamma_s_post = gamma_dof_post/(self.sigma_y + np.sum((self.y - dot(self.P.T, self.b))**2))
        #return gamma.rvs(gamma_dof_post/2, scale=2*gamma_s_post/gamma_dof_post)

    
    def sample_predictive_posterior_y(self, P_test, samples_idx):
        """
        Sample from the posterior predictive distribution. 
        The way bayesians like.
        """
        traces_b = self.traces['b'][samples_idx]        
        traces_s_y = self.traces['s_y'][samples_idx]        
          
        T_test = P_test.shape[1]
        #s_y_inv = np.identity(T_test)/self.s_y
        y_preds = np.zeros((traces_b.shape[0],T_test))
        for i in xrange(traces_b.shape[0]):
            b = traces_b[i]
            s_y = traces_s_y[i]
            
            #s_y_inv = np.identity(T_test)/s_y # same variance for all y points
            # TODO: bottleneck if P_test is too big. Do it one by one.
            #y_preds[i] = mnorm(mean=means, cov=s_y_inv)
            
            means = dot(P_test.T, b)
            y_preds[i] = np.random.normal(loc=means, scale=1/np.sqrt(s_y)) 
        return y_preds

    
    def step(self, iteration):
        """
        One iteration of Gibbs sampler
        """

        self.iteration = iteration # for debugging
        
        # Progress bar
        completed = 100*iteration/self.total_iters
        if completed == 0: print ""
        #print '\r[{0}] {1}%'.format('#'*(int(completed/5)), completed),
        if (iteration%100) == 0:
            print '\r[{0}] {1}%'.format('#'*(int(completed/5)), completed), self.worker_exp, iteration,
                      
            # Write process to file so that we can monitor the set pf parallel processes                      
            filepath = ".workers/" + self.worker_exp + ".progress"
            with open(filepath, 'w') as f:
                f.write(str(completed))        

        if (iteration%5000) == 0:
            self.save_traces()
                    
          
        ###############################
        # Mixture variables (sometimes called Factor)
        ############################### 
        # Sample z but not auxiliary z_0 (intercept) 
        if True:
            for u in xrange(self.U):
                if self.DP:
                    z_new = self.sample_z_CRP(u+1)
                else:
                    z_new = self.sample_z(u+1)
                self.z_update(u+1, z_new)                 
                self.traces['z'][iteration, u] = z_new-1
        else:
            # skip intercept. Offset back indices
            self.traces['z'][iteration,:] = self.z[1:]-1
        
        ###############################
        # Clustering variables
        ############################## 
        if True:
            # Components parameters
            ########################
            if True:
                for r in xrange(self.R): 
                    self.mu_ar[:,r] = self.sample_mu_ar(r)
                    self.traces['mu_ar'][iteration,:,r] = self.mu_ar[:,r].copy()
            self.traces['mu_ar'][iteration,:,:self.R] = self.mu_ar[:,:self.R]
            
            if True:
                for r in xrange(self.R):
                    self.S_ar[r] = self.sample_S_ar(r)
                self.S_ar_inv[:self.R] = inv(self.S_ar[:self.R]) # update the inverses !
            
            # Common parameters
            ####################
            if True:
                self.mu_a0 = self.sample_mu_a0()
    
            if True:
                self.R_a0 = self.sample_R_a0()
                self.R_a0_inv = inv(self.R_a0)
                
            if True:
                self.W_a0 = self.sample_W_a0()
                self.W_a0_inv = inv(self.W_a0)
            
            if True:
                self.beta_a0 = self.sample_beta_a0() 
                try:
                    assert(self.beta_a0>self.F-1), self.beta_a0
                except AssertionError as e:
                    print "ILEGAL BETA_A0. PICKLING STATE---------------------------"
                    self.pickle_state()
                    
            self.traces['beta_a0'][iteration] = self.beta_a0
 

        ###############################
        # Concentration parameter
        ###############################
        if True:
            self.alpha = self.sample_alpha()
            self.traces['alpha'][iteration] = self.alpha
            
            
        ###############################
        # Prediction variables
        ############################### 
        if True:
            # Components parameters
            ########################  
            try:
                if True:
                    # TODO: debugging. Uncomment!
                    #self.b = self.sample_b()
                    pass
                else:
                    self.b = self.b_MLE
                self.traces['b'][iteration,:] = self.b
                assert(self.b.mean() < np.inf)
                    # TODO: excepction here for iris dataset.
                    # caused by what?
                    # also LinAlgError: SVD did not converge
                    # only for low number of threads?
            except Exception as e:
                    print e
                    print "(sampling b)"
                    self.pickle_state()   
                    
            if True:
                for r in xrange(self.R+1):
                    self.mu_br[r] = self.sample_mu_br(r)
                    assert(self.mu_br[r] < np.inf), self.mu_br[r]

                    assert(self.mu_br[r] < 1000000), self.mu_br[r]
                    self.traces['mu_br'][iteration,:] = self.mu_br[r]
            self.traces['mu_br'][iteration,:self.R+1] = self.mu_br[:self.R+1]
            
            if True:
                for r in xrange(self.R+1):
                    self.s_br[r] = self.sample_s_br(r)
                    assert(self.s_br[r] < np.inf)
                    
                #if(np.var(self.s_br[:self.R+1])<1/1000000):
                if(np.var(self.s_br[:self.R+1])<1/1000000000):
                    print "Suspicious low variance in s_br"
                    print "*******************************"
                    print "sample #", iteration
                    print "***********"
                    print "n:", self.n[:self.R+1]
                    print "R:", self.R
                    print "prior_dof:", self.beta_b0
                    print "prior_w",self.w_b0
                    print "suspicious s_br:", self.s_br[:self.R+1]
                    print "variance:", np.var(self.s_br[:self.R+1])
                    for r in xrange(self.R+1):
                        print "----component", r
                        mask = self.Z[r,:]==1
                        sumSq = np.sum((self.b[mask]-self.mu_br[r])**2)
                        gamma_dof_post = self.beta_b0 + self.n[r]
                        gamma_s_post = 1/(self.beta_b0*self.w_b0 + sumSq) 
                        print "dof_post", gamma_dof_post, "(", self.beta_b0, self.n[r], ")"
                        print "scale_post", gamma_s_post, '(', sumSq ,')'
                                

                    self.pickle_state(fname="low_var_sbr.pck", 
                                      fnamedata="low_var_sbr_data.pck")

            self.traces['s_br'][iteration,:self.R+1] = self.s_br[:self.R+1]
            
            # Common parameters
            ####################
            try:
                if True:
                    self.mu_b0 = self.sample_mu_b0()
                    assert(self.mu_b0 < np.inf), (self.mu_b0, self.mu_br, self.r_b0)
                    assert(self.mu_b0 > -np.inf), (self.mu_b0, self.mu_br, self.r_b0)
                self.traces['mu_b0'][iteration] = self.mu_b0
            except Exception as e:
                print e
                self.pickle_state()
                
                    
            if True:
                r_b0_last = self.r_b0
                self.r_b0 = self.sample_r_b0()
                if self.r_b0 == 0: self.r_b0 = r_b0_last
                assert(self.r_b0 < np.inf), self.r_b0
                assert(self.r_b0 > 0), self.r_b0
            self.traces['r_b0'][iteration] = self.r_b0
            
            if True:
                self.w_b0 = self.sample_w_b0()
                assert(self.w_b0 < np.inf)  
            self.traces['w_b0'][iteration] = self.w_b0
            
            if True:
                self.beta_b0 = self.sample_beta_b0()
                assert(self.beta_b0 > 0)
            self.traces['beta_b0'][iteration] = self.beta_b0
            
            # Noise on threads
            ###################
            if True:
                self.s_y = self.sample_s_y()
                assert(self.s_y > 0)
            self.traces['s_y'][iteration] = self.s_y  

                
        ######################
        # Perplexity qnd joint likelihood
        ######################
        # keep track of perplexity
        if self.animated:
            self.trace['perplexity'][iteration] = self.perplexity()
                   
        # keep track of last iteration 
        # so that we can do concatenated executions
        self.last_iter = iteration
          
    def init_animation(self):
        """
        Initialize animation
        """ 
        for cluster in self.clusters:
            cluster.set_data([], [])
        
        for centroid in self.centroids:
            centroid.set_data([], [])
        
        for mline in self.trace_mu_br_plot:
            mline.set_data([],[])   
        
        return (self.clusters, 
                self.centroids,
                self.ncomponents_plot,
                self.rects_w, 
                self.trace_mu_br_plot)


    def animate(self, iteration):
        """ Calls step to get next state to plot"""
        
        # Do not overwrite former traces of former executions
        iteration = self.start_iter + iteration

        # Perform 1 new Gibbs sample        
        self.step(iteration)
            
        #################
        # Plot traces of number of components
        #################
        self.ncomponents[iteration] = self.R
        self.ncomponents_plot.set_data(np.arange(iteration), self.ncomponents[:iteration])

        ################
        # Confussion matrix
        ################
        self.confusion_matrix = dot(self.Z[1:,1:].T, self.Z[1:,1:])
        self.confusion_matrix_plot.set_data(self.confusion_matrix)
        self.confusion_matrix_plot.autoscale()
        
        ##################
        # Perplexity
        ##################    
        self.perplexity_plot.set_data(np.arange(iteration), self.traces['perplexity'][:iteration])        

        ##################
        # Plot means, covariances, and data points
        ##################        
        self.ax1.set_title("Clusters $z$ and centroids $a_r$ %d %d" % (iteration,self.R))       
        for r in xrange(self.R):
            
            # Clusters assignments
            mask = np.where(self.z == r+1)[0]-1
            atts = self.A[:,mask]
            x1 = np.array(atts[0,:])
            y1 = np.array(atts[1,:])
            self.clusters[r].set_data((x1, y1))

            # Centroids
            ar1x = self.mu_ar[0,r] # feature 0 of role r
            ar1y = self.mu_ar[1,r] # feature 1 of role r
            self.centroids[r].set_data((ar1x, ar1y))
            
            # Covariances. Remove old ones before plotting the new
            # since countour has no set_data method
            x = np.arange(-3.0, 10.0, 0.1)
            y = np.arange(-3.0, 10.0, 0.1)
            X, Y = np.meshgrid(x, y)   
            Z1 = mlab.bivariate_normal(X, Y, 
                                       np.sqrt(self.S_ar_inv[r,0,0]), 
                                       np.sqrt(self.S_ar_inv[r,1,1]), 
                                       ar1x, ar1y, self.S_ar_inv[r,0,1])
                                       
            # it throws exception zhen collection is empty
            # warning: how can a collection of existing cluster be empty?
            try:
                for coll in self.variances[r].collections:
                    coll.remove()
            except:
                pass
            self.variances[r] = self.ax1.contour(X, Y, Z1, linewidths=1)

        # Remove covariances of extincted clusters
        # if covarience is empty raise exception but continue
        try:
            for centroid in self.centroids[self.R:]:
                centroid.set_data(([], []))
            
            for variance in self.variances[self.R:]:
                for coll in variance.collections:
                    coll.remove()
        except:
            pass
 
         
        ##################
        # Plot regressions
        ##################   
        # True lengths       
        self.regressions[0].set_data(np.arange(len(self.y)), self.y[np.argsort(self.y)])
        
        # Bayesian estimation
        y_pred = dot(self.P.T, self.b)
        self.regressions[1].set_data(np.arange(len(self.y)), y_pred[np.argsort(self.y)])
       
        # MLE estimation
        y_pred_MLE = dot(self.P.T, self.b_MLE)
        self.regressions[2].set_data(np.arange(len(self.y)), y_pred_MLE[np.argsort(self.y)])
        
        ##########################################################
        # Plot means, covariances, and data points of coefficients
        ##########################################################  
        for i in xrange(self.R):       
            mask = np.where(self.z==i)[0]
            coeffs = self.b[mask]
            self.clusters_b[i].set_data((coeffs, [0]*len(coeffs)))
            
            x = np.arange(-500,500, 0.2)
            self.variances_b[i].set_data(x, 5+1000*mlab.normpdf(x,self.mu_br[i],np.sqrt(1/self.s_br[i])))
        
        ##################
        # Plot weights
        ##################  
        for rect, h in zip(self.rects_b, self.b):
            rect.set_height(h)
        
        for rect, h in zip(self.rects_w, self.mu_br)[:self.R]:
            rect.set_height(h)
                
        ##################
        # Plot all traces
        ##################
        # Attributes  
        for i, line in enumerate(self.trace_mu_ar_plot):
            line.set_data(self.traces['mu_ar'][:,0,i], self.traces['mu_ar'][:,1,i])
            
        self.trace_beta_a0_plot.set_data(np.arange(iteration), self.traces['beta_a0'][:iteration])
            
        # Prediction
        for i, line in enumerate(self.trace_mu_br_plot):
            line.set_data(np.arange(iteration), self.traces['mu_br'][:iteration,i])

        for i, line in enumerate(self.trace_s_br_plot):
            line.set_data(np.arange(iteration), 1/np.sqrt(self.traces['s_br'][:iteration,i]))

        self.trace_r_b0_plot.set_data(np.arange(iteration), 1/np.sqrt(self.traces['r_b0'][:iteration]))
        self.trace_mu_b0_plot.set_data(np.arange(iteration), self.traces['mu_b0'][:iteration])        

        self.trace_w_b0_plot.set_data(np.arange(iteration), self.traces['w_b0'][:iteration])    
        self.trace_beta_b0_plot.set_data(np.arange(iteration), self.traces['beta_b0'][:iteration])


        return (self.clusters, self.clusters_b,
               self.centroids,
               self.ncomponents_plot,
               self.confusion_matrix_plot,
               self.perplexity_plot,
               self.rects_w,
               self.variances, self.variances_b,
               self.trace_mu_br_plot, self.trace_mu_ar_plot,
               self.trace_mu_b0_plot, self.trace_r_b0_plot,
               self.trace_w_b0_plot, self.trace_beta_b0_plot, #TODO: w_b0?
               self.trace_beta_a0_plot
               )        
        
        
    def fit(self, iters, animated=False, videofile=False, title = "Animation"):
        """ 
        Run the sampler
        """
        self.animated = animated
        self.start_iter = self.last_iter
        self.total_iters = self.start_iter + iters
        
        if animated:

            fig = plt.figure(title)
            plt.clf()
            plt.title("Gibbs sampling")
            nrows, ncols = (5,4)
            max_clusters = self.traces['mu_ar'].shape[2] # same max clusters than traces
            
            # Z clustering and extras
            gs0 = gridspec.GridSpec(1, 3) 
            gs0.update(left=0.33, right=0.66, top=0.95, bottom=0.8, wspace=0.5)

            # Attributes
            gs1 = gridspec.GridSpec(nrows, ncols) 
            gs1.update(left=0.05, right=0.48, top=0.75, bottom=0.50, wspace=0.05)

            gs2 = gridspec.GridSpec(2, 1)
            gs2.update(left=0.05, right=0.48, top=0.45, bottom=0.05, wspace=0.05)            

            # Regression
            gs3 = gridspec.GridSpec(nrows, ncols)
            gs3.update(left=0.50, right=0.98, top=0.75, bottom=0.35, hspace=1)
            
            gs4 = gridspec.GridSpec(2, 8)
            gs4.update(left=0.50, right=0.98, top=0.3, bottom=0.05, wspace=0.25)
            
            # Confusion matrix to visualize clustering
            ##########################################
            ax = plt.subplot(gs0[0,0])
            self.confusion_matrix_plot = ax.matshow(self.confusion_matrix)

            # Number of active components
            #############################
            ax = plt.subplot(gs0[0,1], xlim=(0, self.total_iters), ylim=(0, max_clusters))
            ax.set_title('# of clusters')
            self.ncomponents_plot, =  plt.plot([],[], '+')
            
            # Perplexity
            ##############################
            ax = plt.subplot(gs0[0,2])
            ax.grid()
            ax.set_title("Perplexity")
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(0,400)            
            self.perplexity_plot, = plt.plot([],[])
 
           
            # Clusters
            ############
            # ax1 class property so that it can be accessed to draw contours on it 
            self.ax1 = plt.subplot(gs1[:, :], xlim=(-2, 5), ylim=(-2.5, 5.5))
            self.ax1.set_title("Clusters $z$ and centroids $a_r$")            
            self.ax1.grid()            

            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.6',  '0.8', '1.0', '0.2','0.3','0.5','0.7','0.9',
                      'b', 'r', 'g', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.6',  '0.8', '1.0', '0.2','0.3','0.5','0.7','0.9',
                      'b', 'r', 'g', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.6',  '0.8', '1.0', '0.2','0.3','0.5','0.7','0.9',
                      'b', 'r', 'g', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.6',  '0.8', '1.0', '0.2','0.3','0.5','0.7','0.9',
                      'b', 'r', 'g', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.6',  '0.8', '1.0', '0.2','0.3','0.5','0.7','0.9']

            self.clusters = [plt.plot([],[],  'o', c=colors[i], alpha=0.50)[0] for i in range(max_clusters)]
            self.centroids = [plt.plot([],[], marker='$a_%d$' % int(i+1), c=colors[i], markersize=10)[0] 
                              for i in range(max_clusters)]
                                  
            # Covariances
            ##############
            x = np.arange(-3.0, 0.0, 0.1)
            y = np.arange(-2.0, 0.0, 0.1)
            X, Y = np.meshgrid(x, y)
            
            Z = [mlab.bivariate_normal(X, Y, 
                                   self.S_ar[r,0,0], self.S_ar[r,1,1], 
                                   self.mu_ar[0,r], self.mu_ar[1,r]) for r in range(self.R)]
                                   
             
            self.variances = [plt.contour(X, Y, Z[r], linewidths=0) for r in range(self.R)]

           # more covariances just in case it creates more clusters
            for i in xrange(max_clusters-self.R):
                self.variances.append(plt.contour(X, Y, Z[0],  linewidths=0))

            # Trace mu_ar
            ##############
            ax = plt.subplot(gs2[0,0], xlim=(-2, 3), ylim=(-2.5, 2.5))
            ax.set_title("$\mu_{a_r}$")
            
            self.trace_mu_ar_plot = [plt.plot([],[], 'p', c=colors[i])[0] for i in range(max_clusters)]

            # Traces beta_a0
            ###############
            ax = plt.subplot(gs2[1,0])
            ax.grid()
            ax.set_title('$\\beta_{a0}$')
            ax.set_xlim(0, self.total_iters)
            ax.set_ylim(0,200)
            self.trace_beta_a0_plot, =  plt.plot([],[])



            # Regressions   
            ###############     
            ax = plt.subplot(gs3[0:2,:], xlim = (0, len(self.y)), ylim=(-30,30))            
            ax.set_title("Thread length")   
            self.regressions = []
            self.regressions.append(plt.plot([],[], label="True")[0])
            self.regressions.append(plt.plot([],[], label="Bayes")[0])
            self.regressions.append(plt.plot([],[], label="MLE")[0])
            ax.legend()

            # Coefficients centroids and sigmas
            ####################################
            ax = plt.subplot(gs3[2:4,:], xlim = (-500,500 ), ylim=(-30,30))            
            ax.set_title("Clusters $z$ and centroids $b_r$")
            ax.set_yticks([])   

            self.variances_b = [plt.plot([],[], c=colors[i])[0] 
                                for i in range(max_clusters+1)]

            self.clusters_b = [plt.plot([],[],  'o', c=colors[i], alpha=0.5)[0] 
                                for i in range(max_clusters+1)]

            self.variances_b[0] = plt.plot([],[], lw=0)[0] 
            self.clusters_b[0] = plt.plot([],[], 'o', ms=5, c='k')[0] 
                                
            # Coefficients bars   
            ###################                       
            ax = plt.subplot(gs3[4,0:2], xlim=(0, self.U+1), ylim=(-500, 500))
            ax.set_title("Individual coefficients $b_u$")
            ax.grid()
            self.rects_b = plt.bar(range(self.U+1), np.ones(self.U+1), align='center', alpha=0.3)
                        
            ax = plt.subplot(gs3[4,2:4], xlim=(-2, max_clusters), ylim=(-200, 200))
            ax.set_title("Centroids (or role coefficients)  $\mu_{b_r}$")
            ax.grid()
            self.rects_w = plt.bar(range(max_clusters), np.ones(max_clusters), align='center', alpha=0.3)


            # Traces mu_br
            ###############
            ax = plt.subplot(gs4[0,1:3])
            ax.grid()
            ax.set_title('$\mu_{b_r}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(-120,120)
            self.trace_mu_br_plot =  [plt.plot([],[])[0] for j in range(max_clusters+1)]

            # Traces s_br
            ###############
            ax = plt.subplot(gs4[0,5:7])
            ax.grid()
            ax.set_title('$s_{b_r}^{-1/2}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(0,120)
            self.trace_s_br_plot =  [plt.plot([],[])[0] for j in range(max_clusters+1)]
            
            # Traces mu_b0
            ###############
            ax = plt.subplot(gs4[1,0:2])
            ax.grid()
            ax.set_title('$\mu_{b0}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(-100,100)
            self.trace_mu_b0_plot, =  plt.plot([],[])            

            # Traces r_b0
            ###############
            ax = plt.subplot(gs4[1,2:4])
            ax.grid()
            ax.set_title('$r^{-1/2}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(0,300)
            self.trace_r_b0_plot, =  plt.plot([],[])

            # Traces w_b0
            ###############
            ax = plt.subplot(gs4[1,4:6])
            ax.grid()
            ax.set_title('$w_{b0}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(0,10)
            self.trace_w_b0_plot, =  plt.plot([],[])

            # Traces beta_b0
            ###############
            ax = plt.subplot(gs4[1,6:8])
            ax.grid()
            ax.set_title('$\\beta_{b0}$')
            ax.set_xlim(0,self.total_iters)
            ax.set_ylim(0,100)
            self.trace_beta_b0_plot, =  plt.plot([],[])
            
            print "Animate run..."
            animation.ani = animation.FuncAnimation(fig, self.animate, 
                                                    frames=iters, interval=1, 
                                                    blit=False, 
                                                    init_func=self.init_animation,
                                                    repeat=False)  
            if videofile:
                # Uncomment to save into file                                                                
                plt.rcParams['animation.ffmpeg_path']='/usr/local/bin/ffmpeg'
                plt.rcParams['animation.ffmpeg_path']='ffmpeg.exe'
                mywriter = animation.FFMpegWriter()
                animation.ani.save('2IGMM.mp4', fps=30, extra_args=['-vcodec', 'libx264'], writer=mywriter)
        
            plt.show()
            
        else:
            for i in xrange(self.start_iter, self.start_iter + iters):
                self.step(i)
        
        
        
        return {'alpha': self.traces['alpha'],
                'z': self.traces['z'],
                'b': self.traces['b'],
                'w': self.traces['mu_br'],
                'ar': self.traces['mu_ar'],
                's_y': self.traces['s_y']}


    def save_traces(self):
        """ Save traces to files"""
        keys = ['alpha', 'z', 'b', 's_y', 
                'beta_a0',
                'beta_b0', 'mu_b0', 'w_b0', 'r_b0',
                'mu_br', 's_br']
        
        # original. It works
        # traces_selection = {key:self.traces[key] for key in keys}
        
        # untested        
        traces_selection = {key:self.traces[key][:self.last_iter] for key in keys}
        
        for key in traces_selection.keys():
            #outfile = os.path.join(self.traces_dir, "traces." + self.worker_exp + "." + key + ".trc")
            outfile = os.path.join(self.traces_dir, key + ".trc")
            pd.DataFrame(traces_selection[key]).to_csv(outfile, sep='\t')   
       
    def pickle_state(self, fname='status.pck', fnamedata='data.pck'):
        """Pickle current state so that we can make post-mortem debugs"""
        print "picking traces in file"
        with open(fname, 'wb') as f:             
            st = {'beta_b0': self.beta_b0,
                  'beta_a0': self.beta_a0,
                  'R_a0': self.R_a0,
                  'r_b0': self.r_b0,
                  'W_a0': self.W_a0,
                  'w_b0': self.w_b0,
                  'mu_a0': self.mu_a0,
                  'mu_b0': self.mu_b0,
                  'S_ar': self.S_ar,
                  's_br': self.s_br,
                  'mu_ar': self.mu_ar,
                  'mu_br': self.mu_br,
                  'b': self.b,
                  's_y': self.s_y,
                  'z': self.z,
                  'alpha': self.alpha}
            pickle.dump(st, f)
        
        # data and fixed parameters
        with open('data.pck', 'wb') as f:             
            data = {'R': self.R,
                   'A': self.A,
                   'P': self.P,
                   'y': self.y,
                   'DP': self.DP}    
            pickle.dump(data, f)
               
             
    
def rwishart_matlab(dof, sigma):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    Use this intead of R libraries since R libraries only allow dof >= p
    and not dof > p-1
    (see http://stats.stackexchange.com/questions/136305/rwishart-should-be-dofp-1-or-dof-ge-p)
    Taken from: http://www.mit.edu/~mattjj/released-code/hsmm/stats_util.py
    '''
    # TODO: replaced the tri line: 2. -> 2 to use integers.
    # deprectation warning. But why they use floats for size 
    # in the first place?

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing 
    # between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=int(n*(n-1)/2))        
        #A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    try:
        v = np.dot(X,X.T)
    except Exception as e:
        print "Exception in rwishart"
        print e
        import pdb; pdb.pm()
    try:
        np.linalg.cholesky(inv(v))
    except Exception as e:
        print "rwishart proposed non sdp"
        print e
        print "Proposed:\n", repr(inv(v))
        
    return np.dot(X,X.T)