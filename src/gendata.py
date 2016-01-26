# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 16:09:58 2015
Generate synthetic data
@author: lumbrerasa
"""
from __future__ import division
import numpy as np
from matplotlib import pylab as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnorm
from scipy.stats import powerlaw
from numpy import dot, isnan

# Same covariance matrix for every cluster
cov = np.array([[0.01,0.001],
                [0.001,0.01]])
                   
def gen_line(F=2,U=None,z=None):
    """
    Generates a set of clusters ligned up in a row
    """
    R = len(set(z))
    A = np.ones((F,U), dtype=np.float32)
    for r in xrange(R):
        x = r
        y = r
        A[:,z==r] = np.random.multivariate_normal([x,y], cov, sum(z==r)).T    
    return A
    
def gen_cross(F=2,U=None,z=None):
    """
    Generates a set of clusters around a cross
    """
    R = len(set(z))
    A = np.ones((F,U), dtype=np.float32)
    for r in xrange(int(R/2)):
        x = int(R/4)
        y = r
        A[:,z==r] = np.random.multivariate_normal([x,y], cov, sum(z==r)).T  
    for r in xrange(int(R/2),R):
        x = r-int(R/2)
        y = int(R/4)
        A[:,z==r] = np.random.multivariate_normal([x,y], cov, sum(z==r)).T  
    return A
    
def gen_circle(F=2,U=None,z=None):
    """
    Generates a set of clusters around a circle
    """
    R = len(set(z))
    A = np.ones((F,U), dtype=np.float32)
    for r in xrange(R):
        x = np.cos(2*3.1416*r/R)
        y = np.sin(2*3.1416*r/R)
        A[:,z==r] = np.random.multivariate_normal([x,y], cov, sum(z==r)).T
    return A

def gen_b(z, mu_br, s_br):
    """
    Generate individual coefficients according to cluster means
    """
    U = len(z)
    b = np.zeros(U+1) # includes intercept
    for u in range(U):
        r = z[u]
        b[u+1] = norm.rvs(loc=mu_br[r], scale=1/np.sqrt(s_br[r]))
    return b

def gen_b_confused(z, mu_br, s_br):
    """
    Generate individual coefficients according to cluster means
    but confusing the assignment of users of one role by considering from 
    another role
    """
    U = len(z)
    b = np.zeros(U+1) # includes intercept
    mu_br[1] = mu_br[2] # overlap cluster means
    for u in range(U):
        r = z[u]
        b[u+1] = norm.rvs(loc=mu_br[r], scale=1/np.sqrt(s_br[r]))
    return b    
    
def gen_participations(U=100, T=100, density=0.5, power=False):
    """ 
    Create a random participations matrix
    Returned P does NOT include artificial user/intercept corresponding to b_0
    
    Parameters:
    U: number of users
    T: number of threads or instances
    """
    
    # Draw participations
    P = np.zeros((U,T))
    if power:
        for u in xrange(U):
            alpha = 0.085
            alpha = 0.9
            p = powerlaw(alpha).rvs()
            P[u,:] = np.random.choice([0, 1], size=(T,), p=[1-p, p])        
    else:
        for t in xrange(T):
            P[:,t] = np.random.choice([0, 1], size=(U,), p=[1-density, density])   

    # Normalize
    row_sums = P.sum(axis=0)
    if (P[:,:].sum(axis=1)==0).sum() > 0:
        raise ValueError("Some users in 0 threads")    
    P = P / row_sums[np.newaxis, :]
    P[isnan(P)]=0
    
    return P

def gen_lengths(P, b, s_y, b0=0):
    """
    Generate thread lengths
    """

    # add intercept (imaginary user)
    P_ = np.vstack((np.ones(P.shape[1]), P)) 
    b_ = np.hstack((b0, b))
    
    y = mnorm.rvs(mean = dot(P_.T,b_), cov = 1/s_y) 
    return y
    