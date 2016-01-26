# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:21:56 2015

@author: alumbreras
"""
import numpy as np

def plot_posterior(samples, ax=None):
    """
    Plot posterior mean with 50 and 95 confidence bands.

    Parameters:
    ==========
    samples (numpy.ndarray): array with one sample of posterior in every row
    ax: axis to plot in
    """
    dims = samples.shape[1]
    ci_upper50 = np.percentile(samples, 75, axis=0)
    ci_lower50 = np.percentile(samples, 25, axis=0)
    ci_upper95 = np.percentile(samples, 98.5, axis=0)
    ci_lower95 = np.percentile(samples, 2.5, axis=0)
    E_y_preds = samples.mean(axis=0)
    
    fc="grey"
    ax.plot(np.arange(dims), E_y_preds, c='k')
    ax.fill_between(np.arange(dims), ci_upper50, ci_lower50, 
                       facecolor=fc, alpha=0.3)
    ax.fill_between(np.arange(dims), ci_upper95, ci_lower95, 
                       facecolor=fc, alpha=0.2)

    return ax
