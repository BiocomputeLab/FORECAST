
import numpy as np
import pandas as pd
#import random
import scipy as sc
import scipy.stats as stats
from scipy.special import factorial,digamma
import numdifftools as nd
from scipy.optimize import minimize
from joblib import Parallel, delayed



class Simulation():
    def __init__(self,bins,diversity,size,reads,fmax,distribution,ratio_amplification,theta1,theta2,bias_library):
        self.bins=bins
        self.diversity=diversity
        self.size=size
        self.reads=reads #number of reads in total for your simulation
        self.fmax=fmax
        self.distribution=distribution
        if distribution=='lognormal':
            # Working in log-space 
            self.partitioning=np.log(np.logspace(0,np.log10(self.fmax),bins+1))
        elif distribution=='gamma':
            partitioning=np.logspace(0,np.log10(self.fmax),bins+1)
            partitioning[0]=0
            self.partitioning=partitioning
        self.ratio_amplification=ratio_amplification #post sorting PCR: What is the PCR amplification ratio?
        self.theta1=theta1 #first parameter of the distribution (mu for a normal distribution or shape for a gamma distribution) in the form of an array
        self.theta2=theta2 #second parameter of the distribution (sigma for a normal distribution or scale for a gamma distribution) in the form of an array
        self.bias_library=bias_library


class Experiment():
    def __init__(self,bins,diversity,nj,reads,sequencing,fmax,distribution):
        self.bins=bins
        self.diversity=diversity
        self.nj=nj
        self.size=np.sum(self.nj)
        self.reads=reads
        self.sequencing=sequencing
        self.fmax=fmax
        self.distribution=distribution
        if distribution=='lognormal':
            # Working in log-space 
            self.partitioning=np.log(np.logspace(0,np.log10(fmax),bins+1))
        elif distribution=='gamma':
            #Working in normal fluorescence space
            partitioning=np.logspace(0,np.log10(fmax),bins+1)
            partitioning[0]=0
            self.partitioning=partitioning
        self.mean_assigned=[(self.partitioning[j+1]+self.partitioning[j])/2 for j in range(bins)]
        self.enrich=np.divide(nj, reads, out=np.zeros_like(nj), where=reads!=0, dtype=float)
        self.nijhat=np.multiply(self.sequencing,self.enrich).astype(int)
        self.nihat=self.nijhat.sum(axis=1)

