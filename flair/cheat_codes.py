import numpy as np
from scipy import special,integrate
import scipy.stats as stats


def log_moments_mu(m,s):
    #Takes as input the parameters m and s of the log normal distribution and outputs the mean and variance
    return(np.exp(m+0.5*(s**2)))

def log_moments_sigma(m,s):
    #Takes as input the parameters m and s of the log normal distribution and outputs the variance
    return(np.exp(2*m+(s**2))*(np.exp(s**2)-1))

def MAPE(X,X_hat):
    return(np.abs(X-X_hat)/X)

def WD(m,s,m0,s0):
    #Computes 1-Wasserstein distance between the two normals parametersied by m,s and m0,s0
    f=lambda x: np.abs(m-m0 +np.sqrt(2)*(s-s0)*special.erfinv(x))
    return(integrate.quad(f, 0, 1))

def  Visibility(m,s,Fm):
    #Computes the probability mass up to Fm of the normal distribution parameterised by m and s
    return(stats.norm.cdf(Fm,loc=m,scale=s)-stats.norm.cdf(0,loc=m,scale=s))