import numpy as np

def log_moments_mu(m,s):
    #Takes as input the parameters m and s of the log normal distribution and outputs the mean and variance
    return(np.exp(m+0.5*(s**2)))

def log_moments_sigma(m,s):
    #Takes as input the parameters m and s of the log normal distribution and outputs the variance
    return(np.exp(2*m+(s**2))*(np.exp(s**2)-1))

def MAPE(X,Xhat):
    return(np.abs(X-Xhat)/X)

