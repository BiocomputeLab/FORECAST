import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import scipy.stats as stats 
from scipy.special import factorial,digamma
import numdifftools as nd
from scipy.optimize import minimize 
from joblib import Parallel, delayed


##############################################################################################################
###########################################  Functions   #####################################################

def intensity_parameter_reparameterisation(i,j,alpha,beta):  # We enforced the positive constraint on a and b by rewriting alpha=log(a) and beta=log(b)
    Number_construct=Ni[i]
    if Nj[j]==0:
        return(0)
    else :
        probability_bin=stats.norm.cdf(Part_conv[j+1],loc=np.exp(alpha),scale=np.exp(beta))-stats.norm.cdf(Part_conv[j],loc=np.exp(alpha),scale=np.exp(beta))
        return Number_construct*probability_bin*READS[j]/Nj[j]

def data_transformation_bins(X):  #New representation of the data enabling the method of moments
    X=np.ceil(X)
    X=X.astype(int)
    T=np.repeat(Mean_expression_bins,X)
    return(T)


def starting_point_binned_reparameterised(X):   #Compute empirical moments from data and return the log reparameterisation of both shape and scale parameters of the gamma distribution 
    X=np.ceil(X)
    X=X.astype(int)
    T=data_transformation_bins(X)
    if np.count_nonzero(X)==1:  #What if all the cells fall into one unique bin?
        j=np.where(X!=0)[0][0]
        mu=np.mean(T)
        std=(Part_conv[j+1]-Part_conv[j])/4
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        mu=np.mean(T)
        std=np.std(T,ddof=1)
    return np.log(np.array([mu,std]))


def neg_ll_rep(theta,construct):
    alpha=theta[0]
    beta=theta[1]
    NL=0
    i=construct
    for j in range(BINS):
        intensity=intensity_parameter_reparameterisation(i,j,alpha,beta)
        if Sij[construct,j]!=0:
            if intensity>0: #Avoid float error with np.log
                NL+=intensity-Sij[i,j]*np.log(intensity)
        else:
            NL+=intensity
    return(NL)

def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)

def ML_inference_reparameterised(i):
    Dataresults=np.zeros(8)
    T=Nihat[i,:]
    if np.sum(T)!=0:     #Can we do inference? has the genetic construct been sequenced?
        Dataresults[7]=(T[0]+T[-1])/np.sum(T) #Scoring of the data- How lopsided is the read count? all on the left-right border?
        a,b=starting_point_binned_reparameterised(T)
        #The four next lines provide the MOM estimates on a,b, mu and sigma
        Dataresults[4]=np.exp(a) #value of mu MOM
        Dataresults[5]=np.exp(b) #Value of sigma MOM
        if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
            Dataresults[6]=3 #Inference grade 3 : Naive inference
        else:  #in the remaining case, we can deploy the mle framework to improve the mom estimation
            res=minimize(neg_ll_rep,starting_point_binned_reparameterised(T),args=(i),method="Nelder-Mead")
            c,d=res.x
            Dataresults[0]=np.exp(c) #value of mu, MLE
            Dataresults[1]=np.exp(d) #value of sigma squared, MLE
            fi = lambda x: neg_ll_rep(x,i)
            fdd = nd.Hessian(fi) 
            hessian_ndt=fdd([c, d])
            if is_pos_def(hessian_ndt)==True:
                inv_J=np.linalg.inv(hessian_ndt)
                e,f=np.sqrt(np.diag(np.matmul(np.matmul(np.diag((np.exp(c),np.exp(d))),inv_J),np.diag((np.exp(c),np.exp(d))))))
                Dataresults[2]=e
                Dataresults[3]=f
                Dataresults[6]=1 #Inference grade 1 : ML inference  successful
            else:
                Dataresults[6]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
    else:
        Dataresults[6]=4   #Inference grade 4: No inference is possible
    return(Dataresults)


