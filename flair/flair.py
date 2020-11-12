import numpy as np
import pandas as pd
#import random
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


def starting_point(i,FLUORESCENCE_MAX,BINS,Nj,READS,Nijhat,Nihat,distribution,Mean_expression_bins,Part_conv):
#Takes as input the construct number i
#Returns the empirical mean and standard deviation
    T=np.ceil(Nijhat[i,:]).astype(int)
    T=np.repeat(Mean_expression_bins,T)
    if np.max(T) == np.min(T):  #What if all the cells fall into one unique bin?
        j=np.where(Mean_expression_bins==np.max(T))[0][0]
        mu=np.max(T)
        std=(Part_conv[j+1]-Part_conv[j])/4
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        mu=np.mean(T)
        std=np.std(T,ddof=1)
    return (np.array([mu,std**2]))


def neg_ll_rep(theta,i,BINS,Part_conv,READS,Nj,Nihat,distribution,Sij):
#takes as input the parameter theta=(alpha,beta), the construct number i
#Returns the likelihood
    alpha=theta[0]
    beta=theta[1]
    NL=0
    for j in range(BINS):
    #Compute intensity parameter
        if Nj[j]==0:
            intensity=0
        else :
            if distribution=='lognormal':
                probability_bin=stats.norm.cdf(Part_conv[j+1],loc=np.exp(alpha),scale=np.exp(beta))-stats.norm.cdf(Part_conv[j],loc=np.exp(alpha),scale=np.exp(beta))
            else:
                probability_bin=stats.gamma.cdf(Part_conv[j+1],a=np.exp(alpha),scale=np.exp(beta))-stats.gamma.cdf(Part_conv[j],a=np.exp(alpha),scale=np.exp(beta))
            intensity=Nihat[i]*probability_bin*READS[j]/Nj[j]
    #Compute Likelihood
        if Sij[i,j]!=0:
            if intensity>0: #Avoid float error with np.log
                NL+=intensity-Sij[i,j]*np.log(intensity)
        else:
            NL+=intensity
    return(NL)

def ML_inference_reparameterised(i,FLUORESCENCE_MAX,BINS,Nj,READS,Nijhat,Nihat,distribution,Mean_expression_bins,Part_conv,Sij):
#Takes as input the construct number i
#Returns a numpy array containing the FLAIR inference,confidence intervals, MOM inference, scoring and validity of ML inference
    Dataresults=np.zeros(8)
    T=Nijhat[i,:]
    if np.sum(T)!=0:     #Can we do inference? has the genetic construct been sequenced?
        Dataresults[7]=(T[0]+T[-1])/np.sum(T) #Scoring of the data- How lopsided is the read count? all on the left-right border?
        SP=starting_point(i,FLUORESCENCE_MAX,BINS,Nj,READS,Nijhat,Nihat,distribution,Mean_expression_bins,Part_conv)
        #The four next lines provide the MOM estimates on a,b, mu and sigma
        Dataresults[4]=SP[0] #value of mu MOM
        Dataresults[5]=np.sqrt(SP[1]) #Value of sigma MOM
        if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
            Dataresults[6]=3 #Inference grade 3 : Naive inference
        else:  #in the remaining case, we can deploy the mle framework to improve the mom estimation
            if distribution=='lognormal':
                IV=np.log(np.array([SP[0],np.sqrt(SP[1])]))  #initial value for log(mu,sigma)
            else:
                IV=np.log(np.array([(SP[0]**2)/SP[1],(SP[1])/SP[0]]))  #initial value for log(a,b)
            res=minimize(neg_ll_rep,IV,args=(i,BINS,Part_conv,READS,Nj,Nihat,distribution,Sij),method="Nelder-Mead")
            c,d=res.x
            if distribution=='lognormal':
                Dataresults[0]=np.exp(c) #value of mu, MLE
                Dataresults[1]=np.exp(d) #value of sigma , MLE
            else:
                Dataresults[0]=np.exp(c+d) #value of mu, MLE
                Dataresults[1]=np.exp(c/2+d) #value of sigma , MLE
            fi = lambda x: neg_ll_rep(x,i,BINS,Part_conv,READS,Nj,Nihat,distribution,Sij)
            fdd = nd.Hessian(fi)
            hessian_ndt=fdd([c, d])
            if np.all(np.linalg.eigvals(hessian_ndt)>0)==True:
                inv_J=np.linalg.inv(hessian_ndt)
                if distribution=='lognormal':
                    jacobian=np.diag((np.exp(c),np.exp(d)))
                else:
                    a=np.exp(c)
                    b=np.exp(d)
                    jacobian=np.array([[b,a],[b/(2*np.sqrt(a)),np.sqrt(a)]])
                e,f=np.sqrt(np.diag(np.matmul(np.matmul(jacobian.T,inv_J),jacobian)))
                Dataresults[2]=e
                Dataresults[3]=f
                Dataresults[6]=1 #Inference grade 1 : ML inference  successful
            else:
                Dataresults[6]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
    else:
        Dataresults[6]=4   #Inference grade 4: No inference is possible
    return(Dataresults)


def inference(p,FLUORESCENCE_MAX,BINS,Nj,READS,Nijhat,Nihat,distribution,Mean_expression_bins,Part_conv,Sij):
    Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i,FLUORESCENCE_MAX,BINS,Nj,READS,Nijhat,Nihat,distribution,Mean_expression_bins,Part_conv,Sij)for i in range(p))
    Data_results=np.array(Data_results)
    df= pd.DataFrame(Data_results)
    df.rename(columns={0: "mu_MLE", 1: "sigma_MLE", 2: "mu_std",3: "sigma_std",4: "mu_MOM", 5: "sigma_MOM", 6: "Inference_grade",7: "Score"}, errors="raise",inplace=True)
    return(df)


