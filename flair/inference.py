import numpy as np
import pandas as pd
#import random
import scipy as sc
import scipy.stats as stats

from scipy.special import factorial,digamma
import numdifftools as nd
from scipy.optimize import minimize
from joblib import Parallel, delayed

##############################################################################################################
###########################################  Functions   #####################################################


def starting_point(i,Experiment):
    #Takes as input the construct number i
    #Returns the empirical mean and variance
    T=np.ceil(Experiment.nijhat[i,:]).astype(int)
    T=np.repeat(Experiment.mean_assigned,T)
    if np.max(T) == np.min(T):  #What if all the cells fall into one unique bin?
        j=np.where(Experiment.mean_assigned==np.max(T))[0][0]
        mu=np.max(T)
        std=(Experiment.partitioning[j+1]-Experiment.partitioning[j])/4
    elif not np.any(T):
        return(np.array([0,0]))
    else:
        mu=np.mean(T)
        std=np.std(T,ddof=1)
    return (np.array([mu,std**2]))


def neg_ll_rep(theta,i,Experiment):
    #takes as input the logparameter theta=(alpha,beta), the construct number i
    #Returns the likelihood
    alpha=theta[0]
    beta=theta[1]
    NL=0
    for j in range(Experiment.bins):
        if Experiment.nj[j]==0:
            pass
        else :
            #Compute intensity parameter
            if Experiment.distribution=='lognormal':
                probability_bin=stats.norm.cdf(Experiment.partitioning[j+1],loc=np.exp(alpha),scale=np.exp(beta))-stats.norm.cdf(Experiment.partitioning[j],loc=np.exp(alpha),scale=np.exp(beta))
            else:
                probability_bin=stats.gamma.cdf(Experiment.partitioning[j+1],a=np.exp(alpha),scale=np.exp(beta))-stats.gamma.cdf(Experiment.partitioning[j],a=np.exp(alpha),scale=np.exp(beta))
            intensity=Experiment.nihat[i]*probability_bin*Experiment.reads[j]/Experiment.nj[j]        
            #Compute Likelihood
            if Experiment.sequencing[i,j]!=0:
                if intensity>0: #Avoid float error with np.log
                    NL+=intensity-Experiment.sequencing[i,j]*np.log(intensity)
            else:
                NL+=intensity
    return(NL)


def neg_ll(theta,i,Experiment):
    #takes as input the parameter theta=(alpha,beta), the construct number i
    #Returns the likelihood
    alpha=theta[0]
    beta=theta[1]
    NL=0
    for j in range(Experiment.bins):
        if Experiment.nj[j]==0:
            pass
        else :
            #Compute intensity parameter
            if Experiment.distribution=='lognormal':
                probability_bin=stats.norm.cdf(Experiment.partitioning[j+1],loc=alpha,scale=beta)-stats.norm.cdf(Experiment.partitioning[j],loc=alpha,scale=beta)
            else:
                probability_bin=stats.gamma.cdf(Experiment.partitioning[j+1],a=alpha,scale=beta)-stats.gamma.cdf(Experiment.partitioning[j],a=alpha,scale=beta)
            intensity=Experiment.nihat[i]*probability_bin*Experiment.reads[j]/Experiment.nj[j]          
            #Compute Likelihood
            if Experiment.sequencing[i,j]!=0:
                if intensity>0: #Avoid float error with np.log
                    NL+=intensity-Experiment.sequencing[i,j]*np.log(intensity)
            else:
                NL+=intensity
    return(NL)

def ML_inference_reparameterised(i,Experiment):
    #Takes as input the construct number i
    #Returns a numpy array containing the FLAIR inference,confidence intervals, MOM inference, scoring and validity of ML inference
    Dataresults=np.zeros(8)
    T=Experiment.nijhat[i,:]
    if np.sum(T)!=0:     #Can we do inference? has the genetic construct been sequenced?
        Dataresults[7]=(T[0]+T[-1])/np.sum(T) #Scoring of the data- How lopsided is the read count? all on the left-right border?
        SP=starting_point(i,Experiment)
        #The four next lines provide the MOM estimates on a,b, mu and sigma
        if Experiment.distribution=='lognormal':
            Dataresults[4]=SP[0] #value of mu MOM
            Dataresults[5]=np.sqrt(SP[1]) #Value of sigma MOM
            if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
                Dataresults[6]=3 #Inference grade 3 : Naive inference
        
            else:  #in the remaining case, we can deploy the mle framework to improve the mom estimation
                IV=np.log(np.array([SP[0],np.sqrt(SP[1])]))  #initial value for log(mu,sigma)
                res=minimize(neg_ll_rep,IV,args=(i,Experiment),method="Nelder-Mead")
                c,d=res.x
                Dataresults[0],Dataresults[1]=np.exp(c),np.exp(d)
                #Compute confidence intervals with hessian
                fi = lambda x: neg_ll(x,i,Experiment)
                fdd = nd.Hessian(fi)
                hessian_ndt=fdd([np.exp(c), np.exp(d)])
                if np.isfinite(hessian_ndt).all():
                    with np.errstate(invalid='ignore'):
                        if np.all(np.linalg.eigvals(hessian_ndt)>0)==True: 
                            inv_J=np.linalg.inv(hessian_ndt)
                            if Experiment.distribution=='lognormal':
                                jacobian=np.diag((1,1))
                            e,f=np.sqrt(np.diag(np.matmul(np.matmul(jacobian,inv_J),jacobian.T)))
                            Dataresults[2]=e
                            Dataresults[3]=f
                            Dataresults[6]=1 #Inference grade 1 : ML inference  successful

                        else:
                            Dataresults[6]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
                else:
                    Dataresults[6]=2
        elif Experiment.distribution=='gamma':
            Dataresults[4]=SP[0]**2/SP[1] #value of mu MOM
            Dataresults[5]=SP[1]/SP[0] #Value of sigma MOM
            if np.count_nonzero(T)==1: #is there only one bin to be considered? then naive inference
                Dataresults[6]=3 #Inference grade 3 : Naive inference
        
            else:  #in the remaining case, we can deploy the mle framework to improve the mom estimation
                IV=np.log(np.array([(SP[0]**2)/SP[1],(SP[1])/SP[0]]))  #initial value for log(a,b) 
                res=minimize(neg_ll_rep,IV,args=(i,Experiment),method="Nelder-Mead")
                c,d=res.x
                Dataresults[0],Dataresults[1]=np.exp(c),np.exp(d)
                #Compute confidence intervals with hessian
                fi = lambda x: neg_ll(x,i,Experiment)
                fdd = nd.Hessian(fi)
                hessian_ndt=fdd([np.exp(c), np.exp(d)])
                if np.isfinite(hessian_ndt).all():
                    with np.errstate(invalid='ignore'):
                        if np.all(np.linalg.eigvals(hessian_ndt)>0)==True: 
                            inv_J=np.linalg.inv(hessian_ndt)
                            a=np.exp(c)
                            b=np.exp(d)
                            jacobian=np.diag((1,1))#np.array([[b,a],[b/(2*np.sqrt(a)),np.sqrt(a)]])
                            e,f=np.sqrt(np.diag(np.matmul(np.matmul(jacobian,inv_J),jacobian.T)))
                            Dataresults[2]=e
                            Dataresults[3]=f
                            Dataresults[6]=1 #Inference grade 1 : ML inference  successful

                        else:
                            Dataresults[6]=2 #Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
                else:
                    Dataresults[6]=2
    else:
        Dataresults[6]=4   #Inference grade 4: No inference is possible
    return(Dataresults)


def inference(p,q,Experiment):
    Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(ML_inference_reparameterised)(i,Experiment)for i in range(p,q))
    Data_results=np.array(Data_results)
    df= pd.DataFrame(Data_results)
    if Experiment.distribution=='lognormal':
        df.rename(columns={0: "mu_MLE", 1: "sigma_MLE", 2: "mu_std",3: "sigma_std",4: "mu_MOM", 5: "sigma_MOM", 6: "Inference_grade",7: "Score"}, errors="raise",inplace=True)
    elif Experiment.distribution=='gamma':
        df.rename(columns={0: "a_MLE", 1: "b_MLE", 2: "a_std",3: "b_std",4: "mu_MOM", 5: "sigma_MOM", 6: "Inference_grade",7: "Score"}, errors="raise",inplace=True)
    return(df)


