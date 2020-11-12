import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import scipy.stats as stats
from scipy.special import gamma, factorial,digamma
import numdifftools as nd
from scipy.optimize import minimize
from joblib import Parallel, delayed



def Sorting_and_Sequencing(BINS,Diversity,N,BUDGET_READS,Ratio_amplification,BIAS_LIBRARY,distribution,Part_conv,A,B):
    # take as input the number of bins (BINS), the diversity (Diversity) and size of the library sorted (N), the number of reads to allocate in total (BUDGET_READS), the post sorting amplification step (Ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Part_conv),and the parameters of the said distribution.
    # Return the (Diversity*Bins) matrix resulting from the sequencing and the sorting matrix Nj (number of cell sorted in each bin)
    global Sij
    #### STEP 1 - Draw the ratio p_concentration
    
    def sorting_protein_matrix_populate(i,j):
        if distribution=='lognormal':
            element_matrix=stats.norm.cdf(Part_conv[j+1],loc=A[i], scale=B[i])-stats.norm.cdf(Part_conv[j],loc=A[i], scale=B[i])
        else:
            element_matrix=stats.gamma.cdf(Part_conv[j+1],a=A[i], scale=B[i])-stats.gamma.cdf(Part_conv[j],a=A[i], scale=B[i])
        return(stats.norm.cdf(Part_conv[j+1],loc=A[i], scale=B[i])-stats.norm.cdf(Part_conv[j],loc=A[i], scale=B[i]))

    if BIAS_LIBRARY==True:
       params=np.ones(Diversity)
       Dir=[random.gammavariate(a,1) for a in params]
       Dir=[v/sum(Dir) for v in Dir]
       # Sample from the 30,000 simplex to get ratios
       #p_concentration=np.ones(Diversity)/Diversity
       p_concentration=Dir
    else:
       p_concentration=[1/Diversity]*Diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct

    Ni=np.random.multinomial(N, p_concentration, size=1)[0]
    #Ni=Ni[0]

    #### STEP 3 - Compute binning

    ## Compute ratios qji
    Qij=np.fromfunction(sorting_protein_matrix_populate, (Diversity, BINS), dtype=int)
    #Qij=np.fromfunction(lambda i, j: i + j, (Diversity, BINS), dtype=int)
    ## Compute Nij
    Nij=Qij* Ni[:, np.newaxis]
    Nij=np.floor(Nij) #Convert to Integer numbers

    #### STEP 4 - PCR amplification

    Nij_amplified=np.multiply(Nij,Ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    N=np.sum(Nij)
    Nj=np.sum(Nij, axis=0)
    READS=np.floor(Nj*BUDGET_READS/N) #Allocate reads with repsect to the number of cells srted in each bin
    #### STEP 6 - DNA sampling

    Sij=np.zeros((Diversity,BINS))

    #Compute ratios& Multinomial sampling
    for j in range(BINS):
        if np.sum(Nij_amplified,axis=0)[j]!=0:
            Concentration_vector=Nij_amplified[:,j]/np.sum(Nij_amplified,axis=0)[j]
        else:
            Concentration_vector=np.zeros(Diversity)
        Sij[:,j]=np.random.multinomial(READS[j],Concentration_vector,size=1)
    return(Sij,Nj)



def Sorting(BINS,Diversity,N,BIAS_LIBRARY,distribution,Part_conv,A,B):
# take as input the number of bins (BINS), the diversity (Diversity) and size of the library sorted (N), the post sorting amplification step (Ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Part_conv),and the parameters of the said distribution.
# Return the (Diversity*Bins) matrix resulting from the sorting step

    global Sij
    #### STEP 1 - Draw the ratio p_concentration
    
    def sorting_protein_matrix_populate(i,j):
        return(stats.norm.cdf(Part_conv[j+1],loc=A[i], scale=B[i])-stats.norm.cdf(Part_conv[j],loc=A[i], scale=B[i]))

    if BIAS_LIBRARY==True:
       params=np.ones(Diversity)
       Dir=[random.gammavariate(a,1) for a in params]
       Dir=[v/sum(Dir) for v in Dir]
       # Sample from the 30,000 simplex to get ratios
       #p_concentration=np.ones(Diversity)/Diversity
       p_concentration=Dir
    else:
       p_concentration=[1/Diversity]*Diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct

    Ni=np.random.multinomial(N, p_concentration, size=1)[0]
    #Ni=Ni[0]

    #### STEP 3 - Compute binning

    ## Compute ratios qji
    Qij=np.fromfunction(sorting_protein_matrix_populate, (Diversity, BINS), dtype=int)
    #Qij=np.fromfunction(lambda i, j: i + j, (Diversity, BINS), dtype=int)
    ## Compute Nij
    Nij=Qij* Ni[:, np.newaxis]
    Nij=np.floor(Nij) #Convert to Integer numbers
    return(Nij)
    
    
def Sequencing(BINS,Diversity,BUDGET_READS,Ratio_amplification,Nij):
# take as input the number of bins (BINS), the diversity (Diversity) and size of the library sorted (N), the number of reads to allocate in total (BUDGET_READS), the post sorting amplification step (Ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Part_conv),and the parameters of the said distribution.
# Return the (Diversity*Bins) matrix resulting from the sequencing Sij and the sorting matrix Nj (number of cell sorted in each bin)

    #### STEP 4 - PCR amplification

    Nij_amplified=np.multiply(Nij,Ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    N=np.sum(Nij)
    Nj=np.sum(Nij, axis=0)
    READS=np.floor(Nj*BUDGET_READS/N) #Allocate reads with repsect to the number of cells srted in each bin
    #### STEP 6 - DNA sampling

    Sij=np.zeros((Diversity,BINS))

    #Compute ratios& Multinomial sampling
    for j in range(BINS):
        if np.sum(Nij_amplified,axis=0)[j]!=0:
            Concentration_vector=Nij_amplified[:,j]/np.sum(Nij_amplified,axis=0)[j]
        else:
            Concentration_vector=np.zeros(Diversity)
        Sij[:,j]=np.random.multinomial(READS[j],Concentration_vector,size=1)
    return(Sij,Nj)
