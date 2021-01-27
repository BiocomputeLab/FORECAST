
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


def Sorting_and_Sequencing(Simulation):
    # take as input the number of bins (Simulation.bins), the diversity (Simulation.diversity) and size of the library sorted (N), the number of reads to allocate in total (Simulation.reads), the post sorting amplification step (Simulation.ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein Simulation.distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Simulation.partitioning),and the parameters of the said Simulation.distribution.
    # Return the (Simulation.diversity*Bins) matrix resulting from the sequencing and the sorting matrix Nj (number of cell sorted in each bin)
    global Sij
    #### STEP 1 - Draw the ratio p_concentration
    
    def sorting_protein_matrix_populate(i,j):
        if Simulation.distribution=='lognormal':
            element_matrix=stats.norm.cdf(Simulation.partitioning[j+1],loc=Simulation.theta1[i], scale=Simulation.theta2[i])-stats.norm.cdf(Simulation.partitioning[j],loc=Simulation.theta1[i], scale=Simulation.theta2[i])
        else:
            element_matrix=stats.gamma.cdf(Simulation.partitioning[j+1],a=Simulation.theta1[i], scale=Simulation.theta2[i])-stats.gamma.cdf(Simulation.partitioning[j],a=Simulation.theta1[i], scale=Simulation.theta2[i])
        return(element_matrix)

    if Simulation.bias_library==True:
       params=np.ones(Simulation.diversity)
       Dir=[random.gammavariate(a,1) for a in params]
       Dir=[v/sum(Dir) for v in Dir]
       # Sample from the 30,000 simplex to get ratios
       #p_concentration=np.ones(Simulation.diversity)/Simulation.diversity
       p_concentration=Dir
    else:
       p_concentration=[1/Simulation.diversity]*Simulation.diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct

    Ni=np.random.multinomial(Simulation.size, p_concentration, size=1)[0]

    #### STEP 3 - Compute binning

    ## Compute ratios qji
    Qij=np.fromfunction(sorting_protein_matrix_populate, (Simulation.diversity, Simulation.bins), dtype=int)

    ## Compute Nij
    Nij=Qij* Ni[:, np.newaxis]
    Nij=np.floor(Nij) #Convert to Integer numbers

    #### STEP 4 - PCR amplification

    Nij_amplified=np.multiply(Nij,Simulation.ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    N=np.sum(Nij)
    Nj=np.sum(Nij, axis=0)
    READS=np.floor(Nj*Simulation.reads/N) #Allocate reads with repsect to the number of cells srted in each bin
    #### STEP 6 - DNA sampling

    Sij=np.zeros((Simulation.diversity,Simulation.bins))

    #Compute ratios& Multinomial sampling
    for j in range(Simulation.bins):
        if np.sum(Nij_amplified,axis=0)[j]!=0:
            Concentration_vector=Nij_amplified[:,j]/np.sum(Nij_amplified,axis=0)[j]
        else:
            Concentration_vector=np.zeros(Simulation.diversity)
        Sij[:,j]=np.random.multinomial(READS[j],Concentration_vector,size=1)
    return(Sij,Nj)



def Sorting(Simulation):
    # take as input the number of bins (Simulation.bins), the diversity (Simulation.diversity) and size of the library sorted (N), the post sorting amplification step (Simulation.ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein Simulation.distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Simulation.partitioning),and the parameters of the said Simulation.distribution.
    # Return the (Simulation.diversity*Bins) matrix resulting from the sorting step

    global Sij
    #### STEP 1 - Draw the ratio p_concentration
    
    def sorting_protein_matrix_populate(i,j):
        if Simulation.distribution=='lognormal':
            element_matrix=stats.norm.cdf(Simulation.partitioning[j+1],loc=Simulation.theta1[i], scale=Simulation.theta2[i])-stats.norm.cdf(Simulation.partitioning[j],loc=Simulation.theta1[i], scale=Simulation.theta2[i])
        else:
            element_matrix=stats.gamma.cdf(Simulation.partitioning[j+1],a=Simulation.theta1[i], scale=Simulation.theta2[i])-stats.gamma.cdf(Simulation.partitioning[j],a=Simulation.theta1[i], scale=Simulation.theta2[i])
        return(element_matrix)

    if Simualtion.bias_library==True:
       params=np.ones(Simulation.diversity)
       Dir=[random.gammavariate(a,1) for a in params]
       Dir=[v/sum(Dir) for v in Dir]
       # Sample from the 30,000 simplex to get ratios
       #p_concentration=np.ones(Simulation.diversity)/Simulation.diversity
       p_concentration=Dir
    else:
       p_concentration=[1/Simulation.diversity]*Simulation.diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct

    Ni=np.random.multinomial(Simulation.size, p_concentration, size=1)[0]
    #Ni=Ni[0]

    #### STEP 3 - Compute binning

    ## Compute ratios qji
    Qij=np.fromfunction(sorting_protein_matrix_populate, (Simulation.diversity, Simulation.bins), dtype=int)

    ## Compute Nij
    Nij=Qij* Ni[:, np.newaxis]
    Nij=np.floor(Nij) #Convert to Integer numbers
    return(Nij)
    
    
def Sequencing(Simulation,Nij):
    # take as input the number of bins (Simulation.bins), the diversity (Simulation.diversity) and size of the library sorted (N), the number of reads to allocate in total (Simulation.reads), the post sorting amplification step (Simulation.ratio_amplification),if the library is balanced (BIAS_Library), the underlying protein Simulation.distribution (gamma or lognormal), the fluorescence bounds for the sorting machine (Simulation.partitioning),and the parameters of the said Simulation.distribution.
    # Return the (Simulation.diversity*Bins) matrix resulting from the sequencing Sij and the sorting matrix Nj (number of cell sorted in each bin)

    #### STEP 4 - PCR amplification

    Nij_amplified=np.multiply(Nij,Simulation.ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    N=np.sum(Nij)
    Nj=np.sum(Nij, axis=0)
    READS=np.floor(Nj*Simulation.reads/N) #Allocate reads with repsect to the number of cells srted in each bin
    #### STEP 6 - DNA sampling

    Sij=np.zeros((Simulation.diversity,Simulation.bins))

    #Compute ratios& Multinomial sampling
    for j in range(Simulation.bins):
        if np.sum(Nij_amplified,axis=0)[j]!=0:
            Concentration_vector=Nij_amplified[:,j]/np.sum(Nij_amplified,axis=0)[j]
        else:
            Concentration_vector=np.zeros(Simulation.diversity)
        Sij[:,j]=np.random.multinomial(READS[j],Concentration_vector,size=1)
    return(Sij,Nj)
