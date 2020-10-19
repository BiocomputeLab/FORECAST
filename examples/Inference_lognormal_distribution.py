
import flair as fl

##############################################################################################################
#####################################  Technical Parameters  ##############################################

FLUORESCENCE_MAX=10**6
BINS=16
Part_conv=np.log(np.logspace(0,np.log10(FLUORESCENCE_MAX),BINS+1))  #Equally partitioning the fluorescence interval in log-space.Each entry is the lower bound for the fluoresnce in the bin
Mean_expression_bins=np.array([(Part_conv[j+1]+Part_conv[j])/2 for j in range(BINS)])

##############################################################################################################
############################################  Load Data  #####################################################

Nj=np.load('Nj_merged.npy') #FACS events in each bin ( Number of cells sorted in each bin)
Sij=np.load('Sij_merged.npy')  #Filtered Read Counts for each genetic construct (one row) in each bin (one column)
READS=np.array([ 1460332.,  2109815.,  2335533.,  3210865.,  4303324.,  5864139.,
        7490610.,  9922865., 12976416., 15188644., 19094267., 23689418.,
       23664179., 21895118., 17576043.,  5519053.])
Sij=Sij.astype(int)
N=sum(Nj)

##############################################################################################################
########################################  Auxiliary Values ###################################################

if np.any(READS==0):
   Enrich=Nj/(READS+0.001) 
   print('The number of reads allocated in one bin is suprisingly 0! are you sure?') 
else:
    Enrich=Nj/READS
Nihat=np.multiply(Sij,Enrich)
Nihat=np.around(Nihat)
Nihat=Nihat.astype(int)
Ni=Nihat.sum(axis=1)

##############################################################################################################
###########################################  Inference   #####################################################

Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(fl.ML_inference_reparameterised)(i)for i in range(1000))
Data_results=np.array(Data_results)


##############################################################################################################
#########################################  Save Results   ####################################################

df= pd.DataFrame(Data_results)
df.rename(columns={0: "mu_MLE", 1: "sigma_MLE", 2: "mu_std",3: "sigma_std",4: "mu_MOM", 5: "sigma_MOM", 6: "Inference_grade",7: "Score"}, errors="raise",inplace=True)
(df.iloc[:,:2]).to_csv('N.csv', index=False)
               



