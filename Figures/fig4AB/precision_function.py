import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.optimize as optimize
import csv
import scipy.stats as stats 
import pandas as pd
import seaborn as sns
#############################################################
# SETTINGS/PARAMETERS FOR HOW THE GRAPH LOOKS
#############################################################

# Axes titles
matplotlib.rcParams['axes.labelsize']  = 8
# Numbers on each axis
matplotlib.rcParams['ytick.labelsize']  = 8
matplotlib.rcParams['xtick.labelsize']  = 8
# Space between axis and number
matplotlib.rcParams['ytick.major.pad']  = 0.8
matplotlib.rcParams['ytick.minor.pad']  = 0.8
matplotlib.rcParams['xtick.major.pad']  = 1.5
matplotlib.rcParams['xtick.minor.pad']  = 1.5
matplotlib.rcParams['ytick.direction']  = 'out'
matplotlib.rcParams['xtick.direction']  = 'out'
# Lines around the graph
matplotlib.rcParams['axes.spines.left']   = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['axes.spines.top']    = False
matplotlib.rcParams['axes.spines.right']  = False
# Make text editable in Adobe Illustrator
matplotlib.rcParams['pdf.fonttype']          = 42 
matplotlib.rcParams.update({'font.size': 22})
# Colour maps to use for the genetic diagrams
# https://personal.sron.nl/~pault/

#Computing mean and std  accross different simulations-Storing them in a dictionary named MAPE and STDMAPE
MAPE_aux= {}
STDMAPE_aux={}
for st in ['Mean','Var']:
    for b in ['8','16']:
        for i in ['1','2','3','4']:
            for meth in ['MOM','MLE']:
                p1='Diversity_1000_Fmax_10^6_BINS'
                p2='ratio_1e6_lognormal'
                MAPE_aux["{St}_{Meth}_{B}_{I}".format(St=st,Meth=meth,B=b,I=i) ]=100*np.mean(np.load('{St}_sur_{Meth}_{P1}_{B}_{P2}_{I}.npy'.format(St=st,Meth=meth,B=b,I=i,P1=p1,P2=p2)),axis=0)
                STDMAPE_aux["{St}_{Meth}_{B}_{I}".format(St=st,Meth=meth,B=b,I=i) ]=100*np.std(np.load('{St}_sur_{Meth}_{P1}_{B}_{P2}_{I}.npy'.format(St=st,Meth=meth,B=b,I=i,P1=p1,P2=p2)),axis=0)
                
                
                
#Concatenate
MAPE= {}
STDMAPE={}
for st in ['Mean','Var']:
    for b in ['8','16']:
        for meth in ['MOM','MLE']:
            MAPE["{St}_{Meth}_{B}".format(St=st,Meth=meth,B=b)]=np.block([
    [MAPE_aux["{St}_{Meth}_{B}_1".format(St=st,Meth=meth,B=b) ],MAPE_aux["{St}_{Meth}_{B}_2".format(St=st,Meth=meth,B=b) ]],
    [MAPE_aux["{St}_{Meth}_{B}_3".format(St=st,Meth=meth,B=b) ], MAPE_aux["{St}_{Meth}_{B}_4".format(St=st,Meth=meth,B=b) ]    ]
])
            STDMAPE["{St}_{Meth}_{B}".format(St=st,Meth=meth,B=b)]=np.block([
    [STDMAPE_aux["{St}_{Meth}_{B}_1".format(St=st,Meth=meth,B=b) ],STDMAPE_aux["{St}_{Meth}_{B}_2".format(St=st,Meth=meth,B=b) ]],
    [STDMAPE_aux["{St}_{Meth}_{B}_3".format(St=st,Meth=meth,B=b) ], STDMAPE_aux["{St}_{Meth}_{B}_4".format(St=st,Meth=meth,B=b) ]    ]
])

def Cost_16(D,Capacity=5e6, Price=500):
    """
    Takes as input the number of genetic design and return the price for the sequencing step and average number of reads per construct
    To be better than ML inference with 16 bins, you need at least 500 sequencing reads and 200 cells sorted per different genetic design.
    
    D: Int
        Number of genetic designs
        
    Capacity:Int
        average number of reads in one sequencing lane. Default number is 5e6 
    
    Price: Int
        price for one sequecing lane. Default number is $500
    
    """
    #how many sequencing lanes do we need?
    n_lanes=np.ceil((500*D)/(16*Capacity))
    n_lanes=n_lanes*16
    #how many reads per construct do we have on average then?
    n_reads=np.floor(n_lanes*Capacity/D)
    return(n_lanes*Price,n_reads)

def Cost_8(D,Capacity=5e6, Price=500):
    """
    Takes as input the number of genetic design and return the price for the sequencing step and average number of reads per construct
    To be better than ML inference with 8 bins, you need at least 100 sequencing reads and 50 cells sorted per different genetic design.
    
    D: Int
        Number of genetic designs
        
    Capacity:Int
        average number of reads in one sequencing lane. Default number is 5e6 
    
    Price: Int
        price for one sequecing lane. Default number is $500
    
    """
    #how many sequencing lanes do we need?
    n_lanes=np.ceil((100*D)/(8*Capacity))
    n_lanes=n_lanes*8
    #how many reads per construct do we have on average then?
    n_reads=np.floor(n_lanes*Capacity/D)
    return(n_lanes*Price,n_reads)

Reads_range=[5,10,15,20,25,35,50,75,100,150,200,250,500,750,1000,5000,10000,100000]

import sklearn
from sklearn.linear_model import LinearRegression
x=(np.log([5,10,15,20,25,35,50,75,100,150,200,250,500,750,1000,5000,10000,100000])).reshape(-1, 1)
y1=np.log(MAPE['Mean_MLE_8'][17,:])
y2=np.log(MAPE['Mean_MLE_8'][17,:])
reg8 = LinearRegression().fit(x, y1)
reg16 = LinearRegression().fit(x, y2)



def Precision_8(D,Normalised_cells=10):
    myNumber=Cost_8(D)[1]
    MAP=reg8.predict(np.log([myNumber]).reshape(-1, 1))
    return(np.exp(MAP))

def Precision_16(D,Normalised_cells=10):
    myNumber=Cost_16(D)[1]
    MAP=reg16.predict(np.log([myNumber]).reshape(-1, 1))
    return(np.exp(MAP))

# Create the figure
fig = plt.figure(figsize=(1.8, 2.2))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])



fig, ax = plt.subplots(figsize=(15, 10))
D=np.linspace(1e2,int(1e7),int(1e5))
plt.plot(D,Precision_8(D),linewidth=2,label='Sorting in 8 bins')
plt.plot(D,Precision_16(D),linewidth=2,label='Sorting in 16 bins')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Imprecision (MAPE) ',fontsize=8)
sns.despine()
plt.xlabel('Number of genetic designs',fontsize=8)

plt.legend(frameon=False ,fontsize=8,markerscale=3)

# ax[1].plot(D,Precision_8(D),linewidth=2.5,label='Sorting in 8 bins')
# ax[1].plot(D,Precision_16(D),linewidth=2.5,label='Sorting in 16 bins')
# ax[1].yscale('log')
# ax[1].xscale('log')
# ax[1].ylabel('Imprecision (MAPE) ',fontsize=16)
# sns.despine()
# ax[1].xlabel('Number of genetic designs',fontsize=20)
# #ax[1].text(0.17, 1e4, r'$\langle \frac{\sigma_{MOM}-\sigma_{ML}}{\sigma_{ML}} \rangle \sim $%s'%approx_bias, fontsize=26)
# #ax[1].savefig(‘Cambray_comparison_Ml_mom_4_experiments_merged_logfluorescence_data_distribution_upward_bias_variance.png',transparent=True,bbox_inches='tight',dpi=600)
# ax[1].legend(frameon=False ,fontsize=20,markerscale=3)
# # ax[1].ticklabel_format(axis="x",style="sci", scilimits=(0,0))




plt.show()
sns.despine()
#plt.show()



width=3.54
height=3.54
fig.set_size_inches(width, height)

#Tom's remaining code
plt.subplots_adjust(hspace=.0 , wspace=.00, left=.15, right=.95, top=.95, bottom=.13)
fig.savefig('cost_precision.pdf', transparent=True)
# plt.close('all')

#############################################################
