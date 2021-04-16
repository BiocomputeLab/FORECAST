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



#Data for the experiment
q=pd.read_csv('VI.csv')

# Create the figure
fig = plt.figure(figsize=(1.8, 2.2))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])


# X_mu=df.sort_values('Mean',ascending=True)['Mean'][:-4]
# Y_mu=df.sort_values('Mean',ascending=True)['mu_MLE'][:-4]
# Z_mu=df.sort_values('Mean',ascending=True)['mu_MOM'][:-4]
# X_sigma=df.sort_values('variance',ascending=True)['variance'][:-4]
# Y_sigma=df.sort_values('variance',ascending=True)['sigma_squared_MLE'][:-4]
# Z_sigma=df.sort_values('variance',ascending=True)['sigma_squared_MOM'][:-4]

# ax1.scatter(X_mu,Y_mu,s=20,c='tab:brown',label='ML estimate')
# ax1.scatter(X_mu,Z_mu,s=20,c='#4f7942',label='MOM estimate')
# ax1.plot(X_mu,X_mu,c='tab:orange',label='Ground Truth',linewidth=2.5,linestyle='dashed')
# ax1.legend()
# ax1.set(xlabel='True Fluorescence Mean (a.u.)', ylabel='Estimated Fluorescence Mean (a.u.)')
# #plt.title('')
# ax1.legend(frameon=False,fontsize=10,markerscale=2)

X_mu=q.sort_values('Mean',ascending=True)['Mean']  
Y_mu=q.sort_values('Mean',ascending=True)['mu_ML']  
Z_mu=q.sort_values('Mean',ascending=True)['mu_MOM']  
X_sigma=q.sort_values('standard deviation',ascending=True)['standard deviation']   
Y_sigma=q.sort_values('standard deviation',ascending=True)['sigma_ML']   
Z_sigma=q.sort_values('standard deviation',ascending=True)['sigma_MOM']   

ax1.scatter(X_mu,Y_mu,s=12,c='tab:brown',label='ML estimate')
ax1.scatter(X_mu,Z_mu,s=12,c='#4f7942',label='MOM estimate')
ax1.plot(X_mu,X_mu,c='tab:orange',label='Ground Truth',linewidth=1.5,linestyle='dashed')
ax1.legend()
ax1.set(xlabel='Mean Fluorescence (a.u.)', ylabel='Estimated mean Fluorescence (a.u.)')
#plt.title('')
ax1.legend(frameon=False,fontsize=10,markerscale=2)
ax1.set_yscale('log')
ax1.set_xscale('log')

plt.show()
sns.despine()
#plt.show()



width=3.54
height=3.54
fig.set_size_inches(width, height)
fig.savefig('plot.pdf')

#Tom's remaining code
plt.subplots_adjust(hspace=.0 , wspace=.00, left=.15, right=.95, top=.95, bottom=.13)
fig.savefig('gamma_mu_mom_mu.pdf', transparent=True)
# plt.close('all')

#############################################################
