
import matplotlib
#matplotlib.use('TkAgg')
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

# Colour maps to use for the genetic diagrams
# https://personal.sron.nl/~pault/
cmap = {}
cmap['vl_purple'] = (214/255.0, 193/255.0, 222/255.0)
cmap['l_purple']  = (177/255.0, 120/255.0, 166/255.0)
cmap['purple']    = (136/255.0,  46/255.0, 114/255.0)
cmap['blue']      = ( 25/255.0, 101/255.0, 176/255.0)
cmap['l_blue']    = ( 82/255.0, 137/255.0, 199/255.0)
cmap['vl_blue']   = (123/255.0, 175/255.0, 222/255.0)
cmap['green']     = ( 78/255.0, 178/255.0, 101/255.0)
cmap['l_green']   = (144/255.0, 201/255.0, 135/255.0)
cmap['vl_green']  = (202/255.0, 224/255.0, 171/255.0)
cmap['yellow']    = (247/255.0, 238/255.0,  85/255.0)
cmap['vl_orange'] = (246/255.0, 193/255.0,  65/255.0)
cmap['l_orange']  = (241/255.0, 147/255.0,  45/255.0)
cmap['orange']    = (232/255.0,  96/255.0,  28/255.0)
cmap['red']       = (220/255.0,   5/255.0,  12/255.0)
cmap['grey']      = (119/255.0, 119/255.0, 119/255.0)
cmap['vl_grey']   = (230/255.0, 230/255.0, 230/255.0)

error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}



#Data for the experiment
q=pd.read_csv('VI.csv')

# Create the figure
fig = plt.figure(figsize=(1.8, 2.2))
gs = gridspec.GridSpec(1, 1)
ax2 = plt.subplot(gs[0])


X_mu=q.sort_values('Mean',ascending=True)['Mean']  
Y_mu=q.sort_values('Mean',ascending=True)['mu_ML']  
Z_mu=q.sort_values('Mean',ascending=True)['mu_MOM']  
X_sigma=q.sort_values('standard deviation',ascending=True)['standard deviation']   
Y_sigma=q.sort_values('standard deviation',ascending=True)['sigma_ML']   
Z_sigma=q.sort_values('standard deviation',ascending=True)['sigma_MOM'] 




ax2.scatter(X_sigma,Y_sigma,s=12,c='tab:brown',label='ML estimate')
ax2.scatter(X_sigma,Z_sigma,s=12,c='#4f7942',label='MOM estimate')
ax2.plot(X_sigma,X_sigma,c='tab:orange',label='Ground Truth',linewidth=1.5,linestyle='dashed')
ax2.set_yscale('log')
ax2.set_xscale('log')



ax2.legend()
ax2.set(xlabel=' Fluorescence standard deviation (a.u.)', ylabel='Estimated Fluorescence standard deviation (a.u.)')
ax2.legend(frameon=False,fontsize=10,markerscale=2)
sns.despine()

plt.show()
width=3.54
height=3.54
fig.set_size_inches(width, height)
fig.savefig('plot.pdf')

#Tom's remaining code
plt.subplots_adjust(hspace=.0 , wspace=.00, left=.15, right=.95, top=.95, bottom=.13)
fig.savefig('mu_ml_variance_gamma.pdf', transparent=True)
# plt.close('all')

#############################################################
