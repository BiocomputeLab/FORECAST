
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

#############################################################
# PLOT THE DATA
#############################################################

# Data to use (update if necessary)

# af_avg = 148
# af_sd = 35.6
# # Data for the designs [[OFF,OFF,OFF], [ON,ON,ON]]
# Ptac_points = [[154.4545455, 142.4545455, 164.4545455],[1203.454545, 1361.454545, 1479.454545]]
# THS_points = [[147.4545455, 173.4545455, 183.4545455],[79484.45455, 73052.95455, 76770.45455]]
# STAR_points = [[215.4545455, 240.4545455, 256.4545455],[5928.454545, 6760.454545, 6321.454545]]
# DC_points = [[147.4545455, 147.4545455, 168.4545455],[14923.45455, 16847.45455, 15529.45455]]
# # Average values for [OFF, ON]
# Ptac_avg = [154,1348]
# THS_avg = [167,76437]
# STAR_avg = [238,6337]
# DC_avg = [155,15767]
# # SD values for [OFF, ON]
# Ptac_sd = [11.0,138.5]
# THS_sd = [18.6,3229]
# STAR_sd = [20.7,416.2]
# DC_sd = [12.1,984]

# print('*** DYNAMIC RANGE ***')
# print('Ptac: ', np.mean(Ptac_points[1])-np.mean(Ptac_points[0]), np.std(np.array(Ptac_points[1])-np.array(Ptac_points[0])) )
# print('THS: ', np.mean(THS_points[1])-np.mean(THS_points[0]), np.std(np.array(THS_points[1])-np.array(THS_points[0])) )
# print('STAR: ', np.mean(STAR_points[1])-np.mean(STAR_points[0]), np.std(np.array(STAR_points[1])-np.array(STAR_points[0])) )
# print('DC: ', np.mean(DC_points[1])-np.mean(DC_points[0]), np.std(np.array(DC_points[1])-np.array(DC_points[0])) )

# print('*** FOLD-CHANGE ***')

# print('Ptac: ', np.mean(np.array(Ptac_points[1])-af_avg) / np.mean((np.array(Ptac_points[0])-af_avg)))
# print('THS: ', np.mean(np.array(THS_points[1])-af_avg) / np.mean((np.array(THS_points[0])-af_avg)))
# print('STAR: ', np.mean(np.array(STAR_points[1])-af_avg) / np.mean((np.array(STAR_points[0])-af_avg)))
# print('DC: ', np.mean(np.array(DC_points[1])-af_avg) / np.mean((np.array(DC_points[0])-af_avg)))


d_mean = {'MAPE': [100*i for i in ([0.1674891 , 0.14371818, 0.12273398, 
       0.16679492, 0.13970324, 0.1015513 ,
       0.16319497, 0.12743953, 0.06931147]+[0.51141972, 0.51385324, 0.51403695,
       0.52769436, 0.51004928, 0.51341036, 
       0.53446   , 0.52250617, 0.5075517  ])]+[15.29211367, 14.14405139, 14.05101411]+[12.61702118, 10.50428435,  9.82247402]+[10.31754068,  7.2084087 ,  4.77361639]+[16.35151345, 16.9359747 , 17.78217523]+[14.38362791, 14.93895699, 15.7100954 ]+[13.14528142, 13.4672431 , 14.25780018], 'distribution': ['Gamma']*18+['Lognormal']*18,'inference':['ML']*9+['MOM']*9+['ML']*9+['MOM']*9}
df_mean = pd.DataFrame(data=d_mean)
df_mean.head()


# Create the figure
fig = plt.figure(figsize=(11.7,8.3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
my_pal = {"ML": "#2463A3", "MOM": "#B5520E"}
ax=sns.violinplot(x="distribution", y="MAPE", hue="inference",
                 data=df_mean, palette=my_pal)


ax.set_ylabel('MAPE (mean) %')    
ax.set_xlabel('')
# my_pal = ['#2463A3', '#B5520E','#2463A3', '#B5520E']
# INF=['ML','MOM','ML','MOM']
# color_dict = dict(zip(INF, my_pal ))

# for i in range(0,4):
#     mybox = ax.artists[i]
#     mybox.set_facecolor(color_dict[INF[i]])

#plt.legend(frameon=False,fontsize=12)
ax.get_legend().remove()

sns.despine()

width=3.54
height=3.54
fig.set_size_inches(width, height)

plt.subplots_adjust(hspace=.0 , wspace=.00, left=.15, right=.95, top=.95, bottom=.13)
plt.show()

fig.savefig('gamma-log_mu.pdf', transparent=True)
plt.close('all')

#############################################################
