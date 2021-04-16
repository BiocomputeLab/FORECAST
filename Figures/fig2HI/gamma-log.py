
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


d_var = {'MAPE': [56.51961891, 50.47877742, 46.13735704,
       56.41471139, 48.30979619, 39.03006257,
       56.08137685, 44.53477141, 27.01354216]+[287.74453306, 298.1863082 , 298.21313797,299.7961364 , 300.44014621, 311.36703739,
       324.08161946, 323.83104867, 327.57942772]+[67.89211699, 64.24130949, 63.92732816]+[60.43748406, 50.92945822, 46.84127056]+[54.94239969, 39.2380389 , 24.5262507 ]+[195.21194215, 232.21351093, 238.5230456 ]+[219.98637949, 221.72468045, 217.98143615]+[226.76576441, 196.59937264, 221.02871965], 'distribution': ['Gamma']*18+['Lognormal']*18,'inference':['ML']*9+['MOM']*9+['ML']*9+['MOM']*9}
df_var = pd.DataFrame(data=d_var)



# Create the figure
fig = plt.figure(figsize=(11.7,8.3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
my_pal = {"ML": "#2463A3", "MOM": "#B5520E"}
ax=sns.violinplot(x="distribution", y="MAPE", hue="inference",
                 data=df_var, palette=my_pal)


ax.set_ylabel('MAPE (standard deviation) %')    
ax.set_xlabel('')
# my_pal = ['#2463A3', '#B5520E','#2463A3', '#B5520E']
# INF=['ML','MOM','ML','MOM']
# color_dict = dict(zip(INF, my_pal ))

# for i in range(0,4):
#     mybox = ax.artists[i]
#     mybox.set_facecolor(color_dict[INF[i]])

ax.get_legend().remove()

sns.despine()

width=3.54
height=3.54
fig.set_size_inches(width, height)

plt.subplots_adjust(hspace=.0 , wspace=.00, left=.15, right=.95, top=.95, bottom=.13)
plt.show()

fig.savefig('gamma-log.pdf', transparent=True)
plt.close('all')

#############################################################
