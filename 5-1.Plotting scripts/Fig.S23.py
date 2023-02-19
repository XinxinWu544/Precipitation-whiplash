#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:52:46 2023

@author: dai
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy
from scipy import signal  
from scipy import stats
import os
from datetime import datetime
import pandas as pd
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import tqdm
import multiprocessing
import gc
import glob
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib


#%% Source data
#basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'

ensemble_std = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S23.ensemble_std_compare.npy',allow_pickle=True).tolist() 
cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv').values
cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv').values  
#%%
title_aux = list(map(chr, range(97, 123)))[:10]
Colors_limits = np.arange(0.002,0.02,.002)

col=["#FFFFE4","#FEF4CF","#FEEDB0", "#FBCF93", "#F7B07A", "#F19164" ,"#E97356" ,"#C96775","#87518E"] 
#%%
p_num=-1
fig = plt.figure(figsize = (14/2.54, 15/2.54)) # 宽、高

for period in ['current','future']:
    for ex in ['dry_to_wet','wet_to_dry']:
        for ds in ['lens','cmip6']:
            if ds == 'lens':
                loc_x = 0
            else:
                loc_x = 0.5
            if ex =='dry_to_wet':
                e_loc = 0.25
                ex_name= 'dry-to-wet'
            else:
                e_loc =0
                ex_name= 'wet-to-dry'
            if period == 'current':
                loc_y = 0.5
            else:
                loc_y = 0
            if ds =='lens':
                ds_name='CESM-LENS'
            else:
                ds_name='CMIP6'
                
                
            ax = plt.axes([loc_x,loc_y+e_loc,0.47,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            p_num+=1
            
        
            cycle_value, cycle_lon = add_cyclic_point( ensemble_std.get(period+'_sd_'+ds+'_'+ex), coord=ensemble_std.get(period+'_sd_'+ds+'_'+ex).lon)
            
            
            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_value,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,
                         extend='both',
                         colors=col
                         )
            
            
            if ds == 'cmip6':
                
                ax.contourf( ensemble_std.get(period+'_sd_'+ds+'_'+ex).lon,
                            ensemble_std.get(period+'_sd_'+ds+'_'+ex).lat,
                            ensemble_std.get('sig_'+period+'_'+ex)  ,
                            transform=ccrs.PlateCarree(),hatches=[ '......',None],colors="none",width=0.001,zorder=10)
                '''  
                ax.plot(x,y,#sig_future,
                                        transform=ccrs.PlateCarree(),
                                        #s=0.1,
                                        markerfacecolor ='black',
                                        color='black',marker='.',
                                        markersize=0.1,linewidth=0,zorder=10  )         
            ax.set_title(r"$\bf{("+ title_aux[p_num] +")}$"+ds_name,
                             pad=2, size=7, loc='left')
            ##基本属性
            ax.set_global()
            #ax.stock_img()
                '''
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            ax.set_title(r"$\bf{("+title_aux[p_num] +")}$" +' '+ex_name+' ('+period+'; '+ds_name+')',
                             pad=2, size=7, loc='left')
            
            
            a=ax.get_position()
            pad=0.025
            height=0.015
            
            if p_num ==3:
                ax_f = fig.add_axes([  0.02+a.xmax, a.ymin + pad, height, (a.ymax - a.ymin)*1.8 ,  ]) #长宽高
                cb=fig.colorbar(cf, orientation='vertical',cax=ax_f)
                cb.set_label( label = "Standard deviation in trends of frequency \n(x"+u'$10^{-3}$'+ " times/41 yrs)",fontdict={'size':6.5},labelpad=1)
                cb.set_ticklabels(np.arange(2,20,2),size=6)
                ax_f.tick_params(length=1,pad=0.2)
            
            if p_num ==7:
                ax_f = fig.add_axes([  0.02+a.xmax, a.ymin + pad, height, (a.ymax - a.ymin)*1.8 ,  ]) #长宽高
                cb=fig.colorbar(cf, orientation='vertical',cax=ax_f)
                cb.set_label( label = "Standard deviation in trends of frequency  \n(x"+u'$10^{-3}$'+ " times/40 yrs)",fontdict={'size':6.5},labelpad=1)
                cb.set_ticklabels(np.arange(2,20,2),size=6)
                ax_f.tick_params(length=1,pad=0.2)
                
                
                
                
            '''
            ax_f = fig.add_axes([  0.02+loc_x, a.ymin - pad,  (a.xmax - a.xmin)*0.9 , height ]) #长宽高
            cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
            cb.set_label( label = "Standard deviation in frequency trends  \n(x"+u'$10^{-3}$'+ " times/41 yrs)",fontdict={'size':6.5},labelpad=1)
            cb.set_ticklabels(np.arange(2,20,2),size=6)
            ax_f.tick_params(length=1,pad=0.2)
            '''
plt.rcParams['hatch.linewidth'] = 0.5
    
#fig.savefig("E:/research/4.East_Asia/Again/code/34-1.plots/Fig.x.std_CMIP6_vs_CESM_2.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S23.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S23.pdf",dpi=1500, bbox_inches='tight')