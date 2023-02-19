#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:52:10 2023

@author: dai
"""



import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy
from scipy import signal  
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
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib

#%% Source data
#basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'


#%%
color_climate= ['#FA9B58','#FECE7C','#FFF5AE','#FBFAE6','#B9E176','#96D268','#69BE63','#33A456','#108647']

ex='dry_to_wet'
feature='frequency'

title_aux = list(map(chr, range(97, 123)))[:17]


datasets_new=['ERA5','MERRA2','JRA-55','CHIRPS', 
              'GPCC','REGEN_LongTermStns'] #4 grond-base land only

locs=((0.8,0.2),(0.8,0.4),(0.8,0.6),(0.8,0.8),
(0.6,0.2),(0.6,0.4),(0.6,0.6),(0.6,0.8),
(0.4,0.2),(0.4,0.4),(0.4,0.6),(0.4,0.8),
(0.2,0.2),(0.2,0.4),(0.2,0.6),(0.2,0.8))
#%%

global_mean_map = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S6-7.global_mean_map_of_datasets.npy',allow_pickle=True).tolist()

cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv')
cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv')


for feature in ['frequency']:
    for ex in ['dry_to_wet','wet_to_dry']:
        #%
        Colors_limits = np.round(np.arange(0.12,0.5,0.07),2)
        p_num=-1
        fig = plt.figure(figsize = (17/2.54, 10/2.54)) # 宽、高
        #######################################   LENS   ###################################
        
        
        p_num=p_num+1
        subplot_title=title_aux[p_num]
        
        
        ax = plt.axes([locs[p_num][1],locs[p_num][0],0.2,0.2] ,projection=ccrs.Robinson(central_longitude=150))
        
        cycle_current_event = global_mean_map.get('map_'+ex+'_LENS')
        cf=ax.contourf(cycle_LON,cycle_LAT,cycle_current_event,
                     transform=ccrs.PlateCarree(),
                     levels=Colors_limits,extend='both',colors=color_climate)
        
        ##基本属性
        ax.set_global()
        #ax.stock_img()
        ax.coastlines(linewidth=0.3)
        ax.outline_patch.set_visible(False)
        
        #({})  {} {}'.format(subplot_title, ex, feature)
        ax.set_title( r"$\bf{("+subplot_title+")}$ " + 'CESM-LENS' ,  #fontweight='bold',
                         pad=1, size=6.5, loc='left')
        
        
        #######################################  CMIP6   ###################################
        
        
        p_num=p_num+1
        subplot_title=title_aux[p_num]
        
        
        ax = plt.axes([locs[p_num][1],locs[p_num][0],0.2,0.2] ,projection=ccrs.Robinson(central_longitude=150))
        
        cycle_current_event = global_mean_map.get('map_'+ex+'_CMIP6')
        cf=ax.contourf(cycle_LON,cycle_LAT,cycle_current_event,
                     transform=ccrs.PlateCarree(),
                     levels=Colors_limits,extend='both',colors=color_climate)
        
        ##基本属性
        ax.set_global()
        #ax.stock_img()
        ax.coastlines(linewidth=0.3)
        ax.outline_patch.set_visible(False)
        
        #({})  {} {}'.format(subplot_title, ex, feature)
        ax.set_title( r"$\bf{("+subplot_title+")}$ " + 'CMIP6' ,  #fontweight='bold',
                         pad=1, size=6.5, loc='left')
        
        
        #######################################  DATASETS   ###################################
        
        
        for n in range(len(datasets_new)):
            
            
           
            #event=event.interp(lat=np.arange(90,-90,-4),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
            #event=event.interp(lon=np.arange(0,360,4),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
            
                
            p_num=p_num+1
            subplot_title=title_aux[p_num]
            
            
            ax = plt.axes([locs[p_num][1],locs[p_num][0],0.2,0.2] ,projection=ccrs.Robinson(central_longitude=150))
            
            cycle_current_event = global_mean_map.get('map_'+ex+'_'+datasets_new[n])
            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_current_event,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,extend='both',colors=color_climate)
            ##基本属性
            ax.set_global()
            #ax.stock_img()
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            #({})  {} {}'.format(subplot_title, ex, feature)
            ax.set_title( r"$\bf{("+subplot_title+")}$ " + datasets_new[n] ,  #fontweight='bold',
                             pad=1, size=6.5, loc='left')
            
            
        #colorbar位置
        a=ax.get_position()
        pad=0.04
        height=0.022
        ax_f = fig.add_axes([ 0.4 +0.02, a.ymin - pad,  (a.xmax - a.xmin)*2 , height ]) #长宽高
        cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
        '''
        cb.set_label( label = "Ocurrence",fontdict={'size':6.5},labelpad=1)
        '''
        #colorbar标签刻度位置
        cb.set_ticklabels(Colors_limits,size=6)
        cb.outline.set_linewidth(0.7)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
                
        #fig.savefig("/media/dai/suk_code/research/4.East_Asia/Again/code_new/32-2.plots/Fig.Sx.datasets_Climatology_"+ex+".png",dpi=1500, bbox_inches='tight')  
        if ex == 'dry_to_wet':
            fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S6.png",dpi=1500, bbox_inches='tight')
            fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S6.pdf",dpi=1500, bbox_inches='tight') 
        else:
            fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S7.png",dpi=1500, bbox_inches='tight')
            fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S7.pdf",dpi=1500, bbox_inches='tight') 
    