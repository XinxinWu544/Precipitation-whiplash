#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:42:47 2023

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
circulation = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/6_S20.circulation.npy',allow_pickle=True).tolist()
#%%
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

interval=[(1,5),(1,10),(1,15),(1,20),(1,25),(1,30)]

k=5
region='northeastern-china'

lon1=120 - 30
lon2=130 + 30
lat1=40 - 25
lat2=50 + 25    

period = ['current','future']
states = ['before','transition','after']

ct_breaks= np.linspace(-21,21,15)

d='dmean'
title_aux = list(map(chr, range(97, 123)))[:8]

#%%
for whiplash_type in ['dry_to_wet','wet_to_dry']:
    
    fig = plt.figure(figsize = (17/2.54, 10/2.54))
    p_num = -1
    for p in [0,1]:
        for i in [0,1,2]:
        
    
            p_num += 1
            
            region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+region+'.shp').geometries()
            if p_num in [0,1,2]:
                loc_y = 0.5
            else :
                loc_y = 0.1
                
            if p_num in [0,3]:
                loc_x = 0.0
            elif p_num in [1,4]:
                loc_x = 0.3
            else :
                loc_x = 0.6
                
            ax = fig.add_axes([loc_x, loc_y, 0.3, 0.32], 
                              projection = ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40)))
            
            ax.set_aspect('auto') 
            if i in [0,1]:
                Z=circulation.get('before'+'_'+'Z_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                UQ=circulation.get('before'+'_'+'UQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                VQ=circulation.get('before'+'_'+'VQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
            else:
                Z=circulation.get('after'+'_'+'Z_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                UQ=circulation.get('after'+'_'+'UQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                VQ=circulation.get('after'+'_'+'VQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
            Z = Z.where(np.isnan(VQ.values)==False)
            cf = ax.contourf(Z.lon, Z.lat, Z.values,# norm=diversity_norm,
                              cmap = 'RdBu_r', transform = ccrs.PlateCarree(),levels = ct_breaks, 
                              extend = "both")
    
            h=ax.quiver(  Z.lon,Z.lat,
                         UQ.values,VQ.values,
                         transform = ccrs.PlateCarree(),headwidth=4,
                         color='black',zorder=100,scale_units = 'inches',
                         angles = 'uv', regrid_shape = 17,scale=600)
            
            if p_num == 0:
                qk = ax.quiverkey(h, 0.93, 0.77, 100, r'100 $kg·m^{-1} s^{-1}$', labelpos='S',
                      coordinates='figure',zorder=200,
                      fontproperties={'size':7.5})
            
            
                
        
            ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
            bounds = [lon1, lon2, lat1, lat2]
            ax.set_extent(bounds)
            ax.axis('off')
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.5)
            #plt.show()
        
        
            ax.set_title(r"$\bf{("+title_aux[p_num]+")}$ ",
                             pad=4, size=8.5, loc='left')
            '''
            if p_num == 3:
                ax.set_title(r"$\bf{(b)}$ 2060-2099",
                                 pad=4, size=8.5, loc='left')
            '''
            if p_num == 0:
                ax.annotate('1979-2019', xy=(0,0.33), xycoords="axes fraction",size=8,rotation=90)
            if p_num == 3:
                ax.annotate('2060-2099', xy=(0,0.33), xycoords="axes fraction",size=8,rotation=90)
                 
                
            if p_num == 0:
                ax.annotate('30 days before transition', xy=(0.21,1.04), xycoords="axes fraction",size=8)
            if p_num == 1:
                ax.annotate('days of transition start', xy=(0.21,1.04), xycoords="axes fraction",size=8)
            if p_num == 2:
                ax.annotate('days of transition end', xy=(0.21,1.04), xycoords="axes fraction",size=8)
            '''        
            if whiplash_type=='wet_to_dry':    
                if p_num == 0:
                    ax.annotate('wet', xy=(0.5,1), xycoords="axes fraction",size=7)
                if p_num == 1:
                    ax.annotate('transition', xy=(0.4,1), xycoords="axes fraction",size=7)
                if p_num == 2:
                    ax.annotate('dry', xy=(0.5,1), xycoords="axes fraction",size=7)
            '''
        
        #figu.add_panel_label(ax, 'b', x = 0.1, y = 1.3)
        '''
        ax.text(x = 0.6, y = 1.2, s='before', transform=ax.transAxes,
          fontsize=8, fontweight='bold', va='top', ha='right')
        '''
        
    cbaxes = fig.add_axes([0.91, 0.2, 0.015, 0.5]) 
    cbar = fig.colorbar(cf, cax = cbaxes, shrink = 2, )
    
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(label = "500-hPa GPH anomaly",fontsize=7)        
 #%%   
    if whiplash_type == 'dry_to_wet':
        
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.6.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.6.pdf",dpi=1500, bbox_inches='tight')
    else:
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S21.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S21.pdf",dpi=1500, bbox_inches='tight')

#fig.savefig(basic_loc+"/research/4.East_Asia/Again/code_new/40-1.plots/Fig._"+whiplash_type+".png",dpi=1500, bbox_inches='tight')



#%%
