#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:28:24 2023

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
circulation_from_to = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S21_22.circulation_from_to.npy',allow_pickle=True).tolist()
#%%

region='northeastern-china'

lon1=120 - 30
lon2=130 + 30
lat1=40 - 25
lat2=50 + 25    

whiplash_type='dry_to_wet'
whiplash_type='wet_to_dry'
interval=[(1,5),(1,10),(1,15),(1,20),(1,25),(1,30)]

period = ['current','future']
states = ['before','transition','after']

i=1
p=0

ct_breaks= np.linspace(-28,28,15)
#%%

for whiplash_type in ['dry_to_wet','wet_to_dry']:
    #%%
    
    fig = plt.figure(figsize = (18/2.54, 10/2.54))
    for p in [0,1]:
        if p == 0:
            loc_y=0.5
        else:
            loc_y=0
        
        for i in [0,2]:
    
#%
            
            
            if i == 0 :
            
                p_num = -1
                for k in [5,4,3,2,1,0]:
                    
                    
                    Z=circulation_from_to.get(str(k)+'_'+'Z_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    UQ=circulation_from_to.get(str(k)+'_'+'UQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    VQ=circulation_from_to.get(str(k)+'_'+'VQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    Z = Z.where(np.isnan(VQ.values)==False)
                    p_num += 1
                    region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+region+'.shp').geometries()
                    loc_x = p_num*0.14

                        
                    ax = fig.add_axes([loc_x, 0.25+loc_y, 0.14, 0.185], 
                                      projection = ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40)))
                    
                    ax.set_aspect('auto') ##设置不固定长宽
                    cf = ax.contourf(Z.lon, Z.lat, Z.values,# norm=diversity_norm,
                                      cmap = 'RdBu_r', transform = ccrs.PlateCarree(),levels = ct_breaks, 
                                      extend = "both")
                
                    
                    h=ax.quiver(  Z.lon,Z.lat,
                                 UQ.values,VQ.values,
                                 transform = ccrs.PlateCarree(),headwidth=6,linewidth=3,headlength=3,
                                 color='black',zorder=100,scale_units = 'inches',angles = 'uv', regrid_shape = 15,scale=600)
                    
                    if p_num == 0:
                        qk = ax.quiverkey(h, 1, 0.77, 50, r'50 $kg·m^{-1} s^{-1}$', labelpos='S',
                              coordinates='figure',zorder=200,
                              fontproperties={'size':7.5})
                
                    ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
                    bounds = [lon1, lon2, lat1-3, lat2]
                    ax.set_extent(bounds)
                    ax.axis('off')
                    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.3)
                    ax.annotate( str(interval[k][1])+' days', xy=(.4,1), xycoords="axes fraction",size=6.5)
                    
                    
                 
                 
                 
                ################################找到转换日#######################################################################
                Z=circulation.get('after'+'_'+'Z_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                UQ=circulation.get('after'+'_'+'UQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                VQ=circulation.get('after'+'_'+'VQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                Z = Z.where(np.isnan(VQ.values)==False)
            
                ##出图
                p_num += 1
                region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+region+'.shp').geometries()
                
                loc_x = p_num*0.14
                
                    
                ax = fig.add_axes([loc_x, 0.25+loc_y, 0.14, 0.185], 
                                  projection = ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40)))
                
                ax.set_aspect('auto') ##设置不固定长宽
                cf = ax.contourf(Z.lon, Z.lat, Z.values,# norm=diversity_norm,
                                  cmap = 'RdBu_r', transform = ccrs.PlateCarree(),levels = ct_breaks, 
                                  extend = "both")
                
                
                h=ax.quiver(  Z.lon,Z.lat,
                             UQ.values,VQ.values,
                             transform = ccrs.PlateCarree(),headwidth=6,linewidth=3,headlength=3,
                             color='black',zorder=100,scale_units = 'inches',angles = 'uv', regrid_shape = 15)
                
                
                
                ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
                bounds = [lon1, lon2, lat1-3, lat2]
                ax.set_extent(bounds)
                ax.axis('off')
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.3)
                ax.annotate('transition day', xy=(0.25,1), xycoords="axes fraction",size=6.5)
                    
                
  ####################################################################################################################3              
                #plt.show()
            if i == 2 :
            
                p_num = -1
                
                ####################################先放转换日
                Z=circulation.get('after'+'_'+'Z_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                UQ=circulation.get('after'+'_'+'UQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                VQ=circulation.get('after'+'_'+'VQ_'+states[1]+'_'+period[p]+'_'+whiplash_type)
                Z = Z.where(np.isnan(VQ.values)==False)
                ##出图
                p_num += 1
                region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+region+'.shp').geometries()
                
                loc_x = p_num*0.14
                
                    
                ax = fig.add_axes([loc_x, loc_y, 0.14, 0.185], 
                                  projection = ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40)))
                
                ax.set_aspect('auto') ##设置不固定长宽
                cf = ax.contourf(Z.lon, Z.lat, Z.values,# norm=diversity_norm,
                                  cmap = 'RdBu_r', transform = ccrs.PlateCarree(),levels = ct_breaks, 
                                  extend = "both")
                
                
                h=ax.quiver(  Z.lon,Z.lat,
                             UQ.values,VQ.values,
                             transform = ccrs.PlateCarree(),headwidth=6,linewidth=3,headlength=3,
                             color='black',zorder=100,scale_units = 'inches',angles = 'uv', regrid_shape = 15)
                
                
                
                ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
                bounds = [lon1, lon2, lat1-3, lat2]
                ax.set_extent(bounds)
                ax.axis('off')
                ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.3)
                ax.annotate('transition day', xy=(0.25,1), xycoords="axes fraction",size=6.5)
                
                ##############################################再放顺序##################################################
                print('!!!!!!!!!')
                
                
                for k in [0,1,2,3,4,5]:
                
                    
                    
                    Z=circulation_from_to.get(str(k)+'_'+'Z_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    UQ=circulation_from_to.get(str(k)+'_'+'UQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    VQ=circulation_from_to.get(str(k)+'_'+'VQ_'+states[i]+'_'+period[p]+'_'+whiplash_type)
                    Z = Z.where(np.isnan(VQ.values)==False)
                    ##出图
                    p_num += 1
                    region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+region+'.shp').geometries()
                    
                    loc_x = p_num*0.14
                    
                   
                   
                        
                    ax = fig.add_axes([loc_x, loc_y, 0.14, 0.185], 
                                      projection = ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40)))
                    
                    ax.set_aspect('auto') ##设置不固定长宽
                    cf = ax.contourf(Z.lon, Z.lat, Z.values,# norm=diversity_norm,
                                      cmap = 'RdBu_r', transform = ccrs.PlateCarree(),levels = ct_breaks, 
                                      extend = "both")
                    #
                    '''
                    ct = ax.contour(Z.lon, Z.lat, Z.values,
                                    transform = ccrs.PlateCarree(), levels = ct_breaks, colors = "#808080", linewidths = 0.6)
                    '''
                    
                    
                    
                    h=ax.quiver(  Z.lon,Z.lat,
                                 UQ.values,VQ.values,
                                 transform = ccrs.PlateCarree(),headwidth=6,linewidth=3,headlength=3,
                                 color='black',zorder=100,scale_units = 'inches',angles = 'uv', regrid_shape = 15)
                    
                    
                
                    ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
                    bounds = [lon1, lon2, lat1-3, lat2]
                    ax.set_extent(bounds)
                    ax.axis('off')
                    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth = 0.3)
                    ax.annotate( str(interval[k][1])+' days', xy=(.4,1), xycoords="axes fraction",size=6.5)
    ax.annotate( r"$\bf{(a)}$ 1979-2019", xy=(0,1), xycoords="figure fraction",size=8.5)    
    ax.annotate( r"$\bf{(b)}$ 2060-2099", xy=(0,0.495), xycoords="figure fraction",size=8.5) 
    cbaxes = fig.add_axes([0.985, 0.2, 0.01, 0.5]) 
    cbar = fig.colorbar(cf, cax = cbaxes, shrink = 2, )
    
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(label = "500-hPa GPH anomaly",fontsize=7)  
    if whiplash_type == 'dry_to_wet':
        
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S20.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S20.pdf",dpi=1500, bbox_inches='tight')
    else:
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S22.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S22.pdf",dpi=1500, bbox_inches='tight')

          