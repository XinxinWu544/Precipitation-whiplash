#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:39:56 2023

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
basic_dir = 'E:/research/4.East_Asia/Again/'
zonal_mean = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.zonal_mean_of_features.csv',index_col=(0))

#%% user-defined colorbars and variables

datasets_new=['ERA5','MERRA2','JRA-55',
              'CHIRPS',
              'GPCC','REGEN_LongTermStns'] #4 grond-base land only

color_climate= ['#FA9B58','#FECE7C','#FFF5AE','#FBFAE6','#B9E176','#96D268','#69BE63','#33A456','#108647']


col=("#468CBC","#62C4C4","#55BF7F","#97C97B","#EAEA6E","#EAAD5E",
     "#ED965A","#E56A55","#D8324A","#AA4B9F","#43619E","#496BAD")

month_col=("#5461A6","#4180B7","#53A3B0","#71C5A4","#96D4A6","#B9E3A6","#DCEFA0",
  "#FFFDC2","#FCECA4","#FCD589","#F8B871","#F4975F","#EF6F4D",
  "#DD514E","#C93451","#A60F47","#A13F7B","#6E5197")

title_aux = list(map(chr, range(97, 123)))[:8]


#%%

p_num=-1
fig = plt.figure(figsize = (17.5/2.54, 20/2.54)) # 宽、高
for feature in ['frequency','duration','intensity','occurtime']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        p_num=p_num+1
        subplot_title=title_aux[p_num]
        
        if ex =='dry_to_wet':
            position_x = 0
            ex_title = 'dry-to-wet'
        else :
            position_x = 0.5
            ex_title = 'wet-to-dry'
            
        if feature == 'frequency':
            position_y= 0.77
            Colors_limits = np.round(np.arange(0.15,0.55,0.05),2)
            feature_title = 'occurence frequency'
        elif feature == 'duration':
            position_y= 0.53
            Colors_limits = np.round(np.arange(16,24,1),1)
            feature_title = 'transition duration'
        elif feature == 'intensity':
            position_y= 0.29
            #Colors_limits = np.array([3.3,3.4,3.5,3.6,3.7,3.8,3.9,4])
            feature_title = 'intensity'
            Colors_limits = np.array([3.3,3.35,3.5,3.65,3.8,3.95,4.1,4.25])
        else:
            position_y= 0.05
            Colors_limits =np.array([30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350])-10
            feature_title = 'average timing'
        
        if feature != 'occurtime':
            
            cycle_current_event=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.current_'+feature+'_'+ex+'_map'+'.csv')
            cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv')
            cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv')
            
            ax = plt.axes([position_x,position_y,0.4,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            
            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_current_event,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,extend='both',colors=color_climate)
           
            ##formatting settings
            ax.set_global()
            #ax.stock_img()
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            ax.set_title( r"$\bf{("+subplot_title+")}$" + ' '+ex_title+' '+feature_title ,  #fontweight='bold',
                             pad=1, size=8.5, loc='left')
            
            if (feature == 'frequency') :
                region_plot = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/regions_shp/'+'northeastern-china'+'.shp').geometries()
                ax.add_geometries(region_plot, ccrs.PlateCarree(), facecolor='none', edgecolor='m', linewidth=0.8, zorder=1) # 添加中国边界
                
            
            #colorbar
            a=ax.get_position()
            pad=0.02
            height=0.01
            ax_f = fig.add_axes([ position_x +0.02, a.ymin - pad,  (a.xmax - a.xmin)*0.9 , height ]) #长宽高
            cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
       
            cb.set_ticklabels(Colors_limits,size=7.5)
            cb.outline.set_linewidth(0.7)

            ax_f.tick_params(length=1,pad=0.2)
            
            
            #ax1 = plt.axes([0.55,0.6,0.15,0.3])
            a=ax.get_position()
            pad=0.006
            width=0.015
            ax1 = fig.add_axes([a.xmax + pad, a.ymin, 0.06, (a.ymax-a.ymin) ])
            

            ax1.plot(zonal_mean[feature+'_'+ex+'_LENS'],zonal_mean.index,label='CESM-LENS',linewidth=0.7)
            
            ax1.fill_betweenx(x1=zonal_mean[feature+'_'+ex+'_LENS_quan_0.05'],
                              x2=zonal_mean[feature+'_'+ex+'_LENS_quan_0.95'],
                              y=zonal_mean.index,alpha=0.25)

            ax1.plot(zonal_mean[feature+'_'+ex+'_CMIP6'],zonal_mean.index,label='CMIP6',linewidth=0.7)
            
            ax1.fill_betweenx(x1=zonal_mean[feature+'_'+ex+'_CMIP6_quan_0.05'],
                              x2=zonal_mean[feature+'_'+ex+'_CMIP6_quan_0.95'],
                              y=zonal_mean.index,alpha=0.25)
            

            for n in range(len(datasets_new)):
                
                ax1.plot(zonal_mean[feature+'_'+ex+'_'+datasets_new[n]],zonal_mean.index,label=datasets_new[n],linewidth=0.7)
            
            if (feature == 'intensity') :
                ax1.set_xticks([3,4.5])
            if (feature == 'duration') :
                ax1.set_xticks([15,25])
            if (feature == 'frequency') :
                ax1.set_xticks([0.2,0.5])
            
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right") 
            ax1.set_yticks([-80,-40,0,40,80])
            #ax1.set_yticklabels( ['80°S','40°S','0°','40°N','80°N'] ,size=5,rotation=-40)
            ax1.set_yticks([])
            ax1.tick_params(axis='x', labelsize=6,pad=0.5 ,length=1.5)
            ax1.tick_params(axis='y', labelsize=6,pad=0.3,length=1.5 )
            ax1.margins(y=0)
            lines, labels = fig.axes[-1].get_legend_handles_labels()
            print(labels)
            #%
        else:
            current_events_time=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.current_time_'+ex+'_map'+'.csv')
            
           
            ax = plt.axes([position_x,position_y,0.4,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            
            cf=ax.contourf(cycle_LON,cycle_LAT,current_events_time,
                         transform=ccrs.PlateCarree(),colors=month_col,
                         levels=Colors_limits,extend='both')
        
            ##formatting settings
            ax.set_global()
            #ax.stock_img()
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)

            #({})  {} {}'.format(subplot_title, ex, feature)
            ax.set_title( r"$\bf{("+subplot_title+")}$" + ' '+ex_title+' '+feature_title ,  #fontweight='bold',
                             pad=1, size=8.5, loc='left')
     
        ax = plt.axes([0.385,0.138,0.13,0.13] )
        #ax.pie(np.repeat(20,18),colors=month_col, counterclock=False,startangle=90, wedgeprops=dict(width=0.3))
        
        ax.pie(np.append(np.repeat(20,17),25),colors=month_col, counterclock=False,startangle=90, wedgeprops=dict(width=0.3))

        #month = [16,75,136,197,258,319]
        month = [0,59,120,181,243,304]
        month_name=['Jan','Mar','May','Jul','Sept','Nov']
        for i in range(len(month)):
            
            
            ang = 90-(month[i]/365)*360
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            print(x)
            print(y)
            if ang< -90:
                ang1=ang+180
            else:
                ang1=ang
            if (i ==0)|(i==3) :
                
                ax.annotate(month_name[i], xy=(1, 1), xytext=(0.46+x*0.18,0.44+y*0.18) ,xycoords='axes fraction',rotation=ang1,fontsize=5.5)
            else:
                ax.annotate(month_name[i], xy=(1, 1), xytext=(0.4+x*0.18,0.44+y*0.18) ,xycoords='axes fraction',rotation=ang1,fontsize=5.5)
           
fig.legend(lines, labels, bbox_to_anchor=(0.8, 0.15),ncol=4, framealpha=0,fontsize=7)


fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.1.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.1.pdf",dpi=1500, bbox_inches='tight')    