#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:02:37 2023

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

signal_to_noise_stats = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S12-13.global_signal_to_noise.npy',allow_pickle=True).tolist()

cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv')
cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv')


#%%

Colors_limits = [1940,1960,1980,2000,2020,2040,2060]
title_aux = list(map(chr, range(97, 123)))[:6]


#%%



for ex in ['dry_to_wet','wet_to_dry']:
    
    for ds in ['LENS','CMIP6']:
        vars()['global_mean_'+ds] = signal_to_noise_stats.get('global_mean_'+ex+'_'+ds)
        vars()['global_sd_'+ds] = signal_to_noise_stats.get('global_sd_'+ex+'_'+ds)
        vars()['land_mean_'+ds] = signal_to_noise_stats.get('land_mean_'+ex+'_'+ds)
        vars()['land_sd_'+ds] = signal_to_noise_stats.get('land_sd_'+ex+'_'+ds)
        year_range =  vars()['land_sd_'+ds].year
    #%
    
    
    p_num=-1
    fig = plt.figure(figsize = (17/2.54, 15.5/2.54)) # 宽、高
    for ds in ['LENS','CMIP6']:
        if ds == 'LENS':
            loc_x = 0
        else:
            loc_x = 0.47
        if ds =='LENS':
            ds_name='CESM-LENS'
        else:
            ds_name='CMIP6'
        ax = plt.axes([loc_x,0.7,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
        p_num+=1
        
    
        cycle_value, cycle_lon = add_cyclic_point(  signal_to_noise_stats.get('first_year_'+ex+'_'+ds)   , coord=  cycle_LON.iloc[0,:180].values   )
        
    
        cf=ax.contourf(cycle_LON,cycle_LAT,cycle_value,
                     transform=ccrs.PlateCarree(),
                     levels=Colors_limits,
                     extend='both',
                     #colors=col_sf_change
                     )
        
        ax.set_title(r"$\bf{("+ title_aux[p_num] +")}$"+ds_name,
                         pad=2, size=7, loc='left')
        ##基本属性
        ax.set_global()
        #ax.stock_img()
    
        ax.coastlines(linewidth=0.3)
        ax.outline_patch.set_visible(False)
    
        a=ax.get_position()
        pad=0.025
        height=0.015
        ax_f = fig.add_axes([  0.02+loc_x, a.ymin - pad,  (a.xmax - a.xmin)*0.9 , height ]) #长宽高
        cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
    
        cb.set_label( label = "Ocurrence",fontdict={'size':6.5},labelpad=1)
    
    
    
    
        #colorbar标签刻度位置
        cb.set_ticklabels(cb.boundaries[1:-1].astype(int) ,size=6)
        cb.outline.set_linewidth(0.5)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
        #%
        #####################################################################################################    
        ax = plt.axes([loc_x+0.03,0.45,0.4,0.25] )
        p_num+=1
        
        
        
        ax.plot( year_range[1:], vars()['global_mean_'+ds][1:])
        ax.fill_between( x=year_range[1:], y1=-vars()['global_sd_'+ds][1:],y2=vars()['global_sd_'+ds][1:],alpha=0.2)
        
        year_s=np.where(vars()['global_mean_'+ds]>vars()['global_sd_'+ds])[0][0]
        ax.scatter(year_range[year_s] , vars()['global_mean_'+ds][year_s] ,zorder=10)
        print(year_range[year_s])
        ax1= ax.twinx()
        ax1.fill_between( x=year_range[1:],y1=0,y2=signal_to_noise_stats.get('min_size_global_'+ex+'_'+ds),color='grey',alpha=0.2)
        
        xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
        xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
        
        
        ymajorLocator = MultipleLocator(50) #将y轴主刻度标签设置为0.5的倍数
        yminorLocator = MultipleLocator(25) #将此y轴次刻度标签设置为0.1的倍数
    
        xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
        ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
        
        
        #设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    
        
        #设置次刻度标签的位置,没有标签文本格式
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.yaxis.set_major_locator(MultipleLocator(10))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.yaxis.set_minor_locator(MultipleLocator(5))
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
        ax1.tick_params(axis='y', labelsize=6.5,pad=0.5 )
        
        ax.set_xlabel('Year',fontsize=6.5,labelpad=1)
        
        xticks = [1930,1970,2010,2050,2090]
            #y_ = np.arange(0,101,20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,fontsize=6)
        ax.margins(x=0)    
        
        ax.set_title(r"$\bf{("+ title_aux[p_num] +")}$"+ds_name+' global mean ',
                         pad=2, size=7, loc='left')
        if p_num in [1,2]:
            
            ax.set_ylabel('Occurence\n(times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)
        #%
        #####################################################################################################    
        ax = plt.axes([loc_x+0.03,0.14,0.4,0.25] )
        p_num+=1
        ax.plot(year_range[1:], vars()['land_mean_'+ds][1:])
        ax.fill_between( x=year_range[1:], y1=-vars()['land_sd_'+ds][1:],y2=vars()['land_sd_'+ds][1:],alpha=0.2)
        
        year_s=np.where(vars()['land_mean_'+ds]>vars()['land_sd_'+ds])[0][0]
        ax.scatter(year_range[year_s] , vars()['land_mean_'+ds][year_s] ,zorder=10)
        print(year_range[year_s])
        ax1= ax.twinx()
        ax1.fill_between( x=year_range[1:],y1=0,y2=signal_to_noise_stats.get('min_size_land_'+ex+'_'+ds),color='grey',alpha=0.2)
        
        xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
        xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
        
        
        ymajorLocator = MultipleLocator(50) #将y轴主刻度标签设置为0.5的倍数
        yminorLocator = MultipleLocator(25) #将此y轴次刻度标签设置为0.1的倍数
    
        xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
        ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
        
        
        #设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    
        
        #设置次刻度标签的位置,没有标签文本格式
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.yaxis.set_major_locator(MultipleLocator(10))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.yaxis.set_minor_locator(MultipleLocator(5))
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
        ax1.tick_params(axis='y', labelsize=6.5,pad=0.5 )
        
        ax.set_xlabel('Year',fontsize=6.5,labelpad=1)
        
        xticks = [1930,1970,2010,2050,2090]
            #y_ = np.arange(0,101,20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,fontsize=6)
        ax.margins(x=0)    
        if p_num in [1,2]:
            
            ax.set_ylabel('Occurence\n(times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)
        ax.set_title(r"$\bf{("+ title_aux[p_num] +")}$ "+ds_name+' land mean ',
                         pad=2, size=7, loc='left')
   #%%
#fig.savefig("/media/dai/suk_code/research/4.East_Asia/Again/code/34-1.plots/Fig.x.signal_to_noise_of_"+ex+".png",dpi=1500, bbox_inches='tight')
    if ex == 'dry_to_wet':
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S12.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S12.pdf",dpi=1500, bbox_inches='tight') 
    else:
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S13.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S13.pdf",dpi=1500, bbox_inches='tight') 
