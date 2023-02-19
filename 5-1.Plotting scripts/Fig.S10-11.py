#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:37:17 2023

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

global_land_mean = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/2_S9-11.global_and_land_mean_of_features.npy',allow_pickle=True).tolist()

cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv')
cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv')

global_map = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S10-11.CMIP6_change_map_of_features.npy',allow_pickle=True).tolist()


#%%
datasets_new=['CHIRPS','GPCC','REGEN_LongTermStns',
              'ERA5','MERRA2','JRA-55',
              ] #4 grond-base land only

color_climate= ['#FA9B58','#FECE7C','#FFF5AE','#FBFAE6','#B9E176','#96D268','#69BE63','#33A456','#108647']
color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']
colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']

title_aux = list(map(chr, range(97, 123)))[:6]


#%%

for ex in ['dry_to_wet','wet_to_dry']:
    fig = plt.figure(figsize = (17/2.54, 17/2.54)) # 宽、高

    p_num=0
    for feature in ['frequency','duration','intensity']:
        for type in ['mean change','distribution']:
        #for ex in ['wet_to_dry']:    
        
        #for ex in ['dry','wet']: 
            if feature == 'frequency':
                feature_title = 'frequency'
            elif feature == 'duration':
                feature_title = 'transition duration '
            else:
                feature_title = 'intensity'
                
            p_num=p_num+1
            
            if p_num in [1,2]:
                position_y=0.65
            elif p_num in [3,4]:
                position_y=0.35
            else :
                position_y=0.05
            
            if p_num in [1,3,5]:
                position_x=0
            else:
                position_x=0.45
            
            if p_num in [2,4,6]:
                ax = plt.axes([position_x+0.06,position_y+0.1,0.4,0.22] )
                
                
                #1.LENS global 模型平均
                a = global_land_mean.get(ex+'~'+feature+'~'+'global-CMIP6').mean('ensemble')
                ax.plot(a.year,a,label='CESM-LENS-global',linewidth=1)
                
                a1= global_land_mean.get(ex+'~'+feature+'~'+'global-CMIP6').quantile(dim='ensemble',q=0.05)
                a2= global_land_mean.get(ex+'~'+feature+'~'+'global-CMIP6').quantile(dim='ensemble',q=0.95)
                
                ax.fill_between(x=a1.year,y1=a1,y2=a2,alpha=.3,color=color_d_type[0])
            
                #1.LENS land 模型平均
                a = global_land_mean.get(ex+'~'+feature+'~'+'land-CMIP6').mean('ensemble')
                ax.plot(a.year,a,label='CESM-LENS-land',linewidth=1,linestyle='--',color=color_d_type[0])
                
                a1= global_land_mean.get(ex+'~'+feature+'~'+'land-CMIP6').quantile(dim='ensemble',q=0.05)
                a2= global_land_mean.get(ex+'~'+feature+'~'+'land-CMIP6').quantile(dim='ensemble',q=0.95)
                
                ax.fill_between(x=a1.year,y1=a1,y2=a2,alpha=.8,color=color_d_type[0],facecolor="none",linewidth=0.5,linestyle='--')
                

                
                ax.hlines(y=0,xmin=1920,xmax=2100,linestyle='--',linewidth=1,color='grey')
                ax.axvspan(xmin=1979, xmax=2019,alpha=0.1) 
                
                n=1
                for n in range(len(datasets_new)):
                    
                    
                    event_region_mean=global_land_mean.get(ex+'~'+feature+'~'+datasets_new[n])

                    ax.plot(event_region_mean.year,event_region_mean,label=datasets_new[n],linewidth=0.9)
                #plt.legend(loc = "upper left")
                
                
                xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
                xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
                
                if p_num == 2:
                    ymajorLocator = MultipleLocator(40) #将y轴主刻度标签设置为0.5的倍数
                    yminorLocator = MultipleLocator(10) #将此y轴次刻度标签设置为0.1的倍数

                elif p_num == 4 :
                    ymajorLocator = MultipleLocator(4) #将y轴主刻度标签设置为0.5的倍数
                    yminorLocator = MultipleLocator(2) #将此y轴次刻度标签设置为0.1的倍数
                elif p_num == 6 :
                    ymajorLocator = MultipleLocator(4) #将y轴主刻度标签设置为0.5的倍数
                    yminorLocator = MultipleLocator(2) #将此y轴次刻度标签设置为0.1的倍数
                
                
                
                xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
                ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
                
                
                #设置主刻度标签的位置,标签文本的格式
                ax.xaxis.set_major_locator(xmajorLocator)
                ax.xaxis.set_major_formatter(xmajorFormatter)
                ax.yaxis.set_major_locator(ymajorLocator)
                ax.yaxis.set_major_formatter(ymajorFormatter)
            
                
                #设置次刻度标签的位置,没有标签文本格式
                ax.xaxis.set_minor_locator(xminorLocator)
                ax.yaxis.set_minor_locator(yminorLocator)
                
                
                
                
                
                
                ax.tick_params(axis='x', labelsize=7,pad=0.5 )
                ax.tick_params(axis='y', labelsize=7,pad=0.5 )
            
                ax.set_ylabel('Relative change (%)',fontsize=7,labelpad=1)
                ax.set_xlabel('Year',fontsize=7,labelpad=1)
                
                xticks = [1930,1970,2010,2050,2090]
                    #y_ = np.arange(0,101,20)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks,fontsize=7)
                ax.margins(x=0)    
                
                ax.set_title(r"$\bf{("+ title_aux[p_num-1] +")}$ "+'global and land mean of '+feature_title  ,
                                 pad=2, size=8.5, loc='left')
                

            if p_num in [1,3,5]:
                
                
                if p_num == 1:
                    Colors_limits=[-120,-80,-60,-40,-20,0,40,80,120,160,248,280,320]

                elif p_num == 3 :
                    Colors_limits=[-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]
                elif p_num == 5 :
                    #Colors_limits=[-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28]
                    Colors_limits=[-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21]
                
                ax = plt.axes([position_x,position_y,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
                
                
                
                
                monsoon = shpreader.Reader(basic_dir+ 'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp').geometries()

               
                cf=ax.contourf(cycle_LON,cycle_LAT, global_map.get(feature+'_'+ex+'_trend'),
                             transform=ccrs.PlateCarree(),
                             levels=Colors_limits,
                             extend='both',colors=colorbar_change )
                
                
              
                ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                                   facecolor='none', edgecolor='purple', linewidth=0.35, zorder=10) # 添加季风区
                
                ax.contourf(cycle_LON,cycle_LAT, global_map.get(feature+'_'+ex+'_trend_sig'),
                        transform=ccrs.PlateCarree(),hatches=['//////', None],colors="none",width=0.001)
              
                ##基本属性
                ax.set_global()
                #ax.stock_img()
                ax.coastlines(linewidth=0.3)
                ax.outline_patch.set_visible(False)
                
                a=ax.get_position()
                pad=0.025
                height=0.015
                ax_f = fig.add_axes([ position_x +0.02, a.ymin - pad,  (a.xmax - a.xmin)*0.9 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                '''
                cb.set_label( label = "Ocurrence",fontdict={'size':6.5},labelpad=1)
                '''
                #colorbar标签刻度位置
                cb.set_ticklabels(Colors_limits,size=6)
                cb.outline.set_linewidth(0.5)
                #colorbar标签刻度值
                ax_f.tick_params(length=1,pad=0.2)
                
                
                
                ax.set_title(r"$\bf{("+ title_aux[p_num-1] +")}$" +' '+feature_title,
                                 pad=2, size=8.5, loc='left')
                
    
    plt.rcParams['hatch.linewidth'] = 0.3
        
    plt.subplots_adjust(wspace=0.12,hspace=0.15)        
       
    lines, labels = fig.axes[2].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0.8, 0.1),ncol=4,
               columnspacing=0.4, labelspacing=0.3,framealpha=0,fontsize=7)
    
    
    if ex == 'dry_to_wet':
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S10.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S10.pdf",dpi=1500, bbox_inches='tight') 
    else:
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S11.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S11.pdf",dpi=1500, bbox_inches='tight') 


#%%
