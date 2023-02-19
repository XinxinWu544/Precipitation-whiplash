#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:16:38 2023

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

datasets_new=['ERA5','MERRA2','JRA-55',
              'CHIRPS',
              'GPCC','REGEN_LongTermStns'] #4 grond-base land only

color_climate= ['#FA9B58','#FECE7C','#FFF5AE','#FBFAE6','#B9E176','#96D268','#69BE63','#33A456','#108647']


'''
col=("#32A4CD" ,"#2FA17D" ,"#32A351", "#F4E547", "#F0B53B", "#E05F34","#D81D43" ,"#D71564" ,"#AA1C81" ,"#333085", "#1A62A5" , "#3098D1" )
ListedColormap(col)
col=("#0066ff","#0099ff","#00ccff","#00ffcc","#00ff99","#00ff00","#99ff33","#ccff33",
      "#ffff00","#ffcc00","#ff9933","#ff6600","#ff5050","#ff0066","#ff3399","#ff33cc","#ff00ff",
      "#cc33ff","#9966ff","#6666ff","#3366ff")
month_col=["#32A4CD" ,"#2FA17D" ,"#32A351", "#F4E547", "#F0B53B", "#E05F34","#D81D43" ,"#D71564" ,"#AA1C81" ,"#333085", "#1A62A5" , "#3098D1" ]

'''
col=("#468CBC","#62C4C4","#55BF7F","#97C97B","#EAEA6E","#EAAD5E",
     "#ED965A","#E56A55","#D8324A","#AA4B9F","#43619E","#496BAD")

month_col=("#5461A6","#4180B7","#53A3B0","#71C5A4","#96D4A6","#B9E3A6","#DCEFA0",
  "#FFFDC2","#FCECA4","#FCD589","#F8B871","#F4975F","#EF6F4D",
  "#DD514E","#C93451","#A60F47","#A13F7B","#6E5197")
ListedColormap(month_col)



#%%

ex='dry_to_wet'
feature='frequency'

title_aux = list(map(chr, range(97, 123)))[:8]

#%%

p_num=-1
fig = plt.figure(figsize = (17.5/2.54, 10/2.54)) # 宽、高
for feature in ['occurtime']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        p_num=p_num+1
        subplot_title=title_aux[p_num]
        
        if ex =='dry_to_wet':
            position_x = 0
            ex_title = 'dry-to-wet'
        else :
            position_x = 0.5
            ex_title = 'wet-to-dry'
            
    
        position_y= 0.05
        Colors_limits =np.array([30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350])-10
        feature_title = 'average timing'
        
    #%
        cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv')
        cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv')
        
        current_events_time=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S8.future_time_'+ex+'_map'+'.csv')
        
       
        ax = plt.axes([position_x,position_y,0.4,0.4] ,projection=ccrs.Robinson(central_longitude=150))
        
        cf=ax.contourf(cycle_LON,cycle_LAT,current_events_time,
                     transform=ccrs.PlateCarree(),colors=month_col,
                     levels=Colors_limits,extend='both')
    
        ##基本属性
        ax.set_global()
        #ax.stock_img()
        ax.coastlines(linewidth=0.3)
        ax.outline_patch.set_visible(False)
        
        
        #({})  {} {}'.format(subplot_title, ex, feature)
        ax.set_title( r"$\bf{("+subplot_title+")}$" + ' '+ex_title+' '+feature_title ,  #fontweight='bold',
                         pad=1, size=8.5, loc='left')

    ax = plt.axes([0.35,0.06,0.2,0.2] )
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
           


fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S8.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S8.pdf",dpi=1500, bbox_inches='tight')    