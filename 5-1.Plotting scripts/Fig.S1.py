#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:42:13 2023

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

from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib

#%% Source data
#basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'


cycle_LON=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv').values
cycle_LAT=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv').values
clim_prec=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.climatology_prcp_map.csv').values
change_prec=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.future_change_prcp_map.csv')
global_change_prec = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_original_and_detrended_prec.csv')
global_change_prec.index=np.arange(1920,2101,1)

d2w_global_mean = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_dry_to_wet_whiplash.csv')
w2d_global_mean = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_wet_to_dry_whiplash.csv')
d2w_global_mean.index=np.arange(1920,2101,1)
w2d_global_mean.index=np.arange(1920,2101,1)
#%%

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

colorbar_prec = ['#FFFFE4','#FEF4CF','#FEEDB0','#FBCF93','#F7B07A','#F19164','#E97356','#C96775','#87518E']

colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366','#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']

colorbar_change_b=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#FEF4CF','#FECF80','#FCB366','#F89053','#F26B43']

color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']

#%%


fig = plt.figure(figsize = (17.5/2.54, 12/2.54)) 
##################################  a  #######################################
Colors_limits = np.array([0,30,100,300,800,1200,1500,2000,3000])

ax = plt.axes([0,0.73,0.4,0.3] ,projection=ccrs.Robinson(central_longitude=150))

cf=ax.contourf(cycle_LON,cycle_LAT,clim_prec,
             transform=ccrs.PlateCarree(),
             levels=Colors_limits,extend='both', colors=colorbar_prec)


ax.set_global()
#ax.stock_img()
ax.coastlines(linewidth=0.35)
ax.outline_patch.set_visible(False)
'''
#colorbar位置
a=ax.get_position()
pad=0.015
width=0.01
ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ]) #长宽高
cb=fig.colorbar(cf, cax=ax_f)
cb.set_label( label = "Annual mean (mm)",fontdict={'size':6.5},labelpad=1)
#标签刻度位置
ax_f.yaxis.set_ticks_position('right')
ax_f.set_yticklabels(Colors_limits,size=6)
cb.outline.set_linewidth(0.7)
'''

ax.set_title('a',  fontweight='bold',
                 pad=1, size=8, loc='left')


#colorbar位置
a=ax.get_position()
pad=0.02
height=0.01
ax_f = fig.add_axes([ 0.025, a.ymin - pad,  0.37 , height ]) #长宽高
cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
cb.set_label( label = "Annual mean (mm)",fontdict={'size':6.5},labelpad=1)

cb.set_ticklabels(Colors_limits,size=6)
cb.outline.set_linewidth(0.7)

ax_f.tick_params(length=1,pad=0.2)

##################################  b  #######################################

change_level=[-60,-30,-20,-10,0,10,20,30,60]

ax1 = plt.axes([0.42,0.73,0.4,0.3] ,projection=ccrs.Robinson(central_longitude=150))

cf1=ax1.contourf(cycle_LON,cycle_LAT,change_prec,
             transform=ccrs.PlateCarree(),
             levels=change_level,extend='both', colors=colorbar_change_b)


ax1.set_global()
#ax.stock_img()
ax1.coastlines(linewidth=0.35)
ax1.outline_patch.set_visible(False)
ax1.set_title('b',  fontweight='bold',
                 pad=1, size=8, loc='left')



a=ax1.get_position()
pad=0.02
height=0.01
ax1_f = fig.add_axes([ 0.47 , a.ymin - pad,  0.3 , height ]) #长宽高
cb1=fig.colorbar(cf1, orientation='horizontal',cax=ax1_f)
cb1.set_label( label = "Precipitation trend (%)",fontdict={'size':6.5},labelpad=1)

cb1.set_ticklabels(change_level,size=6)
cb1.outline.set_linewidth(0.7)

ax1_f.tick_params(length=1,pad=0.2)


##################################  c  #######################################
#
ax2 = plt.axes([0.07,0.4,0.7,0.25] ) 

ax2.plot(np.arange(1920,2101,1 ).astype(int),global_change_prec.iloc[:,2],label='Raw')
ax2.fill_between(x=np.arange(1920,2101,1 ).astype(int),
                 y1=global_change_prec.iloc[:,0],
                 y2=global_change_prec.iloc[:,1],alpha=0.2 )

ax2.plot(np.arange(1920,2101,1 ).astype(int),global_change_prec.iloc[:,8],label='Polynomial detrended',color=color_d_type[1] )
ax2.fill_between(x=np.arange(1920,2101,1 ).astype(int),
                 y1=global_change_prec.iloc[:,6],
                 y2=global_change_prec.iloc[:,7],alpha=0.2 ,color=color_d_type[1] )

ax2.plot(np.arange(1920,2101,1 ).astype(int),global_change_prec.iloc[:,11],label='Ensemble mean scaling' ,color=color_d_type[2])
ax2.fill_between(x=np.arange(1920,2101,1 ).astype(int),
                 y1=global_change_prec.iloc[:,9],
                 y2=global_change_prec.iloc[:,10],alpha=0.2 ,color=color_d_type[2] )


ax2.plot(np.arange(1920,2101,1 ).astype(int),global_change_prec.iloc[:,5],label='Linear detrended' ,color=color_d_type[3])
ax2.fill_between(x=np.arange(1920,2101,1 ).astype(int),
                 y1=global_change_prec.iloc[:,3],
                 y2=global_change_prec.iloc[:,4],alpha=0.2 ,color=color_d_type[3] )




ax2.hlines(y=global_change_prec.loc[1979:2019].iloc[:,2].mean(),xmin=1920,xmax=2100,linestyle='--',linewidth=0.7)
ax2.axvspan(xmin=1979, xmax=2019,alpha=0.2) 


#修改主刻度
xmajorLocator = MultipleLocator(20) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
ymajorLocator = MultipleLocator(50) #将y轴主刻度标签设置为0.5的倍数
ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
#设置主刻度标签的位置,标签文本的格式
ax2.xaxis.set_major_locator(xmajorLocator)
ax2.xaxis.set_major_formatter(xmajorFormatter)
ax2.yaxis.set_major_locator(ymajorLocator)
ax2.yaxis.set_major_formatter(ymajorFormatter)

#修改次刻度
xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
yminorLocator = MultipleLocator(10) #将此y轴次刻度标签设置为0.1的倍数
#设置次刻度标签的位置,没有标签文本格式
ax2.xaxis.set_minor_locator(xminorLocator)
ax2.yaxis.set_minor_locator(yminorLocator)
ax2.margins(x=0)
 
ax2.set_xlabel('Year',fontsize=7,labelpad=1)

ax2.tick_params(axis='x', labelsize=6.5 )
ax2.tick_params(axis='y', labelsize=6.5 )

yticks=[1100,1150,1200]
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticks,fontsize=6.5)
ax2.set_ylabel('Annual mean (mm) ',fontsize=6.5,labelpad=1)

ax2.set_title('c',  fontweight='bold',
                 pad=1, size=8, loc='left')

ax2.legend( framealpha=0,fontsize=6.5)

#%
#############################################  d  ###############################################

ax3 = plt.axes([0.05,0.12,0.35,0.2] ) #左-右，下-上，宽，高

ex='dry_to_wet'
pnum=0
for d_p in [0,1,2,3]:   
    
    ax3.plot(d2w_global_mean.index,d2w_global_mean.iloc[:,d_p],linewidth=1,color=color_d_type[pnum],alpha=1)
    pnum=pnum+1
    
event_region_mean_baseline= d2w_global_mean.loc[1979:2019].iloc[:,2].mean()    
ax3.hlines(y=event_region_mean_baseline,xmin=1920,xmax=2100,linestyle='--',linewidth=0.7)
ax3.axvspan(xmin=1979, xmax=2019,alpha=0.2) 
ax3.margins(x=0)
 

xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
ymajorLocator = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
#设置主刻度标签的位置,标签文本的格式
ax3.xaxis.set_major_locator(xmajorLocator)
ax3.xaxis.set_major_formatter(xmajorFormatter)
ax3.yaxis.set_major_locator(ymajorLocator)
ax3.yaxis.set_major_formatter(ymajorFormatter)

#修改次刻度
xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
yminorLocator = MultipleLocator(0.05) #将此y轴次刻度标签设置为0.1的倍数
#设置次刻度标签的位置,没有标签文本格式
ax3.xaxis.set_minor_locator(xminorLocator)
ax3.yaxis.set_minor_locator(yminorLocator)
 
ax3.set_xlabel('Year',fontsize=7,labelpad=1)

ax3.tick_params(axis='x', labelsize=6.5 )
ax3.tick_params(axis='y', labelsize=6.5 )

ax3.set_ylabel('Occurence (times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)

ax3.set_title('d',  fontweight='bold',
                 pad=1, size=8, loc='left')

ax3.annotate("Dry-to-wet", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)


#############################################  e  ###############################################

#%
ax4 = plt.axes([0.45,0.12,0.35,0.2] ) #左-右，下-上，宽，高

ex='wet_to_dry'
pnum=0
for d_p in [0,1,2,3]:   
    
    ax4.plot(w2d_global_mean.index,w2d_global_mean.iloc[:,d_p],linewidth=1,color=color_d_type[pnum],alpha=1)
    pnum=pnum+1
    
event_region_mean_baseline= w2d_global_mean.loc[1979:2019].iloc[:,2].mean()    
ax4.hlines(y=event_region_mean_baseline,xmin=1920,xmax=2100,linestyle='--',linewidth=0.7)
ax4.axvspan(xmin=1979, xmax=2019,alpha=0.2) 
ax4.margins(x=0)
 
ax4.xaxis.set_major_locator(xmajorLocator)
ax4.xaxis.set_major_formatter(xmajorFormatter)
ax4.yaxis.set_major_locator(ymajorLocator)
ax4.yaxis.set_major_formatter(ymajorFormatter)


#设置次刻度标签的位置,没有标签文本格式
ax4.xaxis.set_minor_locator(xminorLocator)
ax4.yaxis.set_minor_locator(yminorLocator)
 
ax4.set_xlabel('Year',fontsize=7,labelpad=1)

ax4.tick_params(axis='x', labelsize=6.5 )
ax4.tick_params(axis='y', labelsize=6.5 )

ax4.annotate("Wet-to-dry", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)

ax4.set_title('e',  fontweight='bold',
                 pad=1, size=8, loc='left')



fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S1.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S1.pdf",dpi=1500, bbox_inches='tight')
