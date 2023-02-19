#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:01:53 2023

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
from matplotlib.patches import ConnectionPatch
#%% Source data
#basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'


#%%

feature = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/S2.features_of_a_dry_to_wet_example.npy',allow_pickle=True).tolist()
Day_cycle=feature.get('Day_cycle')
event_loc=feature.get('event_loc')
Sd=feature.get('Sd')
Quan_prec=feature.get('Quan_prec')
prec=feature.get('prec')
cum_prec=feature.get('cum_prec')
cum_prec_detrended=feature.get('cum_prec_detrended')
#%%
examples =((14399,16))
i=14399

day_cycle = np.tile(Day_cycle.values ,181)
sd = np.tile(Sd.values ,181)
quan = Quan_prec

d1 = event_loc[0] #转换事件发生的第一天
fig = plt.figure(figsize=(14/2.54,20/2.54)) 
ax2=plt.axes([0,0.6,0.8,0.2] )

range_d_start = (d1 - 500).astype(int)
range_d_end = (d1 + 500).astype(int)

plot_days= prec[range_d_start:range_d_end ].time
ax2.fill_between(plot_days  ,y1=0,y2=prec[range_d_start:range_d_end ],color='grey')
ax2.plot(plot_days  ,cum_prec[range_d_start:range_d_end],color='#67BB66',linewidth=0.8)
ax2.plot(plot_days  ,cum_prec_detrended[range_d_start:range_d_end ],color='black',linewidth=0.8)
ax2.plot(plot_days  ,day_cycle[range_d_start:range_d_end])
ax2.margins(x=0)  

alpha_year =0.1
nyear = np.unique(plot_days['time.year'])[1:]
if len(nyear)>1:
    for y in range(len(nyear)):
        if y ==0 :
            ax2.axvspan(xmin=plot_days[0].values,xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values ,
                       alpha=alpha_year,color ='green')
            ax2.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='orange')
        
        elif y == len(nyear)-1:
            ax2.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='orange')
        else:
            ax2.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='green')
    
t= [
plot_days[np.where(plot_days['time.year']==nyear[0])[0][0]].values,
plot_days[np.where(plot_days['time.year']==nyear[0])[0][181]].values,
plot_days[np.where(plot_days['time.year']==nyear[1])[0][0]].values,
plot_days[np.where(plot_days['time.year']==nyear[1])[0][181]].values,
plot_days[np.where(plot_days['time.year']==nyear[2])[0][0]].values,
plot_days[np.where(plot_days['time.year']==nyear[2])[0][181]].values,
]        
ax2.set_xticks(ticks=t,fontsize=6.5)
ax2.set_xlabel('Time',fontsize=8)
ax2.set_ylabel('Precipitation (mm)',fontsize=7) 
ax2.tick_params(axis='x', labelsize=6.5,pad=0.5 )
ax2.tick_params(axis='y', labelsize=6.5,pad=0.5 )
ax2.annotate('annual cycle', xy=(plot_days[np.where(plot_days['time.year']==nyear[0])[0][181]].values, 130), xycoords="data",c='black',size=7)
ax2.quiver(plot_days[np.where(plot_days['time.year']==nyear[0])[0][230]].values, 130,
     0,-90,width=0.003,angles='xy', scale_units='xy', scale=1)
ax2.set_title(r"$\bf{(a)}$" +' '+'Cumulative precipitation totals within 30 days',
                 pad=2, size=7.5, loc='left')


ax3=plt.axes([0,0.35,0.8,0.2] )

index_non_detrend = (cum_prec[: ] - day_cycle)/sd
ax3.plot(plot_days  ,index_non_detrend[range_d_start:range_d_end],color='#67BB66',linewidth=0.8,alpha=0.8)

index = (cum_prec_detrended[: ] - day_cycle)/sd
ax3.plot(plot_days  ,index[range_d_start:range_d_end],color='black',linewidth=0.8)

ax3.hlines(y=quan.sel(threshold=0.9), xmin=plot_days[0], xmax=plot_days[-1],linestyles='--',color='green')
ax3.hlines(y=quan.sel(threshold=0.1), xmin=plot_days[0], xmax=plot_days[-1],linestyles='--',color='orange')
ax3.margins(x=0)  
ax3.fill_between(x=plot_days,
            y2=quan.sel(threshold=0.1).values, 
            y1=index[range_d_start:range_d_end].values, 
            where = quan.sel(threshold=0.1) >= index[range_d_start:range_d_end],
            color='orange')
ax3.fill_between(x=plot_days,
            y2=quan.sel(threshold=0.9).values, 
            y1=index[range_d_start:range_d_end].values, 
            where = quan.sel(threshold=0.9) <= index[range_d_start:range_d_end],
            color='green')

if len(nyear)>1:
    for y in range(len(nyear)):
        if y ==0 :
            ax3.axvspan(xmin=plot_days[0].values,xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values ,
                       alpha=alpha_year,color ='green')
            ax3.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='orange')
        
        elif y == len(nyear)-1:
            ax3.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='orange')
        else:
            ax3.axvspan(xmin=plot_days[np.where(plot_days['time.year']==nyear[y])[0][0]].values,
                        xmax=plot_days[np.where(plot_days['time.year']==nyear[y])[0][-1]].values ,
                       alpha=alpha_year,color ='green')
    
   
        
ax3.set_xticks(ticks=t,fontsize=6.5)
ax3.set_xlabel('Time',fontsize=8)
ax3.set_ylabel('Standardized precipitation anomalies',fontsize=7) 
ax3.tick_params(axis='x', labelsize=6.5,pad=0.5 )
ax3.tick_params(axis='y', labelsize=6.5,pad=0.5 )
ax3.set_title(r"$\bf{(b)}$" +' '+'Standardized cumulative precipitation anomalies',
                 pad=2, size=7.5, loc='left')
#%

ax=plt.axes([0,0.1,0.8,0.2] )
range_d_start = (d1 - 70).astype(int)
range_d_end = (d1 + 85).astype(int)
plot_days= prec[range_d_start:range_d_end ].time

ax3.axvspan(xmin=plot_days[0].values,
            xmax=plot_days[-1].values ,
           alpha=0.1,color ='purple')

xy1 = (plot_days[0].values,0)
xy2 = (plot_days[0].values,4)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",axesA=ax3, axesB=ax, color="black")
ax.add_artist(con)

xy1 = (plot_days[-1].values,0)
xy2 = (plot_days[-1].values,4)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",axesA=ax3, axesB=ax, color="black")
ax.add_artist(con)

#ax.plot(plot_days  ,prec[range_d_start:range_d_end ],color='grey')

index = (cum_prec_detrended[: ] - day_cycle)/sd


#plt.plot(plot_days  ,prec[range_d_start:range_d_end ],color='grey')

ax.plot(plot_days  ,index[range_d_start:range_d_end],color='black')

ax.fill_between(x=plot_days,
            y2=quan.sel(threshold=0.1).values, 
            y1=index[range_d_start:range_d_end].values, 
            where = quan.sel(threshold=0.1) >= index[range_d_start:range_d_end],
            color='orange')
ax.fill_between(x=plot_days,
            y2=quan.sel(threshold=0.9).values, 
            y1=index[range_d_start:range_d_end].values, 
            where = quan.sel(threshold=0.9) <= index[range_d_start:range_d_end],
            color='green')
ax.hlines(y=quan.sel(threshold=0.9), xmin=plot_days[0], xmax=plot_days[-1],linestyles='--',color='green')
ax.hlines(y=quan.sel(threshold=0.1), xmin=plot_days[0], xmax=plot_days[-1],linestyles='--',color='orange')

#ax.scatter(x=prec.time[d1.astype(int)],y=event_loc[1],color='red',zorder=10,s=10)
#ax.scatter(x=prec.time[event_loc[3].astype(int)],y=event_loc[4],color='red',zorder=10,s=10)
ax.scatter(x=prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].min())[0].astype(int)],
           y=event_loc[1],color='red',zorder=10,s=10)
ax.scatter(x=prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].max())[0].astype(int)],
           y=event_loc[4],color='red',zorder=10,s=10)


#
ax.axvspan(xmin=prec.time[event_loc[0].astype(int)].values,xmax=prec.time[event_loc[3].astype(int)].values,
       alpha=0.15,color ='green')
#%
ax.hlines(xmin = prec.time[event_loc[0].astype(int)] , xmax = prec.time[event_loc[3].astype(int)],
      y = event_loc[4]+0.5 ,linestyles='solid',color='red',linewidth=1,)
#%
ax.vlines(ymin = event_loc[4]+0.48 , ymax = event_loc[4]+0.6,
      x =  prec.time[event_loc[0].astype(int)],linewidth=1,color='red')
ax.vlines(ymin = event_loc[4]+0.48 , ymax = event_loc[4]+0.6,
      x =  prec.time[event_loc[3].astype(int)],linewidth=1,color='red')
#%
ax.quiver(prec.time[event_loc[0].astype(int)-15],0.5,
     15+((event_loc[3]-event_loc[0])/2).astype(int),event_loc[4],
              width=0.003,angles='xy', scale_units='xy', scale=1)
ax.annotate('transition duration', xy=(prec.time[event_loc[0].astype(int)-25], 0.2), xycoords="data",c='black',size=7)
#%
##标记强度
ax.hlines(xmin = prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].min())[0].astype(int)] , 
          xmax = prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].max())[0].astype(int)+10],
      y = event_loc[1] ,linestyles='solid',linewidth=1,color='red')
ax.hlines(xmin = prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].max())[0].astype(int)] , 
          xmax = prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].max())[0].astype(int)+10],
      y = event_loc[4] ,linestyles='solid',linewidth=1,color='red')
ax.vlines(ymin = event_loc[1], ymax = event_loc[4],
          x = prec.time[range_d_start+np.where(index[range_d_start:range_d_end]==index[range_d_start:range_d_end].max())[0].astype(int)+10],
          linewidth=1,color='red')

ax.quiver(prec.time[event_loc[3].astype(int)+20],0,
     16,0.6,
              width=0.003,angles='xy', scale_units='xy', scale=1)
#
ax.annotate('intensity', xy=(prec.time[event_loc[3].astype(int)+14], -0.4), xycoords="data",c='black',size=7)
ax.set_title(r"$\bf{(c)}$" +' '+'An example of dry-to-wet whiplash',
                 pad=2, size=7.5, loc='left')  

ax1=ax.twinx() #右边
ax1.fill_between(plot_days  ,y1=0,y2=cum_prec_detrended[range_d_start:range_d_end ],color='grey',alpha=0.15)
ax1.fill_between(plot_days  ,y1=0,y2=prec[range_d_start:range_d_end ],color='grey',alpha=0.3)
#ax1.plot(plot_days  ,day_cycle[range_d_start:range_d_end],color='grey')


ax1.set_ylabel('Precipitation (mm)',fontsize=7)
   
t= [prec.time[d1.astype(int)-60].values,prec.time[d1.astype(int)-30].values,
prec.time[d1.astype(int)].values,
prec.time[d1.astype(int)+30].values,prec.time[d1.astype(int)+60].values        
]
   
#t=plot_days.values[np.arange(18,len(plot_days),len(plot_days)/4).astype(int)]
ax.set_xticks(ticks=t,fontsize=6.5)
ax.set_xlabel('Time',fontsize=8)
ax.set_ylabel('Standardized precipitation anomalies',fontsize=7) 

ax.margins(x=0)  
ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
ax1.tick_params(axis='y', labelsize=6.5,pad=0.5 )



fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S2.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S2.pdf",dpi=1500, bbox_inches='tight')