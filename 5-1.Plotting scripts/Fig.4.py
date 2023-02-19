#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:37:15 2023

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
import shapefile,cmaps
from matplotlib.path import Path
from matplotlib.patches import PathPatch

basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'


each_regimes = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/4.each_regimes.npy',allow_pickle=True).tolist()

#%%

col=np.array(["#FA9B58" ,"#FECE7C" ,"#FFF5AE",
       "#69BE63","#B9E176","#DDF2A6",
         "#E2F2F1","#C6E6F2","#89BED9",
           "#FADAC8","#E8A285","#DD726B"])

ListedColormap(col)

col =np.array(["#E2F2F1","#C6E6F2","#89BED9",
          "#FADAC8","#E8A285","#DD726B",
          "#DDF2A6","#B9E176","#69BE63",
          "#FFF5AE","#FECE7C","#FA9B58"
          ])

#%%            
feature='frequency'
pn=2

#%%
fig = plt.figure(figsize = (7/2.54, 10/2.54)) # 宽、高


for ex in ['dry_to_wet','wet_to_dry']:
    if ex == 'dry_to_wet':
        
        ax = plt.axes([0,0.4,1,0.9] ,projection=ccrs.Robinson(central_longitude=150))
    else :
        ax = plt.axes([0,0,1,0.9] ,projection=ccrs.Robinson(central_longitude=150))
        '''
    a_m = global_change.get('event~'+ex+'~'+feature+'~ensemble_mean-period'+str(pn))
    p = global_change.get('prec~global-period'+str(pn)) 
    a = global_change.get('event~'+ex+'~'+feature+'~global-period'+str(pn)) 
    
    mPmE = (a.where(( (a/p) >=0 ) & (a>= 0)) >=0 ) .astype(int).sum('ensemble') # more p more e
    mPlE = (a.where(( (a/p) <=0 ) & (a<= 0)) <=0 ) .astype(int).sum('ensemble') # more p more e,pink
    lPlE = (a.where(( (a/p) >=0 ) & (a<= 0)) <=0 ) .astype(int).sum('ensemble') # more p more e,green
    lPmE = (a.where(( (a/p) <=0 ) & (a>= 0)) >=0 ) .astype(int).sum('ensemble') # more p more e
    
    #sig = (mPmE.where((mPmE>=36) | (mPlE>=36) | (lPlE>=36) | (lPmE>=36))>0).astype(int)  ##90%数据集同意
    sig1 = (mPmE.where((mPmE>=36) )>0).astype(int)#.values
    #sig1=sig1.where(sig>0)
    
    sig2 = (mPlE.where((mPlE>=36) )>0).astype(int)#.values
    #sig2=sig2.where(sig>0)
    
    sig3 = (lPlE.where((lPlE>=36) )>0).astype(int)#.values
    #sig3=sig3.where(sig>0)
    
    sig4 = (lPmE.where((lPmE>=36) )>0).astype(int)
    #sig4=sig4.where(sig>0)
    
    sig = (sig1+sig2+sig3+sig4).where((sig1+sig2+sig3+sig4)>0).values
    
    
    '''
    
    
    '''
    a1 = (a_m/p).where((a_m/p >0) & (a_m > 0))
    a1.quantile([0.33,0.66])
    a2 = (a_m/p).where((a_m/p <0) & (a_m < 0))
    a2.quantile([0.33,0.66])
    a3 = (a_m/p).where((a_m/p >0) & (a_m < 0))
    a3.quantile([0.33,0.66])
    a4 = (a_m/p).where((a_m/p <0) & (a_m > 0))
    a4.quantile([0.33,0.66])
    
    
    b=a1.values
    b[np.where(b>=6)]=3
    b[np.where( (b>3)&(b<6) )]=2
    b[np.where( (b>=0)&(b<3)&(b!=2) )]=1
    a1[:]=b
    
    
    b=a2.values
    b[np.where(b<= -6 )]=9
    b[np.where( (b>-6)&(b< -3) )]=8
    b[np.where( (b>=-3)&(b<=0) )]=7
    a2[:]=b
    
    
    b=a3.values
    b[np.where(b>=6)]= 12
    b[np.where( (b>=3)&(b<6) )]= 11
    b[np.where( (b>=0)&(b<3) )]= 10
    a3[:]=b
    
    
    b=a4.values
    b[ np.where(b<= -6) ]  = 19
    b[np.where( (b> -6)&(b<= -3 )&(b!=-9) )]= 18
    b[np.where( (b<=0)&(b> -3) )]= 17
    a4[:]=b
    
    
    np.isnan(a1.values)
    
    c = xr.concat([a1,a2,a3,a4], dim=pd.Index([1,2,3,4],name='regime'))
    d=np.nansum(c.values,axis=0)
    '''
    
    d=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/4.prec_and_whiplash_regimes_'+ex+'.csv')
    
    
    color_limit = [0,1.5,2.5,3.5,7.5,8.5,9.5,  #1 2 3   ,7,8,9
                   10.5,11.5,12.5,17.5,18.5,19.5] # 10,11,12, 17,18,19
    
    a1=each_regimes.get('a1_'+ex)    
    cycle_regime, cycle_lon = add_cyclic_point(np.array(d), coord=a1.lon)
    cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, a1.lat)
    
    
    ##基本属性
    ax.set_global()
    #ax.stock_img()
    ax.coastlines(linewidth=0.3)
    ax.outline_patch.set_visible(False)
    
    if ex =='wet_to_dry':
        cf_a1 = ax.contourf(a1.lon,a1.lat,a1,
                     transform=ccrs.PlateCarree(),
                     levels=np.array(color_limit)[[0,1,2,3]],
                     #extend='both',
                     colors= col[[0,1,2]])
        cf_a2 = ax.contourf(a1.lon,a1.lat,each_regimes.get('a2_'+ex)   ,
                     transform=ccrs.PlateCarree(),
                     levels=np.array(color_limit)[[3,4,5,6]],
                     #extend='both',
                     colors= col[[3,4,5]])
        cf_a3 = ax.contourf(a1.lon,a1.lat,each_regimes.get('a3_'+ex)   ,
                     transform=ccrs.PlateCarree(),
                     levels=np.array(color_limit)[[6,7,8,9]],
                     #extend='both',
                     colors= col[[6,7,8]])
        cf_a4 = ax.contourf(a1.lon,a1.lat,each_regimes.get('a4_'+ex)   ,
                     transform=ccrs.PlateCarree(),
                     levels=np.array(color_limit)[[9,10,11,12]],
                     #extend='both',
                     colors= col[[9,10,11]])
        
        
        a=ax.get_position()
        pad=0.06
        height=0.022
        ax_f = fig.add_axes([ 0.25, a.ymin - pad,  (a.xmax - a.xmin)*0.2 , height ]) #长宽高
        cb1=fig.colorbar(cf_a1, orientation='horizontal',cax=ax_f)
        ax_f.xaxis.set_ticks_position('top')
        #colorbar标签刻度位置
        #cb1.set_ticks(np.array(color_limit)[[7,8]],fontsize=6)
        cb1.set_ticklabels(np.array(color_limit)[[6,7,8,9]],size=6)
        cb1.set_ticklabels(ticklabels=['','3','6',''],size=6)
        cb1.outline.set_linewidth(0.5)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
        cb1.set_label( label = "morePmoreE",fontdict={'size':7},labelpad=1)
        
        
        
        
        ax_f = fig.add_axes([ 0.6, a.ymin - pad,  (a.xmax - a.xmin)*0.2 , height ]) #长宽高
        cb2=fig.colorbar(cf_a2, orientation='horizontal',cax=ax_f)
        ax_f.xaxis.set_ticks_position('top')
        #colorbar标签刻度位置
        #cb2.set_ticks(np.array(color_limit)[[10,11]])
       
        cb2.set_ticklabels(np.array(color_limit)[[9,10,11,12]],size=6)
        cb2.set_ticklabels(ticklabels=['','3','6',''],size=6)
        
        cb2.outline.set_linewidth(0.5)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
        cb2.set_label( label = "morePfewerE",fontdict={'size':7},labelpad=1)
        
        ax_f = fig.add_axes([ 0.6, a.ymin - 2*pad-0.02,  (a.xmax - a.xmin)*0.2 , height ]) #长宽高
        cb3=fig.colorbar(cf_a3, orientation='horizontal',cax=ax_f)
        ax_f.xaxis.set_ticks_position('top')
        #colorbar标签刻度位置get
        cb3.set_ticklabels(np.array(color_limit)[[3,4,5,6]],size=6)
        cb3.set_ticklabels(ticklabels=['','3','6',''],size=6)
        cb3.outline.set_linewidth(0.5)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
        cb3.set_label( label = "lessPfewerE",fontdict={'size':7},labelpad=1)
        
        ax_f = fig.add_axes([ 0.25, a.ymin - 2*pad-0.02,  (a.xmax - a.xmin)*0.2 , height ]) #长宽高
        cb4=fig.colorbar(cf_a4, orientation='horizontal',cax=ax_f)
        ax_f.xaxis.set_ticks_position('top')
        #colorbar标签刻度位置
        cb4.set_ticklabels(np.array(color_limit)[[0,1,2,3]],size=6)
        cb4.set_ticklabels(ticklabels=['','3','6',''],size=6)
        cb4.outline.set_linewidth(0.5)
        #colorbar标签刻度值
        ax_f.tick_params(length=1,pad=0.2)
        cb4.set_label( label = "lessPmoreE",fontdict={'size':7},labelpad=1)
        
        
    cf=ax.contourf(cycle_LON,cycle_LAT,cycle_regime,
                 transform=ccrs.PlateCarree(),
                 levels=color_limit,
                 extend='both',colors= col)
    
    ax.contourf(a1.lon,a1.lat,each_regimes.get('sig_'+ex)   ,
                            transform=ccrs.PlateCarree(),hatches=['//////', None],colors="none",width=0.001)
                      
    ax.set_global()
    #ax.stock_img()
    ax.coastlines(linewidth=0.3)
    ax.outline_patch.set_visible(False)    
    
    if ex == 'dry_to_wet':
        feature_title = 'dry-to-wet'
        ax.set_title(r"$\bf{(a)}$" +' '+feature_title,
                         pad=2, size=7, loc='left')
    else:
        feature_title = 'wet-to-dry'
        ax.set_title(r"$\bf{(b)}$" +' '+feature_title,
                         pad=2, size=7, loc='left')
    
    
plt.rcParams['hatch.linewidth'] = 0.3
    

fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.4.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.4.pdf",dpi=1500, bbox_inches='tight') 
#%%
