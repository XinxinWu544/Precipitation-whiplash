#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:16:11 2023

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

#%% Source data
basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'
regional_mean = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/3_S14-15.monsoon_regional_mean_trend_and_SNR.npy',allow_pickle=True).tolist()
global_change = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/3_S14-15.global_change_distribution_bottom_map.npy',allow_pickle=True).tolist()

#%% user-defined colorbars and variables

col=("#5E4FA2" ,"#4F61AA" ,"#4173B3", "#4198B6", "#51ABAE", "#77C8A4" ,"#A4DAA4" ,"#CBEA9D", \
       "#ffffff",\
       "#FEF0A5", "#FDD985", "#FDC978", "#F99254", "#F67D4A", "#E85A47", "#D33C4E" ,"#C1284A")
ListedColormap(col)
color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']
colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']
color_shp =["#F6A975" ,"#F08E63", "#E97356", "#DE5953" ,"#CD4257", "#B8315D"]
ListedColormap(colorbar_change)

monsoon_name = ['WAfriM','SAsiaM','SAmerM','NAmerM','EAsiaM','AusMCM']

#Colors_limits_intensity=[-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140]
Colors_limits_intensity=[-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21]
Colors_limits_frequency=[-100,-80,-60,-40,-20,0,30,60,90,120,150,180,210]
#Colors_limits_duration=[-150,-120,-90,-60,-30,0,30,60,90,120,150,180,210]
Colors_limits_duration=[-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]

title_aux = list(map(chr, range(97, 123)))[:6]
ex_name =  ['dry-to-wet','wet-to-dry']
#%% user-defined funtions
def shp2clip(originfig, ax, shpfile):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i + 1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
    for contour in originfig.collections:
        contour.set_clip_path(clip)
    return contour

#%%

for feature in ['frequency','intensity','duration' ]:
    print(feature)
    for ex in ['dry_to_wet','wet_to_dry']:
        print(ex)
        for ms in [0,1,2,3,4,5]:
            
            mean_value = regional_mean.get("LENS~"+ex+'~'+feature+'~'+monsoon_name[ms])
            print(monsoon_name[ms]+": "+str(mean_value.iloc[:,2][-1:].values) )

#%%

for feature in ['frequency','intensity','duration' ]:
    print(feature)
    for ex in ['dry_to_wet','wet_to_dry']:
        print(ex)
        for ms in [0,1,2,3,4,5]:
            
            mean_value = regional_mean.get("LENS~"+ex+'~'+feature+'~'+monsoon_name[ms])
            
            #
            year_p = np.where(  (mean_value.iloc[:,2]>mean_value.iloc[:,1])  &  (mean_value.iloc[:,2]>0)  )[0]
            if len(year_p) !=0:
                
                print('positive '+monsoon_name[ms]+": "+str(mean_value.index[0]+year_p[0]))
            year_n = np.where(  (mean_value.iloc[:,2]<mean_value.iloc[:,0])  &  (mean_value.iloc[:,2]<0)  )[0]
            if len(year_n) !=0:
               
                print('negative '+monsoon_name[ms]+": "+str(mean_value.index[0]+year_n[0]))





#%%

for feature in ['frequency','intensity','duration' ]:
    fig = plt.figure(figsize = (17.5/2.54, 18.5/2.54)) # 宽、高
    
    p_num=0
    
    for ex in ['dry_to_wet','wet_to_dry']:
        p_num=p_num+1
        
        if p_num == 1 :
            y_loc =0.5
        else:
            y_loc =0
            
            
        ax = plt.axes([0,y_loc,0.9,0.46] ,projection=ccrs.PlateCarree())
        
        
        lon=global_change.get("LENS~"+ex+'~'+feature+'~global').lon
        lat=global_change.get("LENS~"+ex+'~'+feature+'~global').lat
        
        cycle_change, cycle_lon = add_cyclic_point(global_change.get("LENS~"+ex+'~'+feature+'~global') , coord=lon)
        cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)
        
        cf=ax.contourf(cycle_LON,cycle_LAT,cycle_change,
                     transform=ccrs.PlateCarree(),
                     levels=vars()['Colors_limits_'+feature],
                     extend='both',colors= colorbar_change,alpha=0.3)
        
        cf1=ax.contourf(cycle_LON,cycle_LAT,cycle_change,
                     transform=ccrs.PlateCarree(),
                     levels=vars()['Colors_limits_'+feature],
                     extend='both',colors=colorbar_change)
        shp2clip(cf1, ax, basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp')
    
        
        for i in range(6):
                
            monsoon = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/'+monsoon_name[i]+'.shp').geometries()
            
            ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor=color_shp[i] , linewidth=0.75, zorder=10) # 添加季风区
        
        #ax.set_extent([-80, 150, -30, 40])
        ax.coastlines(linewidth=0.3,alpha=0.5)
        #ax.outline_patch.set_visible(False)
        ax.set_xlim(-120,160)
        ax.set_ylim(-30,60)
        ax.set_aspect('auto') ##Non-fixed aspect
        gl=ax.gridlines(draw_labels=True,linestyle=":",linewidth=0.3,color='k')
        gl.top_labels=False                              
        gl.right_labels=False
        gl.xlabel_style={'size':7.5}                       
        gl.ylabel_style={'size':7.5}
        ax.spines['geo'].set_linewidth(0.5)
        
        ax.set_ylabel('Latitude (°N)',fontsize=6,labelpad=1)
        ax.set_xlabel('Longitude (°E)',fontsize=6,labelpad=1)
    
        feature_name = feature
        if feature == 'frequency':
            feature_name='transition duration'
        ax.set_title(r"$\bf{("+ title_aux[p_num-1] +")}$"+" Change in "+feature_name+" of "+ex_name[p_num -1]+" (%)",
                         pad=2, size=8.5, loc='left')
        
        for ms in [0,1,2,3,4,5]:
            if ms == 0:
                ms_x = 0.28;ms_y=0.022
            elif ms == 1:
                ms_x = 0.315;ms_y=0.27
            elif ms == 2 :
                ms_x = 0.08;ms_y=0.17
            elif ms == 3:
                ms_x = 0.085;ms_y=0.34
            elif ms == 4:
                ms_x = 0.53;ms_y=0.34
            elif ms == 5:
                ms_x = 0.513;ms_y=0.022
            ax1 = plt.axes([ms_x, y_loc+ms_y, 0.19,0.12] )

            mean_value = regional_mean.get("CMIP6~"+ex+'~'+feature+'~'+monsoon_name[ms])
           
            ax1.fill_between(x=mean_value.index,
                             y1=mean_value.iloc[:,0],y2=mean_value.iloc[:,1],
                             label='CMIP6',linewidth=1,color='#A5D2E0',alpha=0.7) #蓝色是CMIP6
            
            ax1.plot(mean_value.index,mean_value.iloc[:,2],label='CMIP6',linewidth=1,color='#699BC2')
            ax1.annotate(monsoon_name[ms], xy=(0.25, 0.85), xycoords="axes fraction",c=color_shp[ms],size=10)
            
            year_p = np.where(  (mean_value.iloc[:,2]>mean_value.iloc[:,1])  &  (mean_value.iloc[:,2]>0)  )[0]
            if len(year_p) !=0:
                ax1.scatter(mean_value.index[0]+year_p[0],mean_value.iloc[year_p[0],1],color='blue',s=8,zorder=20)
            
            year_n = np.where(  (mean_value.iloc[:,2]<mean_value.iloc[:,0])  &  (mean_value.iloc[:,2]<0)  )[0]
            if len(year_n) !=0:
                ax1.scatter(mean_value.index[0]+year_n[0],mean_value.iloc[year_n[0],0],color='blue',s=8,zorder=20)
            
            ax1.hlines(y=0,xmin=mean_value.index[0],xmax=mean_value.index[-1],color='grey',linestyle='--')
            
            
            ### LENS
            mean_value = regional_mean.get("LENS~"+ex+'~'+feature+'~'+monsoon_name[ms])
            print(monsoon_name[ms]+": "+str(mean_value.iloc[:,2][-1:].values) )
            ax1.fill_between(x=mean_value.index,
                             y1=mean_value.iloc[:,0],y2=mean_value.iloc[:,1],
                             label='CESM-LENS',linewidth=1,color='#F7CA7F' ,alpha=0.4) #橙色是CESM
            ax1.plot(mean_value.index,mean_value.iloc[:,2],label='CESM-LENS',linewidth=1,color='#EE8C53')
            
            
            year_p = np.where(  (mean_value.iloc[:,2]>mean_value.iloc[:,1])  &  (mean_value.iloc[:,2]>0)  )[0]
            if len(year_p) !=0:
                ax1.scatter(mean_value.index[0]+year_p[0],mean_value.iloc[year_p[0],1],color='red',s=8,zorder=20)
                print('year='+str(mean_value.index[0]+year_p[0]))
            year_n = np.where(  (mean_value.iloc[:,2]<mean_value.iloc[:,0])  &  (mean_value.iloc[:,2]<0)  )[0]
            if len(year_n) !=0:
                ax1.scatter(mean_value.index[0]+year_n[0],mean_value.iloc[year_n[0],0],color='red',s=8,zorder=20)
                print('year='+str(mean_value.index[0]+year_n[0]))
            
            xticks = [1940,2010,2080]
                
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticks)
            ax1.margins(x=0)  
            ax1.tick_params(axis='x', labelsize=7,pad=0.5 )
            ax1.tick_params(axis='y', labelsize=7,pad=0.5 )
            
        for ms in [0,1,2,3,4,5]:
            if ms == 0:
                qui_x1 = -5;qui_y1= -2 ;qui_x2 = 10;qui_y2=10
            elif ms == 1:
                qui_x1 = 37;qui_y1=27;qui_x2 = 80;qui_y2=25
            elif ms == 2 :
                qui_x1 = -55;qui_y1= 3 ; qui_x2 = -55 ;qui_y2= -10
            elif ms == 3:
                qui_x1 = -90;qui_y1=50 ; qui_x2 = -110 ;qui_y2= 30
            elif ms == 4:
                qui_x1 = 104;qui_y1=50;qui_x2 = 110;qui_y2= 30
            elif ms == 5:
                qui_x1 = 100;qui_y1=-18;qui_x2 = 135;qui_y2=-18
            ax.arrow(qui_x1, qui_y1, qui_x2-qui_x1  , qui_y2 - qui_y1,
                     width=0.003,
                     length_includes_head=True, # 增加的长度包含箭头部分
                      head_width=1,
                      head_length=2,
                     fc='black',
                     ec='black')
    
    ##formatting settings
    a=ax.get_position()
    pad=0.04
    height=0.015
    ax_f = fig.add_axes([ 0.08, a.ymin - pad,  (a.xmax - a.xmin)*0.8 , height ]) #长宽高
    cb=fig.colorbar(cf1, orientation='horizontal',cax=ax_f)
    cb.set_ticklabels(vars()['Colors_limits_'+feature],size=7)
    cb.outline.set_linewidth(0.5)
    ax_f.tick_params(length=1,pad=0.2)
    #%
    if feature == 'frequency':
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.3.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.3.pdf",dpi=1500, bbox_inches='tight') 
    elif feature == 'duration':
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S14.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S14.pdf",dpi=1500, bbox_inches='tight') 
    else :
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S15.png",dpi=1500, bbox_inches='tight')
        fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S15.pdf",dpi=1500, bbox_inches='tight') 
    

 






















