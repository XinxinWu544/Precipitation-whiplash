#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:36:49 2023

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


#%%
all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))
for m in [20,25,35,40]:
    #inter_period=np.ceil(m/2).astype(int)
    inter_period = m
    all_plans.append((2,'Series_mean',m,0.9, inter_period   ))
for q in [0.8,0.95]:
    all_plans.append((2,'Series_mean',30,q,30))    


datasets_new=['ERA5','MERRA2','JRA-55','CHIRPS',
              'GPCC','REGEN_LongTermStns',] #4 grond-base land only

#%%

diff_detrend = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S3.global_mean_datasets_diff_detrend_methods.csv',index_col=(0))
diff_cumulative = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S4.global_mean_datasets_diff_cumulative_prec_days.csv',index_col=(0))
diff_threshold = pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S5.global_mean_datasets_diff_threshold.csv',index_col=(0))

#%%

fig = plt.figure(figsize = (17/2.54, 7/2.54)) 

p_num=0
for plan in [0,1]:    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    #for ex in ['dry','wet']: 
        p_num=p_num+1
        ax = plt.subplot(2,2,p_num)
        
        result=diff_detrend
        
        years=result.index
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_1'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_2'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_3'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_4'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_5'],linewidth=0.5,color='grey',alpha=0.3)
        
        ax.plot(years ,result[str(plan)+'_'+ex+'_ensemble_mean'],label='CESM-LENS',linewidth=1)
        
        event_region_mean_baseline = result[str(plan)+'_'+ex+'_ensemble_mean'].loc[1979:2019].mean()
        ax.hlines(y=event_region_mean_baseline,xmin=1920,xmax=2100,linestyle='--',linewidth=1)
        ax.axvspan(xmin=1979, xmax=2019,alpha=0.2) 
        
        
        for n in range(len(datasets_new)):
            
            data = result[str(plan)+'_'+ex+'_'+datasets_new[n]].dropna()
            
            
            ax.plot(data.index,data,label=datasets_new[n],linewidth=0.7)
        #plt.legend(loc = "upper left")
        
        
        xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
        xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
        if p_num in [1,2]:
            ymajorLocator = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
        else:   
            ymajorLocator = MultipleLocator(0.2) #将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
        #设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ymajorFormatter)

        #修改次刻度
        xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
        yminorLocator = MultipleLocator(0.05) #将此y轴次刻度标签设置为0.1的倍数
        #设置次刻度标签的位置,没有标签文本格式
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
    
        if p_num in [1,2]:
            ax.axes.xaxis.set_visible(False)
        else:
            ax.set_xlabel('Year',fontsize=6.5,labelpad=1)
     
        if p_num in [1,3]:
            ax.set_ylabel('Occurence (times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)
            ax.annotate("Dry-to-wet", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        else:
            ax.annotate("Wet-to-dry", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        if p_num in [1]:    
            ax.set_title('(a) Original',  fontweight='bold',
                             pad=2, size=7, loc='left')
        if p_num in [3]:    
            ax.set_title('(b) Linear detrending',  fontweight='bold',
                             pad=2, size=7, loc='left')
            

        ax.margins(x=0)
plt.subplots_adjust(wspace=0.12,hspace=0.15)        
       
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.88, 0.05),ncol=7,columnspacing=0.4, labelspacing=0.3,framealpha=0,fontsize=6.5)

fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S3.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S3.pdf",dpi=1500, bbox_inches='tight')

#%%
fig = plt.figure(figsize = (17/2.54, 15/2.54)) # 宽、高

p_num=0
for plan in [4,5,1,6,7]:    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    #for ex in ['dry','wet']: 
        p_num=p_num+1
        ax = plt.subplot(5,2,p_num)
        
        result=diff_cumulative
        
        years=result.index
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_1'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_2'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_3'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_4'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_5'],linewidth=0.5,color='grey',alpha=0.3)
        
        ax.plot(years ,result[str(plan)+'_'+ex+'_ensemble_mean'],label='CESM-LENS',linewidth=1)
        
        event_region_mean_baseline = result[str(plan)+'_'+ex+'_ensemble_mean'].loc[1979:2019].mean()
        ax.hlines(y=event_region_mean_baseline,xmin=1920,xmax=2100,linestyle='--',linewidth=1)
        ax.axvspan(xmin=1979, xmax=2019,alpha=0.2) 
        
        
        for n in range(len(datasets_new)):
            
            data = result[str(plan)+'_'+ex+'_'+datasets_new[n]].dropna()

            ax.plot(data.index,data,label=datasets_new[n],linewidth=0.7)
        
        
        xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
        xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
        ymajorLocator = MultipleLocator(0.2) #将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
        #设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ymajorFormatter)

        #修改次刻度
        xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
        yminorLocator = MultipleLocator(0.1) #将此y轴次刻度标签设置为0.1的倍数
        #设置次刻度标签的位置,没有标签文本格式
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
    
        if p_num in [1,2,3,4,5,6,7,8]:
            ax.axes.xaxis.set_visible(False)
        else:
            ax.set_xlabel('Year',fontsize=6.5,labelpad=1)

        yticks=[0,0.2,0.4,0.6,0.8]
        ax.set_yticks(yticks)
        if p_num in [1,3,5,7,9]:
            
            ax.set_ylabel('Occurence\n(times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)
            ax.annotate("Dry-to-wet", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        else:
            ax.annotate("Wet-to-dry", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        if p_num in [1]:    
            ax.set_title('(a) 20 days',  fontweight='bold',
                             pad=2, size=7, loc='left')
        if p_num in [3]:    
            ax.set_title('(b) 25 days',  fontweight='bold',
                             pad=2, size=7, loc='left')
            
        if p_num in [5]:    
            ax.set_title('(c) 30 days',  fontweight='bold',
                             pad=2, size=7, loc='left')
        if p_num in [7]:    
            ax.set_title('(d) 35 days',  fontweight='bold',
                             pad=2, size=7, loc='left')
        if p_num in [9]:    
            ax.set_title('(e) 40 days',  fontweight='bold',
                             pad=2, size=7, loc='left')
            

        ax.margins(x=0)
plt.subplots_adjust(wspace=0.12,hspace=0.17)        
       
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.88, 0.08),ncol=7,columnspacing=0.4, labelspacing=0.3,framealpha=0,fontsize=6.5)

fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S4.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S4.pdf",dpi=1500, bbox_inches='tight')

#%%
fig = plt.figure(figsize = (17/2.54, 9/2.54)) # 宽、高

p_num=0
for plan in [8,1,9]:    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    #for ex in ['dry','wet']: 
        p_num=p_num+1
        ax = plt.subplot(3,2,p_num)
        
        result=diff_threshold
        
        years=result.index
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_1'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_2'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_3'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_4'],linewidth=0.5,color='grey',alpha=0.3)
        ax.plot(years, result[str(plan)+'_'+ex+'_ensemble_5'],linewidth=0.5,color='grey',alpha=0.3)
        
        ax.plot(years ,result[str(plan)+'_'+ex+'_ensemble_mean'],label='CESM-LENS',linewidth=1)
        
        event_region_mean_baseline = result[str(plan)+'_'+ex+'_ensemble_mean'].loc[1979:2019].mean()
        ax.hlines(y=event_region_mean_baseline,xmin=1920,xmax=2100,linestyle='--',linewidth=1)
        ax.axvspan(xmin=1979, xmax=2019,alpha=0.2) 
        
        
        for n in range(len(datasets_new)):
            
            data = result[str(plan)+'_'+ex+'_'+datasets_new[n]].dropna()

            ax.plot(data.index,data,label=datasets_new[n],linewidth=0.7)
        
        
        xmajorLocator = MultipleLocator(50) #将x主刻度标签设置为20的倍数
        xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
        if p_num in [1,2,3,4]:
            ymajorLocator = MultipleLocator(0.2) #将y轴主刻度标签设置为0.5的倍数
        else:
            ymajorLocator = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
        #设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ymajorFormatter)

        #修改次刻度
        xminorLocator = MultipleLocator(10) #将x轴次刻度标签设置为5的倍数
        yminorLocator = MultipleLocator(0.1) #将此y轴次刻度标签设置为0.1的倍数
        #设置次刻度标签的位置,没有标签文本格式
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5 )
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )
    
        if p_num in [1,2,3,4]:
            ax.axes.xaxis.set_visible(False)
        else:
            ax.set_xlabel('Year',fontsize=6.5,labelpad=1)
      
        #yticks=[0,0.2,0.4]
        #ax.set_yticks(yticks)
        if p_num in [1,3,5]:
            
            ax.set_ylabel('Occurence\n(times '+u'$yr^{-1}$)',fontsize=6.5,labelpad=1)
            ax.annotate("Dry-to-wet", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        else:
            ax.annotate("Wet-to-dry", xy=(0.02, 0.88), xycoords="axes fraction",c='grey',size=6)
        if p_num in [1]:    
            ax.set_title('(a) 80%',  fontweight='bold',
                             pad=2, size=7, loc='left')
        if p_num in [3]:    
            ax.set_title('(b) 90%',  fontweight='bold',
                             pad=2, size=7, loc='left')
            
        if p_num in [5]:    
            ax.set_title('(c) 95%',  fontweight='bold',
                             pad=2, size=7, loc='left')
   
            
        ax.margins(x=0)
        
plt.subplots_adjust(wspace=0.12,hspace=0.17)        
       
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.88, 0.07),ncol=7,columnspacing=0.4, labelspacing=0.3,framealpha=0,fontsize=6.5)

fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S5.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S5.pdf",dpi=1500, bbox_inches='tight')
