#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:00:26 2023

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

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'

#%%
all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))
for m in [20,25,35,40]:
    #inter_period=np.ceil(m/2).astype(int)
    inter_period = m
    all_plans.append((2,'Series_mean',m,0.9, inter_period   ))
for q in [0.8,0.95]: ##这俩不记得了
    all_plans.append((2,'Series_mean',30,q,30))    
for int_q in [10,15]:
    all_plans.append((2,'Series_mean',30,0.9,int_q))    
for method in ['Daily_SPI_proxy','Time_varying_standardized']:
    all_plans.append((2,method,30,0.9,30))   


datasets_new=['ERA5','MERRA2','JRA-55','CHIRPS',
              'GPCC','REGEN_LongTermStns',] #4 grond-base land only


'''
mask=xr.open_dataarray( basic_dir+  'data/combined_gridded_prcp/GPCC_daily_prcp.nc'  ).sel(time=slice('1979-01-01','2020-12-31') )[0,:,:].values
mask[np.where(mask>=0)]=1
pd.DataFrame(mask).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/land_mask.csv',index=False)
'''

mask=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/land_mask.csv').values
#%% 2、compare different detrend methods
result = pd.DataFrame()
for plan in [0,1]:    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events_counts=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/5-1.CESM_LENS_event_frequency_duration_intensity_sensitivity/frequency_5ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        events_counts = events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
        
        event=events_counts.mean('ensemble') *mask
        lats=event['lat']
        event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        event_region_mean_1=(events_counts.sel(ensemble=1)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_2=(events_counts.sel(ensemble=5)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_3=(events_counts.sel(ensemble=9)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_4=(events_counts.sel(ensemble=13)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_5=(events_counts.sel(ensemble=17)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        result=pd.concat([result,pd.Series(event_region_mean,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_mean')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_1,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_1')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_2,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_2')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_3,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_3')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_4,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_4')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_5,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_5')],axis=1)

        
        
        for n in range(len(datasets_new)):
            
            events_counts=xr.open_dataarray( basic_dir + 'code_whiplash/2-2.Original whiplash events/5-2.gridded_dataset_event_frequency_duration_intensity_sensitivity/frequency_'+datasets_new[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            events_counts=events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
            event=events_counts *mask
            #event=event.sel(lon=slice(70,90),lat=slice(40,30))  #################
            lats=event['lat']
            event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            
            result=pd.concat([result,pd.Series(event_region_mean,index=event.year,name=str(plan)+'_'+ex+'_'+datasets_new[n])],axis=1)


result.to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S3.global_mean_datasets_diff_detrend_methods.csv')
#%% 3、compare different cumulative prec days
result = pd.DataFrame()
for plan in [4,5,1,6,7]:    
    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events_counts=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/5-1.CESM_LENS_event_frequency_duration_intensity_sensitivity/frequency_5ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        events_counts = events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
        
        event=events_counts.mean('ensemble') *mask
        lats=event['lat']
        event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        event_region_mean_1=(events_counts.sel(ensemble=1)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_2=(events_counts.sel(ensemble=5)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_3=(events_counts.sel(ensemble=9)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_4=(events_counts.sel(ensemble=13)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_5=(events_counts.sel(ensemble=17)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        result=pd.concat([result,pd.Series(event_region_mean,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_mean')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_1,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_1')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_2,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_2')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_3,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_3')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_4,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_4')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_5,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_5')],axis=1)

        
        
        for n in range(len(datasets_new)):
            
            events_counts=xr.open_dataarray( basic_dir + 'code_whiplash/2-2.Original whiplash events/5-2.gridded_dataset_event_frequency_duration_intensity_sensitivity/frequency_'+datasets_new[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            events_counts=events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
            event=events_counts *mask
            #event=event.sel(lon=slice(70,90),lat=slice(40,30))  #################
            lats=event['lat']
            event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            
            result=pd.concat([result,pd.Series(event_region_mean,index=event.year,name=str(plan)+'_'+ex+'_'+datasets_new[n])],axis=1)


result.to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S4.global_mean_datasets_diff_cumulative_prec_days.csv')



#%% 4、compare different threshold
result = pd.DataFrame()
for plan in [8,1,9]:    
    
    
    for ex in ['dry_to_wet','wet_to_dry']:
    
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events_counts=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/5-1.CESM_LENS_event_frequency_duration_intensity_sensitivity/frequency_5ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        events_counts = events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
        
        event=events_counts.mean('ensemble') *mask
        lats=event['lat']
        event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        event_region_mean_1=(events_counts.sel(ensemble=1)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_2=(events_counts.sel(ensemble=5)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_3=(events_counts.sel(ensemble=9)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_4=(events_counts.sel(ensemble=13)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        event_region_mean_5=(events_counts.sel(ensemble=17)*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
        
        result=pd.concat([result,pd.Series(event_region_mean,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_mean')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_1,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_1')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_2,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_2')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_3,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_3')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_4,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_4')],axis=1)
        result=pd.concat([result,pd.Series(event_region_mean_5,index=np.arange(1921,2100),name=str(plan)+'_'+ex+'_ensemble_5')],axis=1)

        
        
        for n in range(len(datasets_new)):
            
            events_counts=xr.open_dataarray( basic_dir + 'code_whiplash/2-2.Original whiplash events/5-2.gridded_dataset_event_frequency_duration_intensity_sensitivity/frequency_'+datasets_new[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            events_counts=events_counts.sel(year=slice(events_counts.year[1],events_counts.year[-2]))
            event=events_counts *mask
            #event=event.sel(lon=slice(70,90),lat=slice(40,30))  #################
            lats=event['lat']
            event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            
            result=pd.concat([result,pd.Series(event_region_mean,index=event.year,name=str(plan)+'_'+ex+'_'+datasets_new[n])],axis=1)
result.to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S5.global_mean_datasets_diff_threshold.csv')
