#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:18:20 2023

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


basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
monsoon_regions = xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/monsoon_regions_2deg_mask/monsoon_regions.nc')

#%%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))

plan=1
monsoon_name = ['WAfriM','SAsiaM','SAmerM','NAmerM','EAsiaM','AusMCM']


color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']
colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']

color_shp =["#F6A975" ,"#F08E63", "#E97356", "#DE5953" ,"#CD4257", "#B8315D"]


#%%
running_year = 10

regional_mean= {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        for ms in range(6) :
            print(ms)
            
            
            dtrd_typ=all_plans[plan][0]
            thes_typ=all_plans[plan][1]
            min_period=all_plans[plan][2]
            q=all_plans[plan][3]
            inter_period=all_plans[plan][4]
            
            
            ## CESM-LENS
            events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                    '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            # 1.不同地区 
            reg = monsoon_regions[:,:,ms]
            event_m = events * reg
            lats=event_m.lat
           
            vars()[ex+'_region'] = event_m.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099)).rolling(year=running_year,center=True,min_periods=1).mean().sel( year=slice(1921+ running_year/2, 2099 -running_year/2 +1 ))
            
            vars()[ex+'_region_mean'] = (  (vars()[ex+'_region'] - vars()[ex+'_region'][0,:])*100/vars()[ex+'_region'][0,:]  ).mean('ensemble')
            vars()[ex+'_region_sd'] = (  (vars()[ex+'_region'] - vars()[ex+'_region'][0,:])*100/vars()[ex+'_region'][0,:]  ).std('ensemble')
            
            
            #a=event_m.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099)).values
            
            a = pd.concat([pd.Series(-vars()[ex+'_region_sd']),
                       pd.Series(vars()[ex+'_region_sd']),
                       pd.Series(vars()[ex+'_region_mean'])],axis=1)
            a.index=vars()[ex+'_region'].year.values
            regional_mean.update({ 'LENS'+'~'+ex+'~'+feature+'~'+monsoon_name[ms] :a  })
            
            
            
            ## CMIP6
            events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+feature+'_55ensemble'+
                                    '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            # 1.不同地区 
            reg = monsoon_regions[:,:,ms]
            event_m = events * reg
            lats=event_m.lat
           
            vars()[ex+'_region'] = event_m.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099)).rolling(year=running_year,center=True,min_periods=1).mean().sel( year=slice(1921+ running_year/2, 2099 -running_year/2 +1 ))
            vars()[ex+'_region_mean'] = (  (vars()[ex+'_region'] - vars()[ex+'_region'][0,:])*100/vars()[ex+'_region'][0,:]  ).mean('ensemble')
            vars()[ex+'_region_sd'] = (  (vars()[ex+'_region'] - vars()[ex+'_region'][0,:])*100/vars()[ex+'_region'][0,:]  ).std('ensemble')
            
            
            a = pd.concat([pd.Series(-vars()[ex+'_region_sd']),
                       pd.Series(vars()[ex+'_region_sd']),
                       pd.Series(vars()[ex+'_region_mean'])],axis=1)
            a.index=vars()[ex+'_region'].year.values
            regional_mean.update({ 'CMIP6'+'~'+ex+'~'+feature+'~'+monsoon_name[ms] :a  })
            
#%%

np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/3_S14-15.monsoon_regional_mean_trend_and_SNR.npy',regional_mean)

#%%


regional_change = {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
      
            
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
       
        e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        
        #reg = monsoon_regions[:,:,ms]
        #event_m = trend * reg
        regional_change.update({ 'LENS~'+ex+'~'+feature+'~global' :trend  })
        #regional_change.update({ ex+'~'+feature+'~'+monsoon_name[ms] :event_m  })
        
        ## CMIP6
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+feature+'_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
       
        e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        
        #reg = monsoon_regions[:,:,ms]
        #event_m = trend * reg
        regional_change.update({ 'CMIP6~'+ex+'~'+feature+'~global' :trend  })
            
np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/3_S14-15.global_change_distribution_bottom_map.npy',regional_change)
     