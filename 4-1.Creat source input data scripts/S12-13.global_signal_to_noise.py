#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:43:34 2023

@author: dai
"""

#> 4h ; 48 core 

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
import glob
basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'

#%%
mask_o=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/land_mask.csv')

#%%


def cal_min_ensemble(y):
    #print(y)    
    df_year = df.iloc[y,:]
    
    
    for en_n in range(1,en_num-1):
        #print(en_n)
        sum_n =0 
        for i in range(100000):
            
            subset = df_year.sample(n= (en_num-en_n) )
            
            if np.abs(subset.mean()/subset.std())>1:
                sum_n=sum_n+1
        #print(sum_n)        
        if sum_n<95000 :
            break
    return (en_num-en_n)


#%%
signal_to_noise_stats = {}
for feature in ['frequency']:
    print(feature)
    for ex in ['dry_to_wet','wet_to_dry']:
        
        event_LENS =xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_40ensemble_'+ex+'*')[0]).sel(year=slice(1921,2099) )
        event_CMIP6 =xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_55ensemble_'+ex+'*')[0]).sel(year=slice(1921,2099) )
        mask= xr.DataArray(mask_o,dims=('lat','lon'),coords={'lon':event_LENS.lon,'lat':event_LENS.lat})
#%
         ##计算全球分布
        for ds in ['LENS','CMIP6']:
            
            event_rolling =  vars()['event_'+ds].rolling(year=30,center=True).mean().dropna('year')
            
            event_rolling_change_mean = (event_rolling - event_rolling[0,:,:,:].values).mean('ensemble')
            #event_rolling_change_sd = (event_rolling ).std('ensemble')
            event_rolling_change_sd = (event_rolling - event_rolling[0,:,:,:].values).std('ensemble')
            snr_rolling = (  (event_rolling_change_mean/event_rolling_change_sd >= 1) |  (event_rolling_change_mean/event_rolling_change_sd <= -1)       ).astype(int)
            
            #%
            first_year = np.full(shape= (90,180),fill_value=np.nan)
            num_of_year = np.full(shape= (90,180),fill_value=np.nan)
            
            year0=snr_rolling.year[0].values
            
            for j in range(90):
                for k in range(180):
                    loc = np.where(snr_rolling[:,j,k].values == 1 )[0]
                    loc_0 = np.where(snr_rolling[:,j,k].values == 0 )[0]
                    
                    l =  len(loc)
                    
                    if l >0 :
                        num_of_year[j,k] = l
                        first_year[j,k] = year0 + loc_0[-1]
                        
            #first_year[first_year>=2080]=np.nan            
            
            #first_year1 = first_year.copy()
            #first_year1[first_year1>0] =1
            
            #vars()['first_year_'+ds] = first_year.copy()
            signal_to_noise_stats.update({'first_year_'+ex+'_'+ds:first_year.copy()})
#%
        
            event_global_mean =  vars()['event_'+ds].weighted(np.cos(np.deg2rad( vars()['event_'+ds].lat))).mean(('lon','lat'))
            event_change_global_mean = event_global_mean - event_global_mean[0,:]
            df=pd.DataFrame(event_change_global_mean)
            
            en_num = df.shape[1]
            
            
            pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
            vars()['stats_global_'+ds] = list(tqdm.tqdm(pool.imap( cal_min_ensemble, range(1,179)), 
                                           total=len(range(1,179)), position=0, leave=True))
            pool.close()    
            del(pool)
            gc.collect()
            ########################################
            event_land_mean =  (vars()['event_'+ds]*mask).weighted(np.cos(np.deg2rad( vars()['event_'+ds].lat))).mean(('lon','lat'))
            event_change_land_mean = event_land_mean - event_land_mean[0,:]
            df=pd.DataFrame(event_change_land_mean)
            
            en_num = df.shape[1]
            
            
            pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
            vars()['stats_land_'+ds] = list(tqdm.tqdm(pool.imap( cal_min_ensemble, range(1,179)), 
                                           total=len(range(1,179)), position=0, leave=True))
            pool.close()    
            del(pool)
            gc.collect()
    
    
            event_global_mean =  vars()['event_'+ds].weighted(np.cos(np.deg2rad( vars()['event_'+ds].lat))).mean(('lon','lat'))
            
            vars()['global_mean_'+ds] = (event_global_mean - event_global_mean[0,:]).mean('ensemble')
            vars()['global_sd_'+ds] = (event_global_mean - event_global_mean[0,:]).std('ensemble')
            
            event_land_mean =  (vars()['event_'+ds]*mask).weighted(np.cos(np.deg2rad( vars()['event_'+ds].lat))).mean(('lon','lat'))
            
            vars()['land_mean_'+ds] = (event_land_mean - event_land_mean[0,:]).mean('ensemble')
            vars()['land_sd_'+ds] = (event_land_mean - event_land_mean[0,:]).std('ensemble')
            
            
            signal_to_noise_stats.update({'min_size_global_'+ex+'_'+ds: vars()['stats_global_'+ds] })
            signal_to_noise_stats.update({'min_size_land_'+ex+'_'+ds: vars()['stats_land_'+ds] })
            signal_to_noise_stats.update({'global_mean_'+ex+'_'+ds: vars()['global_mean_'+ds] })
            signal_to_noise_stats.update({'global_sd_'+ex+'_'+ds: vars()['global_sd_'+ds] })
            signal_to_noise_stats.update({'land_mean_'+ex+'_'+ds: vars()['land_mean_'+ds] })
            signal_to_noise_stats.update({'land_sd_'+ex+'_'+ds: vars()['land_sd_'+ds] })
    #%%

        np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/S12-13.global_signal_to_noise.npy',signal_to_noise_stats)
