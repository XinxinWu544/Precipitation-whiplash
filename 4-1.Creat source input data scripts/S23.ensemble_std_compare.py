#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:44:58 2023

@author: dai
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy
from scipy import signal  
from scipy import stats
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
import glob
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib
col=["#FFFFE4","#FEF4CF","#FEEDB0", "#FBCF93", "#F7B07A", "#F19164" ,"#E97356" ,"#C96775","#87518E"] 
#%%
basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'
names_cmip6= glob.glob(basic_dir + 'code_whiplash/2-2.Original whiplash events/4-4.CMIP6_1920_2100_models_annual_mean_prec/*f1.nc')

n=[]
for i in range(len(names_cmip6)):
    if 'r1i1p1' in names_cmip6[i]:
        n.append(i)
        
ensemble_std={}        
for ex in ['dry_to_wet','wet_to_dry']:
    lens = xr.open_dataarray(basic_dir+'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/frequency_40ensemble_'+ex+'_Series_mean_lens_detrend_2_of_30_days_quantile_0.9_inter_period_30.nc')
    cmip6 = xr.open_dataarray(basic_dir+'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/frequency_55ensemble_'+ex+'_Series_mean_lens_detrend_2_of_30_days_quantile_0.9_inter_period_30.nc')
    cmip6=cmip6[:,:,:,n]
    #%%
    lens_current = lens.sel(year=slice(1979,2019))
    cmip6_current = cmip6.sel(year=slice(1979,2019))
    lens_slope = lens_current.polyfit('year',deg = 1)['polyfit_coefficients'][0,:,:,:]
    cmip6_slope = cmip6_current.polyfit('year',deg = 1)['polyfit_coefficients'][0,:,:,:]
    
    
    f_test_current = np.zeros(shape=(90,180))
    for i in range(90):
        #print(i)
        for j in range(180):        
            f_test_current[i,j] = scipy.stats.levene(lens_slope[i,j,:].values , cmip6_slope[i,j,:].values)[1]
    
    np.sum(f_test_current<0.05)
    
    vars()['current_sd_lens_'+ex] = lens_slope.std('ensemble')
    vars()['current_sd_cmip6_'+ex] = cmip6_slope.std('ensemble')
    
    sig_current  = (f_test_current<0.05).astype(float)
    sig_current[sig_current==0]=np.nan
    
    vars()['sig_current_'+ex]=sig_current
    
    vars()['current_ratio_'+ex] = (vars()['current_sd_lens_'+ex]/vars()['current_sd_cmip6_'+ex]).values
    print('0.5')
    print((vars()['current_ratio_'+ex]>0.5).sum()/16200)
    print('0.75')
    print((vars()['current_ratio_'+ex]>0.75).sum()/16200)
    #%%
    lens_future = lens.sel(year=slice(2060,2099))
    cmip6_future = cmip6.sel(year=slice(2060,2099))
    lens_slope = lens_future.polyfit('year',deg = 1)['polyfit_coefficients'][0,:,:,:]
    cmip6_slope = cmip6_future.polyfit('year',deg = 1)['polyfit_coefficients'][0,:,:,:]
    
    
    f_test_future = np.zeros(shape=(90,180))
    for i in range(90):
        #print(i)
        for j in range(180):        
            f_test_future[i,j] = scipy.stats.levene(lens_slope[i,j,:].values , cmip6_slope[i,j,:].values)[1]
    
    np.sum(f_test_future<0.05)
    
    #future_sd_lens = lens_slope.std('ensemble')
    #future_sd_cmip6 = cmip6_slope.std('ensemble')
    vars()['future_sd_lens_'+ex] = lens_slope.std('ensemble')
    vars()['future_sd_cmip6_'+ex] = cmip6_slope.std('ensemble')
    vars()['future_ratio_'+ex] = (vars()['future_sd_lens_'+ex]/vars()['future_sd_cmip6_'+ex]).values
    
    sig_future  = (f_test_future<0.05).astype(float)
    sig_future[sig_future==0]=np.nan
    x=lens_slope.lon
    y=lens_slope.lat
    '''
    x=future_sd_lens.lon
    y=future_sd_lens.lat
    x=future_sd_lens.lon[np.where(sig_future==1)[1]]
    y=future_sd_lens.lat[np.where(sig_future==1)[0]]
    '''
    vars()['sig_future_'+ex]=sig_future
    
    print('0.5')
    print((vars()['future_ratio_'+ex]>0.5).sum()/16200)
    print('0.75')
    print((vars()['future_ratio_'+ex]>0.75).sum()/16200)
    
    
    ensemble_std.update({'sig_current_'+ex:vars()['sig_current_'+ex]})
    ensemble_std.update({'current_sd_lens_'+ex:vars()['current_sd_lens_'+ex]})
    ensemble_std.update({'current_sd_cmip6_'+ex:vars()['current_sd_cmip6_'+ex]})
    ensemble_std.update({'sig_future_'+ex:vars()['sig_future_'+ex]})
    ensemble_std.update({'future_sd_lens_'+ex:vars()['future_sd_lens_'+ex]})
    ensemble_std.update({'future_sd_cmip6_'+ex:vars()['future_sd_cmip6_'+ex]})
    
    #%%
np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/S23.ensemble_std_compare.npy',ensemble_std)   