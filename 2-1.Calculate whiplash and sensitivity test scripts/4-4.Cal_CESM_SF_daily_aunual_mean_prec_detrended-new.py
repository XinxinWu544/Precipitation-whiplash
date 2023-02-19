#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 13:56:57 2023

@author: dai
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy import signal  
import os
from datetime import datetime
import pandas as pd
import multiprocessing
import tqdm

basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'
dai_basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'


data_year=xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')
data_year_1 = data_year.sel(year=slice(1920,2005))
pfc_1=data_year_1.polyfit('year',1)
data_year_2 = data_year.sel(year=slice(2006,2100))
pfc_2=data_year_2.polyfit('year',1)

data_year_mean = data_year.sel(year=slice(1979,2019)).mean('year')
#%%
forcing=['AER','GHG','BMB']
num_forcing=[20,20,15]    

for x in [0,1,2]:
    for n in range(num_forcing[x]):

        print(n)
        p=xr.open_dataarray('/media/dai/suk_code/research/4.East_Asia/Again/code/4-3.CESM_SF_ensemble_annual_mean_prec/X'+forcing[x]+'_annual_mean_'+str(n).zfill(3)+'.nc')
    
        p_hist = p.sel(year=slice(1920,2005))
        trend_1 = xr.polyval(coord=p_hist['year'], coeffs=pfc_1.polyfit_coefficients)    ##注意这个系数！！！！
        p_detrended_hist = (   (p_hist - trend_1) ) + data_year_mean.values
    
        p_future = p.sel(year=slice(2006,2080))
        trend_2 = xr.polyval(coord=p_future['year'], coeffs=pfc_2.polyfit_coefficients)    ##注意这个系数！！！！
        p_detrended_future =(   (p_future - trend_2) ) + data_year_mean.values
        
        p_detrended = xr.concat([p_detrended_hist,p_detrended_future], dim='year')
    
    
        p_detrended.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-3.CESM_SF_ensemble_annual_mean_prec/X'+forcing[x]+'_annual_mean_'+str(n).zfill(3)+'_detrend1.nc')

