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

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'

j=79
k=179
examples =((14399,16))
i=14399
#%%
prec = xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/PRECT_001_interp2deg.nc')[:,j,k]
cum_prec = prec.rolling(time=30,center=False).sum()

#%%
n=1

data_year=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')[:,j,k]
data_year_dtrd=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend'+str(1)+'.nc')[:,j,k]
prec1=prec.copy(deep=True).assign_coords({'lat':data_year_dtrd.lat})
prec_detrended =prec1.groupby('time.year')-(data_year-data_year_dtrd) 
cum_prec_detrended = prec_detrended.rolling(time=30,center=False).sum()

#%%
Sd = xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_sd.nc').mean('ensemble')[:,j,k]
Day_cycle = xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_day_cycle.nc').mean('ensemble')[:,j,k]
Quan_prec = xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_quan_prec.nc').mean('ensemble')[:,j,k]

#%%


event = np.load(basic_dir+'code_whiplash/2-2.Original whiplash events/6-4.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble_new_intensity/001/dry_to_wet_Series_mean_lens_detrend_2_of_30_days_quantile_0.9_inter_period_15.npy',allow_pickle= True).tolist()
event_loc = event[i][16]

#%%
feature= ({'prec':prec,
     'cum_prec':cum_prec,
     'cum_prec_detrended':cum_prec_detrended,
     'Day_cycle':Day_cycle,
     'Quan_prec':Quan_prec,
     'Sd':Sd,
     'event_loc':event_loc})
np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/S2.features_of_a_dry_to_wet_example',feature)