#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:55:01 2023

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


#%%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))

plan=1
### 计算变化趋势和降水变化趋势的同步变化

global_change = {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
      
        print(feature)
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
       
        e_future = events.sel(year=slice(2060,2099)).mean('year')#.mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year')#.mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~global-period2' :trend  })

        e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~ensemble_mean-period2' :trend  })
        
        ##
        '''
        e_future = events.sel(year=slice(1979,2019)).mean('year')#.mean(('ensemble'))
        e_current = events.sel(year=slice(1921,1960)).mean('year')#.mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~global-period1' :trend  })

        e_future = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1921,1960)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~ensemble_mean-period1' :trend  })
        '''
        
        
prec=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')
prec_future = prec.sel(year=slice(2060,2099)).mean('year')
prec_current = prec.sel(year=slice(1979,2019)).mean('year')
trend = (prec_future - prec_current)*100/prec_current    
global_change.update({ 'prec~global-period2' :trend  })
'''
prec_future = prec.sel(year=slice(1979,2019)).mean('year')
prec_current = prec.sel(year=slice(1921,1960)).mean('year')
trend = (prec_future - prec_current)*100/prec_current    
global_change.update({ 'prec~global-period1' :trend  })
'''

np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/4.prec_and_whiplash_regimes.npy',global_change)
     