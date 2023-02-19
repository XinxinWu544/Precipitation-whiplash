#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:24:51 2023

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
import glob
basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'

datasets_new=['ERA5','MERRA2','JRA-55',
              'CHIRPS',
              'GPCC','REGEN_LongTermStns'] #4 grond-base land only


#%%
global_mean_map={}
for feature in ['frequency']:
    print(feature)
    for ex in ['dry_to_wet','wet_to_dry']:
        
        
        event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_40ensemble_'+ex+'*')[0]).sel(year=slice(1979,2019)).mean('year')
        
        lon=event.lon
        lat=event.lat
        
        cycle_current_event, cycle_lon = add_cyclic_point(event.mean('ensemble'), coord=lon)
        
        global_mean_map.update({'map_'+ex+'_LENS':cycle_current_event})

        ##### CMIP6 
        event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_55ensemble_'+ex+'*')[0]).sel(year=slice(1979,2019)).mean('year')

        cycle_current_event, cycle_lon = add_cyclic_point(event.mean('ensemble'), coord=lon)
        global_mean_map.update({'map_'+ex+'_CMIP6':cycle_current_event})
        
        #### gridded
        for n in range(len(datasets_new)):
            
            event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-1.gridded_dataset_event_frequency_duration_intensity/'+
                                              feature+'_'+datasets_new[n]+'_'+ex+'*')[0])
            event=event.sel(year=slice(event.year[1],event.year[-2])).mean('year')
            if feature=='frequency':
                event=event.where(event>0)
                
            cycle_current_event, cycle_lon = add_cyclic_point(event, coord=lon)
            global_mean_map.update({'map_'+ex+'_'+datasets_new[n]:cycle_current_event})
            
np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/S6-7.global_mean_map_of_datasets.npy',global_mean_map)

