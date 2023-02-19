#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:04:06 2023

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
import glob 
#%%

Experiments=['ACCESS-ESM1-5','CanESM5','CESM2-WACCM','CMCC-CM2-SR5',
             'CMCC-ESM2','EC-Earth3','EC-Earth3-CC','EC-Earth3-Veg',
             'EC-Earth3-Veg-LR','GFDL-CM4','GFDL-ESM4','INM-CM4-8',
             'INM-CM5-0','IPSL-CM6A-LR','KIOST-ESM','MIROC6',
             'MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM',
             'NorESM2-MM','TaiESM1']


#%%
for n in range(len(Experiments)):
    
    name=np.sort(glob.glob('/media/dai/DATA1/pr/1-COMBINED/pr_day_'+Experiments[n]+'_*'))
    # 查看realization数目
    realizations=np.unique(np.array([name[i].split('_')[len(name[i].split('_'))-1].split('.')[0] for i in range(len(name)) ]))
    for nn in range(len(realizations)):
        print(Experiments[n]+'_'+realizations[nn])
        
        data=xr.open_dataarray('/media/dai/DATA1/pr/1-COMBINED/pr_day_'+Experiments[n]+'_'+realizations[nn]+'.nc')
        data=data.chunk({'lon':20,'lat':20}).groupby('time.year').mean().compute()
        
        data.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-4.CMIP6_1920_2100_models_annual_mean_prec/PRECT_annual_mean_'+Experiments[n]+'_'+realizations[nn]+'.nc')

    