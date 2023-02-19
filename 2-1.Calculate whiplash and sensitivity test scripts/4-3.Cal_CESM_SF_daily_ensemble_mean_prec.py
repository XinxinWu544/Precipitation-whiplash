#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 22:33:25 2022

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
#%%
for n in range(20):
    print(n)
    #1 读入数据
    data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/XAER/XAER_'+str(n).zfill(3)+'_interp2deg.nc')
    data=data.chunk({'lon':20,'lat':20}).groupby('time.year').mean().compute()
    data.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-3.CESM_SF_ensemble_annual_mean_prec/XAER_annual_mean_'+str(n).zfill(3)+'.nc')

for n in range(20):
    print(n)
    #1 读入数据
    data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/XGHG/XGHG_'+str(n).zfill(3)+'_interp2deg.nc')
    data=data.chunk({'lon':20,'lat':20}).groupby('time.year').mean().compute()
    data.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-3.CESM_SF_ensemble_annual_mean_prec/XGHG_annual_mean_'+str(n).zfill(3)+'.nc')

for n in range(15):
    print(n)
    #1 读入数据
    data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/XBMB/XBMB_'+str(n).zfill(3)+'_interp2deg.nc')
    data=data.chunk({'lon':20,'lat':20}).groupby('time.year').mean().compute()
    data.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-3.CESM_SF_ensemble_annual_mean_prec/XBMB_annual_mean_'+str(n).zfill(3)+'.nc')
