# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:45:49 2022

@author: daisukiiiii
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
#basic_dir='E:/research/4.East_Asia/Again/'
#basic_dir='/media/dai/Elements/research/4.East_Asia/Again/'
#basic_dir='/scratch/xtan/suk/4.East_Asia/Again/'

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]


#%%  求多模型平均
n=2
data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/PRECT_'+str(n).zfill(3)+'_interp2deg.nc',chunks=({'lon':20,'lat':20}))

lon=data.lon.values
lat=data.lat.values

DATA=data.groupby('time.year').mean().compute()
DATA=DATA-DATA

for n in num:
    print(n)
    #1 读入数据
    data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/PRECT_'+str(n).zfill(3)+'_interp2deg.nc')
    data=data.assign_coords({'lon':lon,'lat':lat})
    data=data.chunk({'lon':20,'lat':20}).groupby('time.year').mean().compute()
    data.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')

    DATA=DATA+data
    
DATA = DATA/40


#prec_ensemble_mean=DATA.groupby('time.year').mean().compute()
DATA.to_netcdf('/media/dai/suk_code/research/4.East_Asia/Again/code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')

