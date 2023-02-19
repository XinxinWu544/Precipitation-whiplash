#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:41:53 2022

@author: dai
"""

## different detrend method

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy import signal  
import os
from datetime import datetime
import pandas as pd
#os.chdir(r'/media/dai/suk_code/research/4.East_Asia/Again/code')
#from Functions_daily_whiplashes import *
import xarray as xr


basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'

#%%





def detrend_linear_method(var):
    global data_year
    j=var[0]
    k=var[1]
    data_year_detrend=scipy.signal.detrend(data_year[:,j,k], type='linear', overwrite_data=False)    
    data_year_detrended= data_year_detrend + data_year[:,j,k].mean('year').values
    return data_year_detrended


datasets=['CFSR','ERA5','MERRA2','NCEP2','JRA-55','MSWEP_V1.2', #6 reanalysis
'CHIRP','CHIRPS','PERSIANN', #3 satellite
'CPC','GPCC','REGEN_AllStns','REGEN_LongTermStns'] #4 grond-base land only


#%%


for n in range(len(datasets)):
    
    print('dataset='+str(n))
    
    #2. 读入降水数据
    data=xr.open_dataarray( basic_dir+  'data/combined_gridded_prcp/'+datasets[n]+'_daily_prcp.nc'  ).sel(time=slice('1979-01-01','2020-12-31') )
    data_year = data.groupby('time.year').mean()
    print(data['time.year'].values)

     ### method 1 linear trend ### 
    pfc=data_year.polyfit('year',1)
    trend = xr.polyval(coord=data_year['year'], coeffs=pfc.polyfit_coefficients)    
    data_year_detrended_1 = (data_year - trend)+data_year.mean('year')
    
    '''
    plt.plot(range(len(data_year.year)),data_year.mean(('lon','lat')))
    plt.plot(range(len(data_year.year)),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year[:,100,100])
    plt.plot(range(181),data_year_detrended_1[:,100,100])
    '''
    data_year_detrended_1.to_netcdf(basic_dir + 'code/4-2.gridded_dataset_annual_ensemble_mean_prec/PRECT_annual_mean_'+datasets[n]+'_detrend1.nc')

   
    