#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:26:40 2023

@author: dai
"""


## 对年平均数据进行去趋势
## 去完趋势加回1979-2019平均

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
import multiprocessing
import tqdm
import glob
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

#basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'

basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'


#%%


def detrend_linear_method(var):
    global data_year
    j=var[0]
    k=var[1]
    data_year_detrend=scipy.signal.detrend(data_year[:,j,k], type='linear', bp=86, overwrite_data=False)    
    data_year_detrended= data_year_detrend + data_year[:,j,k].sel(year=slice(1979,2019)).mean('year').values
    return data_year_detrended


#%%
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
        
       
        data_year = xr.open_dataarray('/media/dai/suk_code/research/4.East_Asia/Again/code/4-4.CMIP6_1920_2100_models_annual_mean_prec/PRECT_annual_mean_'+Experiments[n]+'_'+realizations[nn]+'.nc')

        input_combo=[]
        for j in range(data_year.shape[1]):
            for k in range(data_year.shape[2]):
                input_combo.append((j,k))

         ### method 1 去除linear trend ### 
        pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
        Summary_Stats = list(tqdm.tqdm(pool.imap(detrend_linear_method, input_combo), 
                                       total=len(input_combo), position=0, leave=True))
        pool.close()
        
        data_year_detrended_1=data_year.copy(deep=True)
        for i in range(len(input_combo)):
            data_year_detrended_1[:,input_combo[i][0],input_combo[i][1]]=Summary_Stats[i]
           
        data_year_detrended_1.to_netcdf(basic_dir + 'code/4-4.CMIP6_1920_2100_models_annual_mean_prec/PRECT_annual_mean_'+Experiments[n]+'_'+ realizations[nn] +'_detrend1.nc')

       

