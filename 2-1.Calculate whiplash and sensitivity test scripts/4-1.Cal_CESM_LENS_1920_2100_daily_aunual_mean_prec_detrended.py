#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:41:53 2022

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

def detrend_scaling_ensemM_method(var):
    global data_year,prec_ensemble_mean
    j=var[0]
    k=var[1]
    ###进行回归
    poly=np.polyfit(prec_ensemble_mean[:,j,k],data_year[:,j,k],deg=1)
    ###得到缩放值
    z = np.polyval(poly, prec_ensemble_mean[:,j,k])
    ###原始值-缩放之后的值+多年平均
    data_year_detrended=((data_year[:,j,k] - z)+ data_year[:,j,k].sel(year=slice(1979,2019)).mean('year') ).values
    
    return data_year_detrended




#%% #用5个ensemble来说明效果

##notice:  mean prec ensemble numbers
prec_ensemble_mean=xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')

for n in [1,5,9,13,17]:
    
    print('ensemble='+str(n))
    data_year = xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')

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
       
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year[:,100,100])
    plt.plot(range(181),data_year_detrended_1[:,100,100])
    '''
    data_year_detrended_1.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend1.nc')

   
    
     ### method 2 polyfit trend ### 
    #求二项式拟合的趋势
    pfc=data_year.polyfit('year',2)
    trend = xr.polyval(coord=data_year['year'], coeffs=pfc.polyfit_coefficients)    

    data_year_detrended_2 = (data_year - trend)+data_year.sel(year=slice(1979,2019)).mean('year')
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_2.mean(('lon','lat')))
    '''
    data_year_detrended_2.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend2.nc')

    
    ### method 3 minus ensemble mean ### 
    pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
    Summary_Stats = list(tqdm.tqdm(pool.imap(detrend_scaling_ensemM_method, input_combo), 
                                   total=len(input_combo), position=0, leave=True))
    pool.close()
    
    data_year_detrended_3=data_year.copy(deep=True)
    for i in range(len(input_combo)):
        data_year_detrended_3[:,input_combo[i][0],input_combo[i][1]]=Summary_Stats[i] 
    
    
    
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_2.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_3.mean(('lon','lat')))
    '''
    
    data_year_detrended_3.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend3.nc')


#%% 只有线性去趋势运用到整个40个集合中

for n in num:
    
    print('ensemble='+str(n))
    data_year = xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')

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
       
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year[:,100,100])
    plt.plot(range(181),data_year_detrended_1[:,100,100])
    '''
    data_year_detrended_1.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend1.nc')

#%%

prec_ensemble_mean=xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')

for n in range(101,106):
    
    print('ensemble='+str(n))
    data_year = xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')

    input_combo=[]
    for j in range(data_year.shape[1]):
        for k in range(data_year.shape[2]):
            input_combo.append((j,k))

   
    
     ### method 2 polyfit trend ### 
    #求二项式拟合的趋势
    pfc=data_year.polyfit('year',2)
    trend = xr.polyval(coord=data_year['year'], coeffs=pfc.polyfit_coefficients)    

    data_year_detrended_2 = (data_year - trend)+data_year.sel(year=slice(1979,2019)).mean('year')
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_2.mean(('lon','lat')))
    '''
    data_year_detrended_2.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend2.nc')

    
    ### method 3 minus ensemble mean ### 
    pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
    Summary_Stats = list(tqdm.tqdm(pool.imap(detrend_scaling_ensemM_method, input_combo), 
                                   total=len(input_combo), position=0, leave=True))
    pool.close()
    
    data_year_detrended_3=data_year.copy(deep=True)
    for i in range(len(input_combo)):
        data_year_detrended_3[:,input_combo[i][0],input_combo[i][1]]=Summary_Stats[i] 
    
    
    
    '''
    plt.plot(range(181),data_year.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_1.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_2.mean(('lon','lat')))
    plt.plot(range(181),data_year_detrended_3.mean(('lon','lat')))
    '''
    
    data_year_detrended_3.to_netcdf(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend3.nc')
