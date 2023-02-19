#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:50:30 2023

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
import tqdm
import multiprocessing
import gc

#basic_dir='E:/research/4.East_Asia/Again/'
basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'
#basic_dir='/scratch/xtan/suk/4.East_Asia/Again/'
basic_dai_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'


threshold=[0.05,0.1,0.15,0.2,0.25,0.75, 0.8,0.85,0.9,0.95]


MIN_period=[10,15,20,25,30,35,40,50,60,90]

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

year=np.array([1920+i for i in range(181)])

rcp85_start=86

#threshold_type = ['time_varying','series_mean','PIC_mean','calendar_mean']

extremes=['wet','dry','dry2wet','wet2dry']

MIN_period_sub=[20,25,30,35,40]

preferred_dims = ('lat', 'lon', 'time') #需要调整为这种维度排序？

threshold_type=['Daily_SPI_proxy','Series_mean','Time_varying','Time_varying_standardized']
result_dir = basic_dir +'code/5-3.Dai_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline/'

n=1
result_dir = basic_dai_dir +'code/5-2.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble/'  #硬盘不够位置了，拷在本机

n=7

all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9))
for m in [20,25,35,40]:
    inter_period=np.ceil(m/2).astype(int)
    all_plans.append((2,'Series_mean',m,0.9))
for q in [0.8,0.95]: 
    all_plans.append((2,'Series_mean',30,q))       
for method in ['Daily_SPI_proxy','Time_varying','Time_varying_standardized']:
    all_plans.append((2,method,30,0.9))   
    

def cal_anom_quantile_Series_mean_without_cum(var):
    global data1

    j=var[0]
    k=var[1]
    cum_prec=data1[:,j,k].rolling(time=min_period,center=False).sum()
    cum_prec=cum_prec.assign_coords({'time':calendar_day})
    day_cycle=pd.Series(cum_prec).groupby(calendar_day).mean()
    sd=pd.Series(cum_prec).groupby(calendar_day).std()
    anom_prec= (cum_prec-np.tile(day_cycle,181))/np.tile(sd,181)
    quan_prec=anom_prec[np.where( (year <= 2019) & (year>=1979) )[0]].quantile(threshold)
    return {'day_cycle':day_cycle,'sd':sd,'quan_prec':quan_prec}

    
#%%

for plan in [1]:
    
    Dtrd_typ=all_plans[plan][0]
    Thes_typ=all_plans[plan][1]
    Min_period=all_plans[plan][2]
    Q=all_plans[plan][3]
   
    
    for n in num:
        
        
        #2. 读入降水数据
        data=xr.open_dataarray('/media/dai/DATA2/CESM-LENS/PRECT/PRECT_'+str(n).zfill(3)+'_interp2deg.nc' )#.sel(lon=slice(10,30) )
    
        print('inputed data:'+str(datetime.now() ))
        
        
        #3. 保存原始日期和calendar日
        original_time=data.time
        month=data['time.month'].values
        day=data['time.day'].values
        year=data['time.year'].values
        calendar_day=[str(month[i]).zfill(2)+'-'+str(day[i]).zfill(2) for i in range(len(month))]
        
        
        
        #1,求不同历时的情况
    
        for dtrd_typ in [Dtrd_typ]: #不去趋势 #linear #二阶趋势 #多模型平均进行缩放
            print('detrend_method='+str(dtrd_typ))
            #dtrd_typ=2
            
            if dtrd_typ == 1:
                data1=data.copy(deep=True)
            if dtrd_typ in [2,3,4]:
                
                
                data_year=xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')
                
                data_year_dtrd=xr.open_dataarray(basic_dir + 'code/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend'+str(dtrd_typ-1)+'.nc')
                data1=data.copy(deep=True).assign_coords({'lat':data_year_dtrd.lat})
                data1 =data1.groupby('time.year')-(data_year-data_year_dtrd) #可能会出现负值但好像不影响
           
                #%%
            print('detrend done ! ') 
            
            min_period=30
            input_combo=[]
            for j in range(data1.shape[1]):
                for k in range(data1.shape[2]):
                    input_combo.append((j,k))
                    
            len_input_combo = []
            for j in range(len(input_combo)):
                len_input_combo.append(j)
                
#%%
    
            start=datetime.now()
            
            pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
            Summary_Stats = list(tqdm.tqdm(pool.imap( cal_anom_quantile_Series_mean_without_cum , input_combo), 
                                           total=len(input_combo), position=0, leave=True))
            pool.close()    
            
            Summary_Stats[0]
            
            np.save('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/'+str(n).zfill(3)+'.npy',Summary_Stats)
#%%
