# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:43:32 2022

@author: daisukiiiii
"""
import cftime
import xarray as xr
import pandas as pd
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import glob
import gc
import multiprocessing
import tqdm
print('Analysis started in:', datetime.now() )


v_name='Z'
#nn=200

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]
print(num)
#%%


#%%
for nn in [500]:
    
    print('level='+str(nn)+' at time:',datetime.now())
    
    variable=v_name+str(nn)
    
    result_loc_detrend='/media/dai/disk2/suk/Data/CESM-LENS/Z500_detrended/'
    result_loc='/media/dai/disk2/suk/Data/CESM-LENS/Z500/'
    
    
    Path(result_loc_detrend).mkdir(parents=True, exist_ok=True) ##创建文件
    Path(result_loc).mkdir(parents=True, exist_ok=True) ##创建文件
    
    
    data_loc='/media/dai/DATA3/CESM-LENS/Z500/'
    os.chdir(data_loc)
    
    
    
    
    

    for n in num:
        
      
        
        #但是按理说这个方法才是对的，因为有些数据集好像2006-01-01重复了
        ############
        print('ensemble hist~')
        if n==1:
            
            d1=xr.open_dataset('b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.18500101-20051231.nc').sel(
              
                time=slice('1920-01-01','2005-12-31'))[variable]
        else:
            d1=xr.open_dataset('b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.19200101-20051231.nc').sel(
               
                time=slice('1920-01-01','2005-12-31'))[variable]
        
        #############
        print('ensemble future~')
        if n<=33:
            d2=xr.open_dataset('b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20060101-20801231.nc').sel(
              
                time=slice('2006-01-01','2080-12-31'))[variable]
            
            d3=xr.open_dataset('b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20810101-21001231.nc').sel(
                
                time=slice('2081-01-01','2100-12-31'))[variable]
            
            Daily=xr.concat([d1,d2,d3],dim='time')
            
            del(d1);del(d2);del(d3)
        else:
            d2=xr.open_dataset('b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20060101-21001231.nc').sel(
               
                time=slice('2006-01-01','2100-12-31'))[variable]
            
            Daily=xr.concat([d1,d2],dim='time')
            
            del(d1);del(d2)
        
    
        
        
        
        ######################################################################################################################
        
        
        ###去趋势
      
        b=Daily.groupby('time.year').mean(dim='time')
        
        pfc=b.polyfit('year',1)
        trend = xr.polyval(coord=b['year'], coeffs=pfc.polyfit_coefficients)    
        #trend1 = trend-trend.sel(year=slice(1920,1920)).mean('year') +b.sel(year=slice(1920,1920)).mean('year')
        data_year_detrended_2 = (b - trend)+trend.sel(year=slice(1920,1920)).mean('year')
        '''
        plt.plot(range(181),b[:,1,1])
        plt.plot(range(181),trend[:,1,1])
        #plt.plot(range(181),trend1[:,1,1])
        plt.plot(range(181),data_year_detrended_2[:,1,1])
        plt.plot(range(181),Daily_detrended[:,1,1].groupby('time.year').mean(dim='time'))
        '''
        
        Daily_detrended = Daily.groupby('time.year') - (b-data_year_detrended_2)
        
        
        '''
        slope=np.zeros(shape=(b.values.shape[1],b.values.shape[2]))
        for s1 in range(0,b.values.shape[1]):
            for s2 in range(0,b.values.shape[2]):
                slope[s1,s2]=np.polyfit(b['year'],b.values[:,s1,s2],1)[0]

        for s1 in range(0,b.values.shape[1]):
            
            for s2 in range(0,b.values.shape[2]):
                #print('detrend_s1='+str(s1)+' and s2='+str(s2)+' at time:',datetime.now())   
                Daily_detrended[:,s1,s2]=Daily[:,s1,s2]-(Daily.time.dt.year-1920)*slope[s1,s2]
                
        
        input_combo=[]
        for j in range(Daily_detrended.shape[1]):
            for k in range(Daily_detrended.shape[2]):
                input_combo.append((j,k))
                
        def cal_detrended_z500(var):
            global Daily
            j=var[0]
            k=var[1]
            slope=np.polyfit(b['year'],b[:,j,k],1)[0]
            daily_detrended = Daily[:,j,k]-(Daily.time.dt.year-1920)*slope
            return daily_detrended
        
        start=datetime.now()
        
        pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
        Summary_Stats = list(tqdm.tqdm(pool.imap( cal_detrended_z500 , input_combo), 
                                       total=len(input_combo), position=0, leave=True))
        pool.close()    
        del(pool)
        gc.collect()
        
        end=datetime.now()
        
        for j in range(len(input_combo)):
            Daily_detrended[:,input_combo[j][0],input_combo[j][1] ] = Summary_Stats[j].values
            
        '''        
        
        
                
                
        
        Daily=Daily.reindex(lat=list(reversed(Daily.lat)))
        Daily.to_netcdf(result_loc+'Z500_'+str(n).zfill(3) +'.nc')
        
        Daily_detrended=Daily_detrended.reindex(lat=list(reversed(Daily_detrended.lat)))
        Daily_detrended.to_netcdf(result_loc_detrend+'Z500_detrend_'+str(n).zfill(3) +'.nc')
        
        del(Daily_detrended);del(Daily)
        gc.collect()
        
