#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 00:28:55 2023

@author: dai
"""
#需要128G
import numpy as np
import xarray as xr
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

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/code_whiplash/Demo/'

#%%
data=xr.open_dataarray(basic_dir+'1.Prcp data/PRECT_annual_mean_001.nc')

lat=data.lat.values

input_combo=[]
for j in range(data.shape[1]):
    for k in range(data.shape[2]):
        input_combo.append((j,k))

area_weight=[]
for j in range(len(input_combo)):
    area_weight.append( np.cos(np.pi* lat[input_combo[j][0]] /180) )

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]


#%%
year=np.repeat(range(1920,2101),365)
current_start = np.where( (year>=1979) & (year<=2019) )[0][0]
current_end = np.where( (year>=1979) & (year<=2019) )[0][-1]

year=np.repeat(range(1920,2101),365)
future_start = np.where( (year>=2060) & (year<=2099) )[0][0]
future_end = np.where( (year>=2060) & (year<=2099) )[0][-1]

#%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))

num=[1,2]
#%%

def cal_event_time(i):
    future_time=[]
    current_time=[]
    for n in num:
       
        #print(n)
        #print(n)
        method_dir=basic_dir +'2-2.Original whiplash events/6-4.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble_new_intensity/' +str(n).zfill(3)+'/'
        
        

        event=np.load(method_dir+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
            '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.npy',allow_pickle=True).tolist()
        
        
        #i=16147
        b=event[i][:,0]
        b_current = b[np.where((b <= current_end) & (b>=current_start) )]
        b_future = b[np.where((b <= future_end) & (b>=future_start) )]
        #time.extend(b)
        future_time.extend(b_future)
        current_time.extend(b_current)
    return {'future time':future_time,'current time':current_time}






#%%
plan=1

for plan in [1]:
    
    dtrd_typ=all_plans[plan][0]
    thes_typ=all_plans[plan][1]
    min_period=all_plans[plan][2]
    q=all_plans[plan][3]
    inter_period=all_plans[plan][4]
    
    
    for ex in ['dry_to_wet','wet_to_dry']  :
        print(ex)
        #ex='dry_to_wet'
        

        pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
        Time_Stats = list(tqdm.tqdm(pool.imap( cal_event_time, range(len(input_combo))), 
                                       total=len(input_combo), position=0, leave=True))
        pool.close()     
            
       
        events_current_time=xr.DataArray(np.full(shape=(3,len(data.lat),len(data.lon)),fill_value=np.nan),
                     dims=('index','lat','lon'),
                     coords=({'lon':data.lon,'lat':data.lat,
                              'index':['simple_mean','cal_by_angle','concentration']}))
        
        events_future_time=xr.DataArray(np.full(shape=(3,len(data.lat),len(data.lon)),fill_value=np.nan),
                     dims=('index','lat','lon'),
                     coords=({'lon':data.lon,'lat':data.lat,
                              'index':['simple_mean','cal_by_angle','concentration']}))
        
        
        for i in range(len(input_combo)):
            print(i)
           
            ############################### current ##############################
            t_c = np.array(Time_Stats[i].get('current time'))
            
            events_current_time[0,input_combo[i][0], input_combo[i][1]] = (t_c %365).mean()
            
            theta=( (t_c%365) *2*np.pi)/365
            x_mean = np.cos(theta).mean()
            y_mean = np.sin(theta).mean()
            
            concentration = np.sqrt( x_mean**2 + y_mean**2 ) 
            
            if (x_mean >0) & (y_mean >= 0):
                d_mean = np.arctan(y_mean / x_mean ) * 365/(2*np.pi)
            elif x_mean <=0 :
                d_mean = (np.arctan(y_mean / x_mean )  + np.pi) * 365/(2*np.pi)
            elif (x_mean > 0) & (y_mean < 0):
                d_mean = (np.arctan(y_mean / x_mean) + 2*np.pi) * 365/(2*np.pi)             
            
            events_current_time[1,input_combo[i][0], input_combo[i][1]] = d_mean
            events_current_time[2,input_combo[i][0], input_combo[i][1]] = concentration
        
            ############################### future ##############################
            t_f = np.array(Time_Stats[i].get('future time'))
            
            events_future_time[0,input_combo[i][0], input_combo[i][1]] = (t_f %365).mean()
            
            theta=( (t_f%365) *2*np.pi)/365
            x_mean = np.cos(theta).mean()
            y_mean = np.sin(theta).mean()
            
            concentration = np.sqrt( x_mean**2 + y_mean**2 ) 
            
            if (x_mean >0) & (y_mean >= 0):
                d_mean = np.arctan(y_mean / x_mean ) * 365/(2*np.pi)
            elif x_mean <=0 :
                d_mean = (np.arctan(y_mean / x_mean )  + np.pi) * 365/(2*np.pi)
            elif (x_mean > 0) & (y_mean < 0):
                d_mean = (np.arctan(y_mean / x_mean) + 2*np.pi) * 365/(2*np.pi)             
            
            events_future_time[1,input_combo[i][0], input_combo[i][1]] = d_mean
            events_future_time[2,input_combo[i][0], input_combo[i][1]] = concentration
             
               #%

        events_current_time.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/current_time_40ensemblemean'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        events_future_time.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/future_time_40ensemblemean'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc') 

