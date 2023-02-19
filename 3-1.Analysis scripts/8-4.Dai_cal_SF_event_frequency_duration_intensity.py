#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:28:43 2023

@author: dai
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy import signal  
import os
from datetime import datetime
import pandas as pd

basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'

basic_dai_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'

#%%
data=xr.open_dataarray(basic_dir+'/code_new/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')

lat=data.lat.values

input_combo=[]
for j in range(data.shape[1]):
    for k in range(data.shape[2]):
        input_combo.append((j,k))


forcing=['AER','GHG','BMB']
num_forcing=[20,20,15]    
x=0

all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean_lens',30,0.9,30))

#%%

for x in [0,1,2]:
        
    for plan in [1]:
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        #
        for ex in ['dry','wet','dry_to_wet','wet_to_dry']  :
            print(ex)
        
            # 1.frequency  
            events_counts=xr.DataArray(np.zeros(shape=(len(data.year),len(data.lat),len(data.lon),num_forcing[x])),
                         dims=('year','lat','lon','ensemble'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,
                                'ensemble':np.arange(0,num_forcing[x],1)  }))
            #duration
            events_duration=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),num_forcing[x]),fill_value=np.nan),
                         dims=('year','lat','lon','ensemble'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,
                                'ensemble':np.arange(0,num_forcing[x],1) }))
            #intensity
            events_intensity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),num_forcing[x]),fill_value=np.nan),
                         dims=('year','lat','lon','ensemble'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,
                                'ensemble':np.arange(0,num_forcing[x],1) }))
            
            
            
            for n in range(num_forcing[x]):
            
                print(n)
                
                method_dir= basic_dai_dir +'code_new/6-6.CESM_SF_daily_whiplash_stats_baseline_new_intensity/'+'X'+forcing[x]+'/'+str(n).zfill(3)+'/'
                
              
                
                
                if (ex=='dry') | (ex=='wet'):
                    event=np.load(method_dir+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                        '_quantile_'+str(q)+'.npy',allow_pickle=True).tolist()
                else:
                    event=np.load(method_dir+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.npy',allow_pickle=True).tolist()
                
                
                
                for i in range(len(input_combo)): #    
                    #print(i)
                    b=event[i][:,0]
                    if (ex=='dry') | (ex=='wet'):
                        duration=event[i][:,1]-event[i][:,0]    
                        intensity=event[i][:,2]
                    else:
                        duration=event[i][:,3]-event[i][:,0]    
                        intensity=np.abs(event[i][:,4]-event[i][:,1])
                        severity=np.abs(event[i][:,5])+np.abs(event[i][:,2])
                    
                    uni_y, count = np.unique( np.floor(b/365) ,return_counts=True )
                    events_counts[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[uni_y.astype(int)]=count
            
                    mean_duration=pd.Series(duration).groupby(np.floor(b/365)).mean()
                    events_duration[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[uni_y.astype(int)]=mean_duration
            
                    mean_intensity=pd.Series(intensity).groupby(np.floor(b/365)).mean()
                    events_intensity[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[uni_y.astype(int)]=mean_intensity
            
            
            dai_basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
            events_counts.to_netcdf(dai_basic_dir + 'code_new/8-4.CESM_SF_event_frequency_duration_intensity_all/frequency_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            events_duration.to_netcdf(dai_basic_dir + 'code_new/8-4.CESM_SF_event_frequency_duration_intensity_all/duration_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
              
            events_intensity.to_netcdf(dai_basic_dir + 'code_new/8-4.CESM_SF_event_frequency_duration_intensity_all/intensity_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
             
