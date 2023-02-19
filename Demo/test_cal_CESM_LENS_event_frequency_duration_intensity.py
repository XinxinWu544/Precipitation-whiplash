# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:05:39 2022

@author: daisukiiiii
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

#num=[1,2,3,4,5,6]
#%%


#%
all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))
   
num=[1,2]
#%%
for plan in [1]:
    
    dtrd_typ=all_plans[plan][0]
    thes_typ=all_plans[plan][1]
    min_period=all_plans[plan][2]
    q=all_plans[plan][3]
    inter_period=all_plans[plan][4]
    
    
    for ex in ['dry_to_wet','wet_to_dry']  :
        print(ex)
    
        # 1.save frequency
        events_counts=xr.DataArray(np.zeros(shape=(len(data.year),len(data.lat),len(data.lon),len(num))),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':num}))
        #transition duration
        events_duration=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),len(num)),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':num}))
        #intensity
        events_intensity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),len(num)),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':num}))
        
        events_severity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),len(num)),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':num}))
        
        
        
        for n in num:
        
            print(n)
            
           
            method_dir=basic_dir +'2-2.Original whiplash events/6-4.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble_new_intensity/' +str(n).zfill(3)+'/'
            
      
           
            event=np.load(method_dir+'dry_to_wet_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                    '_quantile_'+str(round((q),2))+'_inter_period_'+str(inter_period)+'.npy',allow_pickle=True).tolist()
        
            
            
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
                
                mean_severity=pd.Series(severity).groupby(np.floor(b/365)).mean()
                events_severity[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[uni_y.astype(int)]=mean_severity
        
        

        events_counts.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/frequency_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        events_duration.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/duration_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
          
        events_intensity.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/intensity_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
         
        events_severity.to_netcdf(basic_dir + '3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/severity_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')

