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
import glob

basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'

basic_dai_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'

#%%
data=xr.open_dataarray(basic_dir+'/code_new/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')

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
    all_plans.append((dtrd_typ,'Series_mean_lens',30,0.9,30))
   
    
   
Experiments=['ACCESS-ESM1-5','CanESM5','CESM2-WACCM','CMCC-CM2-SR5',
             'CMCC-ESM2','EC-Earth3','EC-Earth3-CC','EC-Earth3-Veg',
             'EC-Earth3-Veg-LR','GFDL-CM4','GFDL-ESM4','INM-CM4-8',
             'INM-CM5-0','IPSL-CM6A-LR','KIOST-ESM','MIROC6',
             'MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM',
             'NorESM2-MM','TaiESM1']    
   
ensemble_index={}
n_num=0

for n in range(22):
    
    name=np.sort(glob.glob('/media/dai/DATA1/pr/1-COMBINED/pr_day_'+Experiments[n]+'_*'))
    # 
    realizations=np.unique(np.array([name[i].split('_')[len(name[i].split('_'))-1].split('.')[0] for i in range(len(name)) ]))
    for nn in range(len(realizations)):
        print(Experiments[n]+'_'+realizations[nn])
        ensemble_index.update({Experiments[n]+'_'+realizations[nn]:n_num})
        n_num=n_num+1
#%%
for plan in [1]:
    
    dtrd_typ=all_plans[plan][0]
    thes_typ=all_plans[plan][1]
    min_period=all_plans[plan][2]
    q=all_plans[plan][3]
    inter_period=all_plans[plan][4]
    
    
    for ex in ['dry_to_wet','wet_to_dry']  :
        print(ex)
    
        # 1.frequency
        events_counts=xr.DataArray(np.zeros(shape=(len(data.year),len(data.lat),len(data.lon),55)),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':range(55)}))
        #duration
        events_duration=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),55),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':range(55)}))
        #intensity
        events_intensity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),55),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':range(55)}))
        
        events_severity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon),55),fill_value=np.nan),
                     dims=('year','lat','lon','ensemble'),
                     coords=({'lon':data.lon,'lat':data.lat,
                            'year':data.year,
                            'ensemble':range(55)}))
        
        
        
        for n in range(22):
            
            name=np.sort(glob.glob('/media/dai/DATA1/pr/1-COMBINED/pr_day_'+Experiments[n]+'_*'))
            
            realizations=np.unique(np.array([name[i].split('_')[len(name[i].split('_'))-1].split('.')[0] for i in range(len(name)) ]))
            for nn in range(len(realizations)):
                print(Experiments[n]+'_'+realizations[nn])
            
                method_dir= basic_dai_dir +'code_new/6-5.CMIP6_daily_whiplash_stats_baseline_models_new_intensity/'+Experiments[n]+'_'+realizations[nn]+'/'
                
              
                
                
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
                    events_counts[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=ensemble_index.get(Experiments[n]+'_'+realizations[nn]))[uni_y.astype(int)]=count
            
                    mean_duration=pd.Series(duration).groupby(np.floor(b/365)).mean()
                    events_duration[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=ensemble_index.get(Experiments[n]+'_'+realizations[nn]))[uni_y.astype(int)]=mean_duration
            
                    mean_intensity=pd.Series(intensity).groupby(np.floor(b/365)).mean()
                    events_intensity[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=ensemble_index.get(Experiments[n]+'_'+realizations[nn]))[uni_y.astype(int)]=mean_intensity
                    
                    mean_severity=pd.Series(severity).groupby(np.floor(b/365)).mean()
                    events_severity[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=ensemble_index.get(Experiments[n]+'_'+realizations[nn]))[uni_y.astype(int)]=mean_severity
            
        
        dai_basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'
        events_counts.to_netcdf(dai_basic_dir + 'code_new/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/frequency_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        events_duration.to_netcdf(dai_basic_dir + 'code_new/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/duration_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
          
        events_intensity.to_netcdf(dai_basic_dir + 'code_new/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/intensity_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
         
        events_severity.to_netcdf(dai_basic_dir + 'code_new/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/severity_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')

