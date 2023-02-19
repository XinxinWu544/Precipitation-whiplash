# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy import signal  
import os
from datetime import datetime
import pandas as pd

#%% 
basic_dir='/media/dai/suk_code/research/4.East_Asia/Again/'
basic_dai_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'

all_combo_situations = np.load( basic_dai_dir+'/code_new/6-3.gridded_datasets_daily_whiplash_statistics_new_intensity/all_combo_situations.npy',allow_pickle=True ).tolist()        

datasets=['ERA5','MERRA2','JRA-55', # reanalysis
'CHIRPS', #satellite
'GPCC','REGEN_LongTermStns'] # grond-base land only


#%
all_sensitivity_plans=[]
for dtrd_typ in [1,2]:
    all_sensitivity_plans.append((dtrd_typ,'Series_mean',30,0.9,30))


#%%
for plan in [1]:
    
    dtrd_typ=all_sensitivity_plans[plan][0]
    thes_typ=all_sensitivity_plans[plan][1]
    min_period=all_sensitivity_plans[plan][2]
    q=all_sensitivity_plans[plan][3]
    inter_period=all_sensitivity_plans[plan][4]
    
    
    for ex in ['dry_to_wet','wet_to_dry']  :
        print(ex)
        for n in range(len(datasets)):
            
            print('dataset='+datasets[n])
            print(datetime.now())
            
            #1.1. read in data
            data=xr.open_dataarray(basic_dir + 'code_new/4-2.gridded_dataset_annual_ensemble_mean_prec/PRECT_annual_mean_'+datasets[n]+'_detrend1.nc')
            data_s=data.sum('year').values
            
            
            input_combo=all_combo_situations.get(datasets[n]+'~'+str(dtrd_typ)+str(min_period))
            '''
            input_combo=[]
            for j in range(cum_prec.shape[1]):
                for k in range(cum_prec.shape[2]):
                    if cum_prec[:,j,k].sum()!=0:
                        input_combo.append((j,k))
            '''            
                        
            events_counts=xr.DataArray(np.zeros(shape=(len(data.year),len(data.lat),len(data.lon))),
                         dims=('year','lat','lon'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,}))
            
                            
            events_duration=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon)),fill_value=np.nan ),
                         dims=('year','lat','lon'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,}))
            
            events_intensity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon)),fill_value=np.nan ),
                         dims=('year','lat','lon'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,}))
            events_severity=xr.DataArray(np.full(shape=(len(data.year),len(data.lat),len(data.lon)),fill_value=np.nan ),
                         dims=('year','lat','lon'),
                         coords=({'lon':data.lon,'lat':data.lat,
                                'year':data.year,}))
            
            method_dir= basic_dai_dir +'code_new/6-3.gridded_datasets_daily_whiplash_statistics_new_intensity/'+datasets[n]+'/'
            
            
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
                    severity=np.abs(event[i][:,5])+ np.abs(event[i][:,2])
                
                uni_y, count = np.unique( np.floor(b/365) ,return_counts=True )
                events_counts[ :,input_combo[i][0], input_combo[i][1]  ][uni_y.astype(int)]=count
                
                mean_duration=pd.Series(duration).groupby(np.floor(b/365)).mean()
                events_duration[ :,input_combo[i][0], input_combo[i][1] ][uni_y.astype(int)]=mean_duration
         
                mean_intensity=pd.Series(intensity).groupby(np.floor(b/365)).mean()
                events_intensity[ :,input_combo[i][0], input_combo[i][1] ][uni_y.astype(int)]=mean_intensity
         
                mean_severity=pd.Series(severity).groupby(np.floor(b/365)).mean()
                events_severity[ :,input_combo[i][0], input_combo[i][1] ][uni_y.astype(int)]=mean_severity
           
            events_counts.to_netcdf(basic_dai_dir + 'code_new/8-1.gridded_dataset_event_frequency_duration_intensity/frequency_'+datasets[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            events_duration.to_netcdf(basic_dai_dir + 'code_new/8-1.gridded_dataset_event_frequency_duration_intensity/duration_'+datasets[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
              
            events_intensity.to_netcdf(basic_dai_dir + 'code_new/8-1.gridded_dataset_event_frequency_duration_intensity/intensity_'+datasets[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
             
            events_severity.to_netcdf(basic_dai_dir + 'code_new/8-1.gridded_dataset_event_frequency_duration_intensity/severity_'+datasets[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
     
