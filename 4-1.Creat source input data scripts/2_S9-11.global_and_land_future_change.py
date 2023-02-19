#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:16:11 2023

@author: dai

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy
from scipy import signal  
import os
from datetime import datetime
import pandas as pd
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import tqdm
import multiprocessing
import gc
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib

basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'


#%%


datasets_new=['CHIRPS','GPCC','REGEN_LongTermStns',
              'ERA5','MERRA2','JRA-55',
              ] #4 grond-base land only


mask_o=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/land_mask.csv')


#%%
color_climate= ['#FA9B58','#FECE7C','#FFF5AE','#FBFAE6','#B9E176','#96D268','#69BE63','#33A456','#108647']
color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']
colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']

Colors_limits=[-120,-80,-60,-40,-20,0,40,80,120,160,248,280,320]
#%%
#%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))

plan=1

dtrd_typ=all_plans[plan][0]
thes_typ=all_plans[plan][1]
min_period=all_plans[plan][2]
q=all_plans[plan][3]
inter_period=all_plans[plan][4]

ex='dry_to_wet'
feature='frequency'

title_aux = list(map(chr, range(97, 123)))[:6]

running_year = 5
#%%

global_land_mean ={}

for feature in ['frequency','duration','intensity']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        print(feature)
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ############################# CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        lats=events['lat']
        mask= xr.DataArray(mask_o,dims=('lat','lon'),coords={'lon':events.lon,'lat':events.lat})
        # 1.全球 
        e_global = events.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099))
        e_global_current = e_global.sel( year=slice(1979,2019) ).mean('year')
        #print(e_global_current.mean())
        e_global_trend = (e_global -e_global_current)*100 /e_global_current
        e_global_trend_rolling = e_global_trend.rolling(dim={'year':running_year},center= True).mean().dropna('year')
    
        # 2.陆地
        e_land = ( (events*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])).sel(year=slice(1921,2099))
        e_land_current = e_land.sel( year=slice(1979,2019) ).mean('year')
        e_land_trend = (e_land -e_land_current)*100 /e_land_current
        e_land_trend_rolling = e_land_trend.rolling(dim={'year':running_year},center= True).mean().dropna('year')
     
        global_land_mean.update( { ex+'~'+feature+'~'+'global-LENS' : e_global_trend_rolling } )
        global_land_mean.update( { ex+'~'+feature+'~'+'land-LENS' : e_land_trend_rolling } )
        
        # ###########################CMIP6 
       
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+feature+'_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        lats=events['lat']
        # 1.全球 
        e_global = events.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099))
        e_global_current = e_global.sel( year=slice(1979,2019) ).mean('year')
        e_global_trend = (e_global -e_global_current)*100 /e_global_current
        e_global_trend_rolling = e_global_trend.rolling(dim={'year':running_year},center= True).mean().dropna('year')
    
        # 2.陆地
        e_land = ( (events*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])).sel(year=slice(1921,2099))
        e_land_current = e_land.sel( year=slice(1979,2019) ).mean('year')
        e_land_trend = (e_land -e_land_current)*100 /e_land_current
        e_land_trend_rolling = e_land_trend.rolling(dim={'year':running_year},center= True).mean().dropna('year')
        
        global_land_mean.update( { ex+'~'+feature+'~'+'global-CMIP6' : e_global_trend_rolling } )
        global_land_mean.update( { ex+'~'+feature+'~'+'land-CMIP6' : e_land_trend_rolling } )
        
        
        ################################ grid datasets
        for n in range(len(datasets_new)):
            
            
            event=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-1.gridded_dataset_event_frequency_duration_intensity/'+feature+'_'+datasets_new[n]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            #event=events_counts *mask
            #event=event.sel(lon=slice(70,90),lat=slice(40,30))  #################
            
           
            lats=event['lat']
            
            if datasets_new[n] == 'CHIRPS':
                event_region_mean=(event.sel(year=slice(1982,2018)))
            elif datasets_new[n] == 'GPCC':
                event_region_mean=(event.sel(year=slice(1984,2018)))
            elif datasets_new[n] == 'REGEN_LongTermStns':
                event_region_mean=(event.sel(year=slice(1980,2015)))
            else:
                event_region_mean=(event.sel(year=slice(1980,2018)))
            '''
            event_region_mean_current = event_region_mean.sel( year=slice(1979,2019) ).mean('year',skipna=True).values
            event_region_mean = (event_region_mean-event_region_mean_current)*100/event_region_mean_current
            event_region_mean=event_region_mean.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            event_region_mean=event_region_mean.rolling(dim={'year':running_year},center= True).mean().dropna('year')
            '''
            
            '''
            event_region_mean_current = event_region_mean.sel( year=slice(1979,2019) ).mean('year',skipna=True)
            event_region_mean=event_region_mean.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            event_region_mean = (event_region_mean-event_region_mean_current.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],))*100/event_region_mean_current.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            
            event_region_mean=event_region_mean.rolling(dim={'year':running_year},center= True).mean().dropna('year')
           '''
            '''
            event_region_mean_current = event_region_mean.sel( year=slice(1979,2019) ).mean('year',skipna=True)
            event_region_mean=event_region_mean.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            event_region_mean=event_region_mean.rolling(dim={'year':running_year},center= True).mean().dropna('year')
            event_region_mean = (event_region_mean-event_region_mean_current.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],))*100/event_region_mean_current.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            '''
            '''
            event_region_mean=event_region_mean.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            event_region_mean_current = event_region_mean.sel( year=slice(1979,2019) ).mean('year',skipna=True)
            event_region_mean = (event_region_mean-event_region_mean_current)*100/event_region_mean_current
          
            event_region_mean=event_region_mean.rolling(dim={'year':running_year},center= True).mean().dropna('year')
            '''
            
            
            event_region_mean=event_region_mean.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'],)
            event_region_mean=event_region_mean.rolling(dim={'year':running_year},center= True).mean().dropna('year')
            event_region_mean_current = event_region_mean.sel( year=slice(1979,2019) ).mean('year',skipna=True)
            event_region_mean = (event_region_mean-event_region_mean_current)*100/event_region_mean_current
        
            global_land_mean.update( { ex+'~'+feature+'~'+datasets_new[n] : event_region_mean } )
        

np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/2_S9-11.global_and_land_mean_of_features.npy',global_land_mean)


#%%



    
    



#%%

change_map={}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        if feature == 'frequency':
            
            e_future = events.sel(year=slice(2060,2099)).sum('year').mean(('ensemble'))
            e_current = events.sel(year=slice(1979,2019)).sum('year').mean(('ensemble'))
            frequency_current = e_current.copy(deep=True)
            e_current = e_current.where(e_current>5)
            
            cycle_change, cycle_lon = add_cyclic_point((e_future - e_current)*100/e_current, coord=events.lon)
            change_map.update({feature+'_'+ex+'_trend': cycle_change})
            
            
            e_future = events.sel(year=slice(2060,2099)).sum('year')
            e_current = events.sel(year=slice(1979,2019)).sum('year')
            e_current = e_current.where(e_current>5)
            trend = (e_future - e_current)*100/e_current
            
        else:
            e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
            e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
            e_current = e_current.where(frequency_current>5)
            
            cycle_change, cycle_lon = add_cyclic_point((e_future - e_current)*100/e_current, coord=events.lon)
            change_map.update({feature+'_'+ex+'_trend': cycle_change})
            
            e_future = events.sel(year=slice(2060,2099)).sum('year')
            e_current = events.sel(year=slice(1979,2019)).sum('year')
            #e_current = e_current.where(e_current>4)
            e_current = e_current.where(frequency_current>5)
            trend = (e_future - e_current)*100/e_current
        
        
        
        inc = (trend>0).sum('ensemble')    
        inc = inc.where( inc >= 36 )
        
        a=np.abs(np.isnan(inc.values).astype(float)-1)
        
        dec = (trend < 0).sum('ensemble')    
        dec = dec.where( dec >= 36 )
    
        b = np.abs(np.isnan(dec.values).astype(float)-1)
        b=b+a
        b[np.where(b==0)]=np.nan
        #vars()[feature+'_'+ex+'_trend_sig'] = b
        b, cycle_lon = add_cyclic_point(b, coord=events.lon)
        change_map.update({feature+'_'+ex+'_trend_sig': b})

np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/2_S9.LENS_change_map_of_features.npy',change_map)

#%%

change_map={}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+feature+'_55ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        if feature == 'frequency':
            
            e_future = events.sel(year=slice(2060,2099)).sum('year').mean(('ensemble'))
            e_current = events.sel(year=slice(1979,2019)).sum('year').mean(('ensemble'))
            frequency_current = e_current.copy(deep=True)
            e_current = e_current.where(e_current>5)
            cycle_change, cycle_lon = add_cyclic_point((e_future - e_current)*100/e_current, coord=events.lon)
            change_map.update({feature+'_'+ex+'_trend': cycle_change})
            
            e_future = events.sel(year=slice(2060,2099)).sum('year')
            e_current = events.sel(year=slice(1979,2019)).sum('year')
            e_current = e_current.where(e_current>5)
            trend = (e_future - e_current)*100/e_current
            
        else:
            e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
            e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
            e_current = e_current.where(frequency_current>5)
            cycle_change, cycle_lon = add_cyclic_point((e_future - e_current)*100/e_current, coord=events.lon)
            change_map.update({feature+'_'+ex+'_trend': cycle_change})
            
            e_future = events.sel(year=slice(2060,2099)).sum('year')
            e_current = events.sel(year=slice(1979,2019)).sum('year')
            #e_current = e_current.where(e_current>4)
            e_current = e_current.where(frequency_current>5)
            trend = (e_future - e_current)*100/e_current
        
        
        
        inc = (trend>0).sum('ensemble')    
        inc = inc.where( inc >= 55*0.9 )
        
        a=np.abs(np.isnan(inc.values).astype(float)-1)
        
        dec = (trend < 0).sum('ensemble')    
        dec = dec.where( dec >= 55*0.9 )
    
        b = np.abs(np.isnan(dec.values).astype(float)-1)
        b=b+a
        b[np.where(b==0)]=np.nan
        b, cycle_lon = add_cyclic_point(b, coord=events.lon)
        change_map.update({feature+'_'+ex+'_trend_sig': b})

np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/S10-11.CMIP6_change_map_of_features.npy',change_map)


#%%


