#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:46:02 2023

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

from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'

#%%
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

#%%

# a 降水气候态
prec=xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')
prec_mean=prec.mean('year')

lon=prec.lon
lat=prec.lat

cycle_value, cycle_lon = add_cyclic_point(prec_mean*365, coord=lon)
cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)

# b 降水变化（2060-2100 - 1979-2019）
prec_future=prec.sel(year=slice(2060,2100)).mean('year')*365
prec_now=prec.sel(year=slice(1979,2019)).mean('year')*365

prec_change = (prec_future-prec_now)*100/prec_now
cycle_change, cycle_lon = add_cyclic_point(prec_change, coord=lon)
cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)


pd.DataFrame(cycle_LON).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lon.csv',index=False)
pd.DataFrame(cycle_LAT).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/meshgrid_lat.csv',index=False)
pd.DataFrame(cycle_value).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.climatology_prcp_map.csv',index=False)
pd.DataFrame(cycle_change).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.future_change_prcp_map.csv',index=False)


#a=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.meshgrid_lon.csv')

#%%

#%
#c 全球降水 
lats=prec['lat']
#区域平均
prec_region_mean=prec.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
prec_global_change = (prec_region_mean-prec_region_mean.sel(year=slice(1979,2019)).mean('year'))*100/prec_region_mean.sel(year=slice(1979,2019)).mean('year')

p_ens=pd.DataFrame()
for n in num:
    print(n)
    p=xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')
    p_global_mean=p.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).values
    p_ens=pd.concat([p_ens,pd.Series(p_global_mean)],axis=1)
    

p_quan= p_ens.quantile(q=[0.05,0.95],axis=1)*365


#去趋势1降水
p_ens_detrend1=pd.DataFrame()
for n in num:
    print(n)
    p=xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend1.nc')
    p_global_mean=p.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).values
    p_ens_detrend1=pd.concat([p_ens_detrend1,pd.Series(p_global_mean)],axis=1)
    
p_quan_detrend1= p_ens_detrend1.quantile(q=[0.05,0.95],axis=1)*365
p_detrend1=p_ens_detrend1.mean(axis=1)*365


#去趋势2降水
p_ens_detrend2=pd.DataFrame()
for n in num:
    print(n)
    p=xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend2.nc')
    p_global_mean=p.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).values
    p_ens_detrend2=pd.concat([p_ens_detrend2,pd.Series(p_global_mean)],axis=1)
    
p_quan_detrend2= p_ens_detrend2.quantile(q=[0.05,0.95],axis=1)*365
p_detrend2=p_ens_detrend2.mean(axis=1)*365


#去趋势3降水
p_ens_detrend3=pd.DataFrame()
for n in num:
    print(n)
    p=xr.open_dataarray(basic_dir+'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend3.nc')
    p_global_mean=p.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).values
    p_ens_detrend3=pd.concat([p_ens_detrend3,pd.Series(p_global_mean)],axis=1)
    
p_quan_detrend3= p_ens_detrend3.quantile(q=[0.05,0.95],axis=1)*365
p_detrend3=p_ens_detrend3.mean(axis=1)*365


pd.Seris(p_quan_detrend3),p_detrend3

data=pd.concat([p_quan.T,    pd.Series(prec_region_mean)*365,
           p_quan_detrend1.T,p_detrend1,
           p_quan_detrend2.T,p_detrend2,
           p_quan_detrend3.T,p_detrend3],axis=1)
pd.DataFrame(data).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_original_and_detrended_prec.csv',index=False)



#%%
all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))
    
#%%
ex='dry_to_wet'
d2w_global_mean = pd.DataFrame()    

for plan in [0,2,3,1]:   
#for ex in ['dry','wet']: 
    ## 不同阈值
    dtrd_typ=all_plans[plan][0]
    thes_typ=all_plans[plan][1]
    min_period=all_plans[plan][2]
    q=all_plans[plan][3]
    inter_period=all_plans[plan][4]

    events_counts=xr.open_dataarray( basic_dir + 'code_whiplash/2-2.Original whiplash events/5-1.CESM_LENS_event_frequency_duration_intensity_sensitivity/frequency_5ensemble'+
                            '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc').mean('ensemble')
    #mask出所选区域
    event=events_counts #*mask
    lats=event['lat']
    #区域平均
    event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
    event_region_mean_baseline = event_region_mean.sel(year=slice(1979,2019) ).mean()
    d2w_global_mean = pd.concat([d2w_global_mean,pd.Series(event_region_mean)],axis=1)
    
d2w_global_mean.columns=['raw','d1','d2','d3']
pd.DataFrame(d2w_global_mean).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_dry_to_wet_whiplash.csv',index=False)
#%%
ex='wet_to_dry'
w2d_global_mean = pd.DataFrame()    

for plan in [0,2,3,1]:   
#for ex in ['dry','wet']: 
    ## 不同阈值
    dtrd_typ=all_plans[plan][0]
    thes_typ=all_plans[plan][1]
    min_period=all_plans[plan][2]
    q=all_plans[plan][3]
    inter_period=all_plans[plan][4]

    events_counts=xr.open_dataarray( basic_dir + 'code_whiplash/2-2.Original whiplash events/5-1.CESM_LENS_event_frequency_duration_intensity_sensitivity/frequency_5ensemble'+
                            '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc').mean('ensemble')
    #mask出所选区域
    event=events_counts #*mask
    lats=event['lat']
    #区域平均
    event_region_mean=event.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon'])
    event_region_mean_baseline = event_region_mean.sel(year=slice(1979,2019) ).mean()
    w2d_global_mean = pd.concat([w2d_global_mean,pd.Series(event_region_mean)],axis=1)
    
w2d_global_mean.columns=['raw','d1','d2','d3']
pd.DataFrame(w2d_global_mean).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S1.global_annual_change_of_wet_to_dry_whiplash.csv',index=False)

