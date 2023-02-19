#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 23:56:31 2023

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
#%%
data=xr.open_dataarray(basic_dir+'/code/4-4.CMIP6_1920_2100_models_annual_mean_prec/PRECT_annual_mean_ACCESS-ESM1-5_r1i1p1f1.nc')

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
n=1
for n in range(1,5):
    print(n)
    Summary_Stats = np.load('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/'+str(n).zfill(3)+'.npy',allow_pickle=True)
    d=Summary_Stats[0].get('day_cycle')
    plt.plot(range(365),d,label=n)
    
plt.legend()
 #%%   
n=1
for n in range(1,5):
    print(n)
    Summary_Stats = np.load('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/'+str(n).zfill(3)+'.npy',allow_pickle=True)
    d=Summary_Stats[0].get('sd')
    plt.plot(range(365),d,label=n)
    
plt.legend()
calendar_day= d.index
#%%
n=1
for n in range(1,5):
    print(n)
    Summary_Stats = np.load('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/'+str(n).zfill(3)+'.npy',allow_pickle=True)
    d=Summary_Stats[0].get('quan_prec')
    plt.plot(threshold,d,label=n)
    
plt.legend()

#%%

sd=xr.DataArray(np.zeros(shape=(365,len(data.lat),len(data.lon),40)),
             dims=('calendar_day','lat','lon','ensemble'),
             coords=({'lon':data.lon,'lat':data.lat,
                    'calendar_day':calendar_day,
                    'ensemble':num}))

day_cycle=xr.DataArray(np.zeros(shape=(365,len(data.lat),len(data.lon),40)),
             dims=('calendar_day','lat','lon','ensemble'),
             coords=({'lon':data.lon,'lat':data.lat,
                    'calendar_day':calendar_day,
                    'ensemble':num}))

quan_prec=xr.DataArray(np.zeros(shape=(10,len(data.lat),len(data.lon),40)),
             dims=('threshold','lat','lon','ensemble'),
             coords=({'lon':data.lon,'lat':data.lat,
                    'threshold':threshold,
                    'ensemble':num}))
#%%
for n in num:
    print(n)
    Summary_Stats = np.load('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/'+str(n).zfill(3)+'.npy',allow_pickle=True)

    for i in range(len(input_combo)): #    
        d = Summary_Stats[i].get('day_cycle')
        s = Summary_Stats[i].get('sd')
        q = Summary_Stats[i].get('quan_prec')
        
        sd[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[:] = s.values
        day_cycle[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[:] = d.values
        quan_prec[:,input_combo[i][0], input_combo[i][1]].sel(ensemble=n)[:] = q.values
#%%
sd.to_netcdf('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_sd.nc')
day_cycle.to_netcdf('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_day_cycle.nc')
quan_prec.to_netcdf('/media/dai/disk2/suk/research/4.East_Asia/Again/code/6-1.CESM_LENS_cumprec_sd_quan/LENS_40ensembles_quan_prec.nc')