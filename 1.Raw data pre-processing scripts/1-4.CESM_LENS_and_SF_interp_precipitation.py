#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:10:18 2022

@author: dai
"""

import xarray as xr
import numpy as np

loc='/media/dai/DATA2/CESM-LENS/PRECT/PRECT/'

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]
#%%
'''
n=1
for n in num:
    print(n)
    data=xr.open_dataarray(loc+'_'+str(n).zfill(3)+'.nc')
    data1=data.interp(lat=np.arange(90,-90,-2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
    print('lat interp done!')
    data1=data1.interp(lon=np.arange(0,360,2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
    print('lon interp done!')
    data1.to_netcdf(loc+'_'+str(n).zfill(3)+'_interp2deg.nc')
    '''
    
#%%    
forcing=['AER','GHG','BMB']
num_forcing=[20,20,15]    

for i in [0,1,2]:
    for j in range(num_forcing[i]):
        data=xr.open_dataarray(loc+'X'+forcing[i]+'/X'+forcing[i]+'_'+str(j).zfill(3)+'.nc')
        data1=data.interp(lat=np.arange(90,-90,-2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
        print('lat interp done!')
        data1=data1.interp(lon=np.arange(0,360,2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
        print('lon interp done!')
        data1.to_netcdf(loc+'X'+forcing[i]+'/X'+forcing[i]+'_'+str(j).zfill(3)+'_interp2deg.nc')