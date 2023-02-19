# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:29:20 2022

@author: daisukiiiii
"""


from datetime import datetime
from datetime import timedelta 
import xarray as xr
import numpy as np
import glob
#%%

forcing=['AER','GHG','BMB']
num_forcing=[20,20,15]

##AER & GHG 需要合并
for i in [0,1]:
    name=np.sort(glob.glob('/scratch/xtan/suk/whiplash_wildfire/data/LENS/single_forcing/X'+forcing[i]+'/b.*'))
    for j in range(num_forcing[i]):
        print('j='+str(j))
        print(name[2*j])
        print(name[2*j+1])
        data1=xr.open_dataset(name[2*j])['PRECT']
        data2=xr.open_dataset(name[2*j+1])['PRECT']
        data=xr.concat([data1,data2], dim='time')
        
        ##反转纬度
        data=data.reindex(lat=list(reversed(data.lat)))
        
        data=data*60*60*24*1000
        data.attrs["units"]='mm/day'
        
        data.to_netcdf('/scratch/xtan/suk/whiplash_wildfire/data/LENS/single_forcing/X'+forcing[i]+'/X'+forcing[i]+'_'+str(j).zfill(3)+'.nc')
        
 ### BMB 不需要
i=2
name=np.sort(glob.glob('/scratch/xtan/suk/whiplash_wildfire/data/LENS/single_forcing/X'+forcing[i]+'/b.*'))
for j in range(num_forcing[i]):
   
    data=xr.open_dataset(name[j])['PRECT']
   
    
    ##反转纬度
    data=data.reindex(lat=list(reversed(data.lat)))
    
    data=data*60*60*24*1000
    data.attrs["units"]='mm/day'
    
    data.to_netcdf('/scratch/xtan/suk/whiplash_wildfire/data/LENS/single_forcing/X'+forcing[i]+'/X'+forcing[i]+'_'+str(j).zfill(3)+'.nc')


    
       