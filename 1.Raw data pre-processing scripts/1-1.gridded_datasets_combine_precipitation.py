#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:28:34 2022

@author: dai
"""

import xarray as xr
import numpy as np
import glob
from datetime import datetime 
from datetime import timedelta

#%%

loc='/media/dai/Elements/research/4.East_Asia/Again/data/gridded_prcp/'
dai_basic_loc='/media/dai/disk2/suk/research/4.East_Asia/Again/'

datasets=['CFSR','ERA5','MERRA2','NCEP2','JRA-55','MSWEP_V1.2', #6 reanalysis
'CHIRP','CHIRPS','PERSIANN', #3 satellite
'CPC','GPCC','REGEN_AllStns','REGEN_LongTermStns'] #4 grond-base land only


#%%


################################## the whole series #########################

for j in range(len(datasets)):
    
    name_series=np.sort(glob.glob(loc+datasets[j]+'/*')) # search all files
    
    
    if j ==3 :  #NCEP
    
        data=[]
        for i in range(len(name_series)):
            print(i)
            d=xr.open_dataset(name_series[i],decode_times=False)['prate']
            d=d.interp(lon=np.arange(0,360,2),lat=np.arange(90,-90,-2),kwargs={"fill_value": "extrapolate"}) #interp
            data.append(d)
        data1=xr.concat(data, dim='time')

        time=[datetime(1979,1,1)+timedelta(i) for i in range(data1.shape[0])]
        data1=data1.assign_coords({'time':time}) #tran to datetime type
        data1=data1*24*3600 # Kg/m^2/s  to  mm/day
        
        data = xr.Dataset({'prcp': (('time','lat', 'lon'), data1.values)}, coords={'time':data1['time'],
                                                                           'lat': data1['lat'],
                                                                           'lon':data1['lon']})
        data=data['prcp']
        
        
    elif j==5 :
        
        lat1=np.arange(89.875,-90,-0.25)
        lon1=np.arange(-179.875,180,0.25)
        
        data=[]
        for i in range(len(name_series)):
            print(i)
            d=xr.open_dataset(name_series[i],decode_times=False)['daily prcp']
            d=d.assign_coords({'longitude':lon1,'latitude':lat1})
            time=[datetime(1979+i,1,1)+timedelta(k) for k in range(d.shape[0])]
            #d=d.assign_coords({'time':time})
            d=d.interp(longitude=np.arange(-178,181,2),latitude=np.arange(90,-90,-2),kwargs={"fill_value": "extrapolate"}) #interp
            p1=d.values.swapaxes(1,2)
            d = xr.Dataset({'prcp': (('time','lat', 'lon'), p1)}, coords={'time':time,
                                                                               'lat': np.arange(90,-90,-2),
                                                                               'lon':np.arange(-178,181,2)})
            data.append(d)
            
        data1=xr.concat(data, dim='time')
        
        lon=data1.lon.values
        lon[lon<0]=lon[lon<0]+360
        data1=data1.assign_coords({'lon':lon}) #trans to 0~360
    
        data1=data1.reindex(lon=np.sort(data1.lon)) 
        data=data1['prcp']
        
      
        
    else:  # all other datasets
        data=[]
        
        for i in range(len(name_series)):
            print(i)
            d=xr.open_dataset(name_series[i])
            d=d.interp(lon=np.arange(-180,181,1),lat=np.arange(-90,91,1),kwargs={"fill_value": "extrapolate"}) #interp
            d=d.sel(lon=np.arange(-178,181,2),lat=np.arange(-88,91,2),)
            d=d.reindex(lat=list(reversed(d.lat))) #reverse lat 
            data.append(d)
        data=xr.concat(data, dim='time')
        
        lon=data.lon.values
        lon[lon<0]=lon[lon<0]+360
        data=data.assign_coords({'lon':lon}) #trans to 0~360
    
        data=data.reindex(lon=np.sort(data.lon)) 
        data=data['prcp']
    
    data.to_netcdf(dai_basic_loc+'data/combined_gridded_prcp/'+
                   datasets[j]+'_daily_prcp.nc')
    
