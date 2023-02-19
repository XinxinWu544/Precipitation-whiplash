
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
import glob
basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'

datasets_new=['ERA5','MERRA2','JRA-55',
              'CHIRPS',
              'GPCC','REGEN_LongTermStns'] #4 grond-base land only


#%%
zonal_mean=pd.DataFrame()
for feature in ['frequency','duration','intensity']:
    print(feature)
    for ex in ['dry_to_wet','wet_to_dry']:
        event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_40ensemble_'+ex+'*')[0]).sel(year=slice(1979,2019)).mean('year')
        
        lon=event.lon
        lat=event.lat
        
        cycle_current_event, cycle_lon = add_cyclic_point(event.mean('ensemble'), coord=lon)
        
        if (ex=='wet_to_dry') & (feature == 'intensity'):
            cycle_current_event=np.abs(cycle_current_event)
            
        ### save map
        pd.DataFrame(cycle_current_event).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.current_'+feature+'_'+ex+'_map'+'.csv',index=False)
        
        
        d = np.abs(event.mean('ensemble').groupby('lat').mean('lon').rolling(dim={'lat':5},center=True).mean())
        zonal_mean=pd.concat([zonal_mean,pd.Series(d,index=d.lat,name= feature+'_'+ex+'_LENS' ) ],axis=1)    
        
        d = np.abs(event.quantile(dim='ensemble',q=[0.05,0.95]).groupby('lat').mean('lon').rolling(dim={'lat':5},center=True).mean())
        zonal_mean=pd.concat([zonal_mean,pd.Series(d[0,:],index=d.lat,name= feature+'_'+ex+'_LENS_quan_0.05' ) ],axis=1)    
        zonal_mean=pd.concat([zonal_mean,pd.Series(d[1,:],index=d.lat,name= feature+'_'+ex+'_LENS_quan_0.95' ) ],axis=1)    
        
        ### quantile 
        event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_40ensemble_'+ex+'*')[0]).sel(year=slice(1979,2019)).mean('year').quantile(dim='ensemble',q=[0.05,0.95])
        

        ##### CMIP6 
        event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-3.CMIP6_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_55ensemble_'+ex+'*')[0]).sel(year=slice(1979,2019)).mean('year')

        d = np.abs(event.mean('ensemble').groupby('lat').mean('lon').rolling(dim={'lat':5},center=True).mean())
        zonal_mean=pd.concat([zonal_mean,pd.Series(d,index=d.lat,name= feature+'_'+ex+'_CMIP6' ) ],axis=1)    
        
        d = np.abs(event.quantile(dim='ensemble',q=[0.05,0.95]).groupby('lat').mean('lon').rolling(dim={'lat':5},center=True).mean())
        zonal_mean=pd.concat([zonal_mean,pd.Series(d[0,:],index=d.lat,name= feature+'_'+ex+'_CMIP6_quan_0.05' ) ],axis=1)    
        zonal_mean=pd.concat([zonal_mean,pd.Series(d[1,:],index=d.lat,name= feature+'_'+ex+'_CMIP6_quan_0.95' ) ],axis=1)    
        
        #### gridded
        for n in range(len(datasets_new)):
            
            event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-1.gridded_dataset_event_frequency_duration_intensity/'+
                                              feature+'_'+datasets_new[n]+'_'+ex+'*')[0])
            event=event.sel(year=slice(event.year[1],event.year[-2])).mean('year')
            if feature=='frequency':
                event=event.where(event>0)
                
            event_lat_average = np.abs(event.groupby('lat').mean('lon'))
            event_lat_average=event_lat_average.where(event_lat_average>0).rolling(dim={'lat':5},center=True).mean()
            zonal_mean=pd.concat([zonal_mean,pd.Series(event_lat_average,
                                                       index=event_lat_average.lat,
                                                       name= feature+'_'+ex+'_'+datasets_new[n] ) ],axis=1)    
        
zonal_mean.to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.zonal_mean_of_features.csv')        


#%% occurence time 

for ex in ['dry_to_wet','wet_to_dry']:
    event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                      'current_time_40ensemblemean_'+ex+'*')[0])

    cycle_current_time, cycle_lon = add_cyclic_point(event.sel(index='cal_by_angle'), coord=lon)
    pd.DataFrame(cycle_current_time).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/1.current_time_'+ex+'_map'+'.csv',index=False)
    ### future 
    event=xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                      'future_time_40ensemblemean_'+ex+'*')[0])

    cycle_current_time, cycle_lon = add_cyclic_point(event.sel(index='cal_by_angle'), coord=lon)
    pd.DataFrame(cycle_current_time).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/S8.future_time_'+ex+'_map'+'.csv',index=False)
    
