#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:25:32 2023

@author: dai
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime


#%%
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

lon1=120 - 40
lon2=130 + 40
lat1=40 - 25
lat2=50 + 25    
#%%
interval=[(1,5),(1,10),(1,15),(1,20),(1,25),(1,30)]


#%%
def cal_before_or_after_days(d,k,b_or_a):
    if b_or_a =='before':
        a=pd.concat([   pd.Series(d-interval[k][1]),    pd.Series(d-interval[k][0])   ],axis=1)  ##这是第k种组合，找到其两端
        a2=[ pd.Series(range(a.iloc[i,0],a.iloc[i,1]+1 ) ) for i in range(0, len(a) )   ] ##将中间的天数填进list里
        a3=[j for i in a2 for j in i]
    else:
        a=pd.concat([   pd.Series(d+interval[k][0]),    pd.Series(d+interval[k][1])   ],axis=1)  ##这是第k种组合，找到其两端
        a2=[ pd.Series(range(a.iloc[i,0],a.iloc[i,1]+1 ) ) for i in range(0, len(a) )   ] ##将中间的天数填进list里
        a3=[j for i in a2 for j in i]
    return a3
    #%%
#uv_type='U'
#x=21
for x in [21,22]:
    for uv_type in ['U','V']:
        
        transition_current={}
        transition_future={}
        dmean_transition_current={}
        dmean_transition_future={}
        
        
        d_before_current={}
        d_after_current={}
        dmean_before_current={}
        dmean_after_current={}
        
        d_before_future={}
        d_after_future={}
        dmean_before_future={}
        dmean_after_future={}
            
        
        for n in num:
            
            ##1、
            uq200 = xr.open_dataarray('/media/dai/disk2/suk/Data/CESM-LENS/'+uv_type+'Q/200/'+uv_type+'Q_'+str(n).zfill(3)+ '.nc',chunks=({'time':1000})).sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
            uq500 = xr.open_dataarray('/media/dai/disk2/suk/Data/CESM-LENS/'+uv_type+'Q/500/'+uv_type+'Q_'+str(n).zfill(3)+ '.nc',chunks=({'time':1000})).sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
            uq850 = xr.open_dataarray('/media/dai/disk2/suk/Data/CESM-LENS/'+uv_type+'Q/850/'+uv_type+'Q_'+str(n).zfill(3)+ '.nc',chunks=({'time':1000})).sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
        
            print(datetime.now())
            u_IVT = ((uq200+uq500+uq850)/9.806)
            #u_IVT=u_IVT.chunk({'time':1000})
            u_IVT_rolling = u_IVT.rolling(time=5, center=True, min_periods=1).mean().compute()
            print(datetime.now())
        
        
            current_start = np.where( (u_IVT_rolling['time.year'].values ==1979 ))[0][0]
            current_end = np.where( (u_IVT_rolling['time.year'].values ==2019 ))[0][-1]
            
            future_start = np.where( (u_IVT_rolling['time.year'].values ==2060 ))[0][0]
            future_end = np.where( (u_IVT_rolling['time.year'].values ==2099 ))[0][-1]
        
        
            actual_days = u_IVT_rolling.indexes['time'].to_datetimeindex() 
            dates_grouped = pd.to_datetime(actual_days).strftime('%m%d')
            
            Smoothed_current = u_IVT_rolling.sel(time=slice('1979-01-01','2019-12-31'))
            Climatology_current = Smoothed_current.assign_coords({'time':pd.to_datetime(Smoothed_current.indexes['time'].to_datetimeindex() ).strftime('%m%d')}).groupby('time').mean()
        
            u_IVT_rolling_c = u_IVT_rolling.assign_coords({'time': dates_grouped}) 
            Anomalies_current = u_IVT_rolling_c.groupby('time') - Climatology_current
            
            
            Smoothed_future = u_IVT_rolling.sel(time=slice('2060-01-01','2099-12-31'))
            Climatology_future = Smoothed_future.assign_coords({'time':pd.to_datetime(Smoothed_future.indexes['time'].to_datetimeindex() ).strftime('%m%d')}).groupby('time').mean()
            Anomalies_future = u_IVT_rolling.assign_coords({'time': dates_grouped}).groupby('time') - Climatology_future
            
            
            print(datetime.now())
        
            ##2、
            for whiplash_type in ['dry_to_wet','wet_to_dry']:
                print('start ensemble:'+str(n)+' here! coming soon ~' +' at time:',datetime.now())    
                a = np.load('/media/dai/disk2/suk/research/4.East_Asia/Again/code_new/7-1.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble/'+str(n).zfill(3)+ '/'+whiplash_type+'_Series_mean_detrend_2_of_30_days_quantile_0.9_inter_period_30.npy',allow_pickle=True)
                d= a[0,:,0].astype(int)
                
                theta=( (d%365) *2*np.pi)/365
                x_mean = np.cos(theta).mean()
                y_mean = np.sin(theta).mean()
                
                concentration = np.sqrt( x_mean**2 + y_mean**2 ) 
                
                if (x_mean >0) & (y_mean >= 0):
                    d_mean = np.arctan(y_mean / x_mean ) * 365/(2*np.pi)
                elif x_mean <=0 :
                    d_mean = (np.arctan(y_mean / x_mean )  + np.pi) * 365/(2*np.pi)
                elif (x_mean > 0) & (y_mean < 0):
                    d_mean = (np.arctan(y_mean / x_mean) + 2*np.pi) * 365/(2*np.pi)         
                    
                d_near_mean = d[np.where ( (d%365 <= d_mean + 60)  & (d%365 >= d_mean - 60) ) ]
                
                dmean_current = (d[   (  d <= (365*99))&(d>= (365*59) )    ]%365).mean()
                dmean_future = (d[d >= (365*140)]%365).mean()
                
                d_near_mean_future = d[np.where ( (d%365 <= dmean_future+ 30)  & (d%365 >= dmean_future - 30) ) ]
                d_near_mean_current = d[np.where ( (d%365 <= dmean_current+ 30)  & (d%365 >= dmean_current - 30) ) ]
                
                
                ## 3、
                a=Anomalies_current [d[ (d>=current_start)&(d<=current_end)  ],:,:].mean('time')
                transition_current.update({str(n)+"~"+whiplash_type :a})
                
                a=Anomalies_current [d_near_mean_current[ (d_near_mean_current>=current_start)&(d_near_mean_current<=current_end)  ],:,:].mean('time')
                dmean_transition_current.update({str(n)+"~"+whiplash_type :a})
                
                a=Anomalies_future [d[ (d>=future_start)&(d<=future_end)  ],:,:].mean('time')
                transition_future.update({str(n)+"~"+whiplash_type :a})
                
                a=Anomalies_future [d_near_mean_future[ (d_near_mean_future>=future_start)&(d_near_mean_future<=future_end)  ],:,:].mean('time')
                dmean_transition_future.update({str(n)+"~"+whiplash_type :a})
                
                
                
                # 4、
                for k in range(len(interval)): 
                    
                    #4-1、前
                    
                    days = cal_before_or_after_days(d=d[ (d>=current_start)&(d<=current_end)], k=k, b_or_a='before')
                    d_before_current.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_current[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d_near_mean_current[ (d_near_mean_current>=current_start)&(d_near_mean_current<=current_end)], k=k, b_or_a='before')
                    dmean_before_current.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_current[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d[ (d>=future_start)&(d<=future_end)], k=k, b_or_a='before')
                    d_before_future.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_future[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d_near_mean_future[ (d_near_mean_future>=future_start)&(d_near_mean_future<=future_end)], k=k, b_or_a='before')
                    dmean_before_future.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_future[days,:,:].mean('time')} )
                    
                    #%%
                    days = cal_before_or_after_days(d=d[ (d>=current_start)&(d<=current_end)], k=k, b_or_a='after')
                    d_after_current.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_current[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d_near_mean_current[ (d_near_mean_current>=current_start)&(d_near_mean_current<=current_end)], k=k, b_or_a='after')
                    dmean_after_current.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_current[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d[ (d>=future_start)&(d<=future_end)], k=k, b_or_a='after')
                    d_after_future.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_future[days,:,:].mean('time')} )
                    
                    days = cal_before_or_after_days(d=d_near_mean_future[ (d_near_mean_future>=future_start)&(d_near_mean_future<=future_end)], k=k, b_or_a='after')
                    dmean_after_future.update({str(n)+'~'+str(k)+'~'+whiplash_type: Anomalies_future[days,:,:].mean('time')} )
                    
                    
                    
                    
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_transition_current.npy',transition_current)
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_dmean_transition_current.npy',dmean_transition_current)   
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_transition_future.npy',transition_future)   
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_dmean_transition_future.npy',dmean_transition_future)          
        
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'d_before_current'+'.npy',d_before_current)         
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'dmean_before_current'+'.npy',dmean_before_current)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'d_before_future'+'.npy',d_before_future)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'dmean_before_future'+'.npy',dmean_before_future)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'d_after_current'+'.npy',d_after_current)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'dmean_after_current'+'.npy',dmean_after_current)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'d_after_future'+'.npy',d_after_future)  
        np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_'+uv_type+'Q_'+'dmean_after_future'+'.npy',dmean_after_future)   
            