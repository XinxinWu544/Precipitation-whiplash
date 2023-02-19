# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:43:32 2022

@author: daisukiiiii
"""
#import cftime
import xarray as xr
import pandas as pd
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
from pathlib import Path
#import glob
print('Analysis started in:', datetime.now() )


w_name='U' ##此处这里为uq
q_name='Q'


#%%
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]
    
#%% 提取LENS时间
time= [datetime(1920,1,1)+timedelta(i) for i in range(66110)]

m=[ int(str(i).split(' ')[0].split('-')[1]) for i in time]
d=[ int(str(i).split(' ')[0].split('-')[2]) for i in time]

leap_day=[]
for i in range(len(m)):
    if ( (m[i] == 2) & (d[i]==29) ):
        print(i)
        leap_day.append(i)

time=np.delete(np.array(time),np.array(leap_day)).tolist()

lens_year = [ int(str(i).split(' ')[0].split('-')[0]) for i in time]
lens_month = [ int(str(i).split(' ')[0].split('-')[1]) for i in time]
lens_day = [ int(str(i).split(' ')[0].split('-')[2]) for i in time]
## 地表气压时间
ps_time=[ datetime(i,j,1) for i in range(1920,2101) for j in range(1,13)]
ps_month = [ int(str(i).split(' ')[0].split('-')[1]) for i in ps_time]
ps_year = [ int(str(i).split(' ')[0].split('-')[0]) for i in ps_time]

#%%
nn=850
n=1

#%%
for nn in [500]:
    
    #######################################################定义一些常量###############################

    print('level='+str(nn)+' at time:',datetime.now()) ##表示这一层的计算开始
    ###选取的变量###
    w_variable=w_name+str(nn)
    q_variable=q_name+str(nn)
    
    ###此处生成我们输出结果的路径####
    result_loc='/media/dai/disk2/suk/Data/CESM-LENS/'+w_name+q_name+'/'+str(nn)+'/'
    Path(result_loc).mkdir(parents=True, exist_ok=True) ##创建文件
    
    ###数据路径
    w_data_loc='/media/dai/DATA3/CESM-LENS/'+w_variable+'/'
    q_data_loc='/media/dai/DATA3/CESM-LENS/'+q_variable+'/'
    ps_data_loc='/media/dai/DATA3/CESM-LENS/PS/'
    
    
    
    ###############################################################################################################################
    
    for n in [3,5,8,9,10,11]:
        try:
            ##我们所需的日期
            print('start ensemble:'+str(n)+' here! ' +' at time:',datetime.now())    
            
         
            ##########################################################   wind  #####################################################################
            print('ensemble wind hist~')
            if n==1:
                
                d1=xr.open_dataset(w_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+w_variable+'.18500101-20051231.nc').sel(
                   
                    time=slice('1920-01-01','2005-12-31'))[w_variable]
            else:
                d1=xr.open_dataset(w_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+w_variable+'.19200101-20051231.nc').sel(
                  
                    time=slice('1920-01-01','2005-12-31'))[w_variable]
            
            #############
            print('ensemble wind future~')
            if n<=33:
                d2=xr.open_dataset(w_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+w_variable+'.20060101-20801231.nc').sel(
                   
                    time=slice('2006-01-01','2080-12-31'))[w_variable]
                
                d3=xr.open_dataset(w_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+w_variable+'.20810101-21001231.nc').sel(
                   
                    time=slice('2081-01-01','2100-12-31'))[w_variable]
                
                Daily_w=xr.concat([d1,d2,d3],dim='time')
                
                del(d1);del(d2);del(d3)
            else:
                d2=xr.open_dataset(w_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+w_variable+'.20060101-21001231.nc').sel(
                   
                    time=slice('2006-01-01','2100-12-31'))[w_variable]
                
                Daily_w=xr.concat([d1,d2],dim='time')
                
                del(d1);del(d2)
            
        
            ##########################################################   q  #####################################################################
            print('ensemble q hist~')
            if n==1:
                
                d1=xr.open_dataset(q_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+q_variable+'.18500101-20051231.nc').sel(
                    
                    time=slice('1920-01-01','2005-12-31'))[q_variable]
            else:
                d1=xr.open_dataset(q_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+q_variable+'.19200101-20051231.nc').sel(
                  
                    time=slice('1920-01-01','2005-12-31'))[q_variable]
            
            #############
            print('ensemble q future~')
            if n<=33:
                d2=xr.open_dataset(q_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+q_variable+'.20060101-20801231.nc').sel(
                  
                    time=slice('2006-01-01','2080-12-31'))[q_variable]
                
                d3=xr.open_dataset(q_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+q_variable+'.20810101-21001231.nc').sel(
                    
                    time=slice('2081-01-01','2100-12-31'))[q_variable]
                
                Daily_q=xr.concat([d1,d2,d3],dim='time')
                
                del(d1);del(d2);del(d3)
            else:
                d2=xr.open_dataset(q_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+q_variable+'.20060101-21001231.nc').sel(
                   
                    time=slice('2006-01-01','2100-12-31'))[q_variable]
                
                Daily_q=xr.concat([d1,d2],dim='time')
                
                del(d1);del(d2)
            
            #%%
            
            #Daily=Daily_q.copy(deep=True)
            #Daily.values=Daily_q.values * Daily_w.values ##Daily则为这两相乘的结果.注意 单位不是真的，单位为 (kg/kg)*(m/s)
            Daily=Daily_q * Daily_w
            Daily.attrs["units"]='(kg/kg)*(m/s)'
        
            del(Daily_q);del(Daily_w) 
            
            
            ###################################################### 如果为850 则需要考虑地表压力############################################################
            variable='PS'
            if nn==850:
                ############
                print('ensemble ps hist~')
                if n==1:
                    
                    d1=xr.open_dataset(ps_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h0.'+variable+'.185001-200512.nc').sel(
                      
                        time=slice('1920-01','2005-12'))[variable]
                else:
                    d1=xr.open_dataset(ps_data_loc+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h0.'+variable+'.192001-200512.nc').sel(
                      
                        time=slice('1920-01','2005-12'))[variable]
                
                #############
                print('ensemble ps future~')
                if n<=33:
                    d2=xr.open_dataset(ps_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h0.'+variable+'.200601-208012.nc').sel(
                       
                        time=slice('2006-01','2080-12'))[variable]
                    
                    d3=xr.open_dataset(ps_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h0.'+variable+'.208101-210012.nc').sel(
                       
                        time=slice('2081-01','2100-12'))[variable]
                    
                    monthly_ps=xr.concat([d1,d2,d3],dim='time')
                    
                    del(d1);del(d2);del(d3)
                else:
                    d2=xr.open_dataset(ps_data_loc+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h0.'+variable+'.200601-210012.nc').sel(
                       
                        time=slice('2006-01','2100-12'))[variable]
                    
                    monthly_ps=xr.concat([d1,d2],dim='time')
                    
                    del(d1);del(d2)
                    
                #发现有缺失的monthly_ps
                if monthly_ps.shape[0]==2170:
                    m1=monthly_ps.interp(time=('2006-01'))
                    m2=monthly_ps.interp(time=('2081-01'))
                    monthly_ps=xr.concat([monthly_ps,m1,m2], dim='time').sortby('time')
                elif monthly_ps.shape[0]==2169:
                    m1=monthly_ps.interp(time=('2006-01'))
                    m2=monthly_ps.interp(time=('2081-01'))
                    m3=monthly_ps.interp(time=('1920-01'))
                    m3.values=monthly_ps[0,:,:].values
                    monthly_ps=xr.concat([monthly_ps,m1,m2,m3], dim='time').sortby('time')
            
                
                ## 将月尺度转换为日尺度
                ps_new=np.zeros(shape=(66065,192,288))
                for i in range(len(ps_year)):
                    m_where=np.where((np.array(lens_year)==ps_year[i]) & (np.array(lens_month)==ps_month[i]))[0]
                    for j in range(len(m_where)):
                        ps_new[m_where[j],:,:]=monthly_ps[i,:,:].values
                
                
                ps_new=np.float32(ps_new)
    
                ps = xr.Dataset({'ps': (('time','lat', 'lon'), ps_new)}, coords={'time':Daily.time,
                                                                                   'lat': monthly_ps['lat'],
                                                                                   'lon':monthly_ps['lon']})
                ps = ps['ps']
                ps.attrs["units"]='Pa' #设置单位
               
                #ps
                ps1=ps.reindex(lat=list(reversed(ps.lat)))
                ps1.to_netcdf('/media/dai/disk2/suk/Data/CESM-LENS/daily_PS/PS_'+str(n).zfill(3) +'.nc')
                Daily_p_diff_sel=ps - 67500  ###
                #Daily_p_diff_sel.values[Daily_p_diff_sel.values<0]= np.nan ##将小于的都设置为nan
                Daily_p_diff_sel = Daily_p_diff_sel.where(Daily_p_diff_sel>=0)
        
            elif nn==500:
                Daily_p_diff_sel=67500-35000
            else :
                Daily_p_diff_sel=35000-10000
            
        
            Daily_var = Daily * Daily_p_diff_sel
            
            Daily_var.attrs["units"]='(kg/kg)*(m/s)*Pa' #设置单位
            Daily_var=Daily_var.reindex(lat=list(reversed(Daily_var.lat)))
            Daily_var.to_netcdf(result_loc+ w_name+q_name+'_'+str(n).zfill(3) +'.nc' )
        except:
            continue
    
    
