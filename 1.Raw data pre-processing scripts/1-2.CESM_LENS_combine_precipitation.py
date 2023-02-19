
from datetime import datetime
from datetime import timedelta 
import xarray as xr
import numpy as np

##目标：
#1.合并数据，确定长度一致，并调整为我设置的时间轴
#2.反转纬度
#3.确定只有三个维度
#4.设置单位

#%%
'''
PS unit:Pa
FSNS unit:W/m2
FSNSC unit:W/m2
PRECT unit:m/s
PSL unit:Pa
QBOT unit:kg/kg
TREFHT unit:K
TREFHTMX unit:K
TREFHTMN unit:K
WSPDSRFAV unit:m/s
'''
variables=['PS','TREFHT','TREFHTMN','TREFHTMX','FSNS','FSNSC','PRECT','PSL','QBOT','WSPDSRFAV']

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

loc='/media/dai/DATA2/CESM-LENS/'

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

#%% PRECT #%%

n=31
for v in [6]:
     
    variable=variables[v]
    loc_var=loc+variable+'/'
    
    #[3,5,6,7,9,22,23,24,25,26,27,28,29,31,32,33]
    for n in num:
        print(n)
        # ----  1. 合并  ----
        print('ensemble historical var~')
        if n==1:
            d1=xr.open_dataset(loc_var+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.18500101-20051231.nc').sel(
                time=slice('1920-01-01','2005-12-31'))[variable]
        else:
            d1=xr.open_dataset(loc_var+'b.e11.B20TRC5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.19200101-20051231.nc').sel(
                time=slice('1920-01-01','2005-12-31'))[variable]
        
        #############
        print('ensemble future var~')
        if n<=33:
            d2=xr.open_dataset(loc_var+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20060101-20801231.nc').sel(
                time=slice('2006-01-01','2080-12-31'))[variable]
            d3=xr.open_dataset(loc_var+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20810101-21001231.nc').sel(
                time=slice('2081-01-01','2100-12-31'))[variable]
            
            try:
                d3[:,1,1].values
            except:
                continue
            
            
            daily_var=xr.concat([d1,d2,d3],dim='time')
            del(d1);del(d2);del(d3)
        else:
            d2=xr.open_dataset(loc_var+'b.e11.BRCP85C5CNBDRD.f09_g16.'+str(n).zfill(3)+'.cam.h1.'+variable+'.20060101-21001231.nc').sel(
                time=slice('2006-01-01','2100-12-31'))[variable]
            
            try:
                d2[:,1,1].values
            except:
                continue
            
            daily_var=xr.concat([d1,d2],dim='time')
            del(d1);del(d2)
            
        if daily_var.shape[0]!=66065:
            print('the len of days != 66065, please recheck')
            
        # ----  1.2 调整为我要的时间轴  ----
        daily_var['time']=time
        
        
        ## ----  2 反转纬度 ----
        daily_var=daily_var.reindex(lat=list(reversed(daily_var.lat)))
        
        
        # ----  3-4 调整单位 ----
        if v==6:
            daily_var=daily_var*60*60*24*1000
            daily_var.attrs["units"]='mm/day'
        elif v==7:
            daily_var=daily_var/100
            daily_var.attrs["units"]='hPa'
        elif v==8:
            daily_var.attrs["units"]='kg/kg'
        elif v==9:
            daily_var.attrs["units"]='m/s'
        
        #  输出
        daily_var.to_netcdf(loc_var+variable+'_'+str(n).zfill(3)+'.nc')
        

#%%
