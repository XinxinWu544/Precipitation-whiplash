'''
module load python/3.9.6
module load scipy-stack
source ~/hzqENV/bin/activate

'''

# CMIP6数据预处理需要的步骤
#1.合并1920-2100,多余的不要
#2.将数据统一为365天
#3.统一时间轴
#4.纬度要进行反转
#5.统一空间分辨率为2x2 
#6.注意所需要的单位


import xarray as xr
import glob 
from datetime import datetime
from datetime import timedelta
import numpy as np

#存到1-COMBINED中
#%% #设置一个时间轴
#只能进行手动改时间轴
full_time=np.array([datetime(1920,1,1)+timedelta(i) for i in range(0,66110)])
month=np.array([i.month for i in full_time])
day=np.array([i.day for i in full_time])

n=np.where( (month==2) & (day==29))[0] ##2月29号的位置，现在要将这些位置去掉
missing_time=[]
for i in n:
    #print(datetime(1850,1,1)+timedelta(int(i)))
    missing_time.append(datetime(1920,1,1)+timedelta(int(i)))
no_leap_time=np.delete(full_time,n) #得到一个全是365天的时间轴

#%% 
Experiments=['ACCESS-ESM1-5','CanESM5','CESM2-WACCM','CMCC-CM2-SR5',
             'CMCC-ESM2','EC-Earth3','EC-Earth3-CC','EC-Earth3-Veg',
             'EC-Earth3-Veg-LR','GFDL-CM4','GFDL-ESM4','INM-CM4-8',
             'INM-CM5-0','IPSL-CM6A-LR','KIOST-ESM','MIROC6',
             'MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM',
             'NorESM2-MM','TaiESM1']

#%% 

for j in range(len(Experiments)):
    
    name=np.sort(glob.glob('/media/dai/DATA1/pr/'+Experiments[j]+'/pr*'))
    
    # 查看realization数目
    realizations=np.unique(np.array([name[i].split('_')[len(name[i].split('_'))-3] for i in range(len(name)) ]))
    
    # %%1.将数据按时间进行合并
    
    for k in range(len(realizations)):
        print(Experiments[j]+'_'+realizations[k])
        name_sub=np.sort(glob.glob('/media/dai/DATA1/pr/'+Experiments[j]+'/pr*_*'+realizations[k]+'*'))
        
        ##有些数据集有2100+的，去掉
        name1=[]
        for i in range(len(name_sub)):
            if int(name_sub[i][len(name_sub[i])-20:len(name_sub[i])-16:1])<=2100:
                name1.append(name_sub[i])
        data=[]
        for i in range(len(name1)):
            #print(i)
            #b=xr.open_dataset(name1[i])[var[k]].sel(lon=slice(110,120),lat=slice(10,20))
            b=xr.open_dataset(name1[i])['pr']
            print(i)
            print(b[1,1,1].values)
            #% 3.反转纬度
            b=b.reindex(lat=list(reversed(b.lat)))
            #% 4.进行重新插值
            b=b.interp(lat=np.arange(90,-90,-2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
            b=b.interp(lon=np.arange(0,360,2),kwargs={"fill_value": "extrapolate"}) #填上边上的缺失值
            
            data.append(b)
        data=xr.concat(data, dim='time')
        # 1850-1920不要
        data=data.sel(time=slice('1920-01-01','2100-12-31'))
        # 去掉重复的时间
        _, index = np.unique(data['time'], return_index=True)
        data=data.isel(time=index)
        
        #%% 2.重新插入datetime格式的时间轴,将只有365天算法的数据进行插值
        if data.shape[0] == 66110:
            print('is a full time dataset')
            data['time']=full_time #将datetime格式的数据放入
            data=data.drop(missing_time,dim='time') #去掉leap days
            
        elif data.shape[0] == 66065:
            print('is a np_leap time dataset')
            data['time']=no_leap_time
        
        else:     
        ##！！！！如果时间长度不对（既不是366也不是365的计数方式），记得报错
            print('time != full or != np_leap, somewhere wrong!!!!!!!!!!!')
            
        
        #%% 5.调整单位
        data = data* 3600 * 24
        data.attrs['units']='mm/day'
        
        data.to_netcdf('/media/dai/DATA1/pr/1-COMBINED/pr_day_'+Experiments[j]+'_'+realizations[k]+'.nc')
        del(data)
    




                
