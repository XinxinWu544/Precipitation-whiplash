
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:41:26 2023

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
import tqdm
import multiprocessing
import gc
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib
import glob
basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'


forcing=['AER','GHG','BMB']
#%%

mask_o=pd.read_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/land_mask.csv')


#%%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean_lens',30,0.9,30))

plan=1
monsoon_name = ['WAfriM','SAsiaM','SAmerM','NAmerM','EAsiaM','AusMCM']

monsoon_regions = xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/monsoon_regions_2deg_mask/monsoon_regions.nc')

color_d_type=['#1f77b4','#2ca02c','#bcbd22','#ff7f0e']

colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']


#%%
####  single forcing 的结果 ###

dtrd_typ=all_plans[plan][0]
thes_typ=all_plans[plan][1]
min_period=all_plans[plan][2]
q=all_plans[plan][3]
inter_period=all_plans[plan][4]

feature='frequency'
ms = 1
ex='wet_to_dry'
forcing=['AER','GHG','BMB']
#%%
#全球、陆地平均
rolling_year = 1

events_mean= {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        ## CESM-LENS
        events=xr.open_dataarray( basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        mask= xr.DataArray(mask_o,dims=('lat','lon'),coords={'lon':events.lon,'lat':events.lat})
        
        lats=events.lat
        # 1.全球 
        vars()[ex+'_region'] = events.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099))
        vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
        vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
        
        a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                   pd.Series(vars()[ex+'_region_quan'][1]),
                   pd.Series(vars()[ex+'_region_mean'])],axis=1)
        
        events_mean.update({ 'LENS'+'~'+ex+'~'+feature+'~global' :a  })
        
        ## 2、陆地
        
        vars()[ex+'_region'] = (events*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099))
        vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
        vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
        
        a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                   pd.Series(vars()[ex+'_region_quan'][1]),
                   pd.Series(vars()[ex+'_region_mean'])],axis=1)
        
        events_mean.update({ 'LENS'+'~'+ex+'~'+feature+'~land' :a  })
        
        
        
        for ms in range(6) :
            print(ms)
    
            reg = monsoon_regions[:,:,ms]
            event_m = events * reg
            
            # 3 不同地区
            vars()[ex+'_region'] = event_m.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,2099))
            vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            ''''''
            plt.fill_between(vars()[ex+'_region_quan'].year, y1=vars()[ex+'_region_quan'][0], y2=vars()[ex+'_region_quan'][1],alpha=0.2)
            plt.plot(vars()[ex+'_region_quan'].year,vars()[ex+'_region_mean'])
            
            ''''''
            a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                       pd.Series(vars()[ex+'_region_quan'][1]),
                       pd.Series(vars()[ex+'_region_mean'])],axis=1)
            
            events_mean.update({ 'LENS'+'~'+ex+'~'+feature+'~'+monsoon_name[ms] :a  })
            


## single forcing 

for x in [0,1,2]:
    for feature in ['frequency','intensity','duration']:
        for ex in ['dry_to_wet','wet_to_dry']:
            print(forcing[x])
            ## CESM-LENS
            events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-4.CESM_SF_event_frequency_duration_intensity_all/'+feature+'_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            lats=events.lat
            # 1.全球 
            if x in [0,1] :
                last_year =2079
            else :
                last_year = 2028
            vars()[ex+'_region'] = events.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,last_year))
            vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            
            a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                       pd.Series(vars()[ex+'_region_quan'][1]),
                       pd.Series(vars()[ex+'_region_mean'])],axis=1)
            a.index=vars()[ex+'_region_quan'].year.values
            events_mean.update({ forcing[x] +'~' + ex+'~'+feature+'~global' :a  })
            
            #陆地
            vars()[ex+'_region'] = (events*mask).weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,last_year))
            vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
            
            a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                       pd.Series(vars()[ex+'_region_quan'][1]),
                       pd.Series(vars()[ex+'_region_mean'])],axis=1)
            a.index=vars()[ex+'_region_quan'].year.values
            events_mean.update({ forcing[x] +'~' + ex+'~'+feature+'~land' :a  })
            
            
            for ms in range(6) :
                print(ms)
        
                reg = monsoon_regions[:,:,ms]
                event_m = events * reg
                
                # 1.全球 
                vars()[ex+'_region'] = event_m.weighted(np.cos(np.deg2rad(lats))).mean(dim=['lat','lon']).sel( year=slice(1921,last_year))
                vars()[ex+'_region_mean'] = vars()[ex+'_region'].mean('ensemble').rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
                vars()[ex+'_region_quan'] = vars()[ex+'_region'].quantile(dim='ensemble',q=[0.05,0.95]).rolling(dim={'year':rolling_year},center= True).mean().dropna('year')
                ''''''
                plt.fill_between(vars()[ex+'_region_quan'].year, y1=vars()[ex+'_region_quan'][0], y2=vars()[ex+'_region_quan'][1],alpha=0.2)
                plt.plot(vars()[ex+'_region_quan'].year,vars()[ex+'_region_mean'])
                
                ''''''
                a = pd.concat([pd.Series(vars()[ex+'_region_quan'][0]),
                           pd.Series(vars()[ex+'_region_quan'][1]),
                           pd.Series(vars()[ex+'_region_mean'])],axis=1)
                a.index=vars()[ex+'_region_quan'].year.values
                
                events_mean.update({ forcing[x] +'~' + ex+'~'+feature+'~'+monsoon_name[ms] :a  })

np.save(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.LENS_and_SF_events_global_and_region_mean.npy',events_mean)

#%% 通过双边t检验来判断是否产生显著影响
'''
from scipy import stats
#stats.ttest_ind(B,A,equal_var= False)
#feature = 'frequency'
forcing=['AER','GHG','BMB']

def cal_student_t(i):
    global events_lens1,events_sf1
    j=i[0]
    k=i[1]
    t_result_all = stats.ttest_ind( events_sf1[:,j,k].mean('ensemble'),events_lens1[:,j,k].mean('ensemble'),nan_policy='omit' )[1]
    return t_result_all

#% 显著性

student_t_forcings= {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry','wet','dry_to_wet','wet_to_dry']:
        
        events_lens=xr.open_dataarray(dai_basic_dir + 'code/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        
        input_combo=[]
        for j in range(events_lens.shape[1]):
            for k in range(events_lens.shape[2]):
                input_combo.append((j,k))
            
        ## single forcing 
        
        x=1
        for x in [0,1]:
            events_sf=xr.open_dataarray(dai_basic_dir + 'code/8-4.CESM_SF_event_frequency_duration_intensity_all/'+feature+'_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            print(datetime.now() )
            #%   #1 、当前时间段的差异
         
            events_lens1=events_lens.sel( year=slice(1979,2019))
            events_sf1=events_sf.sel( year=slice(1979,2019))    
            
            pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
            Summary_Stats = list(tqdm.tqdm(pool.imap( cal_student_t , input_combo), 
                                           total=len(input_combo), position=0, leave=True))
            pool.close()
            
            t_result_all =np.full(shape=(90,180),fill_value=np.nan)
            for j in range(90):
                #print(j)
                for k in range(180):
                    t_result_all[j,k] = Summary_Stats[ j*180+k ]

            student_t_forcings.update({forcing[x]+'~'+ex+'~'+feature+'~1979-2019' : t_result_all} )
            
            
            #%  #2 、 整个时间段
            t_result_all =np.full(shape=(90,180),fill_value=np.nan)
            events_lens1=events_lens.sel( year=slice(1921,2028))
            events_sf1=events_sf.sel( year=slice(1921,2028))    
            
            pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
            Summary_Stats = list(tqdm.tqdm(pool.imap( cal_student_t , input_combo), 
                                           total=len(input_combo), position=0, leave=True))
            pool.close()
            t_result_all =np.full(shape=(90,180),fill_value=np.nan)
            for j in range(90):
                #print(j)
                for k in range(180):
                    t_result_all[j,k] = Summary_Stats[ j*180+k ]    
            student_t_forcings.update({forcing[x]+'~'+ex+'~'+feature+'~1921-2028' : t_result_all} )
            #%
            #  对于 AER 和GHG，补充到未来时段的结果
            if x in [0,1]:
                # 3、
                t_result_all =np.full(shape=(90,180),fill_value=np.nan)
                events_lens1=events_lens.sel( year=slice(1921,2079))
                events_sf1=events_sf.sel( year=slice(1921,2079))    
                pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
                Summary_Stats = list(tqdm.tqdm(pool.imap( cal_student_t , input_combo), 
                                               total=len(input_combo), position=0, leave=True))
                pool.close()
                
                
                t_result_all =np.full(shape=(90,180),fill_value=np.nan)
                for j in range(90):
                    print(j)
                    for k in range(180):
                        t_result_all[j,k] = Summary_Stats[ j*180+k ]    
                        
                student_t_forcings.update({forcing[x]+'~'+ex+'~'+feature+'~1921-2079' : t_result_all} )
                # 4、
                t_result_all =np.full(shape=(90,180),fill_value=np.nan)
                events_lens1=events_lens.sel( year=slice(2040,2079))
                events_sf1=events_sf.sel( year=slice(2040,2079))    
                pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
                Summary_Stats = list(tqdm.tqdm(pool.imap( cal_student_t , input_combo), 
                                               total=len(input_combo), position=0, leave=True))
                pool.close()
                
                
                t_result_all =np.full(shape=(90,180),fill_value=np.nan)
                for j in range(90):
                    print(j)
                    for k in range(180):
                        t_result_all[j,k] = Summary_Stats[ j*180+k ]    
    
                student_t_forcings.update({forcing[x]+'~'+ex+'~'+feature+'~2040-2079' : t_result_all} )
#%%
np.save('/media/dai/suk_code/research/4.East_Asia/Again/code_new/11-1.LENS_SF_global_and_region_stats/student_t_test_of_forcings.npy',student_t_forcings)
'''

#%%  ## forcing的相对影响


IF_forcings_global_and_regional_mean = {}

first_years = [1921,1979,2028,1921,2040,2060,2079]
last_years = [2028,2019,2028,2079,2079,2079,2079]


for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        events_lens=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        
        ## single forcing 
        
        x=0
        for x in [0,1,2]:
            events_sf=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-4.CESM_SF_event_frequency_duration_intensity_all/'+feature+'_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            print(datetime.now() )
            
            if x==2 :
                n_years = [0,1,2]
            else:
                n_years = [0,1,2,3,4,5,6]
                
            for ny in n_years:
                first_year = first_years[ny]
                last_year = last_years[ny]
     
        
                #取时期多年平均
                events_lens1=events_lens.sel( year=slice(first_year,last_year)).mean('year').mean('ensemble')
                events_lens1 = events_lens1.where(events_lens1>0) # 将 =0的设为np.nan
                events_sf1 = events_sf.sel( year=slice(first_year,last_year)).mean('year')
                
                #先球全球平均，再取均值和spread值
                events_sf1_mean_global = events_sf1.weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .mean('ensemble')
                events_sf1_quan_global = events_sf1.weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .quantile([0.05,0.95],'ensemble')
    
                events_lens1_global = events_lens1.weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat'))

                relative_if_mean = (events_lens1_global - events_sf1_mean_global)*100/events_lens1_global
                relative_if_quan = (events_lens1_global - events_sf1_quan_global)*100/events_lens1_global
    
                IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global' : relative_if_mean} )
                IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~quan~global' : relative_if_quan} )
    
                ##陆地平均
                events_sf1_mean_global = (events_sf1*mask).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .mean('ensemble')
                events_sf1_quan_global = (events_sf1*mask).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .quantile([0.05,0.95],'ensemble')
    
                events_lens1_global = (events_sf1*mask).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat'))

                relative_if_mean = (events_lens1_global - events_sf1_mean_global)*100/events_lens1_global
                relative_if_quan = (events_lens1_global - events_sf1_quan_global)*100/events_lens1_global
    
                IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~land' : relative_if_mean} )
                IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~quan~land' : relative_if_quan} )
    
    
    

                for ms in range(6) :
                    print(ms)
                
                    reg = monsoon_regions[:,:,ms]
                    
                    
                    events_sf1_mean_regional = (events_sf1* reg).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .mean('ensemble')
                    events_sf1_quan_regional = (events_sf1* reg).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat')) .quantile([0.05,0.95],'ensemble')
                    
                    
                    events_lens1_regional = (events_lens1 *reg).weighted(np.cos(np.deg2rad(events_sf1.lat))).mean(('lon','lat'))
                    
                    relative_if_mean_regional = (events_lens1_regional - events_sf1_mean_regional)*100/events_lens1_regional
                    relative_if_quan_regional = (events_lens1_regional - events_sf1_quan_regional)*100/events_lens1_regional
                  
                    
                    IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~'+monsoon_name[ms] : relative_if_mean_regional} )
                      
                    IF_forcings_global_and_regional_mean.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~quan~'+monsoon_name[ms] : relative_if_quan_regional} )
                   

np.save(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.IF_forcings_global_and_regional_mean.npy',IF_forcings_global_and_regional_mean)

#%%
IF_forcings_distribution = {}

first_years = [1921,1979,1921,2040,2060]
last_years = [2028,2019,2079,2079,2079]


for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        events_lens=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        
        ## single forcing 
        
        x=0
        for x in [0,1,2]:
            events_sf=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-4.CESM_SF_event_frequency_duration_intensity_all/'+feature+'_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            print(datetime.now() )
            
            if x==2 :
                n_years = [0,1]
            else:
                n_years = [0,1,2,3,4]
                
            for ny in n_years:
                first_year = first_years[ny]
                last_year = last_years[ny]
     
        
                #取时期多年平均
                events_lens1=events_lens.sel( year=slice(first_year,last_year)).mean('year').mean('ensemble')
                events_lens1 = events_lens1.where(events_lens1>0) # 将 =0的设为np.nan
                events_sf1 = events_sf.sel( year=slice(first_year,last_year)).mean('year').mean('ensemble')
                
                #取均值
                
                relative_if_mean = (events_lens1 - events_sf1)*100/events_lens1
               
    
                IF_forcings_distribution.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global' : relative_if_mean} )
                

np.save(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.IF_forcings_distribution.npy',IF_forcings_distribution)

#%% 用多少个集合agree来表示显著性

If_sig_agreement = {}

first_years = [1921,1979,1921,2040,2060]
last_years = [2028,2019,2079,2079,2079]


for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
        
        events_lens=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
        
        ## single forcing 
        
        x=0
        for x in [0,1,2]:
            events_sf=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-4.CESM_SF_event_frequency_duration_intensity_all/'+feature+'_SF_X'+forcing[x]+
                                    '_'+ex+'_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
            
            
            print(datetime.now() )
            
            if x==2 :
                n_years = [0,1]
            else:
                n_years = [0,1,2,3,4]
                
            for ny in n_years:
                first_year = first_years[ny]
                last_year = last_years[ny]
     
        
                #取时期多年平均
                events_lens1=events_lens.sel( year=slice(first_year,last_year)).mean('year').mean('ensemble')
                events_lens1 = events_lens1.where(events_lens1>0) # 将 =0的设为np.nan
                events_sf1 = events_sf.sel( year=slice(first_year,last_year)).mean('year').mean('ensemble')
                events_sf1_ensembles = events_sf.sel( year=slice(first_year,last_year)).mean('year')
                
                #取均值
                
                relative_if_mean = (events_lens1 - events_sf1)#*100/events_lens1
                
                relative_if_ensembles = (events_lens1 - events_sf1_ensembles)#*100/events_lens1
                relative_if_mean [:]=np.sign(relative_if_mean.values)
                relative_if_ensembles [:]=np.sign(relative_if_ensembles.values)
                
                a = relative_if_ensembles.where(relative_if_ensembles == relative_if_mean)
                a[:] = np.abs(a)
                s= a.sum('ensemble')
                
                #s1=s.values
                
    
                If_sig_agreement.update({forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global' : s} )
                

np.save(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.If_sig_agreement.npy',If_sig_agreement)
#%%

for feature in ['frequency']:
    #fig = plt.figure(figsize = (14/2.54, 14/2.54)) # 宽、高
    for ex in ['dry_to_wet','wet_to_dry']:
    
        event_LENS = xr.open_dataarray(glob.glob(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+
                                          feature+'_40ensemble_'+ex+'*')[0]).sel(year=slice(1921,2099) )
        #
         ##计算全球分布

        event_rolling =  event_LENS.rolling(year=10,center=True).mean().dropna('year')

        event_rolling_change_mean = (event_rolling - event_rolling[0,:,:,:].values).mean('ensemble')
        #event_rolling_change_sd = (event_rolling ).std('ensemble')
        event_rolling_change_sd = (event_rolling - event_rolling[0,:,:,:].values).std('ensemble')
        snr_rolling_positive = (  (event_rolling_change_mean/event_rolling_change_sd >= 1)      ).astype(int)

        #%
        first_year = np.full(shape= (90,180),fill_value=0)
        num_of_year = np.full(shape= (90,180),fill_value=0)

        year0=snr_rolling_positive.year[0].values

        for j in range(90):
            for k in range(180):
                loc = np.where(snr_rolling_positive[:,j,k].values == 1 )[0]
                loc_0 = np.where(snr_rolling_positive[:,j,k].values == 0 )[0]
                
                l =  len(loc)
                
                if l >0 :
                    num_of_year[j,k] = l
                    first_year[j,k] = year0 + loc_0[-1]
                    
        #first_year[first_year>=2080]=np.nan            

        #first_year1 = first_year.copy()
        #first_year1[first_year1>0] =1
        first_year_LENS_positive = first_year.copy()

        #%
        ############################
        snr_rolling_negative = (  (event_rolling_change_mean/event_rolling_change_sd <= -1)      ).astype(int)

        #%
        first_year = np.full(shape= (90,180),fill_value=0)
        num_of_year = np.full(shape= (90,180),fill_value=0)

        year0=snr_rolling_positive.year[0].values

        for j in range(90):
            for k in range(180):
                loc = np.where(snr_rolling_negative[:,j,k].values == 1 )[0]
                loc_0 = np.where(snr_rolling_negative[:,j,k].values == 0 )[0]
                
                l =  len(loc)
                
                if l >0 :
                    num_of_year[j,k] = l
                    first_year[j,k] = year0 + loc_0[-1]
                    
        #first_year[first_year>=2080]=np.nan            

        #first_year1 = first_year.copy()
        #first_year1[first_year1>0] =1
        first_year_LENS_negative = first_year.copy()


        first_year_LENS_negative[first_year_LENS_negative>=2080]=0       
        first_year_LENS_positive[first_year_LENS_positive>=2080]=0       


        first_year_LENS_negative[first_year_LENS_negative>0]=1
        first_year_LENS_positive[first_year_LENS_positive>0]=1
        
        vars()['first_year_LENS_negative_' +ex+'_'+feature] = first_year_LENS_negative
        vars()['first_year_LENS_positive_' +ex+'_'+feature] = first_year_LENS_positive
        
        pd.DataFrame(vars()['first_year_LENS_positive_' +ex+'_'+feature]).to_csv(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.'+'first_year_LENS_positive_' +ex+'_'+'feature.csv',index=False)
        pd.DataFrame(vars()['first_year_LENS_negative_' +ex+'_'+feature]).to_csv(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-17.'+'first_year_LENS_negative_' +ex+'_'+'feature.csv',index=False)