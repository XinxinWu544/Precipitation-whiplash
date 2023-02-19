# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:44:22 2023

@author: daisukiiiii
"""
# -*- coding: utf-8 -*-


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

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/code_whiplash/Demo/'
result_dir = basic_dir +'2-2.Original whiplash events/6-4.CESM_LENS_daily_whiplash_stats_baseline_40_ensemble_new_intensity/' 

#%% ------------------------------------------- parameters ---------------------------------------


threshold=[0.05,0.1,0.15,0.2,0.25,0.75, 0.8,0.85,0.9,0.95]
MIN_period=[10,15,20,25,30,35,40,50,60,90]

num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

year=np.array([1920+i for i in range(181)])
rcp85_start=86
extremes=['wet','dry','dry2wet','wet2dry']
MIN_period_sub=[20,25,30,35,40]

threshold_type=['Series_mean']

#%%#%%   -------------------------------  functions  -------------------------------



def cal_w_d_events(j):
    global Summary_Stats,Rough_Stats
    #global anom_prec,extreme_type,daily_event,quan_threshold

    #j=var[0]
    #k=var[1]
    '''
    if (thes_typ == 'Series_mean') | (thes_typ == 'Daily_SPI_proxy'):
        a=Summary_Stats[j].get('anom_prec')
    else :
        a=Summary_Stats[j].get('anom_prec').values
    '''
    a=Summary_Stats[j].get('anom_prec')
    
    #a=anom_prec[:,j,k].values
    b=Rough_Stats[j].get(extreme_type+'_events')
    
    
    if extreme_type=='dry':
        thres=Summary_Stats[j].get('quan_prec').sel(quantile= round(th+mth,2)).values
    else :
        thres=Summary_Stats[j].get('quan_prec').sel(quantile= round(1-th-mth,2)).values
  
        
    w_event=np.where(b>=0)[0] #loc of events
   
    w_interval=np.diff(w_event) 
    w_ind_event_start=np.append(0,np.where(w_interval >1)[0]+1) #index of each independent event starts
    where_independent_event_start=w_event[w_ind_event_start] #index of each independent event in the whold series
    
    ##index of each independent event ends
    w_ind_event_end=np.where(   ( (w_interval[:len(w_interval)-1]==1) & (w_interval[1:]!=1) ) \
                | ((w_interval[:len(w_interval)-1]!=1) & (w_interval[1:]!=1)) )[0]+1
    if w_interval[0]>1: 
        w_ind_event_end=np.append(0,w_ind_event_end)
        
    #add the last day
    where_independent_event_end=np.append(w_event[w_ind_event_end],w_event[len(w_event)-1])
    
    #2.merge events
   
    #cal the intervals between envets 
    w_start_depend=np.where((where_independent_event_start[1:]-where_independent_event_end[0:-1])<=min_period )[0]
    
    c=b.copy()
   
    for i in range(len(w_start_depend)):
        #print(i)
      
        w_interarr_start=where_independent_event_start[w_start_depend[i]]
        
        w_interarr_end=where_independent_event_start[w_start_depend[i]+1]
      
        m=a[w_interarr_start:w_interarr_end].mean() #一开始算错，a[b:c]实际到c-1
        #print(m)
        
        if (thes_typ == 'Series_mean') | (thes_typ == 'Series_mean_lens') :
            m_thres=thres
        elif thes_typ == 'Daily_SPI_proxy':
          
            if (w_interarr_end%365 - w_interarr_start%365)<0:
                m_thres=np.append(thres[:w_interarr_end%365],thres[w_interarr_start%365:]).mean()
            else:
                m_thres=thres[w_interarr_start%365:w_interarr_end%365].mean()
        
        else:  
            m_thres=thres[int(np.floor(w_interarr_start/365))]
        
        if extreme_type == 'dry':
            if m <=  m_thres:
                #print(i)
                
                c[w_interarr_start:where_independent_event_end[w_start_depend[i]+1]+1 ]=c[w_interarr_start]
        elif extreme_type == 'wet':
            if m >=  m_thres:
                #print(i)
                
                c[w_interarr_start:where_independent_event_end[w_start_depend[i]+1]+1 ]=c[w_interarr_start]
      
    
   
    #cal intensity
    df = pd.DataFrame(np.vstack((c,a)).T)
    df.columns = ['index','spi']
    df_mean = df.groupby('index')['spi'].mean() 
    
    
    
    w1=np.where( (c[1:]>=0) & (np.isnan(c[0:len(c)-1])) )[0] +1
    w2=np.where( (c[0:len(c)-1]>=0) & (np.isnan(c[1:])) )[0] 
    
    
    if(c[-1]>=0):
        w2=np.append(w2,int(c[-1]))
    
    w=np.vstack((w1,w2,df_mean)).T
   
    a= np.where( (w2-w1)<3 )[0]
    for i in range(len(a)):
        c[w1[a[i]]:(w2[a[i]]+1) ]=np.nan
       
  
    return {'uni_event':np.float32(w[a,:]),'independent_event':c }




def cal_whiplash_events(j):
    global independent_dry,independent_wet
    #j=var
    
   
    a=Summary_Stats[j].get('anom_prec')
    
    
    independent_dry_bool=independent_dry[j].copy()
    independent_dry_bool[independent_dry_bool>=0]=1
    independent_dry_bool[np.isnan(independent_dry_bool)]=0
    
    independent_wet_bool=independent_wet[j].copy()
    independent_wet_bool[independent_wet_bool>=0]=2
    independent_wet_bool[np.isnan(independent_wet_bool)]=0
    


    e_s=independent_wet_bool+independent_dry_bool
    
    # 5.1 identify dry to wet 
    
    d_end=np.where( (np.diff(e_s)== -1)&(e_s[:-1]==1) )[0] 
  
    dw_d_end=[] 
    dw_w_start=[]
    for i in range(len(d_end)):
        if (2 in e_s[d_end[i]:d_end[i]+inter_period+1]): 
            
            dw_d_end.append(d_end[i])
            dw_w_start.append(d_end[i]+np.where(e_s[d_end[i]:d_end[i]+inter_period+1]==2)[0][0]) 
    
    
    bbb=pd.DataFrame(np.vstack((independent_dry[j],a)).T)
    bbb.columns = ['index','spi']
    bbb_min = bbb.groupby('index')['spi'].min() 
    bbb_dry_sum = bbb.groupby('index')['spi'].sum() 
    
    
    bbb=pd.DataFrame(np.vstack((independent_wet[j],a)).T)
    bbb.columns = ['index','spi']
    bbb_max = bbb.groupby('index')['spi'].max() 
    bbb_wet_sum = bbb.groupby('index')['spi'].sum() 
    
    cumu_start = bbb_min.get(independent_dry[j][dw_d_end])
    cumu_end = bbb_max.get(independent_wet[j][dw_w_start])
    
    cumu_start_sum = bbb_dry_sum.get(independent_dry[j][dw_d_end])
    cumu_end_sum = bbb_wet_sum.get(independent_wet[j][dw_w_start])
    #%
    
    d_to_w=np.vstack((dw_d_end,cumu_start,cumu_start_sum,dw_w_start,cumu_end,cumu_end_sum)).T
   
    # 5.2 identify wet to dry 
   
    w_end=np.where( (np.diff(e_s)== -2)&(e_s[:-1]==2) )[0] 
  
    wd_w_end=[] 
    wd_d_start=[]
    for i in range(len(w_end)):
        if (1 in e_s[w_end[i]:w_end[i]+inter_period+1]): 
            
            wd_w_end.append(w_end[i])
            wd_d_start.append(w_end[i]+np.where(e_s[w_end[i]:w_end[i]+inter_period+1]==1)[0][0]) 
    
 
    cumu_start = bbb_max.get(independent_wet[j][wd_w_end])
    cumu_end = bbb_min.get(independent_dry[j][wd_d_start])
    
 
    
    
    cumu_start_sum = bbb_wet_sum.get(independent_wet[j][wd_w_end])
    cumu_end_sum = bbb_dry_sum.get(independent_dry[j][wd_d_start])
    #%
    w_to_d=np.vstack((wd_w_end,cumu_start,cumu_start_sum,wd_d_start,cumu_end,cumu_end_sum)).T

   
    return ({'wd':w_to_d,'dw':d_to_w})

def cal_anom_quantile_Series_mean(var):
    
    global cum_prec

    j=var[0]
    k=var[1]
    day_cycle=pd.Series(cum_prec[:,j,k]).groupby(calendar_day).mean()
    sd=pd.Series(cum_prec[:,j,k]).groupby(calendar_day).std()
    anom_prec= (cum_prec[:,j,k]-np.tile(day_cycle,nyear))/np.tile(sd,nyear)
    quan_prec=anom_prec[np.where( (year <= 2019) & (year>=1979) )[0]].quantile(threshold)
    return {'anom_prec':anom_prec.values,'quan_prec':quan_prec}


def rough_event_Series_mean(j):
    
    global Summary_Stats    
    a=Summary_Stats[j].get('anom_prec')
    b=Summary_Stats[j].get('quan_prec')
    
    dry_events=(a > b.sel(quantile= round(1-q,2)).values ).astype(float) #找出事件
    c = dry_events.cumsum()
    c[np.where(np.array(dry_events)!=0)]=np.nan
    c[0:(min_period-1)]=np.nan
    

    wet_events=(a < b.sel(quantile= q ).values  ).astype(float) 
    d = wet_events.cumsum()
    d[np.where(np.array(wet_events)!=0)]=np.nan
    d[0:(min_period-1)]=np.nan
    return {'dry_events':c,'wet_events':d}






#%%

# select linear detrend; 30-day cumulative prcp; 90th threshold
all_plans=[]
for dtrd_typ in [1,2,3,4]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9))

#%%

for plan in [1]:
    
    Dtrd_typ=all_plans[plan][0]
    Thes_typ=all_plans[plan][1]
    Min_period=all_plans[plan][2]
    Q=all_plans[plan][3]
       
    for n in [1,2]:
        
        #1. creat folder for each ensemble member.
        if os.path.exists(result_dir+str(n).zfill(3)+'/')==False:
            os.makedirs(result_dir+str(n).zfill(3)+'/')
        method_dir=result_dir+str(n).zfill(3)+'/'
        
        print('Ensemble='+str(n).zfill(3))
        print(datetime.now())
        
        
        #2. read in prcp data 
        data=xr.open_dataarray(basic_dir + '1.Prcp data/PRECT_'+str(n).zfill(3)+'_interp2deg_NEC.nc')
        print('inputed data:'+str(datetime.now() ))
        
        
        #3. change to calendar days
        original_time=data.time
        month=data['time.month'].values
        day=data['time.day'].values
        year=data['time.year'].values
        calendar_day=[str(month[i]).zfill(2)+'-'+str(day[i]).zfill(2) for i in range(len(month))]
        nyear=len(np.unique(year))
        
        #4. detrend daily prcp
        for dtrd_typ in [Dtrd_typ]: 
            print('detrend_method='+str(dtrd_typ))
            #dtrd_typ=2
            if dtrd_typ == 1:
                data1=data.copy(deep=True)
            if dtrd_typ in [2,3,4]:
                data_year = xr.open_dataarray(basic_dir + '1.Prcp data/PRECT_annual_mean_'+str(n).zfill(3)+'.nc')
                data_year_dtrd = xr.open_dataarray(basic_dir + '1.Prcp data/PRECT_annual_mean_'+str(n).zfill(3)+'_detrend'+str(dtrd_typ-1)+'.nc')
                data1=data.copy(deep=True).assign_coords({'lat':data_year_dtrd.lat})
                data1 =data1.groupby('time.year')-(data_year-data_year_dtrd) #detrended daily data
            print('detrend done ! ') 
            
            #5. cal cumulative prcp
            #min_period=30
            for min_period in [Min_period]:
                
                cum_prec=data1.rolling(time=min_period,center=False).sum()
                
                input_combo=[]
                for j in range(data1.shape[1]):
                    for k in range(data1.shape[2]):
                        input_combo.append((j,k))
                        
                len_input_combo = []
                for j in range(len(input_combo)):
                    len_input_combo.append(j)
                
                #6. cal cumu-prec climotology, sd and quantile
                for thes_typ in [Thes_typ] :
                    #thes_typ='Series_mean'
                   
                    start=datetime.now()
                    pool = multiprocessing.Pool(processes = 12) # object for multiprocessing
                    Summary_Stats = list(tqdm.tqdm(pool.imap( globals()['cal_anom_quantile_'+thes_typ] , input_combo), 
                                                   total=len(input_combo), position=0, leave=True))
                    pool.close()    
                    del(pool)
                    gc.collect()
                    
                    end=datetime.now()
                    print('spend cal anom and quan:'+str(end-start)) # about 5 mins 
                    
                    #7. cal rought dry and wet extremes
                    for q in [Q]:
                        #q=0.9
                 
                        start=datetime.now()
                        pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
                        Rough_Stats = list(tqdm.tqdm(pool.imap( globals()['rough_event_'+thes_typ], len_input_combo), 
                                                       total=len(len_input_combo), position=0, leave=True))
                        pool.close()    
                        del(pool)
                        gc.collect()

                        end=datetime.now()
                        print('spend cal rough wet and dry event indexs:'+str(end-start)) #about   11 mins !!!!

                        th= round(1-q,2)
                        
                        #### 8. merge rough events
                        for mth in [0.05]:
                            #mth=0.05

                            print('cal dry and wet events')
                            extreme_type='dry'
                            
                            pool = multiprocessing.Pool(processes = 24) # object for multiprocessing
                            Summary_Stats_dry = list(tqdm.tqdm(pool.imap(cal_w_d_events, len_input_combo), 
                                                           total=len(len_input_combo), position=0, leave=True))
                            pool.close()
                            del(pool)
                            gc.collect()
                            
                            #################################
                            extreme_type='wet'
                      
                            pool = multiprocessing.Pool(processes = 24) # object for multiprocessing
                            Summary_Stats_wet = list(tqdm.tqdm(pool.imap(cal_w_d_events, len_input_combo), 
                                                           total=len(len_input_combo), position=0, leave=True))
                            pool.close()
                            del(pool)
                            
                            
                            UNI_DRY=[]
                            for j in range(len(input_combo)):
                                UNI_DRY.append(Summary_Stats_dry[j].get('uni_event'))
                            
                            UNI_WET=[]
                            for j in range(len(input_combo)):
                                UNI_WET.append(Summary_Stats_wet[j].get('uni_event'))
                            
                            
                            #save
                            np.save(method_dir+'dry_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(round((1-th),2))+'.npy',UNI_DRY)
    
                            np.save(method_dir+'wet_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(round((1-th),2))+'.npy',UNI_WET)
                           
                            del(UNI_WET)
                            del(UNI_DRY)
    
                            print('cal whiplash events')
                            independent_dry=[]
                            for j in range(len(input_combo)):
                                independent_dry.append(Summary_Stats_dry[j].get('independent_event'))
                            
                            
                            independent_wet=[]
                            for j in range(len(input_combo)):
                                independent_wet.append(Summary_Stats_wet[j].get('independent_event'))
                                
                                
                            del(Summary_Stats_dry)
                            del(Summary_Stats_wet)
                            
                            
                            for ip in [1,2,3]:
                                inter_period=np.ceil(min_period/ip).astype(int)
                                
                                pool = multiprocessing.Pool(processes = 24) # object for multiprocessing
                                Summary_Stats_whiplash = list(tqdm.tqdm(pool.imap(cal_whiplash_events, len_input_combo ), 
                                                               total=len(len_input_combo), position=0, leave=True))
                                pool.close()
                                del(pool)
                                gc.collect()
                                
                                dry_to_wet=[]
                                for j in range(len(input_combo)):
                                    dry_to_wet.append(Summary_Stats_whiplash[j].get('dw'))
                                
                                wet_to_dry=[]
                                for j in range(len(input_combo)):
                                    wet_to_dry.append(Summary_Stats_whiplash[j].get('wd'))
                                
                                
                                np.save(method_dir+'dry_to_wet_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(round((1-th),2))+'_inter_period_'+str(inter_period)+'.npy',dry_to_wet)
                                np.save(method_dir+'wet_to_dry_'+thes_typ+'_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                        '_quantile_'+str(round((1-th),2))+'_inter_period_'+str(inter_period)+'.npy',wet_to_dry)
                            
                                del(Summary_Stats_whiplash)
                                del(dry_to_wet)
                                del(wet_to_dry)
                            del(independent_wet)
                            del(independent_dry)                        
                        del(Rough_Stats)
                        gc.collect()
                    del(Summary_Stats)
                    gc.collect()
               
        del(data)
        gc.collect()
