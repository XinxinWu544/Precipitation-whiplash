#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:55:01 2023

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
import shapefile,cmaps
from matplotlib.path import Path
from matplotlib.patches import PathPatch

basic_dir = '/media/dai/disk2/suk/research/4.East_Asia/Again/'


#%%
all_plans=[]
for dtrd_typ in [1,2]:
    all_plans.append((dtrd_typ,'Series_mean',30,0.9,30))

plan=1
### cal concurrent change

global_change = {}
for feature in ['frequency','intensity','duration']:
    for ex in ['dry_to_wet','wet_to_dry']:
      
        print(feature)
        dtrd_typ=all_plans[plan][0]
        thes_typ=all_plans[plan][1]
        min_period=all_plans[plan][2]
        q=all_plans[plan][3]
        inter_period=all_plans[plan][4]
        
        
        ## CESM-LENS
        events=xr.open_dataarray(basic_dir + 'code_whiplash/3-2.Processed data from analysis/8-2.CESM_LENS_event_frequency_duration_intensity_all_new_intensity/'+feature+'_40ensemble'+
                                '_'+ex+'_'+thes_typ+'_lens_detrend_'+str(dtrd_typ)+'_of_'+str(min_period)+'_days'+
                                    '_quantile_'+str(q)+'_inter_period_'+str(inter_period)+'.nc')
        
       
        e_future = events.sel(year=slice(2060,2099)).mean('year')#.mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year')#.mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~global-period2' :trend  })

        e_future = events.sel(year=slice(2060,2099)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~ensemble_mean-period2' :trend  })
        
        ##
        '''
        e_future = events.sel(year=slice(1979,2019)).mean('year')#.mean(('ensemble'))
        e_current = events.sel(year=slice(1921,1960)).mean('year')#.mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~global-period1' :trend  })

        e_future = events.sel(year=slice(1979,2019)).mean('year').mean(('ensemble'))
        e_current = events.sel(year=slice(1921,1960)).mean('year').mean(('ensemble'))
        e_future[:] = np.abs(e_future.values)
        e_current[:] = np.abs(e_current.values)
        trend = (e_future - e_current)*100/e_current     
        global_change.update({ 'event~'+ex+'~'+feature+'~ensemble_mean-period1' :trend  })
        '''
        
        
prec=xr.open_dataarray(basic_dir + 'code_whiplash/2-2.Original whiplash events/4-0.CESM_LENS_1920_2100_ensemble_annual_mean_prec/PRECT_annual_ensemble_mean_40_ensemble.nc')
prec_future = prec.sel(year=slice(2060,2099)).mean('year')
prec_current = prec.sel(year=slice(1979,2019)).mean('year')
trend = (prec_future - prec_current)*100/prec_current    
global_change.update({ 'prec~global-period2' :trend  })
'''
prec_future = prec.sel(year=slice(1979,2019)).mean('year')
prec_current = prec.sel(year=slice(1921,1960)).mean('year')
trend = (prec_future - prec_current)*100/prec_current    
global_change.update({ 'prec~global-period1' :trend  })
'''
pn=2
regimes={}
for feature in ['frequency']:
    for ex in ['dry_to_wet','wet_to_dry']:
    
        a_m = global_change.get('event~'+ex+'~'+feature+'~ensemble_mean-period'+str(pn))
        p = global_change.get('prec~global-period'+str(pn)) 
        a = global_change.get('event~'+ex+'~'+feature+'~global-period'+str(pn)) 
        
        mPmE = (a.where(( (a/p) >=0 ) & (a>= 0)) >=0 ) .astype(int).sum('ensemble') # more p more e
        mPlE = (a.where(( (a/p) <=0 ) & (a<= 0)) <=0 ) .astype(int).sum('ensemble') # more p more e,pink
        lPlE = (a.where(( (a/p) >=0 ) & (a<= 0)) <=0 ) .astype(int).sum('ensemble') # more p more e,green
        lPmE = (a.where(( (a/p) <=0 ) & (a>= 0)) >=0 ) .astype(int).sum('ensemble') # more p more e
        
        #sig = (mPmE.where((mPmE>=36) | (mPlE>=36) | (lPlE>=36) | (lPmE>=36))>0).astype(int)  ##90%数据集同意
        sig1 = (mPmE.where((mPmE>=36) )>0).astype(int)#.values
        #sig1=sig1.where(sig>0)
        
        sig2 = (mPlE.where((mPlE>=36) )>0).astype(int)#.values
        #sig2=sig2.where(sig>0)
        
        sig3 = (lPlE.where((lPlE>=36) )>0).astype(int)#.values
        #sig3=sig3.where(sig>0)
        
        sig4 = (lPmE.where((lPmE>=36) )>0).astype(int)
        #sig4=sig4.where(sig>0)
        
        sig = (sig1+sig2+sig3+sig4).where((sig1+sig2+sig3+sig4)>0).values
     
        a1 = (a_m/p).where((a_m/p >0) & (a_m > 0))
        a1.quantile([0.33,0.66])
        a2 = (a_m/p).where((a_m/p <0) & (a_m < 0))
        a2.quantile([0.33,0.66])
        a3 = (a_m/p).where((a_m/p >0) & (a_m < 0))
        a3.quantile([0.33,0.66])
        a4 = (a_m/p).where((a_m/p <0) & (a_m > 0))
        a4.quantile([0.33,0.66])
        
        b=a1.values
        b[np.where(b>=6)]=3
        b[np.where( (b>3)&(b<6) )]=2
        b[np.where( (b>=0)&(b<3)&(b!=2) )]=1
        a1[:]=b
    
        b=a2.values
        b[np.where(b<= -6 )]=9
        b[np.where( (b>-6)&(b< -3) )]=8
        b[np.where( (b>=-3)&(b<=0) )]=7
        a2[:]=b
    
        b=a3.values
        b[np.where(b>=6)]= 12
        b[np.where( (b>=3)&(b<6) )]= 11
        b[np.where( (b>=0)&(b<3) )]= 10
        a3[:]=b
    
        b=a4.values
        b[ np.where(b<= -6) ]  = 19
        b[np.where( (b> -6)&(b<= -3 )&(b!=-9) )]= 18
        b[np.where( (b<=0)&(b> -3) )]= 17
        a4[:]=b
      
        np.isnan(a1.values)
          
        c = xr.concat([a1,a2,a3,a4], dim=pd.Index([1,2,3,4],name='regime'))
        d=np.nansum(c.values,axis=0)
        regimes.update({'a1_'+ex:a1})
        regimes.update({'a2_'+ex:a2})
        regimes.update({'a3_'+ex:a3})
        regimes.update({'a4_'+ex:a4})
        regimes.update({'sig_'+ex:sig})
        pd.DataFrame(d).to_csv(basic_dir+'code_whiplash/4-2.Input data for plotting/4.prec_and_whiplash_regimes_'+ex+'.csv',index=False)
np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/4.each_regimes.npy',regimes) 