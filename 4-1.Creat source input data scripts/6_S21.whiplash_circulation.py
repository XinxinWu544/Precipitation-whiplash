#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:18:45 2023

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

basic_dir='/media/dai/disk2/suk/research/4.East_Asia/Again/'
#basic_loc = 'E:'
#%%
num=[np.linspace(1,35,35).astype(int)]
num.append(np.linspace(101,105,5).astype(int))
num=[j for i in num for j in i]

interval=[(1,5),(1,10),(1,15),(1,20),(1,25),(1,30)]

k=5
region='northeastern-china'

lon1=120 - 30
lon2=130 + 30
lat1=40 - 25
lat2=50 + 25    

d='dmean'
period = ['current','future']
states = ['before','transition','after']
#%%

circulation={}

for x in [21,22]:
    for whiplash_type in ['dry_to_wet','wet_to_dry']:
    
        
        for p in [0,1]:
            for i in [0,1,2]:
            
    
    #%
                if (d == 'd')&(i==1):
                    uq = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_UQ_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                    vq = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_VQ_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                    z = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Z500_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                else:
                    uq = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_UQ_'+d+'_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                    vq = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Anomalies_IVT_VQ_'+d+'_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                    z = np.load(basic_dir+'code_whiplash/3-2.Processed data from analysis/7-2.current_future_circulation/plan'+str(x)+'_Z500_'+d+'_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                #z = np.load(basic_loc+'/research/4.East_Asia/Again/code/7-1.current_future_circulation_old/Z500_dmean_'+states[i]+'_'+period[p]+'.npy',allow_pickle=True).tolist()
                
                #z
                
                UQ=[]
                for n in num:
                    
                    if states[i] == 'transition':
                        new = uq.get(str(n)+'~'+whiplash_type)
                    else:
                        new = uq.get(str(n)+'~'+str(k)+'~'+whiplash_type)
                    if n>1 :
                        new = new.assign_coords({'lat':UQ[0].lat})
                    UQ.append(new)
                UQ = xr.concat(UQ, pd.Index(num, name='ensemble') ).mean('ensemble').sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
                    
                VQ=[]
                for n in num:
                    if states[i] == 'transition':
                        new = vq.get(str(n)+'~'+whiplash_type)
                    else:
                        new = vq.get(str(n)+'~'+str(k)+'~'+whiplash_type)
                    if n>1 :
                        new = new.assign_coords({'lat':VQ[0].lat})
                    VQ.append(new)    
                VQ = xr.concat(VQ, pd.Index(num, name='ensemble') ).mean('ensemble').sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
                    
                Z=[]
                for n in num:
                    if states[i] == 'transition':
                        new = z.get(str(n)+'~'+whiplash_type)
                    else:
                        new = z.get(str(n)+'~'+str(k)+'~'+whiplash_type)
                    if n>1 :
                        new = new.assign_coords({'lat':Z[0].lat})
                    Z.append(new)    
                Z = xr.concat(Z, pd.Index(num, name='ensemble') ).mean('ensemble').sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
                if x == 21:
                    x_name = 'before'
                else:
                    x_name = 'after'
                circulation.update({x_name+'_'+'Z_'+states[i]+'_'+period[p]+'_'+whiplash_type:Z})
                circulation.update({x_name+'_'+'VQ_'+states[i]+'_'+period[p]+'_'+whiplash_type:VQ})
                circulation.update({x_name+'_'+'UQ_'+states[i]+'_'+period[p]+'_'+whiplash_type:UQ})
                


np.save(basic_dir+'code_whiplash/4-2.Input data for plotting/6_S20.circulation.npy',circulation)