#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:00:33 2023

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


#%% Source data
#basic_dir = '/media/dai/suk_code/research/4.East_Asia/Again/'
basic_dir = 'E:/research/4.East_Asia/Again/'
IF_forcings = np.load(basic_dir +'code_whiplash/4-2.Input data for plotting/5_S16-19.risk_ratio_distribution.npy',allow_pickle=True).tolist()
If_sig_agreement = np.load(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.If_sig_agreement.npy',allow_pickle=True).tolist()
forcing_mean = np.load(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.LENS_and_SF_events_global_and_region_mean.npy',allow_pickle=True).tolist()
IF_forcings_global_and_regional_mean = np.load(basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.risk_ratio_global_and_regional_mean.npy',allow_pickle=True).tolist()




#%% user-defined colorbars and variables

colorbar_change=['#6CA2CC','#89BED9','#A8D8E7','#C6E6F2','#E2F2F1','#F7E5A6','#FECF80','#FCB366',
 '#F89053','#F26B43','#DF3F2D','#C92226','#AB0726']

color_forcing = ['#F0DD30','#288F8B','#67BB66','#496191']
ListedColormap(('#F0DD30','#288F8B','#67BB66','#49325E','#496191'))

col_sf_change=("#5E4FA2" ,"#4F61AA" ,"#4173B3", "#4198B6", "#51ABAE", "#77C8A4" ,"#A4DAA4" ,"#E0EFC5", \
       "#ffffff",\
       "#FFF6D2", "#FEF0A5", "#FDC978", "#F99254", "#F67D4A", "#E85A47", "#D33C4E" ,"#C1284A")

col_sf_change=("#5E4FA2" ,"#4F61AA" ,"#4173B3", "#4198B6", "#51ABAE", "#77C8A4" ,"#A4DAA4" ,"#E0EFC5", \
       
       "#FFF6D2", "#FEF0A5", "#FDC978", "#F99254", "#F67D4A", "#E85A47", "#D33C4E" ,"#C1284A")
'''
col_sf_change=("#5E4FA2" ,"#4F61AA" ,"#4173B3", "#4198B6", "#51ABAE", "#77C8A4" ,"#A4DAA4" ,"#E0EFC5", \
       "#ffffff","#ffffff",
       "#FFF6D2", "#FEF0A5", "#FDC978", "#F99254", "#F67D4A", "#E85A47", "#D33C4E" ,"#C1284A")

'''

    
forcing=['AER','GHG','BMB']

first_years = [1921,1979,1921,2040,2060]
last_years = [2028,2019,2079,2079,2079]


num_forcing = [20,20,15]

monsoon_name = ['WAfriM','SAsiaM','SAmerM','NAmerM','EAsiaM','AusMCM']

title_aux = list(map(chr, range(97, 123)))[:6]
#%%


for feature in ['frequency']:
    
    for ex in ['dry_to_wet','wet_to_dry']:
        print('-------to 2079---------')
        for x in [0,1]:
            
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~global')
            
            print(forcing[x])
            print(m)
            print(q )
            #print(((m-1)*100/78)*100)
            #print(((q)*100/78)*100)
            
            #
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~land')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~land')
            
            print('land')
            print(m)
            print(q )
            #print(((m-1)*100/78)*100)
            #print(((q)*100/78)*100)
            #
            
            for ms in range(6):
                c=IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~'+monsoon_name[ms])
                c_q=IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~'+monsoon_name[ms])
                
                print(monsoon_name[ms]+":"+str(c))
                print(monsoon_name[ms]+":"+str(c_q))
         
        print('-------to 2028----------')
        for x in [0,1,2]:
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~global')
            
            
            print(forcing[x])
            print(m)
            print(q)
            
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~land')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~land')
            
            print('land')
            print(m)
            print(q )
            
            
            
#%%
for feature in ['frequency']:
    print('-------to 2079---------')
    for ex in ['dry_to_wet','wet_to_dry']:
        for x in [0]:     
            
            print(forcing[x])
            a=forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~global').iloc[:,2]
            b=forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~global').iloc[:,2]
            a.index = np.arange(1921,2100,1)
            
            
            a_change = (a.iloc[119:159].mean() - a.iloc[58:99].mean())*100/a.iloc[58:99].mean()
            b_change = (b.iloc[119:159].mean() - b.iloc[58:99].mean())*100/b.iloc[58:99].mean()
            
            c= 100*(a_change - b_change )/a_change
            
            #c=((a-b)*100/a).max()
            #c=((a-b)*100/a).dropna().rolling(10).mean().dropna().iloc[-1]
            #d=100*((a.iloc[:159]-b).iloc[-1] - (a.iloc[:159]-b).iloc[0])/(a.iloc[158]- a.iloc[0])


            print(c)
          
            
            for ms in range(6):
                #print()
                
                a=forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~'+monsoon_name[ms]).iloc[:,2]
                b=forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~'+monsoon_name[ms]).iloc[:,2]
                a.index = np.arange(1921,2100,1)
                c=((a-b)*100/a).dropna().rolling(10).mean().dropna().iloc[-1]
                
                #plt.plot(c.index,c,label=monsoon_name[ms])
                #plt.legend()
                print(monsoon_name[ms]+":"+str(c))
#%%contribution

contribution = np.load(basic_dir+'code_whiplash/4-2.Input data for plotting/5_S16-19.global_contribution_of_forcings.npy',allow_pickle=True).tolist()                
            
#%% plot whiplash frequency

ny=3
p_num=-1      

for feature in ['frequency']:
    fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
    for ex in ['dry_to_wet','wet_to_dry']:

        
        if ex=='dry_to_wet' :
            x_loc=0
            ex_name='dry-to-wet'
        else :
            x_loc  = 0.5
            ex_name='wet-to-dry'
        ax = plt.axes([x_loc,0.7,0.32,0.22] )
        for x in [0,1,2]:
            f_s_mean = forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~global')
            ax.fill_between(x = range(1921,1921+len(f_s_mean)),y1=f_s_mean.iloc[:,0],y2=f_s_mean.iloc[:,1],color = 'none',alpha=0.3,facecolor=color_forcing[x+1])
            ax.plot(range(1921,1921+len(f_s_mean)),f_s_mean.iloc[:,2],c = color_forcing[x+1],linewidth=2,label='X'+forcing[x])
       
            
        f_l_mean = forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~global')
        ax.fill_between(x = range(1921,1921+len(f_l_mean)),y1=f_l_mean.iloc[:,0],y2=f_l_mean.iloc[:,1],color='none',alpha=0.3,facecolor = color_forcing[0])
        ax.plot(range(1921,1921+len(f_l_mean)),f_l_mean.iloc[:,2],c = color_forcing[0],linewidth=2,label='LENS')
        
    
        p_num+=1
        
        
        ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ "+ex_name,
                         pad=2, size=8.5, loc='left')
        
        
        xmajorLocator = MultipleLocator(50) 
        xminorLocator = MultipleLocator(10) 
        ymajorLocator = MultipleLocator(0.1) 
        yminorLocator = MultipleLocator(0.05)
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
       
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        xmajorFormatter = FormatStrFormatter('%d') 
        ymajorFormatter = FormatStrFormatter('%1.1f') 
        
        xticks = [1930,1970,2010,2050,2090]
            #y_ = np.arange(0,101,20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,fontsize=7)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5)
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )

        ax.margins(x=0)    
        if p_num==0:
            plt.legend(framealpha=0,fontsize=7)
            ax.set_ylabel('Occurence(times '+u'$yr^{-1}$)',fontsize=7,labelpad=1)

        
        ax1 = plt.axes([x_loc+0.33,0.7,0.1,0.22] )
        
        
        
        ax1.hlines(xmin=-0.5,xmax=2.5,y=1,color='grey',linestyle='--',linewidth=0.6 )
        ax1.set_xlim((-0.5,2.5))
        
        for x in [0,1]:
            
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~global')

            err = np.zeros(shape=(2,1))
            err[0] =q
            err[1] = q
            
            ax1.errorbar(x=x-0.15,y=m,yerr=err,fmt='o',ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1)
            
            
        for x in [0,1,2]:
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~global')
            
            err = np.zeros(shape=(2,1))
            err[0] = q
            err[1] = q
            
            ax1.errorbar(x=x+0.15,y=m,yerr=err,ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1,markerfacecolor='white',markeredgewidth=0.7,fmt='o',label=' ')
            
            ax1.margins(x=0)
            
            
            ax1.tick_params(axis='x', labelsize=6.5,pad=0.5,rotation=45 )
            ax1.tick_params(axis='y', labelsize=6.5,pad=0.5, labelright=True)
            
            xticks = [0,1,2]
               
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(forcing,fontsize=7)
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            #plt.legend() 
            if p_num ==3:
                ax1.set_ylabel('Risk ratio',fontsize=7,labelpad=1)

        for x in [0,1]:

            if (x==0):
                loc_y=0.32
            else:
                loc_y  = 0.06
            if ex=='dry_to_wet' :
                
                loc_x =0
            else :
                loc_x= .5
                
                
            monsoon = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp').geometries()
            
            first_year = first_years[ny]
            last_year = last_years[ny]
            rif  = IF_forcings.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global')
            #sig  = sig_student.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079')
            #sig[sig>=0.1]=np.nan
            #rif_masked = rif * sig
            sig_agree = If_sig_agreement.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global').copy(deep=True).values
            sig_agree[sig_agree < num_forcing[x]*0.9 ] = np.nan
            sig_agree[sig_agree>0] = 1
            #rif_masked = rif * sig_agree
            #%
            rif.quantile([0.1,0.5,0.9])
            #Colors_limits=[-1.6,-1.2,-0.8,-0.4,-0.2,0,0.2,0.4,0.8,1.2,1.6,2,2.4]
            Colors_limits=[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.2,1.4,1.6,1.8,2,2.2,2.4,]    
            #Colors_limits=[0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]    
            np.arange(-22.5,22.5,3)
            print(len(np.arange(-22.5,23,3)))

            

            ax = plt.axes([loc_x,loc_y,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            p_num+=1
            print(p_num)
            print('map!')
            ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ Risk ratio under "+forcing[x]+' ('+ex_name+")",
                             pad=2, size=8.5, loc='left')
            lon=rif.lon
            lat=rif.lat

            cycle_if, cycle_lon = add_cyclic_point(rif, coord=lon)
            cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)

            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_if,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,
                         extend='both',
                         colors=col_sf_change
                         )
            
            vars()['first_year_LENS_negative_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_negative_' +ex+'_'+'feature.csv')
            vars()['first_year_LENS_positive_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_positive_' +ex+'_'+'feature.csv')

            ax.contour(lon,lat,vars()['first_year_LENS_positive_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='purple',linewidths=0.6)
            
            ax.contour(lon,lat,vars()['first_year_LENS_negative_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='green',linewidths=.8)
            
            cf1 =ax.contourf(lon,lat,sig_agree,
                    transform=ccrs.PlateCarree(),hatches=['///////', None],colors="none",width=0.001)
            
            ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='blue' , linewidth=0.3, zorder=10) # 添加季风区
        
            
            ax.set_global()
           
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            
            
            if p_num==5:
                a=ax.get_position()
                pad=0.025
                height=0.015
                ax_f = fig.add_axes([  0.2, a.ymin - pad,  0.55 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                
                #cb.set_label( label = "Relative influence (%)",fontdict={'size':7},labelpad=1)
               
                cb.set_ticklabels(Colors_limits,size=7)
                cb.outline.set_linewidth(0.5)
                
                ax_f.tick_params(length=1,pad=0.2)
    plt.rcParams['hatch.linewidth'] = 0.3            


fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.5.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.5.pdf",dpi=1500, bbox_inches='tight')    


#%%  plot whiplash duration


p_num=-1      
title_aux = list(map(chr, range(97, 123)))[:6]
  
for feature in ['duration']:
    fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
    for ex in ['dry_to_wet','wet_to_dry']:

        
        if ex=='dry_to_wet' :
            x_loc=0
            ex_name='dry-to-wet'
        else :
            x_loc  = 0.5
            ex_name='wet-to-dry'
        ax = plt.axes([x_loc,0.7,0.32,0.22] )
        for x in [0,1,2]:
            f_s_mean = forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~global')
            ax.fill_between(x = range(1921,1921+len(f_s_mean)),y1=f_s_mean.iloc[:,0],y2=f_s_mean.iloc[:,1],color = 'none',alpha=0.3,facecolor=color_forcing[x+1])
            ax.plot(range(1921,1921+len(f_s_mean)),f_s_mean.iloc[:,2],c = color_forcing[x+1],linewidth=2,label='X'+forcing[x])
 
        f_l_mean = forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~global')
        ax.fill_between(x = range(1921,1921+len(f_l_mean)),y1=f_l_mean.iloc[:,0],y2=f_l_mean.iloc[:,1],color='none',alpha=0.3,facecolor = color_forcing[0])
        ax.plot(range(1921,1921+len(f_l_mean)),f_l_mean.iloc[:,2],c = color_forcing[0],linewidth=2,label='LENS')
 
        p_num+=1

        ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ "+ex_name,
                         pad=2, size=8.5, loc='left')

        xmajorLocator = MultipleLocator(50) 
        xminorLocator = MultipleLocator(10) 
        ymajorLocator = MultipleLocator(1) 
        yminorLocator = MultipleLocator(0.5) 
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
      
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        xmajorFormatter = FormatStrFormatter('%d') 
        ymajorFormatter = FormatStrFormatter('%1.1f') 
        
        
        xticks = [1930,1970,2010,2050,2090]
            #y_ = np.arange(0,101,20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,fontsize=7)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5)
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )

        ax.margins(x=0)    
        if p_num==0:
            plt.legend(framealpha=0,fontsize=7)
            ax.set_ylabel('Occurence(times '+u'$yr^{-1}$)',fontsize=7,labelpad=1)

        
        ax1 = plt.axes([x_loc+0.33,0.7,0.1,0.22] )
        
        
        
        ax1.hlines(xmin=-0.5,xmax=2.5,y=1,color='grey',linestyle='--',linewidth=0.6 )
        ax1.set_xlim((-0.5,2.5))
        
        for x in [0,1]:
            
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~global')

            err = np.zeros(shape=(2,1))
            err[0] = q
            err[1] = q
            
            ax1.errorbar(x=x-0.15,y=m,yerr=err,fmt='o',ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1)
            
            
        for x in [0,1,2]:
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~global')
            
            err = np.zeros(shape=(2,1))
            err[0] = q
            err[1] = q
            
            ax1.errorbar(x=x+0.15,y=m,yerr=err,ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1,markerfacecolor='white',markeredgewidth=0.7,fmt='o',label=' ')
            
            ax1.margins(x=0)
            
            
            ax1.tick_params(axis='x', labelsize=6.5,pad=0.5,rotation=45 )
            ax1.tick_params(axis='y', labelsize=6.5,pad=0.5, labelright=True)
            
            xticks = [0,1,2]
              
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(forcing,fontsize=7)
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            #plt.legend() 
            if p_num ==3:
                ax1.set_ylabel('Risk ratio',fontsize=7,labelpad=1)

        for x in [0,1]:

            #%
            if (x==0):
                loc_y=0.32
            else:
                loc_y  = 0.06
            if ex=='dry_to_wet' :
                
                loc_x =0
            else :
                loc_x= .5

            monsoon = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp').geometries()
            
            first_year = first_years[ny]
            last_year = last_years[ny]
            rif  = IF_forcings.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global')
            #sig  = sig_student.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079')
            #sig[sig>=0.1]=np.nan
            #rif_masked = rif * sig
            sig_agree = If_sig_agreement.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global').copy(deep=True).values
            sig_agree[sig_agree < num_forcing[x]*0.9 ] = np.nan
            sig_agree[sig_agree>0] = 1
            #rif_masked = rif * sig_agree
            #%
            Colors_limits=[-1.6,-1.2,-0.8,-0.4,-0.2,0,0.2,0.4,0.8,1.2,1.6,2,2.4]
            #Colors_limits=[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35]
            Colors_limits=[0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14]

            ax = plt.axes([loc_x,loc_y,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            p_num+=1
            print(p_num)
            print('map!')
            ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ Risk ratio under "+forcing[x]+' ('+ex_name+")",
                             pad=2, size=8.5, loc='left')
            lon=rif.lon
            lat=rif.lat

            cycle_if, cycle_lon = add_cyclic_point(rif, coord=lon)
            cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)

            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_if,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,
                         extend='both',
                         colors=col_sf_change
                         )
            '''
            vars()['first_year_LENS_negative_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_negative_' +ex+'_'+'feature.csv')
            vars()['first_year_LENS_positive_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_positive_' +ex+'_'+'feature.csv')

            ax.contour(lon,lat,vars()['first_year_LENS_positive_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='purple',linewidths=0.3)
            
            ax.contour(lon,lat,vars()['first_year_LENS_negative_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='green',linewidths=0.3)
            '''
            cf1 =ax.contourf(lon,lat,sig_agree,
                    transform=ccrs.PlateCarree(),hatches=['///////', None],colors="none",width=0.001)
            
            ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='blue' , linewidth=0.3, zorder=10) # 添加季风区
        
            ax.set_global()
            
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            
            
            if p_num==5:
                a=ax.get_position()
                pad=0.025
                height=0.015
                ax_f = fig.add_axes([  0.2, a.ymin - pad,  0.55 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                
                #cb.set_label( label = "Relative influence (%)",fontdict={'size':7},labelpad=1)

                cb.set_ticklabels(Colors_limits,size=7)
                cb.outline.set_linewidth(0.5)

                ax_f.tick_params(length=1,pad=0.2)
    plt.rcParams['hatch.linewidth'] = 0.3            
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S17.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S17.pdf",dpi=1500, bbox_inches='tight')    














#%%  plot whiplash intensity
p_num=-1      
  
for feature in ['intensity']:
    fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
    for ex in ['dry_to_wet','wet_to_dry']:

        
        if ex=='dry_to_wet' :
            x_loc=0
            ex_name='dry-to-wet'
        else :
            x_loc  = 0.5
            ex_name='wet-to-dry'
        ax = plt.axes([x_loc,0.7,0.32,0.22] )
        for x in [0,1,2]:
            f_s_mean = forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~global')
            ax.fill_between(x = range(1921,1921+len(f_s_mean)),y1=f_s_mean.iloc[:,0],y2=f_s_mean.iloc[:,1],color = 'none',alpha=0.3,facecolor=color_forcing[x+1])
            ax.plot(range(1921,1921+len(f_s_mean)),f_s_mean.iloc[:,2],c = color_forcing[x+1],linewidth=2,label='X'+forcing[x])

        f_l_mean = forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~global')
        ax.fill_between(x = range(1921,1921+len(f_l_mean)),y1=f_l_mean.iloc[:,0],y2=f_l_mean.iloc[:,1],color='none',alpha=0.3,facecolor = color_forcing[0])
        ax.plot(range(1921,1921+len(f_l_mean)),f_l_mean.iloc[:,2],c = color_forcing[0],linewidth=2,label='LENS')
      
        p_num+=1
       
        ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ "+ex_name,
                         pad=2, size=8.5, loc='left')
       
        xmajorLocator = MultipleLocator(50) 
        xminorLocator = MultipleLocator(10) 
        ymajorLocator = MultipleLocator(0.2) 
        yminorLocator = MultipleLocator(0.1) 
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
        #ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        xmajorFormatter = FormatStrFormatter('%d') 
        ymajorFormatter = FormatStrFormatter('%1.1f') 
     
        xticks = [1930,1970,2010,2050,2090]
            #y_ = np.arange(0,101,20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,fontsize=7)
        
        ax.tick_params(axis='x', labelsize=6.5,pad=0.5)
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5 )

        ax.margins(x=0)    
        if p_num==0:
            plt.legend(framealpha=0,fontsize=7)
            ax.set_ylabel('Occurence(times '+u'$yr^{-1}$)',fontsize=7,labelpad=1)

        ax1 = plt.axes([x_loc+0.33,0.7,0.1,0.22] )
      
        ax1.hlines(xmin=-0.5,xmax=2.5,y=1,color='grey',linestyle='--',linewidth=0.6 )
        ax1.set_xlim((-0.5,2.5))
        
        for x in [0,1]:
            
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~global')

            err = np.zeros(shape=(2,1))
            err[0] = q
            err[1] = q
            
            ax1.errorbar(x=x-0.15,y=m,yerr=err,fmt='o',ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1)
            
            
        for x in [0,1,2]:
            m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~global')
            q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~global')
            
            err = np.zeros(shape=(2,1))
            err[0] = q
            err[1] = q
            
            ax1.errorbar(x=x+0.15,y=m,yerr=err,ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                         elinewidth=0.7,capsize=2,capthick=1,markerfacecolor='white',markeredgewidth=0.7,fmt='o',label=' ')
            
            ax1.margins(x=0)
            
            
            ax1.tick_params(axis='x', labelsize=6.5,pad=0.5,rotation=45 )
            ax1.tick_params(axis='y', labelsize=6.5,pad=0.5, labelright=True)
            
            xticks = [0,1,2]
                #y_ = np.arange(0,101,20)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(forcing,fontsize=7)
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            #plt.legend() 
            if p_num ==3:
                ax1.set_ylabel('Risk ratio',fontsize=7,labelpad=1)
    
        for x in [0,1]:

            #%
            if (x==0):
                loc_y=0.32
            else:
                loc_y  = 0.06
            if ex=='dry_to_wet' :
                
                loc_x =0
            else :
                loc_x= .5
                
                
            monsoon = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp').geometries()
            
            first_year = first_years[ny]
            last_year = last_years[ny]
            rif  = IF_forcings.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global')
            #sig  = sig_student.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079')
            #sig[sig>=0.1]=np.nan
            #rif_masked = rif * sig
            sig_agree = If_sig_agreement.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global').copy(deep=True).values
            sig_agree[sig_agree < num_forcing[x]*0.9 ] = np.nan
            sig_agree[sig_agree>0] = 1
            #rif_masked = rif * sig_agree
            #%
            Colors_limits=[-1.6,-1.2,-0.8,-0.4,-0.2,0,0.2,0.4,0.8,1.2,1.6,2,2.4]
            Colors_limits=[0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14]
            

            ax = plt.axes([loc_x,loc_y,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            p_num+=1
            print(p_num)
            print('map!')
            ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ Risk ratio under "+forcing[x]+' ('+ex_name+")",
                             pad=2, size=8.5, loc='left')
            lon=rif.lon
            lat=rif.lat

            cycle_if, cycle_lon = add_cyclic_point(rif, coord=lon)
            cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)

            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_if,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,
                         extend='both',
                         colors=col_sf_change
                         )
            '''
            vars()['first_year_LENS_negative_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_negative_' +ex+'_'+'feature.csv')
            vars()['first_year_LENS_positive_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_positive_' +ex+'_'+'feature.csv')

            ax.contour(lon,lat,vars()['first_year_LENS_positive_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='purple',linewidths=0.3)
            
            ax.contour(lon,lat,vars()['first_year_LENS_negative_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='green',linewidths=0.3)
            '''
            cf1 =ax.contourf(lon,lat,sig_agree,
                    transform=ccrs.PlateCarree(),hatches=['///////', None],colors="none",width=0.001)
            
            ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='blue' , linewidth=0.3, zorder=10) # 添加季风区
        
          
            ax.set_global()
           
            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            
            if p_num==5:
                a=ax.get_position()
                pad=0.025
                height=0.015
                ax_f = fig.add_axes([  0.2, a.ymin - pad,  0.55 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                
                #cb.set_label( label = "Relative influence (%)",fontdict={'size':7},labelpad=1)
                
                
                cb.set_ticklabels(Colors_limits,size=7)
                cb.outline.set_linewidth(0.5)
                
                ax_f.tick_params(length=1,pad=0.2)
    plt.rcParams['hatch.linewidth'] = 0.3            
        
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S18.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S18.pdf",dpi=1500, bbox_inches='tight')    

#%%  plot 1921-2028
ny=0
p_num=-1      
title_aux = list(map(chr, range(97, 123)))[:6]
  

col_sf_change_2=("#5E4FA2" ,"#4F61AA" ,"#4173B3", "#4198B6", "#51ABAE", "#77C8A4" ,"#A4DAA4" ,"#E0EFC5", \
       "#ffffff","#ffffff",
       "#FFF6D2", "#FEF0A5", "#FDC978", "#F99254", "#F67D4A", "#E85A47", "#D33C4E" ,"#C1284A")



for feature in ['frequency']:
    fig = plt.figure(figsize = (14/2.54, 14/2.54)) 
    for ex in ['dry_to_wet','wet_to_dry']:

        
        if ex=='dry_to_wet' :
            x_loc=0
            ex_name='dry-to-wet'
        else :
            x_loc  = 0.5
            ex_name='wet-to-dry'

        
        for x in [0,1,2]:

            #%
            if (x==0):
                loc_y=0.52
            elif (x==1):
                loc_y  = 0.26
            else:
                loc_y =0
            if ex=='dry_to_wet' :
                
                loc_x =0
            else :
                loc_x= .5
                
                
            monsoon = shpreader.Reader(basic_dir+'code_whiplash/4-2.Input data for plotting/monsoons_shapefile/monsoons.shp').geometries()
            
            first_year = first_years[ny]
            last_year = last_years[ny]
            rif  = IF_forcings.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global')
            #sig  = sig_student.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079')
            #sig[sig>=0.1]=np.nan
            #rif_masked = rif * sig
            sig_agree = If_sig_agreement.get(forcing[x]+'~'+ex+'~'+feature+'~'+str(first_year)+'-'+str(last_year)+'~mean~global').copy(deep=True).values
            sig_agree[sig_agree < num_forcing[x]*0.9 ] = np.nan
            sig_agree[sig_agree>0] = 1
            #rif_masked = rif * sig_agree
            #%
            rif.quantile([0.1,0.5,0.9])
            Colors_limits=[-1.6,-1.2,-0.8,-0.4,-0.2,0,0.2,0.4,0.8,1.2,1.6,2,2.4]
            Colors_limits=[0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4]    
            np.arange(-22.5,22.5,3)
            print(len(np.arange(-22.5,23,3)))

            

            ax = plt.axes([loc_x,loc_y,0.45,0.4] ,projection=ccrs.Robinson(central_longitude=150))
            p_num+=1
            print(p_num)
            print('map!')
            ax.set_title(r"$\bf{("+ title_aux[p_num] +") }$ Risk ratio under "+forcing[x]+' ('+ex_name+")",
                             pad=2, size=7, loc='left')
            lon=rif.lon
            lat=rif.lat

            cycle_if, cycle_lon = add_cyclic_point(rif, coord=lon)
            cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, lat)

            cf=ax.contourf(cycle_LON,cycle_LAT,cycle_if,
                         transform=ccrs.PlateCarree(),
                         levels=Colors_limits,
                         extend='both',
                         colors=col_sf_change_2
                         )
            
            vars()['first_year_LENS_negative_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_negative_' +ex+'_'+'feature.csv')
            vars()['first_year_LENS_positive_' +ex+'_'+feature] = pd.read_csv (basic_dir + 'code_whiplash/4-2.Input data for plotting/5_S16-19.'+'first_year_LENS_positive_' +ex+'_'+'feature.csv')
            '''
            ax.contour(lon,lat,vars()['first_year_LENS_positive_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='purple',linewidths=0.6)
            
            ax.contour(lon,lat,vars()['first_year_LENS_negative_' +ex+'_'+feature] ,levels=[0,1],
                    transform=ccrs.PlateCarree(),colors='green',linewidths=0.8)
            '''
            
            cf1 =ax.contourf(lon,lat,sig_agree,
                    transform=ccrs.PlateCarree(),hatches=['///////', None],colors="none",width=0.001)
            
            ax.add_geometries(monsoon, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='blue' , linewidth=0.3, zorder=10) # 添加季风区
        
          
            ax.set_global()

            ax.coastlines(linewidth=0.3)
            ax.outline_patch.set_visible(False)
            
            if p_num==5:
                a=ax.get_position()
                pad=0.025
                height=0.015
                ax_f = fig.add_axes([  0.2, a.ymin - pad,  0.55 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                
                #cb.set_label( label = "Relative influence (%)",fontdict={'size':6.5},labelpad=1)
                
                cb.set_ticklabels(Colors_limits,size=6)
                cb.outline.set_linewidth(0.5)
                
                ax_f.tick_params(length=1,pad=0.2)
    plt.rcParams['hatch.linewidth'] = 0.3            
        
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S19.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S19.pdf",dpi=1500, bbox_inches='tight')    


#%%  plot six monsoon regions
p_num=-1      
title_aux = list(map(chr, range(97, 123)))[:12]
monsoon_name = ['WAfriM','SAsiaM','SAmerM','NAmerM','EAsiaM','AusMCM']

for feature in ['frequency']:
    fig = plt.figure(figsize = (17/2.54, 17/2.54)) # 宽、高
    for ex in ['dry_to_wet','wet_to_dry']:

        
        if ex=='dry_to_wet' :
            x_loc=0
            ex_name='dry-to-wet'
        else :
            x_loc  = 0.5
            ex_name='wet-to-dry'
        for ms in range(6):
            
            
            ax = plt.axes([x_loc,1-ms*0.16,0.32,0.16] )
            for x in [0,1,2]:
                f_s_mean = forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~'+monsoon_name[ms])
                ax.fill_between(x = range(1921,1921+len(f_s_mean)),y1=f_s_mean.iloc[:,0],y2=f_s_mean.iloc[:,1],color = 'none',alpha=0.3,facecolor=color_forcing[x+1])
                ax.plot(range(1921,1921+len(f_s_mean)),f_s_mean.iloc[:,2],c = color_forcing[x+1],linewidth=2,label='X'+forcing[x])
                
                
                '''
                f_s_mean = forcing_mean.get(forcing[x] +'~' + ex+'~'+feature+'~land')
                ax.fill_between(x = range(1921,1921+len(f_s_mean)),y1=f_s_mean.iloc[:,0],y2=f_s_mean.iloc[:,1],color = color_forcing[x+1],alpha=0.3,facecolor='none',linestyle ="--")
                ax.plot(range(1921,1921+len(f_s_mean)),f_s_mean.iloc[:,2],c = color_forcing[x+1],linewidth=1,linestyle ="--")
                '''
                
            f_l_mean = forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~'+monsoon_name[ms])
            ax.fill_between(x = range(1921,1921+len(f_l_mean)),y1=f_l_mean.iloc[:,0],y2=f_l_mean.iloc[:,1],color='none',alpha=0.3,facecolor = color_forcing[0])
            ax.plot(range(1921,1921+len(f_l_mean)),f_l_mean.iloc[:,2],c = color_forcing[0],linewidth=2,label='LENS')
            
            
            '''
            f_l_mean = forcing_mean.get('LENS' +'~' + ex+'~'+feature+'~land')
            ax.fill_between(x = range(1921,1921+len(f_l_mean)),y1=f_l_mean.iloc[:,0],y2=f_l_mean.iloc[:,1],color=color_forcing[0],alpha=0.3,facecolor ='none',linestyle='--' )
            ax.plot(range(1921,1921+len(f_l_mean)),f_l_mean.iloc[:,2],c = color_forcing[0],linewidth=1,linestyle ="--")
            '''
            
            p_num+=1
            
            print(p_num )
            if p_num in [0,6] :
                    
                ax.set_title(ex_name,
                                 pad=-2, size=8.5, loc='center')
                
            ax.annotate(r"$\bf{("+ title_aux[p_num] +") }$ "+ monsoon_name[ms] ,xy=(0.1,0.8),xytext=(0.02,0.9),xycoords='axes fraction',fontsize=8.5)
            xmajorLocator = MultipleLocator(50)
            xminorLocator = MultipleLocator(10) 
            ymajorLocator = MultipleLocator(0.3) 
            yminorLocator = MultipleLocator(0.1)
            ax.xaxis.set_major_locator(xmajorLocator)
            ax.xaxis.set_minor_locator(xminorLocator)
            #ax.xaxis.set_major_formatter(xmajorFormatter)
            ax.yaxis.set_major_locator(ymajorLocator)
            ax.yaxis.set_minor_locator(yminorLocator)
            xmajorFormatter = FormatStrFormatter('%d') 
            ymajorFormatter = FormatStrFormatter('%1.1f')
            
            xticks = [1930,1970,2010,2050,2090]
                #y_ = np.arange(0,101,20)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks,fontsize=6)
            
            ax.tick_params(axis='x', labelsize=7,pad=0.5)
            ax.tick_params(axis='y', labelsize=7,pad=0.5 )
    
    
    
            ax.margins(x=0)    
            if p_num==2:
               
                ax.set_ylabel('Occurence(times '+u'$yr^{-1}$)',fontsize=8,labelpad=1)
    
            
            ax1 = plt.axes([x_loc+0.33,1-ms*0.16,0.1,0.16] )
            ax1.hlines(xmin=-0.5,xmax=2.5,y=1,color='grey',linestyle='--',linewidth=0.6 )
            ax1.set_xlim((-0.5,2.5))
            
            if p_num ==8:
                ax1.set_ylabel('Risk ratio',fontsize=8,labelpad=1)
            
            
            for x in [0,1]:
                
                m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~mean~'+monsoon_name[ms])
                q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2079~quan~'+monsoon_name[ms])
    
                err = np.zeros(shape=(2,1))
                err[0] = q
                err[1] = q
                
                ax1.errorbar(x=x-0.15,y=m,yerr=err,fmt='o',ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                             elinewidth=0.7,capsize=2,capthick=1)
                
                
            for x in [0,1,2]:
                m = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~mean~'+monsoon_name[ms])
                q = IF_forcings_global_and_regional_mean.get(forcing[x]+'~'+ex+'~'+feature+'~1921-2028~quan~'+monsoon_name[ms])
                
                err = np.zeros(shape=(2,1))
                err[0] = q
                err[1] = q
                
                ax1.errorbar(x=x+0.15,y=m,yerr=err,ecolor=color_forcing[x+1], ms=3,color=color_forcing[x+1],
                             elinewidth=0.7,capsize=2,capthick=1,markerfacecolor='white',markeredgewidth=0.7,fmt='o',label=' ')
                
                ax1.margins(x=0.2)
                
                ax1.tick_params(axis='x', labelsize=6.5,pad=0.5,rotation=45 )
                ax1.tick_params(axis='y', labelsize=6.5,pad=0.5, labelright=True)
                
                xticks = [0,1,2]
                    #y_ = np.arange(0,101,20)
                ax1.set_xticks(xticks)
                ax1.set_xticklabels(forcing,fontsize=6)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                #plt.legend() 
      
lines, labels = fig.axes[-2].get_legend_handles_labels()                
fig.legend(lines, labels, bbox_to_anchor=(0.67, 0.17),ncol=4, framealpha=0,fontsize=7)
                
            
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S16.png",dpi=1500, bbox_inches='tight')
fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S16.pdf",dpi=1500, bbox_inches='tight')    
            
