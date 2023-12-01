# Precipitation whiplash
## Increasing global precipitation whiplash due to anthropogenic greenhouse gas emissions

Supporting code for  'Increasing global precipitation whiplash due to anthropogenic greenhouse gas emissions' 

If you find meaningful errors in the code of have questions, please contact Xinxin WU [wuxx26@mail2.sysu.edu.cn]



## Organization of repository ##
python scripts to read and analyze data, and create figures:  
* **1. Raw data prec-processing scripts:** Code to pre-process all downloaded raw data before calculating whiplash (not all raw data is included due to size; see details below)  
* **2-1. Calculate whiplash and sensitivity test scripts:** Code to calculate precipitation whiplash and perform sensitivity test for different parameters of the indices. (see details below)  
* **2-2.Original whiplash events**: Original whiplash features generated in scripts in folder 2-1, not provided here as the generated results are over 200G.  
* **3-1.Analysis scripts**: Code to analyze precipitation whiplash events to obtain the features needed for this paper. (see details below)  
* **3-2.Processed data from analysis**: Generated data for this study in scripts in folder 3-1 (about 5 G)  
* **4-1.Creat source input data scripts**: Code to organize the results into csv and npy files that can be used directly to create figures. (see details below)  
* **4-2.Input data for plotting**: All source data generated in scripts in folder 4-1 that is used for creating all figures of this study.  
* **5-1.Plotting scripts**: Code used to create all figures and supplementary figures for this paper (see details below)  
* **Demo**: Provide a simple test to calculate precipitation whiplash and its frequency, duration, intensity and occurence timing.


## Steps to run the scripts ##
1. download this repository  
2. download processed data into the folder 3-2, and 4-2. 
3. install the required python modules using conda. The environment.txt provide information on the required modules.
4. run and/or edit the scripts in folders.

## Data ##
Due to size of the original raw datasets used in this paper (about 10T), we only provide the URLs to download these datasets.  
Data analyzed in the paper is publicly available from  the following sources:  
CESM-LENS data are made available by the CESM Large Ensemble Community Project (https://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html);  
CESM-SF data are made available by the CESM1 “Single Forcing” Large Ensemble Project (https://www.cesm.ucar.edu/working_groups/CVC/simulations/cesm1- single_forcing_le.html).  
The CMIP6 ensemble used for this study are freely available from the Earth System Grid Federation (ESGF, https://esgf-node.llnl.gov/search/cmip6/);  
ERA5 data were obtained from https://cds.climate.copernicus.eu/;  
JRA-55 data were obtained from https://jra.kishou.go.jp/JRA-55/index_en.html;  
MERRA-2 data were obtained from http://gmao.gsfc.nasa.gov/reanalysis/MERRA-2;  
CHIRPS data were obtained from https://www.chc.ucsb.edu/data/chirps;  
GPCC data were obtained from https://climatedataguide.ucar.edu/climate-data/gpcc-global-precipitation-climatology-centre;  
REGEN_LongTermStns data were obtained from https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f6973_9398_8796_3040.  

## Details of the codes in folders  ## 
#See Methods section of the paper for a detailed description of the analysis.   
#For steps that take longer to run and require more computing power, some instructions on calculations are noted below.  
#This project has been tested on Inter Xeon 6248R CPU with 96 cores, and requires at least 192 GB memory.  

******1. Raw data prec-processing scripts******  
Code to pre-process all downloaded raw data before calculating whiplash   
* 1-1.gridded_datasets_combine_precipitation.py: Combine, re-grid and convert units for gridded precipitation datasets.  
* 1-2.CESM_LENS_combine_precipitation.py: Combine and convert units for CESM-LENS precipitation datasets. (> 4 hours)  
* 1-3.CESM_combine_Single_Forcing_precipitation.py: Combine and convert units for CESM-XLENS precipitation datasets. (> 4 hours)  
* 1-4.CESM_LENS_and_SF_interp_precipitation.py: Re-grid CESM-XLENS and CESM-LENS precipitation datasets. (> 5 hours)  
* 1-5.CMIP6_combine_precipitation.py: Combine, re-grid and convert units for CMIP6 precipitation datasets.(> 5 hours)  
* 1-6.CESM_LENS_combine_UQ.py: Combine, re-grid, convert units and calculate IVT for CESM-LENS circulation datasets.(> 10 hours)  
* 1-7.CESM_LENS_combine_VQ.py: Combine, re-grid, convert units and calculate IVT for CESM-LENS circulation datasets.(> 10 hours)  
* 1-8.CESM_LENS_combine_Z500.py: Combine, re-grid, convert units and calculate Z500 for CESM-LENS circulation datasets.(> 3 hours)  

******2-1.Calculate whiplash and sensitivity test scripts******  
Code to calculate precipitation whiplash and perform sensitivity test for different parameters of the indices.    
* 4-0.Cal_CESM_LENS_1920_2100_daily_ensemble_mean_prec.py:   
* 4-1.Cal_CESM_LENS_1920_2100_daily_aunual_mean_prec_detrended.py:   
	Calculate raw and detrended annual mean precipition for CESM-LENS.   
* 4-2.Cal_gridded_dataset_daily_aunual_mean_prec_detrended.py:   
	Calculate raw and detrended annual mean precipition for gridded precipitation datasets.   
* 4-3.Cal_CESM_SF_daily_ensemble_mean_prec.py:   
* 4-4.Cal_CESM_SF_daily_aunual_mean_prec_detrended-new.py:   
	Calculate raw and detrended annual mean precipition for CESM-XLENS.   
* 4-5.Cal_CMIP6_1920_2100_daily_models_mean_prec.py:   
* 4-6.Cal_CMIP6_1920_2100_daily_aunual_mean_prec_detrended.py:    
	Calculate raw and detrended annual mean precipition for CMIP6.   
* 5-1.Cal_CESM_LENS_1920_2100_daily_whiplash_statistics_sensitivity_test.py:   
* 5-2.Cal_gridded_dataset_daily_whiplash_statistics_sensitivity_test.py:   
	Perform sensitivity analysis on the parameters used in calculating whiplash of CESM-LENS and gridded datasets.  (>10 hours with multiprocessing)   
* 6-1-1.Dai_cal_CESM_LENS_cumprec_sd_quan.py:   
* 6-1-2.Dai_set_CESM_LENS_cumprec_sd_quan_to_nc.py:  
	Calculate extreme dry and wet event thresholds for the 40 ensemble members of CESM-LENS. (>3 hours with multiprocessing)  
* 6-2-1.Dai_cal_CMIP6_cumprec_sd_quan.py:   
* 6-2-2.Dai_set_CMIP6_cumprec_sd_quan_to_nc.py:   
	Calculate extreme dry and wet event thresholds for the 55 ensemble members of CMIP6. (>3 hours with multiprocessing)   
* 6-3.Dai_cal_gridded_dataset_daily_whiplash_statistics_new_intensity.py:   
	Calculate the original statistical information of precipitation whiplash for gridded datasets.    
* 6-4.Dai_cal_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline1979_2019_40_ensemble_new_intensity.py:   
	Calculate the original statistical information of precipitation whiplash for CESM-LENS. (>5 hours with multiprocessing)  
* 6-5.Dai_cal_CMIP6_1920_2100_daily_whiplash_statistics_baseline1979_2019_models_new_intensity.py: 
	Calculate the original statistical information of precipitation whiplash for CMIP6. (>5 hours with multiprocessing)  
* 6-6.Dai_cal_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline1979_2019_XAER.py:  
* 6-6.Dai_cal_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline1979_2019_XBMB.py:  
* 6-6.Dai_cal_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline1979_2019_XGHG.py:  
	Calculate the original statistical information of precipitation whiplash for CESM-XLENS. (>10 hours with multiprocessing)   

******3-1.Analysis scripts******  
Code to analyze the features of precipitation whiplash events.    
* 7-1.Dai_cal_Northeastern_China_CESM_LENS_1920_2100_daily_whiplash_statistics_baseline1979_2019.py  
	Calculate the original statistical information of precipitation whiplash for CESM-XLENS in Northeastern China.  
* 7-2.Dai_cal_current_and_future_IVT_[UQ/VQ/Z]_anomalies_[before/after].py   
	Calculte the circulation anomalies of whiplash days  in Northeastern China. (>10 hours)  
* 8-1.Dai_cal_gridded_dataset_event_frequency_duration_intensity.py  
* 8-2.Dai_cal_CESM_LENS_event_frequency_duration_intensity_40_ensemble.py  
* 8-2-1.Dai_cal_CESM_LENS_mean_Occurence_time.py  
* 8-3.Dai_cal_CMIP6_event_frequency_duration_intensity_55_ensemble.py  
* 8-4.Dai_cal_SF_event_frequency_duration_intensity.py  
	Calculate whiplash occurrence frequency, transition duration, transition intensity and average occurrence timing 
	for gridded datasets, CESM-LENS, CESM-XLENS, and CMIP6. (>3 hours)  

******4-1.Creat source input data scripts******  
Code to organize the results into csv and npy files that can be used directly to create figures.  
The number in each script file name indicates the serial number of figures in the paper.  
See each figure caption of the paper for a detailed description of the analysis.  
* 1_S8.climatology_of_features_and_time.py  
	Calculate climatology of occurrence frequency, transition duration, intensity and average timing of precipitation whiplash.  
* 2_S9-11.global_and_land_future_change.py  
	Calculate projected relative changes in the occurrence characteristics of whiplash over 1921-2099.  
* 3_S14-15.monsoon_regions_mean_trend_SNR.py   
	Calculate projected relative changes in the monsoon-regional occurrence characteristics of precipitation whiplash and the emergence of external forcing.  
* 4.prec_and_whiplash_concurrent_regimes.py  
	Calculate concurrent changes in precipitation totals and precipitation whiplash occurrences. Concurrent changes in the occurrence frequency of dry-to-wet whiplash and wet-to-dry whiplash  
* 5_S16-19.relative_influence_of_single_forcings.py  
	Calculate anthropogenic effects on changes in the occurrence frequency of precipitation whiplash.   
* 6_S20.whiplash_circulation.py  
	Calculate large scale atmospheric circulation associated with dry-to-wet whiplash over northeastern China.  
* S1.cal_global_mean_prec_and_change.py  
	Calculate global annual mean precipitation and precipitation trend over 1920-2100 in the CESM-LENS ensemble.  
* S2.an_example_of_whiplash.py  
	Select an example illustrating the principle of the whiplash indices.   
* S3-5.test_sensitivity_of_different_paras.py  
	Calculate the occurrence frequency of dry-to-wet and wet-to-dry whiplash calculated by different parameters.  
* S6-7.climatology_of_global_mean_map_of_datasets.py  
	Calculate frequency of whiplash in CESM-LENS, CMIP6, 6 gridded datasets over 1979-2019.   
* S12-13.global_signal_to_noise.py  
	Calculate the time of emergence when the forced response of whiplash is greater than the internal variability and the number of members required to produce a robust whiplash change.(>3 hours with multiprocessing)   
* S21-22.circulation_from_to.py  
Gradual changing process of large-scale atmospheric circulation of whiplash over northeastern China  
* S23.ensemble_std_compare.py  
	Calculate global standard deviation in frequency trends of whiplash for the current period and future period over CMIP6 and CESM-LENS.  

******5-1.Plotting scripts******  
Code used to create all figures and supplementary figures for this paper  
The name of each script file indicates the serial number of figures in the paper.  
See each figure caption of the paper for a detailed description of the analysis.   
Simply change the input path in the [source data] cell of the python scripts to creat figures.  




