"""
First created on Mon Aug 13 10:01:03 2018

Module used for analysis of the image created from Zernike analysis

Versions:
Oct 31, 2019: 0.22 -> 0.22b introduced verbosity
Mar 10, 2020: 0.22b -> 0.23 if std too small, disregard error calculation
Apr 01, 2020: 0.23 -> 0.24 added options to create_basic_comparison_plot
Apr 29, 2020: 0.24 -> 0.24a added check for image for both side of defocus in create_solution
Jun 17, 2020: 0.24a -> 0.24b cleaned the STAMPS_FOLDER specification
Jun 25, 2020: 0.24b -> 0.25 improved create_res_data
Jul 03, 2020: 0.25 -> 0.26 included multi analysis
Jul 15, 2020: 0.26 -> 0.26a modified to include PSF_DIRECTORY
Sep 08, 2020: 0.26a -> 0.26b small changed around create_chains functions

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""
########################################
#standard library imports
from __future__ import absolute_import, division, print_function
import os
import time
import sys
import math
import socket
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
#print(np.__config__)
from multiprocessing import current_process
from functools import lru_cache

from tqdm import tqdm
#import pyfftw
#import pandas as pd

########################################
# Related third party imports
# none at the moment

########################################
# Local application/library specific imports
# galsim
import galsim
galsim.GSParams.maximum_fft_size=12000

# astropy
import astropy
import astropy.convolution
from astropy.convolution import Gaussian2DKernel

# scipy and skimage
import scipy.misc
import skimage.transform
import scipy.optimize as optimize
from scipy.ndimage.filters import gaussian_filter

# pickle
import pickle

#lmfit
import lmfit

#matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# pandas
import pandas as pd

# needed for resizing routines
from typing import Tuple, Iterable
########################################

__all__ = ['Zernike_Analysis','Zernike_result_analysis','create_mask','resize']

__version__ = "0.26b"

############################################################
# name your directory where you want to have files!
if socket.gethostname()=='IapetusUSA':
    PSF_DIRECTORY='/Users/nevencaplar/Documents/PFS/'
else:
    PSF_DIRECTORY='/tigress/ncaplar/'

  

############################################################   

TESTING_FOLDER=PSF_DIRECTORY+'Testing/'
TESTING_PUPIL_IMAGES_FOLDER=TESTING_FOLDER+'Pupil_Images/'
TESTING_WAVEFRONT_IMAGES_FOLDER=TESTING_FOLDER+'Wavefront_Images/'
TESTING_FINAL_IMAGES_FOLDER=TESTING_FOLDER+'Final_Images/'
   
class Zernike_Analysis(object):
    """!
    Class for analysing results of the cluster run
    """

    def __init__(self, date,obs,single_number,eps,arc=None,dataset=None,multi_var=False,list_of_defocuses=None,verbosity=1):
        """!

        @param[in] date                                           date
        @param[in] obs                                            observatio
        @param[in] single_number                                  single number determining which spot we are analyzing
        @param[in] eps                                            analysis parameter
        @param[in] arc                                            arc-lamp used
        @param[in] dataset                                        dataset number
        @param[in] multi_var                                      is this multi analysis
        @param[in] list_of_defocuses                              at which defocuses we are analyzing           
        
        """
        
        
        ############
        #initializing 
        ###########
        if arc is None:
            arc=''
            

        self.date=date
        self.obs=obs

        
        self.single_number=single_number
        self.eps=eps
        self.arc=arc
        self.multi_var=multi_var
        self.list_of_defocuses=list_of_defocuses 
        
        method='P'
        self.method=method
        self.verbosity=verbosity
        
            
        #############
        # where are poststamps of spots located
        if dataset==0:
            STAMPS_FOLDER=PSF_DIRECTORY+"Data_Nov_14/Stamps_cleaned/"  
        if dataset==1:
            STAMPS_FOLDER=PSF_DIRECTORY+"ReducedData/Data_Feb_5/Stamps_cleaned/"    
        if dataset==2:
            STAMPS_FOLDER=PSF_DIRECTORY+"ReducedData/Data_May_28/Stamps_cleaned/"
        if dataset==3:
            STAMPS_FOLDER=PSF_DIRECTORY+"ReducedData/Data_Jun_25/Stamps_cleaned/"
        if dataset==4 or dataset==5:
            STAMPS_FOLDER=PSF_DIRECTORY+"ReducedData/Data_Aug_14/Stamps_cleaned/"    

        
        # which observation numbers associated with each dataset
        if dataset==0:
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=8603
                elif arc=="Ne":
                    single_number_focus=8693  

        if dataset==1:  
            # F/3.4 stop 
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=11748
                    obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11748,11694,11700,11706,11712,11718,11724,11730,11736])

                elif arc=="Ne":
                    single_number_focus=11748+607  
                    obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12355,12349,12343,12337,12331,12325,12319,12313,12307])
 
        if dataset==2:
            # F/2.8 stop
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=17017+54
                    obs_possibilites=np.array([17023,17023+6,17023+12,17023+18,17023+24,17023+30,17023+36,17023+42,-99,17023+48,\
                               17023+54,17023+60,17023+66,17023+72,17023+78,17023+84,17023+90,17023+96,17023+48])
                if arc=="Ne":
                    single_number_focus=16292  
                    obs_possibilites=np.array([16238+6,16238+12,16238+18,16238+24,16238+30,16238+36,16238+42,16238+48,-99,16238+54,\
                               16238+60,16238+66,16238+72,16238+78,16238+84,16238+90,16238+96,16238+102,16238+54])
                if arc=="Kr":
                    single_number_focus=17310+54  
                    obs_possibilites=np.array([17310+6,17310+12,17310+18,17310+24,17310+30,17310+36,17310+42,17310+48,-99,17310+54,\
                                    17310+60,17310+66,17310+72,17310+78,17310+84,17310+90,17310+96,17310+102,17310+54])
                
        if dataset==3:  
            # F/2.5 stop
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=19238+54
                    obs_possibilites=np.array([19238,19238+6,19238+12,19238+18,19238+24,19238+30,19238+36,19238+42,-99,19238+48,\
                                   19238+54,19238+60,19238+66,19238+72,19238+78,19238+84,19238+90,19238+96,19238+48])
                elif arc=="Ne":
                    single_number_focus=19472  
                    obs_possibilites=np.array([19472+6,19472+12,19472+18,19472+24,19472+30,19472+36,19472+42,19472+48,-99,19472+54,\
                                  19472+60,19472+66,19472+72,19472+78,19472+84,19472+90,19472+96,19472+102,19472+54]) 
                    
        if dataset==4: 
            # F/2.8 stop, July LAM data, full defocus
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=21346+54
                    obs_possibilites=np.array([21346+6,21346+12,21346+18,21346+24,21346+30,21346+36,21346+42,21346+48,-99,21346+54,\
                                   21346+60,21346+66,21346+72,21346+78,21346+84,21346+90,21346+96,21346+102,21346+48])
                if arc=="Ne":
                    single_number_focus=21550+54  
                    obs_possibilites=np.array([21550+6,21550+12,21550+18,21550+24,21550+30,21550+36,21550+42,21550+48,-99,21550+54,\
                                   21550+60,21550+66,21550+72,21550+78,21550+84,21550+90,21550+96,21550+102,21550+54])
                if str(arc)=="Kr":
                    single_number_focus=21754+54    
                    obs_possibilites=np.array([21754+6,21754+12,21754+18,21754+24,21754+30,21754+36,21754+42,21754+48,-99,21754+54,\
                                    21754+60,21754+66,21754+72,21754+78,21754+84,21754+90,21754+96,21754+102,21754+54])
    
        if dataset==5:
            # F/2.8 stop, July LAM data, fine defocus
            
                if arc=='HgAr':
                    obs_possibilites=np.arange(21280,21280+11*6,6)
                if arc=='Ne':
                    obs_possibilites=np.arange(21484,21484+11*6,6)
                if arc=='Kr':
                     obs_possibilites=np.arange(21688,21688+11*6,6)

        #  if multi ??
        if multi_var==True:
            obs_multi=single_number_focus+48
            self.obs_multi=obs_multi
            obs_single=obs
            self.obs_single=obs_single
            

        label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']
        label_fine_defocus=['m05ff','m04ff','m03ff','m02ff','m01ff','0ff','p01ff','p02ff','p03ff','p04ff','p05ff']

        if type(obs)==str:
            labelInput=obs
            obs=obs_possibilites[label.index(labelInput)]
            
        obs_int = int(obs)      
        
        

        
        
        if dataset in [0,1,2,3,4]:
            labelInput=label[list(obs_possibilites).index(obs_int)]
        if dataset in [5]:
            labelInput=label_fine_defocus[list(obs_possibilites).index(obs_int)]
            
        if multi_var==True:
            if self.verbosity==1:
                print('labelInput: ' + str(labelInput))
                print('self.single_number: '+str(self.single_number))
            index_of_single_image_in_list_of_images=self.list_of_defocuses.index(labelInput)
            self.index_of_single_image_in_list_of_images=index_of_single_image_in_list_of_images
            
        list_of_obs=[]
        if multi_var==True:

            for labelInput in self.list_of_defocuses:
                if dataset in [0,1,2,3,4]:
                    obs_single=obs_possibilites[label.index(labelInput)]
                if dataset in [5]:
                    obs_single=obs_possibilites[label_fine_defocus.index(labelInput)]
                    
                list_of_obs.append(obs_single)
        else:
            list_of_obs.append(obs_single)


        ##########################
        # import data
        ##########################
       
        
        if multi_var==True:
            list_of_sci_images=[]
            list_of_mask_images=[]
            list_of_var_images=[]
            
            if self.verbosity==1:
                print('list_of_defocuses: ' +str(self.list_of_defocuses))
                print('list_of_obs: ' +str(list_of_obs))
            
            
            for obs_v in list_of_obs:
                sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs_v)+str(single_number)+str(arc)+'_Stacked.npy')
                mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs_v)+str(single_number)+str(arc)+'_Stacked.npy')
                var_image =np.load(STAMPS_FOLDER+'var'+str(obs_v)+str(single_number)+str(arc)+'_Stacked.npy')
                
                list_of_sci_images.append(sci_image)
                list_of_mask_images.append(mask_image)
                list_of_var_images.append(var_image)     


        sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
        var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')   
                
       
        self.list_of_sci_images=list_of_sci_images
        self.list_of_mask_images=list_of_mask_images
        self.list_of_var_images=list_of_var_images
        
        
        self.sci_image=sci_image
        self.var_image=var_image
        self.mask_image=mask_image
        self.STAMPS_FOLDER=STAMPS_FOLDER
        
        if dataset==1:
            if arc=="HgAr":
                finalArc=finalHgAr_Feb2019
            elif arc=="Ne":
                finalArc=finalNe_Feb2019    
            else:
                print("Not recognized arc-line")  
                
        if dataset==2: 
            
            with open(PSF_DIRECTORY+'ReducedData/Data_May_28/Dataframes/finalNe_May2019.pkl', 'rb') as f:
                finalNe_May2019=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_May_28/Dataframes/finalHgAr_May2019.pkl', 'rb') as f:
                finalHgAr_May2019=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_May_28/Dataframes/finalKr_May2019.pkl', 'rb') as f:
                finalKr_May2019=pickle.load(f)  
            
            if arc=="HgAr":
                finalArc=finalHgAr_May2019
            elif arc=="Ne":
                finalArc=finalNe_May2019    
            elif arc=="Kr":
                finalArc=finalKr_May2019    
            else:
                print("Not recognized arc-line")   
                
        if dataset==3:   
            
            with open(PSF_DIRECTORY+'ReducedData/Data_Jun_25/Dataframes/finalNe_May2019.pkl', 'rb') as f:
                finalNe_May2019=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_Jun_25/Dataframes/finalHgAr_May2019.pkl', 'rb') as f:
                finalHgAr_May2019=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_Jun_25/Dataframes/finalKr_May2019.pkl', 'rb') as f:
                finalKr_May2019=pickle.load(f)  
            
            if arc=="HgAr":
                finalArc=finalHgAr_May2019
            elif arc=="Ne":
                finalArc=finalNe_May2019    
            else:
                print("Not recognized arc-line")   
                
        if dataset==4 or dataset==5:   
                
            with open(PSF_DIRECTORY+'ReducedData/Data_Aug_14/Dataframes/finalHgAr_Feb2020', 'rb') as f:
                print(f)
                finalHgAr_Feb2020_dataset=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_Aug_14/Dataframes/finalNe_Feb2020', 'rb') as f:
                finalNe_Feb2020_dataset=pickle.load(f)  
            with open(PSF_DIRECTORY+'ReducedData/Data_Aug_14/Dataframes/finalKr_Feb2020', 'rb') as f:
                finalKr_Feb2020_dataset=pickle.load(f)  
            
            
            if arc=="HgAr":
                finalArc=finalHgAr_Feb2020_dataset
            elif arc=="Ne":
                finalArc=finalNe_Feb2020_dataset    
            elif arc=="Kr":
                finalArc=finalKr_Feb2020_dataset    
            else:
                print("Not recognized arc-line")            

        ##########################
        # import column names
        ##########################

        
        columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent',
                      'x_ilum','y_ilum',
                      'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']    
        
            
        columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
               'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_radius_illumination',
              'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
        
        columns22_analysis=columns22+['chi2','chi2max']
        
        self.columns=columns
        self.columns22=columns22
        self.columns22_analysis=columns22_analysis
 
        ##########################
        # where are results from Tiger placed
        ##########################
       
        RESULT_FOLDER=PSF_DIRECTORY+'TigerAnalysis/ResultsFromTiger/'+date+'/'
        if os.path.exists(RESULT_FOLDER):
            pass
        else:
            RESULT_FOLDER='/Volumes/My Passport for Mac/Old_Files/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
            if os.path.exists(RESULT_FOLDER):
                pass
            else:
                RESULT_FOLDER='/Volumes/Saturn_USA/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
        
        self.RESULT_FOLDER=RESULT_FOLDER
        
        IMAGES_FOLDER=PSF_DIRECTORY+'/Images/'+date+'/'
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        self.IMAGES_FOLDER=IMAGES_FOLDER
        
        if finalArc['close'].loc[int(single_number)]=='1':
            double_sources=False
        else:
             double_sources=True           
        
        self.double_sources=double_sources
        double_sources_positions_ratios=finalArc.loc[int(single_number)][['second_offset','second_ratio']].values
        self.double_sources_positions_ratios=double_sources_positions_ratios
        
        if self.verbosity==1:
            print('analyzing label: '+str(obs))
            print('double_sources_positions_ratios for this spot is: '+str(double_sources_positions_ratios))
            
            
            
            
        
        

    def return_double_sources(self):
        return self.double_sources,self.double_sources_positions_ratios
    
    def return_lists_of_images(self):
        assert self.multi_var==True

        return self.list_of_sci_images,self.list_of_var_images,self.list_of_mask_images
    
    def return_index_of_single_image_in_list_of_images(self):
        return self.index_of_single_image_in_list_of_images

    def return_columns(self):
        return self.columns,self.columns22,self.columns22_analysis
    
    def create_list_of_var_or_ln_sums(self,sigma_offset=0):
        list_of_var_sums=[]
        for i in range(len(self.list_of_var_images)):
            # taking from create_chi_2_almost function in LN_PFS_single
    
    
            mask_image=self.list_of_mask_images[i]
            var_image=self.list_of_var_images[i]
            # array that has True for values which are good and False for bad values
            inverted_mask=~mask_image.astype(bool)
    
            #         
            var_image_masked=var_image*inverted_mask
            var_image_masked_without_nan = var_image_masked.ravel()[var_image_masked.ravel()>0]
    
            var_sum=-(1/2)*(len(var_image_masked_without_nan)*sigma_offset+np.sum(np.log(2*np.pi*var_image_masked_without_nan)))
    
            list_of_var_sums.append(var_sum)  
            
        array_of_var_sums=np.array(list_of_var_sums)    
        return array_of_var_sums        
        
    
    
    def create_likelihood(self):

        if self.multi_var==True:
            self.obs=self.obs_multi
            
            
        self.len_of_chains()    
            
        # Swarm1
        #likechain_Swarm1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')
        likechain_Swarm1=self.likechain_Swarm1
        
        
        like_min_swarm1=[]
        for i in range(likechain_Swarm1.shape[0]):
            like_min_swarm1.append(np.min(np.abs(likechain_Swarm1[i]))  )  

        #        
        if self.chain_Emcee2 is not None:
            # Emcee1
            #likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
            likechain_Emcee1=self.likechain_Emcee1
            
            like_min_Emcee1=[]
            for i in range(likechain_Emcee1.shape[1]):
                like_min_Emcee1.append(np.min(np.abs(likechain_Emcee1[:,i]))  )     
                
      
            # Swarm2
            likechain_Swarm2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')        
            likechain_Swarm2=self.likechain_Swarm2
    
            like_min_swarm2=[]
            for i in range(likechain_Swarm2.shape[0]):
                like_min_swarm2.append(np.min(np.abs(likechain_Swarm2[i]))  )  
            
            # Emcee 2
            #chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
            #likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
            
            chain_Emcee2=self.chain_Emcee2
            likechain_Emcee2=self.likechain_Emcee2
      
            # get chain number 0, which is has lowest temperature
            #if len(likechain_Emcee3)<=4:
            #    likechain0_Emcee3=likechain_Emcee3[0]
            #    chain0_Emcee3=chain_Emcee3[0]     
            #else:
            #    likechain0_Emcee3=likechain_Emcee3
            #    chain0_Emcee3=chain_Emcee3
            # check the shape of the chain (number of walkers, number of steps, number of parameters)
            if self.verbosity==1:
                print('(number of walkers, number of steps, number of parameters for Emcee): '+str(chain_Emcee2.shape))
            
            # see the best chain
            minchain=chain_Emcee2[np.abs(likechain_Emcee2)==np.min(np.abs(likechain_Emcee2))][0]
            #print(minchain)
            self.minchain=minchain
            
            like_min_Emcee2=[]
                    
            for i in range(likechain_Emcee2.shape[1]):
                like_min_Emcee2.append(np.min(np.abs(likechain_Emcee2[:,i]))  )  
            
   
            like_min=like_min_swarm1+like_min_Emcee1+like_min_swarm2+like_min_Emcee2
        else:
            
            # see the best chain
            minchain=self.chain_Swarm1[np.abs(self.likechain_Swarm1)==np.min(np.abs(self.likechain_Swarm1))][0]
            #print(minchain)
            self.minchain=minchain
            
            like_min=like_min_swarm1            
            
        list_of_var_sums=self.create_list_of_var_or_ln_sums(0)
        
        array_of_var_sum=np.array(list_of_var_sums)
        max_of_array_of_var_sum=np.max(array_of_var_sum)
        renormalization_of_var_sum=array_of_var_sum/max_of_array_of_var_sum
        
        zero_sigma_ln=np.mean(list_of_var_sums/renormalization_of_var_sum)
        self.zero_sigma_ln=zero_sigma_ln
        list_of_var_sums_1=self.create_list_of_var_or_ln_sums(1)
        one_sigma_ln=np.mean(list_of_var_sums_1/renormalization_of_var_sum)
        self.one_sigma_ln=one_sigma_ln

        #print(len(like_min))      
        if self.verbosity==1:                  
            print('minimal likelihood is: '+str(np.min(like_min)))   
        
        min_like_min=np.min(like_min)
        self.min_like_min=min_like_min
        
        
        chi2=(np.array(like_min)*(2)-np.sum(np.log(2*np.pi*self.var_image)))/(self.sci_image.shape[0])**2
       
        min_chi2=-(min_like_min+zero_sigma_ln)/(one_sigma_ln-zero_sigma_ln)
        
        print('average chi2 reduced is: '+str(min_chi2))
        
        return minchain,like_min

    def len_of_chains(self):

        if self.multi_var==True:
            self.obs=self.obs_multi
            
            
        
            
        self.create_chains_Emcee_1()
        self.create_chains_Emcee_2()
        self.create_chains_swarm_1()
        self.create_chains_swarm_2()
   
        # (number of walkers, number of steps, number of parameters) for Emcee 
        # (number of steps, number of walkers, number of parameters) for Swarm 
        if self.chain_Emcee2 is None:
            print(self.chain_Swarm1.shape)       
            return [len(self.chain_Swarm1),0,0,0]
        else:
            print(self.chain_Swarm1.shape,self.chain_Emcee2.shape,self.chain_Swarm2.shape,self.chain_Emcee3.shape)
            return [len(self.chain_Swarm1),(self.chain_Emcee2).shape[1],len(self.chain_Swarm2),(self.chain_Emcee3).shape[1]]
        

    def create_chains(self):
         
        #chain_Emcee1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')
        #likechain_Emcee1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')

        # get chain number 0, which is has lowest temperature
        #likechain0_Emcee1=likechain_Emcee1[0]
        #chain0_Emcee1=chain_Emcee1[0]

        #chain_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        #likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        
        #likechain0_Emcee2=likechain_Emcee2[0]
        #chain0_Emcee2=chain_Emcee2[0]        
        if self.multi_var==True:
            self.obs=self.obs_multi
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3   


    
    def create_chains_Emcee_1(self):
        """
        get chain and likelihood chain for first run of Emcee
        unfortunately the file name is ``Emcee2'', because of historical reasons
        
        
        Returns
        -------
        chain0_Emcee1 : chain
        likechain0_Emcee1 : likelihood chain
        
        
        """
        
        if self.multi_var==True:
            self.obs=self.obs_multi
        try:     
            chain_Emcee1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                 str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
            likechain_Emcee1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                     str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
            
            
            self.chain_Emcee1=chain_Emcee1
            self.likechain_Emcee1=likechain_Emcee1
        except:
            self.chain_Emcee1=None
            self.likechain_Emcee1=None
            
        
        return self.chain_Emcee1,self.likechain_Emcee1   
    
    
    def create_chains_Emcee_2(self):
        """        
        get chain and likelihood chain for the second run of Emcee
        unfortunately the file name is ``Emcee3'', because of historical reasons
        """        

        if self.multi_var==True:
            self.obs=self.obs_multi
            
        try:
            chain_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                 str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
            likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                     str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
    
            self.chain_Emcee2=chain_Emcee2
            self.likechain_Emcee2=likechain_Emcee2
        except:
            self.chain_Emcee2=None
            self.likechain_Emcee2=None
        
        return self.chain_Emcee2,self.likechain_Emcee2  
 
        
    def create_Emcee2_stack(self):

        if self.multi_var==True:
            self.obs=self.obs_multi
        
        
        chain0_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                              str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        likechain0_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                  str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        
        for i in range(chain0_Emcee2.shape[1]):
            if i==0:
                chain0_Emcee2_reshaped=chain0_Emcee2[:,0]
                likechain0_Emcee2_reshaped=likechain0_Emcee2[:,0]
            else:
                chain0_Emcee2_reshaped=np.vstack((chain0_Emcee2_reshaped,chain0_Emcee2[:,i]))
                likechain0_Emcee2_reshaped=np.vstack((likechain0_Emcee2_reshaped,likechain0_Emcee2[:,i]))
    
    
        chain0_stack=chain0_Emcee2_reshaped
        likechain0_stack=likechain0_Emcee2_reshaped.ravel()
        likechain0_stack=likechain0_stack-np.max(likechain0_stack)
        
        
        return chain0_stack,likechain0_stack
    

    def create_chains_swarm_1(self):
        
        """        
        get chain and likelihood chain for the first run of cosmoHammer optimizer

        """  

        if self.multi_var==True:
            self.obs=self.obs_multi
        try:      
            chain_Swarm1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                 str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')
            likechain_Swarm1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                     str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')
        except:
            print('Swarm1 or likechainSwarm1 not found')
            print('Path searched was: '+str(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                 str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy'))

        
        self.chain_Swarm1=chain_Swarm1
        self.likechain_Swarm1=likechain_Swarm1
        
        return chain_Swarm1,likechain_Swarm1       
    
    def create_chains_swarm_2(self):
        
        """        
        get chain and likelihood chain for the second run of cosmoHammer optimizer

        """
              
        if self.multi_var==True:
            self.obs=self.obs_multi
        try:
            chain_Swarm2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                 str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')
            likechain_Swarm2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+\
                                     str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')
      
            self.chain_Swarm2=chain_Swarm2
            self.likechain_Swarm2=likechain_Swarm2
            
        except:
            self.chain_Swarm2=None
            self.likechain_Swarm2=None
            
        
        return self.chain_Swarm2,self.likechain_Swarm2      
     

    def create_allparameters_single(self,mm,array_of_polyfit_1_parameterizations,zmax=None):
        """
        
        copied from multi
        
        transfroms linear fits as a function of defocus of parametrizations into form acceptable for creating single images 
        workhorse function used by create_list_of_allparameters
        
        @param mm [float]                               defocus of the slit
        @param array_of_polyfit_1_parameterizations     parametrs describing linear fit for the parameters as a function of focus
        @param zmax                                     largerst Zernike used
        
        """
        
        if zmax==None:
            zmax=11
        
        #for single case, up to z11
        if zmax==11:
            z_parametrizations=array_of_polyfit_1_parameterizations[:8]
            g_parametrizations=array_of_polyfit_1_parameterizations[8:]
            
            
            allparameters_proposal_single=np.zeros((8+len(g_parametrizations)))
            
            for i in range(0,8,1):
                allparameters_proposal_single[i]=self.value_at_defocus(mm,z_parametrizations[i][0],z_parametrizations[i][1])      
        
            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[i+8]=g_parametrizations[i][1] 
                
        if zmax==22:
            z_parametrizations=array_of_polyfit_1_parameterizations[:19]
            g_parametrizations=array_of_polyfit_1_parameterizations[19:]
            
            
            allparameters_proposal_single=np.zeros((19+len(g_parametrizations)))
            for i in range(0,19,1):
                #print(str([i,mm,z_parametrizations[i]]))
                allparameters_proposal_single[i]=self.value_at_defocus(mm,z_parametrizations[i][0],z_parametrizations[i][1])      
        
            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[19+i]=g_parametrizations[i][1] 
            
        return allparameters_proposal_single           
        
        
        
    
    def entrance_exit_pupil_plot(self):
        
        
        ilum=np.load(TESTING_PUPIL_IMAGES_FOLDER+'ilum.npy')
        radiometricEffectArray=np.load(TESTING_PUPIL_IMAGES_FOLDER+'radiometricEffectArray.npy')
        ilum_radiometric=np.load(TESTING_PUPIL_IMAGES_FOLDER+'ilum_radiometric.npy')
        
        plt.figure(figsize=(30,8))
        plt.subplot(131)
        plt.imshow(ilum,origin='lower',vmax=1,vmin=0)
        plt.title('entrance pupil')
        plt.colorbar()
        plt.subplot(132)
        plt.title('ent->exit pupil')
        plt.imshow(radiometricEffectArray,origin='lower',vmax=1,vmin=0)
        
        plt.colorbar()
        plt.subplot(133)
        plt.title('exit pupil')
        plt.imshow(ilum_radiometric,origin='lower',vmax=1,vmin=0)
        plt.colorbar()
            
    def wavefront_plot(self):
        
        wf_full=np.load(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full.npy') 
        
        
        
        plt.figure(figsize=(36,6))
        plt.subplot(141)
        plt.imshow(wf_full)
        plt.colorbar()
        
        plt.subplot(142)
        plt.imshow(np.real(np.exp(2j*np.pi * wf_full/800)))
        plt.colorbar()
        
        plt.subplot(143)
        plt.imshow(np.imag(np.exp(2j*np.pi * wf_full/800)))
        plt.colorbar()
        
    def illumination_wavefront_plot(self):
        ilum=np.load(TESTING_PUPIL_IMAGES_FOLDER+'ilum.npy')
        wf_full=np.load(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full.npy') 
        wf_full_fake_0=np.load(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full_fake_0.npy') 
        
        midpoint=int(len(ilum)/2)
        
        plt.figure(figsize=(26,6))
        
        plt.subplot(131)
        plt.imshow(ilum[int(midpoint-len(ilum)/4):int(midpoint+len(ilum)/4),int(midpoint-len(ilum)/4):int(midpoint+len(ilum)/4)],origin='lower',vmax=1,vmin=0)
        plt.title('illumination of the pupil',fontsize=25)
        plt.subplot(132)

        ilum_1=np.copy(ilum)
        ilum_1[ilum_1>0.01]=1
        
        wavefront=ilum_1*wf_full
        wavefront=wavefront/800
        plt.imshow(wavefront[int(midpoint-len(ilum)/4):int(midpoint+len(ilum)/4),int(midpoint-len(ilum)/4):\
                             int(midpoint+len(ilum)/4)],cmap=plt.get_cmap('bwr'),vmax=np.max(np.abs(wavefront))*0.75,vmin=-np.max(np.abs(wavefront))*0.75)

        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('wavefront [units of waves]',fontsize=25)
        
        plt.subplot(133)

        ilum_1=np.copy(ilum)
        ilum_1[ilum_1>0.01]=1
        
        wavefront=ilum_1*wf_full_fake_0
        wavefront=wavefront/800
        plt.imshow(wavefront[int(midpoint-len(ilum)/4):int(midpoint+len(ilum)/4),int(midpoint-len(ilum)/4):int(midpoint+len(ilum)/4)],cmap=plt.get_cmap('bwr'),vmax=np.max(np.abs(wavefront))*0.75,vmin=-np.max(np.abs(wavefront))*0.75)

        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('wavefront w.o. defocus [u. of waves]',fontsize=25)        
        
        
    
    def wavefront_gradient_plot(self):
         
        wf_full=np.load(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full.npy') 
        plt.figure(figsize=(30,8))
        plt.subplot(131)
        vgrad = np.gradient(wf_full)
        fulgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
        plt.title('gradient (magnitude)')
        plt.imshow(fulgrad,cmap=plt.get_cmap('hot'), vmin = np.amin(fulgrad),vmax = np.amax(fulgrad))  
        plt.colorbar()
        plt.subplot(132)
        x, y = range(0, len(wf_full)), range(0,len(wf_full))
        xi, yi = np.meshgrid(x, y)
        plt.title('gradient (direction)')
        plt.streamplot(xi, yi, vgrad[0], vgrad[1])
        plt.subplot(133)
        laplace_of_wf = scipy.ndimage.filters.laplace(wf_full)
        plt.title('Laplacian')
        plt.imshow(laplace_of_wf,cmap=plt.get_cmap('hot'), vmin = -1,vmax = 1) 
        plt.colorbar()

    def create_basic_data_image(self,return_Images=False):     
        
        sci_image=self.sci_image
        var_image=self.var_image
        mask_image=self.mask_image
        
        
        plt.figure(figsize=(30,8))
        plt.subplot(131)
        plt.imshow(sci_image,norm=LogNorm(),origin='lower',vmin=1,vmax=np.max(sci_image))
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        
        plt.subplot(132)
        plt.imshow(var_image,norm=LogNorm(),origin='lower',vmin=1,vmax=np.max(sci_image))
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        
        plt.subplot(133)
        plt.imshow(sci_image,norm=LogNorm(),origin='lower',vmin=1,vmax=np.max(sci_image))
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        plt.imshow(mask_image,origin='lower',vmin=0,vmax=np.max(mask_image),alpha=0.2)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        
        if return_Images==True:
            return sci_image,var_image,mask_image


    def create_fitting_evolution_plot(self):     
        
        minchain,like_min=self.create_likelihood()
        len_of_chains=self.len_of_chains()
        #chain0_Emcee3,likechain0_Emcee3=self.create_chains()
        
        
        #size=self.chain_swarm1.shape[1]
        matplotlib.rcParams.update({'font.size': 18})
        plt.figure(figsize=(24,12))
        plt.subplot(211)
        plt.plot(np.linspace(1,len(like_min),len(like_min)),like_min,'blue',ls='-',marker='o')
        plt.ylabel('likelihood')
        plt.xlabel('steps')
        plt.axvline(np.sum(len_of_chains[:1])+0.5,ls='--')
        plt.axvline(np.sum(len_of_chains[:2])+0.5,ls='--')
        plt.axvline(np.sum(len_of_chains[:3])+0.5,ls='--')
        plt.ylim(-self.zero_sigma_ln,1.05*np.max(like_min))
        plt.axhline(self.min_like_min,ls='--')
        plt.axhline(-self.one_sigma_ln,ls='--',color='black')        
        
        plt.subplot(212)
        plt.plot(np.linspace(1,len(like_min),len(like_min)),np.log10(like_min),'blue',ls='-',marker='o')
        plt.ylabel('log10(likelihood)')
        plt.xlabel('steps')
        plt.axvline(np.sum(len_of_chains[:1])+0.5,ls='--')
        plt.axvline(np.sum(len_of_chains[:2])+0.5,ls='--')
        plt.axvline(np.sum(len_of_chains[:3])+0.5,ls='--')

    
    def create_basic_comparison_plot(self,custom_model_image=None,custom_mask=None,\
                                     custom_sci_image=None,custom_var_image=None,use_max_chi_scaling=False,show_flux_mask=False): 
        
        if custom_model_image is None:
            optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
            res_iapetus=optPsf_cut_fiber_convolved_downsampled
        else:
            res_iapetus=custom_model_image
            
            
        
        if custom_sci_image is None:   
            sci_image=self.sci_image
        else:
            sci_image=custom_sci_image
            
        mean_value_of_background=np.mean([np.median(sci_image),np.median(sci_image),\
                              np.median(sci_image),np.median(sci_image)])*3
            
        flux_mask=sci_image>(mean_value_of_background)
            

        if custom_var_image is None:           
            var_image=self.var_image
        else:
            var_image=custom_var_image     
            
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
        
        if size==20:
            x_center=find_centroid_of_flux(res_iapetus)[0]
        else:
            x_center=(size/2)
            
        left_limit=np.round(x_center-3.5)+0.5
        right_limit=np.round(x_center+3.5)-0.5               
        
        
        plt.figure(figsize=(14,14))


        plt.subplot(221)
        plt.imshow(res_iapetus,origin='lower',vmax=np.max(np.abs(sci_image)))
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.4)    
        
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Model')
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image,origin='lower',vmax=np.max(np.abs(sci_image)))
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.colorbar(fraction=0.046, pad=0.04)
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.4)   
            
        plt.title('Data')
        plt.grid(False)
        plt.subplot(223)
        plt.imshow(sci_image-res_iapetus,origin='lower',cmap='bwr',vmin=-np.max(np.abs(sci_image))/20,vmax=np.max(np.abs(sci_image))/20)
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)   

        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='black')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='black')
        
        plt.colorbar(fraction=0.046, pad=0.04)
        
        
        if custom_mask is None:
            pass
        else:
            plt.imshow(custom_mask,origin='lower',alpha=0.25)
        
        plt.title('Residual (data - model)')
        plt.grid(False)
        plt.subplot(224)
        #plt.imshow((res_iapetus-sci_image)/np.sqrt(var_image),origin='lower',cmap='bwr',vmax=np.max(np.abs((res_iapetus-sci_image)/np.sqrt(var_image))),vmin=-np.max(np.abs((res_iapetus-sci_image)/np.sqrt(var_image))))
       

        if use_max_chi_scaling==False:
            plt.imshow((sci_image-res_iapetus)/np.sqrt(var_image),origin='lower',cmap='bwr',vmax=5,vmin=-5)
        else:
            max_chi=np.max(np.abs((sci_image-res_iapetus)/np.sqrt(var_image)))
            plt.imshow((sci_image-res_iapetus)/np.sqrt(var_image),origin='lower',cmap='bwr',vmax=-max_chi,vmin=max_chi)            

        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)    

        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='black')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='black')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('chi map')
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-10.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        if custom_mask is None:
            pass
        else:
            print('chi**2 reduced within mask area is: '+str(np.mean((res_iapetus[custom_mask]-sci_image[custom_mask])**2/(var_image[custom_mask]))))
        
        
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))
  
    def create_basic_comparison_plot_log(self,custom_model_image=None,custom_mask=None,custom_sci_image=None,custom_var_image=None,use_max_chi_scaling=False,\
                                         show_flux_mask=False):   
        
        
        if custom_model_image is None:
            optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
            res_iapetus=optPsf_cut_fiber_convolved_downsampled
        else:
            res_iapetus=custom_model_image
        

        if custom_sci_image is None:   
            sci_image=self.sci_image
        else:
            sci_image=custom_sci_image

        mean_value_of_background=np.mean([np.median(sci_image),np.median(sci_image),\
                              np.median(sci_image),np.median(sci_image)])*3
            
        flux_mask=sci_image>(mean_value_of_background)


        if custom_var_image is None:           
            var_image=self.var_image
        else:
            var_image=custom_var_image  
            
            
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
            
        if size==20:
            x_center=find_centroid_of_flux(res_iapetus)[0]
        else:
            x_center=(size/2)
            
        left_limit=np.round(x_center-3.5)+0.5
        right_limit=np.round(x_center+3.5)-0.5                  
        
        
        
        plt.figure(figsize=(14,14))
        plt.subplot(221)
        plt.imshow(res_iapetus,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Model')
        
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)    
        
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Data')
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)    
            
        plt.grid(False)
        plt.subplot(223)
        plt.imshow(np.abs(sci_image-res_iapetus),origin='lower',vmax=np.max(np.abs(sci_image))/20,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('abs(Residual (  data-model))')
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)        

        plt.grid(False)
        plt.subplot(224)
        plt.imshow((sci_image-res_iapetus)**2/((1)*var_image),origin='lower',vmin=1,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        if show_flux_mask==True:
            plt.imshow(flux_mask,alpha=0.55)    
        
        plt.title('chi**2 map')
        print('chi**2 max reduced is: '+str(np.sum((res_iapetus)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-7.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))     
 

    def create_basic_comparison_plot_log_artifical(self,custom_model_image=None,custom_mask=None,custom_sci_image=None,custom_var_image=None,use_max_chi_scaling=False): 
        
        # need to update for multivar
        if custom_model_image is None:
            optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
            res_iapetus=optPsf_cut_fiber_convolved_downsampled
        else:
            res_iapetus=custom_model_image
        

        noise=self.create_artificial_noise(custom_model_image=custom_model_image,custom_var_image=custom_var_image)
        
        if custom_sci_image is None:   
            sci_image=self.sci_image
        else:
            sci_image=custom_sci_image

        if custom_var_image is None:           
            var_image=self.var_image
        else:
            var_image=custom_var_image  
            
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1


        if size==20:
            x_center=find_centroid_of_flux(res_iapetus)[0]
        else:
            x_center=(size/2)
            
        left_limit=np.round(x_center-3.5)+0.5
        right_limit=np.round(x_center+3.5)-0.5            
        
        plt.figure(figsize=(14,14))
        plt.subplot(221)
        plt.imshow(res_iapetus+noise,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Model with artifical noise')
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Data')
        plt.grid(False)
        plt.subplot(223)
        plt.imshow(np.abs(res_iapetus-sci_image),origin='lower',vmax=np.max(np.abs(sci_image))/20,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('abs(Residual (model - data))')
        plt.grid(False)
        plt.subplot(224)
        plt.imshow((res_iapetus-sci_image)**2/((1)*var_image),origin='lower',vmin=1,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(left_limit),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*(right_limit),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('chi**2 map')
        print(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image)))
        np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-7.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))          
        
    def create_artificial_noise(self, custom_model_image=None,custom_var_image=None):
        
        if custom_var_image is None:           
            var_image=self.var_image
        else:
            var_image=custom_var_image  
        
        if custom_model_image is None:
            optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
            res_iapetus=optPsf_cut_fiber_convolved_downsampled
        else:
            res_iapetus=custom_model_image        
                
        artifical_noise=np.zeros_like(res_iapetus)
        artifical_noise=np.array(artifical_noise)
        for i in range(len(artifical_noise)):
            for j in range(len(artifical_noise)):
                artifical_noise[i,j]=np.random.randn()*np.sqrt(var_image[i,j]+40)       
                
        return artifical_noise
    
    def create_cut_plots(self):
        
        var_image=self.var_image
        artifical_noise=self.create_artificial_noise()
        sci_image=self.sci_image
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        
        mid_point_of_sci_image=int(sci_image.shape[0]/2)
        
        plt.figure(figsize=(25,10))
        
        plt.subplot(121)
        plt.title('horizontal direction')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image])*(1/2)),'--',color='black')
        
        
        plt.legend(fontsize=25)
        
        plt.subplot(122)
        plt.title('wavelength direction')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[:,mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[:,mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image])*(1/2)),'--',color='black')
        plt.legend(fontsize=20)      

        plt.figure(figsize=(30,10))
        
        plt.subplot(121)
        plt.title('horizontal direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[mid_point_of_sci_image]+artifical_noise[mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[mid_point_of_sci_image]*(1/2))),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(res_iapetus[mid_point_of_sci_image]*(1/2))),'-',color='blue',label='FWHM of model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(res_iapetus[mid_point_of_sci_image]-sci_image[mid_point_of_sci_image])),'red',linestyle='--',label='abs(residual)')
        plt.legend(fontsize=15)
        
        plt.subplot(122)
        plt.title('wavelength direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[:,mid_point_of_sci_image]+artifical_noise[:,mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[:,mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image]*(1/2))),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(res_iapetus[:,mid_point_of_sci_image]*(1/2))),'-',color='blue',label='FWHM of model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(res_iapetus[:,mid_point_of_sci_image]-sci_image[:,mid_point_of_sci_image])),'red',linestyle='--',label='abs(residual)')
        plt.legend(fontsize=15)

        plt.figure(figsize=(30,10))
        
        plt.subplot(121)
        plt.title('horizontal direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[mid_point_of_sci_image]+artifical_noise[mid_point_of_sci_image],'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),sci_image[mid_point_of_sci_image],'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[mid_point_of_sci_image]-sci_image[mid_point_of_sci_image],'red',linestyle='--',label='residual')
        plt.errorbar(np.array(range(len(res_iapetus))),sci_image[mid_point_of_sci_image],yerr=1*np.sqrt(var_image[mid_point_of_sci_image]),color='orange',fmt='o')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(sci_image[mid_point_of_sci_image]*(1/2)),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(res_iapetus[mid_point_of_sci_image]*(1/2)),'-',color='blue',label='FWHM of model')
        plt.legend(fontsize=15)
        
        plt.subplot(122)
        plt.title('wavelength direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[:,mid_point_of_sci_image]+artifical_noise[:,mid_point_of_sci_image],'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),sci_image[:,mid_point_of_sci_image],'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[:,mid_point_of_sci_image]-sci_image[:,mid_point_of_sci_image],'red',linestyle='--',label='residual')
        plt.errorbar(np.array(range(len(res_iapetus))),sci_image[:,mid_point_of_sci_image],yerr=1*np.sqrt(var_image[:,mid_point_of_sci_image]),color='orange',fmt='o')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(sci_image[mid_point_of_sci_image]*(1/2)),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(res_iapetus[mid_point_of_sci_image]*(1/2)),'-',color='blue',label='FWHM of model')
        plt.legend(fontsize=15)    

    def create_corner_plots(self):
        # this probably goes into result_analysis
        IMAGES_FOLDER='/Users/nevencaplar/Documents/PFS/Images/'+self.date+'/'
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        print('Images are in folder: '+str(IMAGES_FOLDER))
    
        import corner
        minchain=self.minchain
        columns=self.columns
        chain0_Emcee3=self.chain0_Emcee3
        #likechain_Emcee3=self.likechain_Emcee3

        matplotlib.rcParams.update({'font.size': 16})
        flatchain0=np.reshape(chain0_Emcee3,(chain0_Emcee3.shape[0]*chain0_Emcee3.shape[1],chain0_Emcee3.shape[2]))
        
        figure=corner.corner(flatchain0[:,0:8], labels=columns[0:8],
                          truths=list(minchain[0:8]))
        figure.savefig(IMAGES_FOLDER+'zparameters.png')
        
        flatchain0=np.reshape(chain0_Emcee3,(chain0_Emcee3.shape[0]*chain0_Emcee3.shape[1],chain0_Emcee3.shape[2]))

        figure=corner.corner(flatchain0[:,8:], labels=columns[8:],
                          truths=list(minchain[8:]))
        figure.savefig(IMAGES_FOLDER+'/globalparameters.png')
        
    def return_sci_image(self):
        return self.sci_image
            
    def return_var_image(self):
        return self.var_image
            
    def return_model_image(self):
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        return optPsf_cut_fiber_convolved_downsampled
     

    def plot_1D_residual(self,custom_model_image=None,custom_mask=None,custom_sci_image=None,custom_var_image=None,use_max_chi_scaling=False,title=None):
            
        """ 
    
        @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
        @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
        @array[in] model_image             model (20x20 image)
        @string[in] title                  custom title to appear above the plot
    
    
        @plot[out]                         diagnostic plot
    
        """
        
        if custom_model_image is None:
            optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
            model_image=optPsf_cut_fiber_convolved_downsampled
        else:
            model_image=custom_model_image
            
            
        
        if custom_sci_image is None:   
            sci_image=self.sci_image
        else:
            sci_image=custom_sci_image

        if custom_var_image is None:           
            var_image=self.var_image
        else:
            var_image=custom_var_image     
            
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
        
        if size==20:
            x_center=find_centroid_of_flux(model_image)[0]
        else:
            x_center=(size/2)
            
        left_limit=np.round(x_center-3.5)+0.5
        right_limit=np.round(x_center+3.5)-0.5           
        
        init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=self.residual_1D(sci_image,var_image,model_image)
        
        
        position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
        difference_from_max=range(20)-position_of_max_flux
        pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
        Q=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
     
        plt.figure(figsize=(20,10))
        plt.errorbar(np.array(range(len(init_lamda))),init_lamda,yerr=std_init_lamda,fmt='o',elinewidth=2,capsize=12,markeredgewidth=2,label='data',color='orange')
        plt.plot(np.array(range(len(init_lamda))),init_lamda*0.01,color='gray',label='1% data')
        plt.plot(np.array(range(len(init_lamda))),-init_lamda*0.01,color='gray')    
        
        plt.errorbar(np.array(range(len(init_removal_lamda))),init_removal_lamda,yerr=std_init_removal_lamda,color='red',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='residual')
    
        for i in range(20):
            plt.text(-0.5+i, -1400, str("{:1.0f}".format(init_lamda[i])), fontsize=20,rotation=70.,color='orange')
    
        for i in range(20):
            plt.text(-0.5+i, -2100, str("{:1.1f}".format(init_removal_lamda[i]/std_init_removal_lamda[i])), fontsize=20,rotation=70.,color='red')
        
        if title is None:
            pass
        else:
            plt.title(str(title))
            
        plt.legend(loc=2, fontsize=22)
        plt.plot(np.zeros(20),'--',color='black')
        plt.ylim(-2500,2500)
        plt.ylabel('flux',size=25)
        plt.xlabel('pixel',size=25)
        plt.xticks(range(20))
    
        sci_image_40000,var_image_40000,model_image_40000=add_artificial_noise(sci_image,var_image,model_image)
        init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=self.residual_1D(sci_image_40000,var_image_40000,model_image_40000)
    
        position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
        difference_from_max=range(20)-position_of_max_flux
        pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
        Q_40000=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
        
    
    
        plt.text(19.5,2300, '$Q_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(Q)),
                horizontalalignment='right',
                verticalalignment='top',fontsize=26)
    
        chi2=np.mean((model_image-sci_image)**2/var_image)
    
        plt.text(19.5,2000, '$\chi^{2}_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(chi2)),
                horizontalalignment='right',
                verticalalignment='top',fontsize=26)
    
        chi2_40000=np.mean((model_image_40000-sci_image_40000)**2/var_image_40000)
    
        plt.text(19.5,1650, '$Q_{40000}$='+str("{:1.2f}".format(Q_40000)),
                horizontalalignment='right',
                verticalalignment='top',fontsize=26)
        plt.text(19.5,1300, '$\chi^{2}_{40000}$='+str("{:1.2f}".format(chi2_40000)),
                horizontalalignment='right',
                verticalalignment='top',fontsize=26)
    
        plt.axvspan(pixels_to_test[0]-0.5, pixels_to_test[3]+0.5, alpha=0.3, color='grey')
        plt.axvspan(pixels_to_test[4]-0.5, pixels_to_test[7]+0.5, alpha=0.3, color='grey')
        
            
    def residual_1D(self,sci_image,var_image,res_iapetus):
        
        """
    
        @param[in] sci_image        data (20x20 cutout)
        @param[in] var_image        Variance data (20x20 cutout)
        @param[in] res_iapetus      model (20x20 cutout)
        """
    
        x_center=find_centroid_of_flux(res_iapetus)[0]  
        left_limit=np.round(x_center-3.5)+0.5
        right_limit=np.round(x_center+3.5)-0.5                
    
    
        #sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        #var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
        multiplicative_factor_to_renormalize_to_50000=np.max(sci_image)/50000
        sci_image_smaller=sci_image[:,left_limit:right_limit]/multiplicative_factor_to_renormalize_to_50000
        var_image_smaller=var_image[:,left_limit:right_limit]/multiplicative_factor_to_renormalize_to_50000
        residual_initial_smaller=sci_image_smaller-res_iapetus[:,left_limit:right_limit]/multiplicative_factor_to_renormalize_to_50000
        #residual_RF_smaller=chi_RF_corrected_image[:,8:14]*np.sqrt(var_image_smaller)
    
        #################################
        # step 5 from Horne, very simplified
        inputimage_smaller=sci_image_smaller
        Px=np.sum(inputimage_smaller,axis=0)/np.sum(inputimage_smaller)
        var_inputimage_smaller=var_image_smaller
        #################################
        # Equation 8 from Horne with modification from Robert abut variance for extraction of signal
        # note that this uses profile from full thing, and not "residual profile"
    
        # nominator
        weighted_inputimage_smaller=inputimage_smaller*Px/(1)
        # denominator
        weights_array=np.ones((inputimage_smaller.shape[0],inputimage_smaller.shape[1]))*Px**2
    
        init_lamda=np.array(list(map(np.sum, weighted_inputimage_smaller)))/(np.array(list(map(np.sum,weights_array))))
        init_lamda_boxcar=np.array(list(map(np.sum, inputimage_smaller)))
        # Equation 8.5 from Horne
        var_f_std_lamda=1/np.sum(np.array(Px**2/(var_inputimage_smaller)),axis=1)
        std_init_lamda=np.sqrt(var_f_std_lamda)
        std_init_lamda_boxcar=np.sqrt(np.array(list(map(np.sum, var_inputimage_smaller))))
    
    
        #################################
        # Equation 8 from Horne with modification from Robert abut variance for initial removal
        # note that this uses profile from full thing, and not "residual profile"
    
        # nominator
        weighted_inputimage_smaller=residual_initial_smaller*Px/(1)
        # denominator
        weights_array=np.ones((residual_initial_smaller.shape[0],residual_initial_smaller.shape[1]))*Px**2
    
        init_removal_lamda=np.array(list(map(np.sum, weighted_inputimage_smaller)))/(np.array(list(map(np.sum,weights_array))))
        init_removal_lamda_boxcar=np.array(list(map(np.sum, residual_initial_smaller)))
        # Equation 8.5 from Horne
        var_init_removal_lamda=1/np.sum(np.array(Px**2/(var_inputimage_smaller)),axis=1)
        std_init_removal_lamda=np.sqrt(var_init_removal_lamda)
        return init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda
        

class Zernike_result_analysis(object):
    

    def __init__(self, date,single_number,arc,zmax,dataset,verbosity=None):
    
        
        if verbosity is None:
            verbosity=0
            
        self.verbosity=verbosity
        
        columns11=['z4','z5','z6','z7','z8','z9','z10','z11',
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_radius_illumination',
              'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
    
        columns11_analysis=columns11+['chi2','chi2max']
    
        columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
               'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_radius_illumination',
              'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
    
        columns22_analysis=columns22+['chi2','chi2max']
        
        self.columns22=columns22
        self.columns22_analysis=columns22_analysis
        self.columns=columns11
        self.columns11=columns11
        self.columns11_analysis=columns11_analysis
        
        

        if dataset==0:
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/Data_Nov_14/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=8603
                elif arc=="Ne":
                    single_number_focus=8693  

        if dataset==1:   
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Feb_5/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=11748
                elif arc=="Ne":
                    single_number_focus=11748+607  
 
        if dataset==2:
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=17017+54
                if arc=="Ne":
                    single_number_focus=16292  
                if arc=="Kr":
                    single_number_focus=17310+54  
                
        if dataset==3:  
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Jun_25/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=19238+54
                elif arc=="Ne":
                    single_number_focus=19472  
                    
        if dataset==4 or dataset==5:  
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Aug_14/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=21346+54
                elif arc=="Ne":
                    single_number_focus=21550+54  
                if str(arc)=="Kr":
                    single_number_focus=21754+54                     

        self.STAMPS_FOLDER=STAMPS_FOLDER
        
        columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent',
                      'x_ilum','y_ilum',
                      'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']    
        
        self.columns=columns
        
        RESULT_FOLDER='/Users/nevencaplar/Documents/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
        if os.path.exists(RESULT_FOLDER):
            pass
        else:
            RESULT_FOLDER='/Volumes/My Passport for Mac/Old_Files/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
        
        self.RESULT_FOLDER=RESULT_FOLDER
        
        IMAGES_FOLDER='/Users/nevencaplar/Documents/PFS/Images/'+date+'/'
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        self.IMAGES_FOLDER=IMAGES_FOLDER

        self.date=date
        self.single_number=single_number
        self.eps=5
        self.arc=arc
        self.zmax=zmax
        self.dataset=dataset
        
        method='P'
        self.method=method
    


    def create_results_of_fit_single(self):
        """create solution from a single image, create error and lower and upper erros
    
        @param[in] date             date when the analysis was conducted
        @param[in] single_number    number of the image
        @param[in] arc              which arc was analyzed
        @returns       results_of_fit_single,err_results_of_fit_single,err_results_of_fit_single_lower,err_results_of_fit_single_upper to be fed to solution_at_0_and_plots
        """
       
        if self.verbosity==1:
            print('supplied zMax is '+str(self.zmax))
            print(str(self.STAMPS_FOLDER))
        
        zmax=self.zmax
        date=self.date
        arc=self.arc
        dataset=self.dataset
        single_number=self.single_number
        STAMPS_FOLDER=self.STAMPS_FOLDER
        
                
        columns22=self.columns22
        columns22_analysis=self.columns22_analysis
        columns=self.columns11
        columns11=self.columns11
        columns11_analysis=self.columns11_analysis
            
        if dataset==0:
            if arc=='HgAr':
                obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
            elif arc=='Ne':
                obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])+90
    
        # F/3.2 data
        if dataset==1:
            if arc=='HgAr':
                obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11748,11694,11700,11706,11712,11718,11724,11730,11736])
            elif arc=='Ne':
                # different sequence than for HgAr
                obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12355,12349,12343,12337,12331,12325,12319,12313,12307])    
    
        # F/2.8 data
        if dataset==2:
            if arc=='HgAr':
                obs_possibilites=np.array([17023,17023+6,17023+12,17023+18,17023+24,17023+30,17023+36,17023+42,17023+48,17023+48,\
                                           17023+54,17023+60,17023+66,17023+72,17023+78,17023+84,17023+90,17023+96,17023+48])
            elif arc=='Ne':
                # different sequence than for HgAr
                obs_possibilites=np.array([16238+6,16238+12,16238+18,16238+24,16238+30,16238+36,16238+42,16238+48,16238+54,16238+54,\
                                           16238+60,16238+66,16238+72,16238+78,16238+84,16238+90,16238+96,16238+102,16238+54])
            elif arc=='Kr':
                 obs_possibilites=np.array([17310+6,17310+12,17310+18,17310+24,17310+30,17310+36,17310+42,17310+48,17310+54,17310+54,\
                                            17310+60,17310+66,17310+72,17310+78,17310+84,17310+90,17310+96,17310+102,17310+54])
        
        # F/2.5 data  
        if dataset==3:
            if arc=='HgAr':
                obs_possibilites=np.array([19238+6,19238+12,19238+18,19238+24,19238+30,19238+36,19238+42,19238+48,19238+54,19238+54,\
                                           19238+60,19238+66,19238+72,19238+78,19238+84,19238+90,19238+96,19238+102,19238+54])
            elif arc=='Ne':
            # different sequence than for HgAr
                obs_possibilites=np.array([19472+6,19472+12,19472+18,19472+24,19472+30,19472+36,19472+42,19472+48,19472+54,19472+54,\
                                           19472+60,19472+66,19472+72,19472+78,19472+84,19472+90,19472+96,19472+102,19472+54])    
        # F/2.8 data - July data
        if dataset==4:
            if arc=='HgAr':
                obs_possibilites=np.array([21346+6,21346+12,21346+18,21346+24,21346+30,21346+36,21346+42,21346+48,21346+54,21346+54,\
                                           21346+60,21346+66,21346+72,21346+78,21346+84,21346+90,21346+96,21346+102,21346+48])
            if arc=='Ne':
                obs_possibilites=np.array([21550+6,21550+12,21550+18,21550+24,21550+30,21550+36,21550+42,21550+48,21550+54,21550+54,\
                                           21550+60,21550+66,21550+72,21550+78,21550+84,21550+90,21550+96,21550+102,21550+54])
            if arc=='Kr':
                 obs_possibilites=np.array([21754+6,21754+12,21754+18,21754+24,21754+30,21754+36,21754+42,21754+48,21754+54,21754+54,\
                                            21754+60,21754+66,21754+72,21754+78,21754+84,21754+90,21754+96,21754+102,21754+54])
                      
        # F/2.8 data - fine defocus
        if dataset==5:
            if arc=='HgAr':
                obs_possibilites=np.arange(21280,21280+11*6,6)
            if arc=='Ne':
                obs_possibilites=np.arange(21484,21484+11*6,6)
            if arc=='Kr':
                 obs_possibilites=np.arange(21688,21688+11*6,6)
   
    
        
        if zmax==22:
            columns_analysis=columns22_analysis
        else:
            columns_analysis=columns11_analysis
        
        if dataset in [0,1,2,3,4]:
            results_of_fit_single=pd.DataFrame(np.zeros((18,len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                               index=['-4.0','-3.5','-3.0','-2.5','-2','-1.5','-1','-0.5','0','0','0.5','1','1.5','2','2.5','3.0','3.5','4'],\
                                               columns=columns_analysis)
            err_results_of_fit_single=pd.DataFrame(np.zeros((18,len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                   index=['-4.0','-3.5','-3.0','-2.5','-2','-1.5','-1','-0.5','0','0','0.5','1','1.5','2','2.5','3.0','3.5','4'],\
                                                   columns=columns_analysis)
            err_results_of_fit_single_lower=pd.DataFrame(np.zeros((18,len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                         index=['-4.0','-3.5','-3.0','-2.5','-2','-1.5','-1','-0.5','0','0','0.5','1','1.5','2','2.5','3.0','3.5','4'],\
                                                         columns=columns_analysis)
            err_results_of_fit_single_upper=pd.DataFrame(np.zeros((18,len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                         index=['-4.0','-3.5','-3.0','-2.5','-2','-1.5','-1','-0.5','0','0','0.5','1','1.5','2','2.5','3.0','3.5','4'],\
                                                         columns=columns_analysis)
        if dataset in [5]:
            fine_defocus_values=['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4','0.5']
            
            results_of_fit_single=pd.DataFrame(np.zeros((len(fine_defocus_values),len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                               index=fine_defocus_values,\
                                               columns=columns_analysis)
            err_results_of_fit_single=pd.DataFrame(np.zeros((len(fine_defocus_values),len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                   index=fine_defocus_values,\
                                                   columns=columns_analysis)
            err_results_of_fit_single_lower=pd.DataFrame(np.zeros((len(fine_defocus_values),len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                         index=fine_defocus_values,\
                                                         columns=columns_analysis)
            err_results_of_fit_single_upper=pd.DataFrame(np.zeros((len(fine_defocus_values),len(columns_analysis))).reshape(-1,len(columns_analysis)),\
                                                         index=fine_defocus_values,\
                                                         columns=columns_analysis)
            
            
     
        # arrange all results in one pandas dataframe
        RESULT_FOLDER='/Users/nevencaplar/Documents/PFS/TigerAnalysis/ResultsFromTiger/'+str(date)+'/'
    
        single_defocus_list=obs_possibilites
    
        image_index=single_number
        method='P'
        eps=5
    
        res_likelihood=[]
    
        for single_defocus in range(0,len(single_defocus_list)):
            try:
    
                index=['-4.0','-3.5','-3.0','-2.5','-2','-1.5','-1','-0.5','0','0','0.5','1','1.5','2','2.5','3.0','3.5','4']
                obs=single_defocus_list[single_defocus]
                index_name=index[single_defocus]
                """
                try:
                    chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                    likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                    print(str(single_number)+' obs (Emcee3, defocus): '+str(obs)+' is found!')
                except:    
                    chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee2.npy')
                    likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee2.npy')
                    print(str(single_number)+' obs (Emcee2, defocus): '+str(obs)+' is found!')
                """                    
                chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                print(str(single_number)+' obs (Emcee3, defocus): '+str(obs)+' ('+str(index_name)+') is found!')             
    
                if obs==8600:
                    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
                    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
                else:     
    
                    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
                    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
                    #sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
                    #var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
                #print("test0")
     
                likechain0=likechain
                chain0=chain
                
                minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]
                chi2reduced=2*np.min(np.abs(likechain0))/(sci_image.shape[0])**2

                minchain_err_old=self.create_minchain_err(chain0,likechain0,sci_image,var_image,old=1)
                #print(minchain_err_old)
                #print(" test1")
                minchain_err=self.create_minchain_err(chain0,likechain0,sci_image,var_image)
                #print(" test2")
                results_of_fit_single.iloc[single_defocus]=np.concatenate((minchain,np.array([chi2reduced,np.mean(sci_image**2/var_image)])),axis=0)
                #print(" test3")
                #print(results_of_fit_single.iloc[single_defocus])
                err_results_of_fit_single.iloc[single_defocus]=np.concatenate((minchain_err_old,np.array([1,1])),axis=0)
                #print(" test4")
                err_results_of_fit_single_lower.iloc[single_defocus]=np.concatenate((minchain_err[:,0],np.array([1,1])),axis=0)
                #print(" test5")
                err_results_of_fit_single_upper.iloc[single_defocus]=np.concatenate((minchain_err[:,1],np.array([1,1])),axis=0)
                #print(" test6")
            
            except:
                ValueError
                if self.verbosity==1:
                    print(str(single_number)+'obs '+str(obs)+' '+str(arc)+' is NOT found or failed!')
        #results_of_fit_single=results_of_fit_single[np.abs(results_of_fit_single['z4'])>0]
        #err_results_of_fit_single=err_results_of_fit_single[np.abs(err_results_of_fit_single['z4'])>0]
        #err_results_of_fit_single_lower=err_results_of_fit_single_lower[np.abs(err_results_of_fit_single_lower['z4'])>0]
        #err_results_of_fit_single_upper=err_results_of_fit_single_upper[np.abs(err_results_of_fit_single_upper['z4'])>0]              
        
        return results_of_fit_single,err_results_of_fit_single,err_results_of_fit_single_lower,err_results_of_fit_single_upper


    def create_results_of_fit_single_focus(self):
        """create solution from a single image, create error and lower and upper erros
    
        @param[in] date             date when the analysis was conducted
        @param[in] single_number    number of the image
        @param[in] arc              which arc was analyzed
        @param[in] zMax             biggest Zernike polynomial analysed
        @param[in] dataset          which dataset
        
        @returns       results_of_fit_single,err_results_of_fit_single,err_results_of_fit_single_lower,err_results_of_fit_single_upper to be fed to solution_at_0_and_plots
        """
        
        zmax=self.zmax
        date=self.date
        arc=self.arc
        dataset=self.dataset
        single_number=self.single_number
        STAMPS_FOLDER=self.STAMPS_FOLDER
        
                
        columns22=self.columns22
        columns22_analysis=self.columns22_analysis
        columns=self.columns11
        columns11=self.columns11
        columns11_analysis=self.columns11_analysis
        
            
        if arc=='HgAr':
            obs_possibilites=np.array([11748])
            labels=['11748']
        elif arc=='Ne':
            obs_possibilites=np.array([12355])
            labels=['12355']
            
        if dataset==2:
            if arc=='HgAr':
                obs_possibilites=np.array([17017+54])
                labels=['17071']
            elif arc=='Ne':
                obs_possibilites=np.array([16292])
                labels=['16292']
            elif arc=='Kr':
                obs_possibilites=np.array([17364])
                labels=['17364']
                
                
        if dataset==3:
            if arc=='HgAr':
                obs_possibilites=np.array([19238+54])
                labels=['19292']
            elif arc=='Ne':
                obs_possibilites=np.array([19472+54 ])
                labels=['19526']
                
            
        # dataset=4
        if dataset==4:   
            if str(arc)=="HgAr":
                obs_possibilites=np.array([21346+54 ])
                labels=['21400']
            elif str(arc)=="Ne":
                obs_possibilites=np.array([21550+54  ])
                labels=['21604']
            elif str(arc)=="Kr":
                obs_possibilites=np.array([21754+54 ])
                labels=['21808']
            else:
                print("Not recognized arc-line")                
                
        if zmax==22:
            columns_analysis=columns22_analysis
        else:
            columns_analysis=columns11_analysis        
            
    
        results_of_fit_single=pd.DataFrame(np.zeros((len(labels),len(columns_analysis))).reshape(-1,len(columns_analysis)),index=labels,columns=columns_analysis)
        err_results_of_fit_single=pd.DataFrame(np.zeros((len(labels),len(columns_analysis))).reshape(-1,len(columns_analysis)),index=labels,columns=columns_analysis)
        err_results_of_fit_single_lower=pd.DataFrame(np.zeros((len(labels),len(columns_analysis))).reshape(-1,len(columns_analysis)),index=labels,columns=columns_analysis)
        err_results_of_fit_single_upper=pd.DataFrame(np.zeros((len(labels),len(columns_analysis))).reshape(-1,len(columns_analysis)),index=labels,columns=columns_analysis)
    

        RESULT_FOLDER='/Users/nevencaplar/Documents/PFS/TigerAnalysis/ResultsFromTiger/'+str(date)+'/'
    
        single_defocus_list=obs_possibilites
    
        image_index=single_number
        method='P'
        eps=5
    
        res_likelihood=[]
    
        for single_defocus in range(0,len(single_defocus_list)):
            try:
                obs=single_defocus_list[single_defocus]
                """
                try:
                    chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                    likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                    print(str(single_number)+' obs (Emcee3, focus): '+str(obs)+' is found!')
                except:    
                    chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee2.npy')
                    likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee2.npy')
                    print(str(single_number)+'obs (Emcee2, focus): '+str(obs)+' is found!')
                """
                chain=np.load(RESULT_FOLDER+'chain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                likechain=np.load(RESULT_FOLDER+'likechain'+str(date)+'_Single_'+str(method)+'_'+str(obs)+str(single_number)+str(eps)+str(arc)+'Emcee3.npy')
                print(str(single_number)+' obs (Emcee3, focus): '+str(obs)+' is found!')

    
                if obs==8600:
                    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
                    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
                else:       
                    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
                    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
                    #sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
                    #var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
    
     
                likechain0=likechain
    
                chain0=chain
                minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]
                chi2reduced=2*np.min(np.abs(likechain0))/(sci_image.shape[0])**2
                
                chi2reduced=(2*np.min(np.abs(likechain0))-np.sum(np.log(2*np.pi*var_image)))/(sci_image.shape[0])**2
                
    
                """
                minchain_err=[]
                for i in range(len(columns)):
                    #minchain_err=np.append(minchain_err,np.std(chain0[:,:,i].flatten()))
                    minchain_err=np.append(minchain_err,np.sqrt(chi2reduced)*np.std(chain0[:,:,i].flatten()))
    
                minchain_err=np.array(minchain_err)
                """
                minchain_err_old=self.create_minchain_err(chain0,likechain0,sci_image,var_image,old=1)
                #print(minchain_err_old)
               
                minchain_err=self.create_minchain_err(chain0,likechain0,sci_image,var_image)
                #print(" test2")
                results_of_fit_single.iloc[single_defocus]=np.concatenate((minchain,np.array([chi2reduced,np.mean(sci_image**2/var_image)])),axis=0)
                #print(" test3")
                #print(results_of_fit_single.iloc[single_defocus])
                err_results_of_fit_single.iloc[single_defocus]=np.concatenate((minchain_err_old,np.array([1,1])),axis=0)
                #print(" test4")
                err_results_of_fit_single_lower.iloc[single_defocus]=np.concatenate((minchain_err[:,0],np.array([1,1])),axis=0)
                #print(" test5")
                err_results_of_fit_single_upper.iloc[single_defocus]=np.concatenate((minchain_err[:,1],np.array([1,1])),axis=0)
                #print(" test6")
            
            except:
                ValueError
                if self.verbosity==1:
                    print(str(single_number)+' obs '+str(obs)+' '+str(arc)+' is NOT found or failed!')
        #results_of_fit_single=results_of_fit_single[np.abs(results_of_fit_single['z4'])>0]
        #err_results_of_fit_single=err_results_of_fit_single[np.abs(err_results_of_fit_single['z4'])>0]
        #err_results_of_fit_single_lower=err_results_of_fit_single_lower[np.abs(err_results_of_fit_single_lower['z4'])>0]
        #err_results_of_fit_single_upper=err_results_of_fit_single_upper[np.abs(err_results_of_fit_single_upper['z4'])>0]              
        
        return results_of_fit_single,err_results_of_fit_single,err_results_of_fit_single_lower,err_results_of_fit_single_upper
    


    def create_results_of_fit_single_fine_defocus(self):
                    
        zmax=self.zmax
        date=self.date
        arc=self.arc
        dataset=self.dataset
        single_number=self.single_number
        STAMPS_FOLDER=self.STAMPS_FOLDER
        
        method='P'
        eps=5
        
        columns22=self.columns22
        columns22_analysis=self.columns22_analysis
        columns=self.columns11
        columns11=self.columns11
        columns11_analysis=self.columns11_analysis
            
        # F/2.8 data - fine defocus
        if dataset==5:
            if arc=='HgAr':
                obs_possibilites=np.arange(21280,21280+11*6,6)
            if arc=='Ne':
                obs_possibilites=np.arange(21484,21484+11*6,6)
            if arc=='Kr':
                 obs_possibilites=np.arange(21688,21688+11*6,6)
                    
        if zmax==22:
            columns_analysis=columns22_analysis
        else:
            columns_analysis=columns11_analysis   
            
        list_of_chain0_stack=[]   
        list_of_likechain0_stack=[]
        list_of_i=[]
        for i in range(11):        
            obs=obs_possibilites[i]
            single_analysis=Zernike_Analysis(date,obs,single_number,eps,arc,dataset)
            try:
                chain0_stack,likechain0_stack=single_analysis.create_Emcee2_stack()
                list_of_chain0_stack.append(chain0_stack)
                list_of_likechain0_stack.append(likechain0_stack)
                list_of_i.append(i)
            except:
                pass
            
            
        chain0_superstack=np.concatenate((list_of_chain0_stack))
        likechain0_superstack=np.concatenate((list_of_likechain0_stack))
         
        return np.median(chain0_superstack,axis=0)


    def solution_at_0_and_plots(self,results_of_fit_single,err_results_of_fit_single,err_results_of_fit_single_lower,err_results_of_fit_single_upper,\
                                plot=True,return_solution_at_05_0_05=None):
        """create solution at the focus and plot dependence with defocus
    
        @param[in] date             date when the analysis was conducted
        @param[in] single_number    number of the image
        @param[in] arc              which arc was analyzed
      
        """    
        
        date=self.date
        single_number=self.single_number
        arc=self.arc
        zMax=self.zmax
        
        columns22=self.columns22
        columns22_analysis=self.columns22_analysis
        columns=self.columns11
        columns11=self.columns11
        columns11_analysis=self.columns11_analysis
        
        
        if zMax==22:
            columns_analysis=columns22_analysis
            columns=columns22
            z_addition_factor=11
        else:
            columns_analysis=columns11_analysis
            columns=columns11
            z_addition_factor=0
        
        
        results_of_fit_single=results_of_fit_single[np.abs(results_of_fit_single['z4'])>0]
        len_of_negative_defocus=len(results_of_fit_single[np.abs(results_of_fit_single['z4'])<-1.5])
        len_of_positive_defocus=len(results_of_fit_single[np.abs(results_of_fit_single['z4'])>1.5])
        
        err_results_of_fit_single=err_results_of_fit_single[np.abs(err_results_of_fit_single['z4'])>0]
        err_results_of_fit_single_lower=err_results_of_fit_single_lower[np.abs(err_results_of_fit_single_lower['z4'])>0]
        err_results_of_fit_single_upper=err_results_of_fit_single_upper[np.abs(err_results_of_fit_single_upper['z4'])>0]    

        # in the case when results_of_fit_single consists of 18 lines (from -4 to +4 in steps of 0.5, with two values for focus) this takes values near focus
        index_arr=results_of_fit_single['z4'].index.values.astype(float)
        results_of_fit_single_near_focus=results_of_fit_single[np.abs(index_arr)<=0.5]    
        results_of_fit_single_near_focus=results_of_fit_single_near_focus[np.abs(results_of_fit_single_near_focus['z4'])>0]


        # Do not analyze if :
        # 1.if you have less than 7 observations
        # 2. if you have no observations near focus
        # 3. if you do not have at least 2 observations from each side of defocus
        # else return zeroes
        if len(results_of_fit_single)<=7 or len(results_of_fit_single_near_focus)==0 or len_of_negative_defocus<=2 or len_of_positive_defocus<=2:
            if zMax==22:
                solution_at_0=np.full(31+11,0)
                solution_at_05_0_05=[]
                for i in range(11):
                    solution_at_05_0_05.append(list(solution_at_0))
    
                solution_at_05_0_05=np.array(solution_at_05_0_05)
            else:
                solution_at_0=np.full(31,0)    
                solution_at_05_0_05=[]
                for i in range(11):
                    solution_at_05_0_05.append(list(solution_at_0))
    
                solution_at_05_0_05=np.array(solution_at_05_0_05)

            if return_solution_at_05_0_05 is None:
                return solution_at_0
            else:
                return solution_at_0,solution_at_05_0_05

        else:
            IMAGES_FOLDER='/Users/nevencaplar/Documents/PFS/Images/'+str(date)+'/'
    
            solution_at_0=[]
            solution_at_05_0_05=[]
    
    
            for q in columns_analysis:
    
                z4_arr=np.array(results_of_fit_single[q])
                z4_arr_err=np.array(err_results_of_fit_single[q])
                z4_arr_err_up=np.array(err_results_of_fit_single_upper[q])
                z4_arr_err_low=np.array(err_results_of_fit_single_lower[q])
                index_arr=results_of_fit_single[q].index.values.astype(float)
                #print(q)
                #print(index_arr)
                #print(z4_arr)
                #print(z4_arr_err)
                #print(z4_arr_err_up)
                #print(z4_arr_err_low)
                
                z4_arr_no0=z4_arr[np.abs(index_arr)>0.5]
                z4_arr_no0_err=z4_arr_err[np.abs(index_arr)>0.5]
                z4_arr_no0_err_up=z4_arr_err_up[np.abs(index_arr)>0.5]
                z4_arr_no0_err_low=z4_arr_err_low[np.abs(index_arr)>0.5]
                index_arr_no0=index_arr[np.abs(index_arr)>0.5]
                #print(q)
                #print(index_arr_no0)
                #print(z4_arr_no0)
                #print(z4_arr_no0_err)
                #print(z4_arr_no0_err_up)
                #print(z4_arr_no0_err_low)
                
                
                z4_arr_only0=z4_arr[np.abs(index_arr)<=0.5]
                z4_arr_only0_err=z4_arr_err[np.abs(index_arr)<=0.5]
                z4_arr_only0_err_up=z4_arr_err_up[np.abs(index_arr)<=0.5] 
                z4_arr_only0_err_low=z4_arr_err_low[np.abs(index_arr)<=0.5]
                index_arr_only0=index_arr[np.abs(index_arr)<=0.5]
    
    
    
                fit_res=[]
                fit_res_focus=[]
                interim_zero_solutions=[]
    
                
                if q in columns[:8+z_addition_factor]:
                    # these variables are fit via linear fit, without values at focus
                    # it is z4-z11
                    # for loop below removes 2 points from the fit that create largest deviations from median result in focus
                    for l in range(len(index_arr_no0)):
                        #print('good_index'+str(np.delete(index_arr_no0,l)))
                        #print('index_arr_no0'+str(np.delete(z4_arr_no0,l)))
                        #print('z4_arr_no0_err_low'+str(np.delete(z4_arr_no0_err_low,l)))
                        #print('z4_arr_no0_err_up'+str(np.delete(z4_arr_no0_err_up,l)))
                        popt=optimize.leastsq(curve_fit_custom_lin,x0=[1,1],args=(np.delete(index_arr_no0,l),np.delete(z4_arr_no0,l), np.delete(z4_arr_no0_err_low,l),np.delete(z4_arr_no0_err_up,l)))[0]
                        interim_zero_solutions.append([l,lin_fit_1D(0,popt[0],popt[1])])
                    interim_zero_solutions_arr=np.array(interim_zero_solutions)
                    interim_zero_solutions_arr_dif=np.abs(interim_zero_solutions_arr[:,1]-np.median(interim_zero_solutions_arr[:,1]))
                    second_max_dif=np.sort(interim_zero_solutions_arr_dif)[-2]
                    interim_zero_solutions_arr=interim_zero_solutions_arr[interim_zero_solutions_arr_dif<second_max_dif]
                    #print(interim_zero_solutions_arr[:,0])
                    good_index=interim_zero_solutions_arr[:,0].astype(int)
                    #print('good_index'+str(good_index))
                    #print('index_arr_no0'+str(index_arr_no0))
                    #print('z4_arr_no0_err_low'+str(z4_arr_no0_err_low))
                    #print('z4_arr_no0_err_up'+str(z4_arr_no0_err_up))
                    popt=optimize.leastsq(curve_fit_custom_lin,x0=[1,1],args=(np.array(index_arr_no0)[good_index],np.array(z4_arr_no0)[good_index],np.array(z4_arr_no0_err_low)[good_index],np.array(z4_arr_no0_err_up)[good_index]))[0]
    
                    for i in np.linspace(-4.5,4.5,19):
                        fit_res.append(lin_fit_1D(i,popt[0],popt[1]))
                    for j in np.linspace(-0.5,0.5,11):
                        fit_res_focus.append(lin_fit_1D(j,popt[0],popt[1]))
                        
                    solution_at_0.append(fit_res[9])
                    solution_at_05_0_05.append(fit_res_focus)
    
                interim_zero_solutions=[]
                if q in columns[8+z_addition_factor:25+z_addition_factor]:
                    # these variables are set at mean value (constant fit), without values at focus
                    # these are variables that describe the pupil
                    #print("###########")
                    #print('q: '+str(q))
                    #print('index_arr_no0: '+str(index_arr_no0))
                    #print('z4_arr_no0: '+str(z4_arr_no0))
                    #print('z4_arr_no0_err_low: '+str(z4_arr_no0_err_low))
                    #print('z4_arr_no0_err_up: '+str(z4_arr_no0_err_up))
                    
                    # if all the values are the same, i.e., you did nove this value around in the fit
                    # if not, go through fitting routine
                    #print('len(np.unique(z4_arr_no0))'+str(len(np.unique(z4_arr_no0))))
                    #print('np.unique(z4_arr_no0)'+str(np.unique(z4_arr_no0)))
                    if len(np.unique(z4_arr_no0))==1:
                        fit_res=lin_fit_1DConstant(np.linspace(-4.5,4.5,19),np.unique(z4_arr_no0)[0])
                        fit_res_focus=lin_fit_1DConstant(np.linspace(-0.5,0.5,11),np.unique(z4_arr_no0)[0])                        
                    else:                            
                        for l in range(len(index_arr_no0)):
                            #print('l: '+str(l))
                            popt=optimize.leastsq(curve_fit_custom_con,x0=[1],args=(np.delete(index_arr_no0,l),np.delete(z4_arr_no0,l), np.delete(z4_arr_no0_err_low,l),np.delete(z4_arr_no0_err_up,l)))[0]
                            #print('popt:'+str(popt))
                            interim_zero_solutions.append([l,popt[0]])
                        interim_zero_solutions_arr=np.array(interim_zero_solutions)
                        interim_zero_solutions_arr_dif=np.abs(interim_zero_solutions_arr[:,1]-np.median(interim_zero_solutions_arr[:,1]))
                        third_max_dif=np.sort(interim_zero_solutions_arr_dif)[-3]
                        #print('interim_zero_solutions_arr_dif'+str(interim_zero_solutions_arr_dif))
                        #print('third_max_dif'+str(third_max_dif))
                        interim_zero_solutions_arr=interim_zero_solutions_arr[interim_zero_solutions_arr_dif<=(third_max_dif*1.1)]
                        #print(str(q)+str(interim_zero_solutions_arr[:,0]))
                        good_index=interim_zero_solutions_arr[:,0].astype(int)
                        popt=optimize.leastsq(curve_fit_custom_con,x0=[1],args=(np.array(index_arr_no0)[good_index],np.array(z4_arr_no0)[good_index],np.array(z4_arr_no0_err_low)[good_index],np.array(z4_arr_no0_err_up)[good_index]))[0]
                        
                        fit_res=lin_fit_1DConstant(np.linspace(-4.5,4.5,19),popt[0])
                        fit_res_focus=lin_fit_1DConstant(np.linspace(-0.5,0.5,11),popt[0])
                
                    solution_at_0.append(fit_res[9])
                    solution_at_05_0_05.append(fit_res_focus)
    
                if q in np.concatenate((np.array(columns[25+z_addition_factor:]),np.array(['chi2','chi2max'])),axis=0):
                    # these variables are set at value as measured at 0 - (perhaps is should be close to 0)
                    if z4_arr_only0.size==1:
                        for i in np.linspace(-4.5,4.5,19):
                            fit_res.append(z4_arr_only0) 
                        
                        for j in np.linspace(-0.5,0.5,11):
                            fit_res_focus.append(z4_arr_only0)
                        
                        solution_at_0.append(fit_res[9])
                        solution_at_05_0_05.append(fit_res_focus)
                    else:
                        # these variables are set at mean value (constant fit), without values at focus
                        popt=optimize.leastsq(curve_fit_custom_con,x0=[1],args=(index_arr_only0, z4_arr_only0, z4_arr_only0_err_low,z4_arr_only0_err_up))[0]
                        fit_res=lin_fit_1DConstant(np.linspace(-4.5,4.5,19),popt[0])
                        fit_res_focus=lin_fit_1DConstant(np.linspace(-0.5,0.5,11),popt[0]) 
                        
                        solution_at_0.append(fit_res[9])
                        solution_at_05_0_05.append(fit_res_focus)
    
                #making plots here
                ######
                if plot==True:
                    plt.figure(figsize=(20,10))
                    plt.errorbar(index_arr,z4_arr,yerr=[np.abs(z4_arr_err_low),z4_arr_err_up],color='blue',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='single fit results')
                    plt.plot(np.linspace(-4.5,4.5,19),fit_res,color='orange',label='fit')
                    if q in columns[:8]:
                        plt.plot(np.linspace(-4.5,4.5,19),np.zeros((19,1)),':',color='black')
                    plt.plot([0], [fit_res[9]], marker='o', markersize=10, color="red",label='prediction for focus')
                    #print('prediction for focus: '+str(q)+' '+str(fit_res[9]))
                    plt.title(q,size=40)
                    plt.legend(fontsize=25)
                    plt.xlabel('defocus lab [mm]',size=30)
                    plt.ylabel('defocus model',size=30)
                    #print(IMAGES_FOLDER+'Defocus/'+str(single_number)+'/')
                    if not os.path.exists(IMAGES_FOLDER+'Defocus/'+str(single_number)+'/'):
                        os.makedirs(IMAGES_FOLDER+'Defocus/'+str(single_number)+'/')
                    plt.savefig(IMAGES_FOLDER+'Defocus/'+str(single_number)+'/'+str(arc)+str(q))
        
                    if not os.path.exists(IMAGES_FOLDER+'Defocus/'+str(q)+'/'):
                        os.makedirs(IMAGES_FOLDER+'Defocus/'+str(q)+'/')
                    plt.savefig(IMAGES_FOLDER+'Defocus/'+str(q)+'/'+str(arc)+str(single_number))
        
                    plt.close()  
                else:
                    pass
                ######
    
            solution_at_0=np.array(solution_at_0)[:len(solution_at_0)-2]
            
            solution_at_05_0_05=np.transpose(np.array(solution_at_05_0_05[0:42]))
            if self.verbosity==1:
                print('shape of the solution_at_05_0_05 is: ' +str(solution_at_05_0_05.shape))
            if return_solution_at_05_0_05 is None:
                return solution_at_0
            else:
                return solution_at_0,solution_at_05_0_05
            

    def create_minchain_err(self,chain0,likechain0,sci_image,var_image,old=0):
        """create error on the parameters from the chains
        
        @param chain0      
        @param likechain0  
        @param sci_image  
        @param var_image  
        @param old 
        
        @returns        
        """
        columns22=self.columns22
        columns11=self.columns11

        minchain_err_test=[]
        if len(chain0[0][0])==42:
            columns=columns22          
        else:
            columns=columns11


        for var_number in range(len(columns)):
            #ravel likelihood
            likechain0_Emcee3_ravel=np.ravel(likechain0)
    
            # connect chain and lnchain
            chain0_Emcee3_ravel=np.ravel(chain0[:,:,var_number])
            chain0_Emcee3_ravel_argsort=np.argsort(chain0_Emcee3_ravel)  
            chain0_Emcee3_ravel_sort=chain0_Emcee3_ravel[chain0_Emcee3_ravel_argsort]
            likechain0_Emcee3_ravel_sort=likechain0_Emcee3_ravel[chain0_Emcee3_ravel_argsort]
    
            # move to chi2 space
            # you should take into account mask!
            chi2_Emcee3_ravel_sort=-(np.array(likechain0_Emcee3_ravel_sort)*(2)-np.log(2*np.pi*np.sum(var_image)))/(sci_image.shape[0])**2
            min_chi2_Emcee3_ravel_sort=np.min(chi2_Emcee3_ravel_sort)
    
            # simplest standard deviation
            std_chain=np.std(chain0_Emcee3_ravel_sort)
    
            #best solution
            mean_chain=chain0_Emcee3_ravel_sort[chi2_Emcee3_ravel_sort==np.min(chi2_Emcee3_ravel_sort)][0]
    
            # step size
            step=std_chain/10
            
            # if standard deviation is much much smaller than the mean value aborth the effort
            # and set the erros to be equal to the mean value
            # if the standard deviation is so small it means that there were problems in the fit
            # or you intentionally did not move the variable
            if std_chain<np.abs(mean_chain)/10**6:
                minchain_err_element=[-mean_chain,mean_chain]
            else:
                # create result, go 3*std in each direction
                try:
                    res=[]
                    for i in np.arange(mean_chain-30*step,mean_chain+30*step,step):
                        selected_chi2_Emcee3_ravel_sort=chi2_Emcee3_ravel_sort[(np.array(chain0_Emcee3_ravel_sort<i+step)&np.array(chain0_Emcee3_ravel_sort>i))]
                        if len(selected_chi2_Emcee3_ravel_sort>10):   
                            res.append([i+step/2,np.min(chi2_Emcee3_ravel_sort[(np.array(chain0_Emcee3_ravel_sort<i+step)&np.array(chain0_Emcee3_ravel_sort>i))])])
        
                    res=np.array(res)
        
                    #print(columns[var_number]+' min : '+str(mean_chain))
                    #print(columns[var_number]+' std : '+str(std_chain))
        
                    # find low limit and high limit
                    res_within2_chi=res[res[:,1]<min_chi2_Emcee3_ravel_sort*2]
                    minchain_err_element=[-np.abs(mean_chain-res_within2_chi[0,0]),np.abs(res_within2_chi[-1,0]-mean_chain)]
                except IndexError:   
                    if self.verbosity==1:
                        print(columns[var_number]+': failed!')
                    minchain_err_element=[-mean_chain,mean_chain]
    
            minchain_err_test.append(minchain_err_element)
            #print(columns[var_number]+' min_err : '+str(minchain_err_element[0]))
            #print(columns[var_number]+' max_err : '+str(minchain_err_element[1]))
        if old==1:
            minchain_err_test=np.mean(np.abs(np.array(minchain_err_test)),axis=1)
            #print(minchain_err_test)
            return minchain_err_test
        else:       
            return np.array(minchain_err_test)
         

            

 
def Ifun16Ne (lambdaV,lambda0,Ne):
    """Construct Lorentizan scattering
        @param lambdaV      
        @param lambda0  
        @param Ne           number of effective lines
        @returns              
    """
    
    
    return (lambda0/(Ne*np.pi*np.sqrt(2)))**2/((lambdaV-lambda0)**2+(lambda0/(Ne*np.pi*np.sqrt(2)))**2)


def create_mask(FFTTest_fiber_and_pixel_convolved_downsampled_40,semi=None):
    """!given the image, create a mask
        if semi is not specified, it gives the masks which span in all directions
        if semi is specified is masks towards only towards the region specified
        
    
    @param FFTTest_fiber_and_pixel_convolved_downsampled_40     science data stamp
    @param semi (+,-,l,r)                                       which region to uncover
     """    
    central_position=np.array(find_centroid_of_flux(FFTTest_fiber_and_pixel_convolved_downsampled_40))
    central_position_int=np.round(central_position)
    central_position_int_x=int(central_position_int[0])
    central_position_int_y=int(central_position_int[1])

    size=len(FFTTest_fiber_and_pixel_convolved_downsampled_40)
    
    center_square=np.zeros((size,size))
    if size==20:
        cutsize=3
        small_cutsize=2
    if size==40:
        cutsize=6
        small_cutsize=4
    
    center_square[central_position_int_y-cutsize:+central_position_int_y+cutsize,central_position_int_x-cutsize:central_position_int_x+cutsize]=np.ones((2*cutsize,2*cutsize))
    # need to add l and r here
    horizontal_cross=np.zeros((size,size))
    horizontal_cross[central_position_int_y-cutsize:central_position_int_y+cutsize,0:size,]=np.ones((2*cutsize,size))
    horizontal_cross_full=horizontal_cross
    horizontal_cross=horizontal_cross-center_square



    vertical_cross=np.zeros((size,size))
    if semi is None:
        vertical_cross[0:size,central_position_int_x-cutsize:central_position_int_x+cutsize]=np.ones((size,2*cutsize))
        vertical_cross=vertical_cross-center_square
    if semi=='+':
        vertical_cross[central_position_int_y+cutsize:size,central_position_int_x-cutsize:central_position_int_x+cutsize]=np.ones((size-central_position_int_y-cutsize,2*cutsize))
    if semi=='-':
        vertical_cross[0:central_position_int_y+1-cutsize,central_position_int_x-cutsize:central_position_int_x+cutsize]=np.ones((central_position_int_y+1-cutsize,2*cutsize))
    vertical_cross_full=vertical_cross


    diagonal_cross=np.zeros((size,size))
    if semi is None:
        #print(central_position_int_y)
        #print(central_position_int_x)
        diagonal_cross[0:central_position_int_y-small_cutsize,0:central_position_int_x-small_cutsize]=np.ones((central_position_int_y-small_cutsize,central_position_int_x-small_cutsize))
        diagonal_cross[(central_position_int_y+small_cutsize):size,0:(central_position_int_x-small_cutsize)]=np.ones((size-(central_position_int_y+small_cutsize),(central_position_int_x-small_cutsize)))
        diagonal_cross[0:(central_position_int_y-small_cutsize),(central_position_int_x+small_cutsize):size]=np.ones(((central_position_int_y-small_cutsize),size-(central_position_int_x+small_cutsize)))
        diagonal_cross[(central_position_int_y+small_cutsize):size,(central_position_int_x+small_cutsize):size]=np.ones((size-(central_position_int_y+small_cutsize),size-(central_position_int_x+small_cutsize)))
    if semi=='+':
        diagonal_cross[(central_position_int_y+small_cutsize):size,0:(central_position_int_x-small_cutsize)]=np.ones((size-(central_position_int_y+small_cutsize),(central_position_int_x-small_cutsize)))
        diagonal_cross[(central_position_int_y+small_cutsize):size,(central_position_int_x+small_cutsize):size]=np.ones((size-(central_position_int_y+small_cutsize),size-(central_position_int_x+small_cutsize)))
    if semi=='-':
        diagonal_cross[0:central_position_int_y-small_cutsize,0:central_position_int_x-small_cutsize]=np.ones((central_position_int_y-small_cutsize,central_position_int_x-small_cutsize))
        diagonal_cross[0:(central_position_int_y-small_cutsize),(central_position_int_x+small_cutsize):size]=np.ones(((central_position_int_y-small_cutsize),size-(central_position_int_x+small_cutsize)))
    if semi=='r':
        diagonal_cross[(central_position_int_y+small_cutsize):size,(central_position_int_x+small_cutsize):size]=np.ones((size-(central_position_int_y+small_cutsize),size-(central_position_int_x+small_cutsize)))
        diagonal_cross[0:(central_position_int_y-small_cutsize),(central_position_int_x+small_cutsize):size]=np.ones(((central_position_int_y-small_cutsize),size-(central_position_int_x+small_cutsize)))
    if semi=='l':
        diagonal_cross[0:central_position_int_y-small_cutsize,0:central_position_int_x-small_cutsize]=np.ones((central_position_int_y-small_cutsize,central_position_int_x-small_cutsize))
        diagonal_cross[(central_position_int_y+small_cutsize):size,0:(central_position_int_x-small_cutsize)]=np.ones((size-(central_position_int_y+small_cutsize),(central_position_int_x-small_cutsize)))

    total_mask=np.zeros((size,size))
    if semi is None:
        total_mask=np.ones((size,size))
    if semi=='+':
        total_mask[(central_position_int_y):size,0:size]=np.ones((size-(central_position_int_y),size))
    if semi=='-':
        total_mask[:(central_position_int_y),0:size]=np.ones(((central_position_int_y),size))
    if semi=='r':
        total_mask[:(central_position_int_y),0:size]=np.ones(((central_position_int_y),size))  
    if semi=='l':
        total_mask[:(central_position_int_y),0:size]=np.ones(((central_position_int_y),size))   
        
    return [center_square,horizontal_cross,vertical_cross,diagonal_cross,total_mask]


def create_res_data(FFTTest_fiber_and_pixel_convolved_downsampled_40,mask=None,custom_cent=None,size_pixel=None):
    """!given the small science image, create radial profile in microns
    
    @param FFTTest_fiber_and_pixel_convolved_downsampled_40     science data stamps
    @param mask                                                 mask to cover science data [default: None]
    @param custom_cent                                          if None create new center using the function ``find_centroid_of_flux'' [default:None]
                                                                otherwise pass a list with [x_center, y_center]
    @param size_pixel                                           pixel size in the image, in microns [default:1, dithered=7.5, normal operation=15]
     """     
     
    xs_uniform=find_centroid_of_flux(np.ones(FFTTest_fiber_and_pixel_convolved_downsampled_40.shape))[0] 
    ys_uniform=find_centroid_of_flux(np.ones(FFTTest_fiber_and_pixel_convolved_downsampled_40.shape))[1]
    
    print('uniform values are:' +str([xs_uniform,ys_uniform]))
    
    # changed in 0.25 
    if size_pixel is None:
        size_pixel=1
    
    image_shape=np.array(FFTTest_fiber_and_pixel_convolved_downsampled_40.shape)

    if custom_cent is None:
        xs0=(find_centroid_of_flux(FFTTest_fiber_and_pixel_convolved_downsampled_40)[0]-xs_uniform)*size_pixel
        ys0=(find_centroid_of_flux(FFTTest_fiber_and_pixel_convolved_downsampled_40)[1]-ys_uniform)*size_pixel
        print('center deduced  (in respect to the center of image) at:' +str([xs0,ys0]))
    else:
        xs0,ys0=custom_cent-xs_uniform
        print('center specified (in respect to the center of image) at:' +str([xs0,ys0]))
            
    pointsx = np.linspace(-(int(image_shape[0]*size_pixel)-size_pixel)/2,(int(image_shape[0]*size_pixel)-size_pixel)/2,num=int(image_shape[0]))
    pointsy = np.linspace(-(int(image_shape[0]*size_pixel)-size_pixel)/2,(int(image_shape[0]*size_pixel)-size_pixel)/2,num=int(image_shape[0]))
    xs, ys = np.meshgrid(pointsx, pointsy)
    r0 = np.sqrt((xs-xs0)** 2 + (ys-ys0)** 2)
    
    if mask is None:
        mask=np.ones((FFTTest_fiber_and_pixel_convolved_downsampled_40.shape[0],FFTTest_fiber_and_pixel_convolved_downsampled_40.shape[1]))
    
    distances=range(int(image_shape[0]/2*size_pixel*1.2))

    res_test_data=[]
    for r in tqdm(distances):
        pixels_upper_limit=(mask*FFTTest_fiber_and_pixel_convolved_downsampled_40)[r0<(r+size_pixel)]
        pixels_lower_limit=(mask*FFTTest_fiber_and_pixel_convolved_downsampled_40)[r0<(r)]
        
        mask_upper_limit=mask[r0<(r+size_pixel)]
        mask_lower_limit=mask[r0<(r)]
        
        number_of_valid_pixels=np.sum(mask_upper_limit)-np.sum(mask_lower_limit)
        
        if number_of_valid_pixels==0:
            res_test_data.append(0)
        else:                  
            average_flux=(np.sum(pixels_upper_limit)-np.sum(pixels_lower_limit))/number_of_valid_pixels
            res_test_data.append(average_flux)        

    return res_test_data 

def custom_fftconvolve(array1, array2):
    assert array1.shape==array2.shape
    return np.fft.fftshift(np.real(np.fft.irfft2(np.fft.rfft2(array1)*np.fft.rfft2(array2))))


# Taken from galsim.Aperature() code

    # Some quick notes for Josh:
    # - Relation between real-space grid with size theta and pitch dtheta (dimensions of angle)
    #   and corresponding (fast) Fourier grid with size 2*maxk and pitch stepk (dimensions of
    #   inverse angle):
    #     stepk = 2*pi/theta
    #     maxk = pi/dtheta
    # - Relation between aperture of size L and pitch dL (dimensions of length, not angle!) and
    #   (fast) Fourier grid:
    #     dL = stepk * lambda / (2 * pi)
    #     L = maxk * lambda / pi
    # - Implies relation between aperture grid and real-space grid:
    #     dL = lambda/theta
    #     L = lambda/dtheta
def stepK(pupil_plane_scale, lam, scale_unit=galsim.arcsec):
    """Return the Fourier grid spacing for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
    @returns           Fourier grid spacing.
    """
    return 2*np.pi*pupil_plane_scale/(lam*1e-9) * scale_unit/galsim.radians

def maxK(pupil_plane_size,lam, scale_unit=galsim.arcsec):
    """Return the Fourier grid half-size for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
    @returns           Fourier grid half-size.
    """
    return np.pi*pupil_plane_size/(lam*1e-9) * scale_unit/galsim.radians

def sky_scale( pupil_plane_size,lam, scale_unit=galsim.arcsec):
    """Return the image scale for this aperture at given wavelength.
    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image scale.
    """
    return (lam*1e-9) / pupil_plane_size * galsim.radians/scale_unit

def sky_size(pupil_plane_scale, lam, scale_unit=galsim.arcsec):
    """Return the image size for this aperture at given wavelength.
    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image size.
    """
    return (lam*1e-9) / pupil_plane_scale * galsim.radians/scale_unit

def value_at_defocus(mm,a,b=None):
    if b==None:
        return a
    else:
        return a*mm+b 
    
def create_x(mm,parameters):
    """
    transfrom multi analysis variable into single analysis variables
    only for z11
    
    @param mm     defocu at LAM
    @param parameters multi parameters
    
    """
    #for single case, up to z11
    
    zparameters=parameters[:16]
    globalparameters=parameters[16:]
    
    
    x=np.zeros((8+len(globalparameters)))
    for i in range(0,8,1):
        x[i]=value_at_defocus(mm,zparameters[i*2],zparameters[i*2+1])      

    for i in range(len(globalparameters)):
        x[int(len(zparameters)/2)+i]=globalparameters[i] 
    

    return x

def remove_pupil_parameters_from_all_parameters(parameters):
    lenpar=len(parameters)
    return np.concatenate((parameters[:lenpar-23],parameters[lenpar-17:lenpar-13],parameters[lenpar-7:]))

def add_pupil_parameters_to_all_parameters(parameters,pupil_parameters):
    lenpar=len(parameters)
    return np.concatenate((parameters[:lenpar-11],pupil_parameters[:6],parameters[lenpar-11:lenpar-7],pupil_parameters[6:],parameters[lenpar-7:]),axis=0)


    
    

def curve_fit_custom_lin(V,index_arr,z4_arr,z4_arr_err_low,z4_arr_err_up):
    a,b=V
    yfit=lin_fit_1D(index_arr,a,b)
    weight=np.ones_like(yfit)
    weight[yfit>z4_arr]=z4_arr_err_up[yfit>z4_arr] # if the fit point is above the measure, use upper weight
    weight[yfit<=z4_arr]=z4_arr_err_low[yfit<=z4_arr] # else use lower weight
    return (yfit-z4_arr)**2/weight**2

def curve_fit_custom_con(V,index_arr,z4_arr,z4_arr_err_low,z4_arr_err_up):
    a=V
    #print('a:'+str(a))
    yfit=lin_fit_1DConstant(index_arr,a)
    #print('yfit:'+str(yfit))
    weight=np.ones_like(yfit)
    #print('weight_before'+str(weight))
    weight[yfit>z4_arr]=z4_arr_err_up[yfit>z4_arr] # if the fit point is above the measure, use upper weight
    weight[yfit<=z4_arr]=z4_arr_err_low[yfit<=z4_arr] # else use lower weight
    #print('weight_after'+str(weight))
    #print('(yfit-z4_arr)**2/weight**2 '+str((yfit-z4_arr)**2/weight**2 ))
    return (yfit-z4_arr)**2/weight**2 

"""
def curve_fit_custom_con(V,index_arr,z4_arr,z4_arr_err_low,z4_arr_err_up):
    a=V
    yfit=lin_fit_1DConstant(index_arr,a)
    weight=np.ones_like(yfit)
    weight[yfit>z4_arr]=z4_arr_err_up[yfit>z4_arr] # if the fit point is above the measure, use upper weight
    weight[yfit<=z4_arr]=z4_arr_err_low[yfit<=z4_arr] # else use lower weight
    return (yfit-z4_arr)**2/weight**2 
"""

def lin_fit_1D(x, a, b):
    return a * x + b

def lin_fit_1DConstant(x, b):
    return  np.full(len(x),b)

def lin_fit_2D(x,y, a, b,c):
    return a * x + b*y+c




def chi_50000(sci_image,var_image,res_iapetus):

    #sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
    #var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
    multiplicative_factor_to_renormalize_to_50000=np.max(sci_image)/50000
    sci_image_renormalized=sci_image/multiplicative_factor_to_renormalize_to_50000
    var_image_renormalized=var_image/multiplicative_factor_to_renormalize_to_50000
    res_iapetus_renormalized=res_iapetus/multiplicative_factor_to_renormalize_to_50000
    

    return np.mean((sci_image_renormalized-res_iapetus_renormalized)**2/var_image_renormalized)

def add_artificial_noise(sci_image,var_image,res_iapetus):
    
    """
    add extra noise so that it has comparable noise as if the max flux in the image (in the single pixel) is 40000
    
    
    """

    
    multi_factor=np.max(sci_image)/40000
    Max_SN_now=np.max(sci_image)/np.max(np.sqrt(var_image))
    dif_in_SN=Max_SN_now/200
    artifical_noise=np.zeros_like(res_iapetus)
    artifical_noise=np.array(artifical_noise)
    for i in range(len(artifical_noise)):
        for j in range(len(artifical_noise)):
            artifical_noise[i,j]=np.random.randn()*np.sqrt((dif_in_SN**2-1)*var_image[i,j])   
            
    if dif_in_SN>1:        
        return (sci_image+artifical_noise),((dif_in_SN**2)*var_image),res_iapetus
    else:
        return (sci_image),((dif_in_SN**2)*var_image),res_iapetus 



def _reflect_breaks(size: int) -> np.ndarray:
  """Calculate cell boundaries with reflecting boundary conditions."""
  result = np.concatenate([[0], 0.5 + np.arange(size - 1), [size - 1]])
  assert len(result) == size + 1
  return result

def _interval_overlap(first_breaks: np.ndarray,
                      second_breaks: np.ndarray) -> np.ndarray:
  """Return the overlap distance between all pairs of intervals.

  Args:
    first_breaks: breaks between entries in the first set of intervals, with
      shape (N+1,). Must be a non-decreasing sequence.
    second_breaks: breaks between entries in the second set of intervals, with
      shape (M+1,). Must be a non-decreasing sequence.

  Returns:
    Array with shape (N, M) giving the size of the overlapping region between
    each pair of intervals.
  """
  first_upper = first_breaks[1:]
  second_upper = second_breaks[1:]
  upper = np.minimum(first_upper[:, np.newaxis], second_upper[np.newaxis, :])

  first_lower = first_breaks[:-1]
  second_lower = second_breaks[:-1]
  lower = np.maximum(first_lower[:, np.newaxis], second_lower[np.newaxis, :])

  return np.maximum(upper - lower, 0)

    
def _resize_weights(
    old_size: int, new_size: int, reflect: bool = False) -> np.ndarray:
  """Create a weight matrix for resizing with the local mean along an axis.

  Args:
    old_size: old size.
    new_size: new size.
    reflect: whether or not there are reflecting boundary conditions.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if not reflect:
    old_breaks = np.linspace(0, old_size, num=old_size + 1)
    new_breaks = np.linspace(0, old_size, num=new_size + 1)
  else:
    old_breaks = _reflect_breaks(old_size)
    new_breaks = (old_size - 1) / (new_size - 1) * _reflect_breaks(new_size)

  weights = _interval_overlap(new_breaks, old_breaks)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (new_size, old_size)
  return weights

def resize(array: np.ndarray,
           shape: Tuple[int, ...],
           reflect_axes: Iterable[int] = ()) -> np.ndarray:
  """Resize an array with the local mean / bilinear scaling.

  Works for both upsampling and downsampling in a fashion equivalent to
  block_mean and zoom, but allows for resizing by non-integer multiples. Prefer
  block_mean and zoom when possible, as this implementation is probably slower.

  Args:
    array: array to resize.
    shape: shape of the resized array.
    reflect_axes: iterable of axis numbers with reflecting boundary conditions,
      mirrored over the center of the first and last cell.

  Returns:
    Array resized to shape.

  Raises:
    ValueError: if any values in reflect_axes fall outside the interval
      [-array.ndim, array.ndim).
  """
  reflect_axes_set = set()
  for axis in reflect_axes:
    if not -array.ndim <= axis < array.ndim:
      raise ValueError('invalid axis: {}'.format(axis))
    reflect_axes_set.add(axis % array.ndim)

  output = array
  for axis, (old_size, new_size) in enumerate(zip(array.shape, shape)):
    reflect = axis in reflect_axes_set
    weights = _resize_weights(old_size, new_size, reflect=reflect)
    product = np.tensordot(output, weights, [[axis], [-1]])
    output = np.moveaxis(product, -1, axis)
  return output

def find_centroid_of_flux(image):
    """
    function giving the position of weighted average of the flux in a square image
    
    @param iamge    input image 
    """
    
    
    x_center=[]
    y_center=[]

    I_x=[]
    for i in range(len(image)):
        I_x.append([i,np.sum(image[:,i])])

    I_x=np.array(I_x)

    I_y=[]
    for i in range(len(image)):
        I_y.append([i,np.sum(image[i])])

    I_y=np.array(I_y)


    x_center=(np.sum(I_x[:,0]*I_x[:,1])/np.sum(I_x[:,1]))
    y_center=(np.sum(I_y[:,0]*I_y[:,1])/np.sum(I_y[:,1]))

    return(x_center,y_center)

def cut_Centroid_of_natural_resolution_image(image,size_natural_resolution,oversampling,dx,dy):
    """
    function which takes central part of a larger oversampled image
    
    @param image                          array contaning suggested starting values for a model 
    @param size_natural_resolution        how many natural units to create new image (fix)
    @param oversampling                   oversampling
    @param dx                             how much to move in dx direction (fix)
    @param dy                             how much to move in dy direction (fix)
    """

    positions_from_where_to_start_cut=[int(len(image)/2-size_natural_resolution/2-dx*oversampling),
                                       int(len(image)/2-size_natural_resolution/2-dy*oversampling)]

    res=image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1]+int(size_natural_resolution),
                 positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0]+int(size_natural_resolution)]
    
    return res

def downsample_manual_function(optPsf_cut_fiber_convolved,npixfinal):
    """
    function which downsamples the image ''manually'' i.e., without interpolation
    
    
    @param optPsf_cut_fiber_convolved     image which needs to be downsampled (in our case this will be image of the optical psf convolved with fiber)
    @param npixfinal                      what is the size of the final image
    """
    res=[]
    FFTImage_cropped_split=np.array_split(optPsf_cut_fiber_convolved,npixfinal)
    for j in range(len(FFTImage_cropped_split)):
        FFTImage_cropped_split_split=np.array_split(FFTImage_cropped_split[j],npixfinal,axis=1)
        for i in range(len(FFTImage_cropped_split_split)):
            res.append(np.sum(FFTImage_cropped_split_split[i]))

    res=np.array(res)
    FFTImage_croppedEvenDownsampled32=res.reshape(npixfinal,npixfinal)
    return FFTImage_croppedEvenDownsampled32

def find_nearest(array,value):
    """
    function which fineds the nearest value in the array to the input value (not sure if I am still using this)
    
    
    @param optPsf_cut_fiber_convolved     image which needs to be downsampled (in our case this will be image of the optical psf convolved with fiber)
    @param npixfinal                      what is the size of the final image
    """
    idx = (np.abs(array-value)).argmin()
    return idx

