"""
Created on Mon Mar 30 2020

Code and scripts used for parameter estimation for Zernike analysis

Versions:
Mar 31, 2020: ? -> 0.28 added argparser and extra Zernike
Apr 07, 2020: 0.28 -> 0.28b added /tigress/ncaplar/Result_DirectAndInt_FinalResults/' to the output path of the final product
Apr 08, 2020: 0.28b -> 0.28c playing with masking in the extra Zernike procedure
Apr 09, 2020: 0.28c -> 0.28d fixed a bug where I was not saving res_init_image
Apr 09, 2020: 0.28d -> 0.28e changed the input image results
Apr 14, 2020: 0.28e -> 0.29 integrated analysis of focused images
Jul 01, 2020: 0.29 -> 0.3 introduced LN_PFS_multi_same_spot analysis 
Jul 06, 2020: 0.30 -> 0.31 many minor improvments, modified emcee parameters
Jul 06, 2020: 0.31 -> 0.31a added str2bool
Jul 06, 2020: 0.31a -> 0.31b added list of swarm times
Jul 14, 2020: 0.31b -> 0.32 added skip_initial_state_check=True to work with Emcee3
Jul 14, 2020: 0.32 -> 0.32a added  and num_iter>10 check in the early break
Jul 14, 2020: 0.32a -> 0.32b changed to num_iter>int(nsteps) 
Jul 14, 2020: 0.32b -> 0.32c changed /tigress/ncaplar/Data/ to /tigress/ncaplar/ReducedData/ and Stamps_Cleaned to Stamps_cleaned
Jul 30, 2020: 0.32c -> 0.33 startedTokovinin implentation
Jul 30, 2020: 0.33 -> 0.33a changed eps=6 to be test implementation
Jul 31, 2020: 0.33a -> 0.33b implemented list_of_labelInput_without_focus_or_near_focus
Aug 10, 2020: 0.33b -> 0.34 first implementation of the extra Zernike parameters
Aug 11, 2020: 0.34 -> 0.34a multiple fixes in order to be able to run extra Zernike analysis
Aug 12, 2020: 0.34a -> 0.34b increased initial variance of parameters in multiAnalysis
Sep 09, 2020: 0.34b -> 0.35 added multi_var Tokovinin analysis
Sep 16, 2020: 0.35 -> 0.35a added poor-man analysis of fake images
Sep 19, 2020: 0.35a -> 0.35b changed variations of Zernike in Tokovinin multi analysis
Sep 22, 2020: 0.35b -> 0.36 large changes on how the _std version of the images is treated
Oct 06, 2020: 0.36 -> 0.37 when creating Tokovinin test images, keep normalization and position
Nov 03, 2020: 0.37 -> 0.37a adding that zmax is controlled by input in Tokovinin
Nov 05, 2020: 0.37a -> 0.37b removing Tokovinin function, which is now imported from the Module
Nov 09, 2020: 0.37b -> 0.38 added first implementation of going Tokovin via previous best
Nov 11, 2020: 0.38 -> 0.38a parallelizing Tokovinin
Nov 16, 2020: 0.38a -> 0.38b first version of Tokovinin via previous beest that works
Nov 17, 2020: 0.38b -> 0.38c modified speeds so the code does not get outside limits
Dec 05, 2020: 0.38c -> 0.39 implemented support for November Subaru data and Module v0.37
Dec 09, 2020: 0.39 -> 0.39a transform single_number variable to int asap
Jan 09, 2021: 0.39a -> 0.39b take nearby points where input is unavaliable
Jan 13, 2021: 0.39b -> 0.39c print out number of accepted images, do not analyze if below 60%
Jan 28, 2021: 0.39c -> 0.39d simplified when to take account wide parameters
Feb 02, 2021: 0.39d -> 0.39e take correct finalAr (not finalAr_Feb2020.pkl)
Apr 08, 2021: 0.39e -> 0.40 implented use_only_chi=True 
Apr 08, 2021: 0.40 -> 0.40a print time when each step is starting
Apr 20, 2021: 0.40a -> 0.40b do not hang when image is not found
Apr 22, 2021: 0.40b -> 0.40c experiment with multi_background_factor
Apr 27, 2021: 0.40c -> 0.40d further experiment with multi_background_factor
May 06, 2021: 0.40d -> 0.40e set w=0.51
May 07, 2021: 0.40e -> 0.40f array_of_multi_background_factors, changed to correct number of factors
May 13, 2021: 0.40f -> 0.41 updated to again to be able to get focus analysis

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""

#standard library imports
from __future__ import absolute_import, division, print_function
import socket
import time
print(str(socket.gethostname())+': Start time for importing is: '+time.ctime()) 
import sys
import os 
import argparse
from datetime import datetime
from copy import deepcopy
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

import pickle 

#Related third party imports
#from multiprocessing import Pool
#from multiprocessing import current_process

# MPI imports, depending on configuration
from schwimmbad import MPIPool
from schwimmbad import MultiPool
#import mpi4py
from functools import partial

#Local application/library specific imports

#pandas
#import pandas

#cosmohammer
import cosmoHammer

#galsim
#import galsim

#emcee
import emcee

#Zernike_Module
from Zernike_Module import LN_PFS_single,LN_PFS_multi_same_spot,create_parInit,PFSLikelihoodModule,svd_invert,Tokovinin_multi,check_global_parameters

__version__ = "0.41"

parser = argparse.ArgumentParser(description="Starting args import",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog='Done with import')

############################################################################################################
print('############################################################################################################')  

############################################################################################################
print('Start time is: '+time.ctime())  
time_start=time.time()  

print('version of Zernike_parameter_estimation is: '+str(__version__))

################################################
# nargs '+' specifies that the algorithm can accept one or more arguments
parser.add_argument("-obs", help="name of the observation (actually a number/numbers) which we will analyze", nargs='+',type=int,default=argparse.SUPPRESS)
#parser.add_argument("obs", help="name of the observation (actually a number) which we will analyze",type=int,default=argparse.SUPPRESS)
#parser.add_argument('--obs',dest='obs',default=None)
################################################
# name of the spot (again, actually number) which we will analyze
parser.add_argument("-spot", help="name of the spot (again, actually a number) which we will analyze", type=int)
################################################
# number of steps each walker will take 
parser.add_argument("-nsteps", help="number of steps each walker will take ",type=int)
################################################
# input argument that controls the paramters of the cosmo_hammer process
# if in doubt, eps=5 is probably a solid option
parser.add_argument("-eps", help="input argument that controls the paramters of the cosmo_hammer process; if in doubt, eps=5 is probably a solid option ",type=int)
################################################    
# which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5]   
parser.add_argument("-dataset", help="which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5] ",type=int, choices=[0, 1, 2,3,4,5,6])
################################################    
parser.add_argument("-arc", help="which arc lamp is being analyzed (HgAr for Mercury-Argon, Ne for Neon, Kr for Krypton)  ", choices=["HgAr","Ar", "Ne", "Kr"])
################################################ 
parser.add_argument("-double_sources", help="are there two sources in the image (True, False) ", default='False',type=str, choices=['True','False'])
################################################ 
parser.add_argument("-double_sources_positions_ratios", help="parameters for second source ",action='append')        
################################################        
parser.add_argument("-twentytwo_or_extra", help="number of Zernike components (22 or number larger than 22 which leads to extra Zernike analysis)",type=int)             
################################################       
parser.add_argument("-date_of_input", help="input date")   
################################################       
parser.add_argument("-direct_or_interpolation", help="direct or interpolation ", choices=["direct", "interpolation"])   
################################################   
parser.add_argument("-date_of_output", help="date_of_output " )       
################################################  
parser.add_argument("-analysis_type", help="defocus or focus? " )       
################################################   

 # Finished with specifying arguments     
 
################################################  
# Assigning arguments to variables
args = parser.parse_args()
################################################ 
# put all passed observations in a list
list_of_obs=args.obs
len_of_list_of_obs=len(list_of_obs)

# if you passed only one value, somehow put in a list and make sure that the code runs
print('all values in the obs_list is/are: '+str(list_of_obs))
print('number of images analyzed is: '+str(len_of_list_of_obs))

if len_of_list_of_obs > 1:
    multi_var=True
else:
    multi_var=False

# obs variable is assigned to the first number in the list
obs_init = list_of_obs[0]
print('obs_init is: '+str(obs_init))  
################################################ 
single_number= int(args.spot)
print('spot number (single_number) is: '+str(single_number)) 
################################################ 
nsteps = args.nsteps
print('nsteps is: '+str(nsteps)) 
################################################ 
eps=args.eps
print('eps parameter is: '+str(eps))  

if eps==1:
    #particle count, c1 parameter (individual), c2 parameter (global)
    options=[390,1.193,1.193]
if eps==2:
    options=[790,1.193,1.193]
    nsteps=int(nsteps/2)
if eps==3:
    options=[390,1.593,1.193]
if eps==4:
    options=[390,0.993,1.193]
if eps==5:
    options=[390,2.793,1.593]
if eps==6:
    options=[480,2.793,1.193]
if eps==7:
    options=[160,2.793,1.193]
if eps==8:
    options=[240,2.793,1.593]
if eps==9:
    options=[190,1.193,1.193]
    nsteps=int(2*nsteps)
if eps==10:
    options=[390,1.893,2.893]
    
c1=options[1]
c2=options[2]
################################################ 
dataset=args.dataset

print('Dataset analyzed is: '+str(dataset))
if dataset==0 or dataset==1:
    print('ehm.... old data, not analyzed')

# folder contaning the data from December 2017
# dataset 0
#DATA_FOLDER='/tigress/ncaplar/Data/Dec18Data/'

# folder contaning the data from February 2019
# dataset 1
#DATA_FOLDER='/tigress/ncaplar/Data/Feb5Data/'

# folder containing the data taken with F/2.8 stop in April and May 2019
# dataset 2
if dataset==2:
    DATA_FOLDER='/tigress/ncaplar/ReducedData/Data_May_28/'

# folder containing the data taken with F/2.8 stop in April and May 2019
# dataset 3
if dataset==3:
    DATA_FOLDER='/tigress/ncaplar/ReducedData/Data_Jun_25/'

# folder containing the data taken with F/2.8 stop in July 2019
# dataset 4 (defocu) and 5 (fine defocus)
if dataset==4 or dataset==5:
    DATA_FOLDER='/tigress/ncaplar/ReducedData/Data_Aug_14/'      
    
    if single_number==120:
        DATA_FOLDER='/tigress/ncaplar/ReducedData/Data_Aug_14/'   

# folder contaning the data taken with F/2.8 stop in November 2020 on Subaru
if dataset==6:
    DATA_FOLDER='/tigress/ncaplar/ReducedData/Data_Nov_20/'   

          
    

STAMPS_FOLDER=DATA_FOLDER+'Stamps_cleaned/'
DATAFRAMES_FOLDER=DATA_FOLDER+'Dataframes/'


RESULT_FOLDER='/tigress/ncaplar/Results/'
################################################ 
arc=args.arc

print('arc lamp is: '+str(arc))

# inputs for Dec 2017 data
# dataset=0

#if arc=="HgAr":
#    single_number_focus=8603
#elif arc=="Ne":
#    single_number_focus=8693    
#else:
#    print("Not recognized arc-line")
 
# inputs for Februaruy 2019 data, F/3.2 stop
# dataset=1

#if str(arc)=="HgAr":
#    single_number_focus=11748
#elif str(arc)=="Ne":
#    single_number_focus=12355 
#else:
#    print("Not recognized arc-line")
    
# inputs for Februaruy 2019 data, F/2.8 stop   
# dataset=2
if dataset==2:
    if str(arc)=="HgAr":
        single_number_focus=17017+54
    elif str(arc)=="Ne":
        single_number_focus=16292 
    elif str(arc)=="Kr":
        single_number_focus=17310+54
    else:
        print("Not recognized arc-line")    
    
# inputs for April/May 2019 data, F/2.5 stop   
# dataset=3
if dataset==3:
    if str(arc)=="HgAr":
        single_number_focus=19238+54
    elif str(arc)=="Ne":
        single_number_focus=19472+54 
    else:
        print("Not recognized arc-line")  

# defocused data from July 2019    
# dataset=4
if dataset==4:   
    if str(arc)=="HgAr":
        single_number_focus=21346+54
    elif str(arc)=="Ne":
        single_number_focus=21550+54 
    elif str(arc)=="Kr":
        single_number_focus=21754+54
    else:
        print("Not recognized arc-line")     
        
# defocused data from November 2020    
# dataset=6
if dataset==6:   
    if str(arc)=="Ar":
        single_number_focus=34341+48
    elif str(arc)=="Ne":
        single_number_focus=34217+48
    elif str(arc)=="Kr":
        single_number_focus=34561+48
    else:
        print('Arc line specified is '+str(arc))
        print("Not recognized arc-line")  
        
################################################  
print('args.double_sources: '+str(args.double_sources))        

def str2bool(v):
    """ 
    Small function that take a string and gives back boolean value
    
    """
    
    
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')        
        
double_sources=str2bool(args.double_sources)



print('double sources argument is: '+str(double_sources))
################################################  
double_sources_positions_ratios=eval(args.double_sources_positions_ratios[0])
print('double source parameters are: '+str(double_sources_positions_ratios))   
################################################  
twentytwo_or_extra=args.twentytwo_or_extra
print('22_or_extra parameter is: '+str(twentytwo_or_extra))       
        
################################################       
parser.add_argument("date_of_input", help="input date")   
date_of_input=args.date_of_input
print('date of input database: '+str(date_of_input))  
################################################       
parser.add_argument("direct_or_interpolation", help="direct or interpolation ", choices=["direct", "interpolation"])   
direct_or_interpolation=args.direct_or_interpolation
print('direct or interpolation: '+str(direct_or_interpolation))  
################################################   
parser.add_argument("date_of_output", help="date_of_output " )       
date_of_output=args.date_of_output
print('date of output: '+str(date_of_output))   
################################################  
parser.add_argument("analysis_type", help="analysis_type " )       
analysis_type=args.analysis_type
print('analysis_type: '+str(analysis_type))   
################################################  

# if you passing a fake stamps and dataframe (placed as index=120 and ``HgAr'' lamp)
if single_number>=120 and arc=='HgAr':
    STAMPS_FOLDER=DATA_FOLDER+'Stamps_cleaned_fake/'
    DATAFRAMES_FAKE_FOLDER=DATA_FOLDER+'Dataframes_fake/'   
    
    single_number_original=np.copy(single_number)    
    single_number=120
    print('Single number changed to 120')
else:
    single_number_original=np.copy(single_number)    

# here we should check that version of Zernike_Module one is using is correct?
# not implemented



 

# Finished with all arguments


    


################################################  
################################################  
################################################  
################################################  
################################################  
################################################  
# import data

list_of_sci_images=[]
list_of_mask_images=[]
list_of_var_images=[]
list_of_sci_images_focus=[]
list_of_var_images_focus=[]
list_of_obs_cleaned=[]

list_of_times=[]


for obs in list_of_obs:
    if single_number<120:
        try:
            sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
            mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
            var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')   
            print('sci_image loaded from: '+STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        except:
            # change to that code does not fail and hang if the image is not found
            # this will lead to pass statment in next step because np.sum(sci_image)=0
            print('sci_image not found')
            sci_image=np.zeros((20,20))
            var_image=np.zeros((20,20))
            mask_image=np.zeros((20,20))
            
            
        try:
            sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
            var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
        except:
            pass

    else:
        STAMPS_FOLDER='/tigress/ncaplar/ReducedData/Data_Aug_14/Stamps_cleaned_fake/'
        sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
        var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy') 
        print('sci_image loaded from: '+STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        # for fake images below 120 I did not create _large images

    # If there is no science image, do not add images
    if int(np.sum(sci_image))==0:
        print('No science image - passing')
        pass
    else:    
        # do not analyze images where a large fraction of the image is masked
        if np.mean(mask_image)>0.1:
            print(str(np.mean(mask_image)*100)+'% of image is masked... when it is more than 10% - exiting')
            pass
        else:
            print('adding images for obs: '+str(obs))
            list_of_sci_images.append(sci_image)
            list_of_mask_images.append(mask_image)
            list_of_var_images.append(var_image)
            try:
                if single_number<120:
                    list_of_sci_images_focus.append(sci_image_focus_large)
                    list_of_var_images_focus.append(var_image_focus_large)
            except:
                pass
            # observation which are of good enough quality
            list_of_obs_cleaned.append(obs)
            
print('len of list_of_sci_images: '+str(len(list_of_sci_images)) ) 

print('len of accepted images '+str(len(list_of_obs_cleaned)) + ' / len of asked images '+ str(len(list_of_obs)))          

# If there is no valid images imported, exit 
if list_of_sci_images==[]:
    print('No valid images - exiting')
    sys.exit(0)
    
# if you were able only to import only a fraction of images
# if this fraction is too low - exit
if (len(list_of_obs_cleaned)/len(list_of_obs) ) <0.6:
    print('Fraction of images imported is too low - exiting')
    sys.exit(0)


# name of the outputs
# should update that it goes via date
list_of_NAME_OF_CHAIN=[]
list_of_NAME_OF_LIKELIHOOD_CHAIN=[]
for obs in list_of_obs_cleaned:
    NAME_OF_CHAIN='chain'+str(date_of_output)+'_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    NAME_OF_LIKELIHOOD_CHAIN='likechain'+str(date_of_output)+'_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    
    # if you are passing higher than 120
    if single_number_original>=120:
        NAME_OF_CHAIN='chain'+str(date_of_output)+'_Single_P_'+str(obs)+str(single_number_original)+str(eps)+str(arc)
        NAME_OF_LIKELIHOOD_CHAIN='likechain'+str(date_of_output)+'_Single_P_'+str(obs)+str(single_number_original)+str(eps)+str(arc)        
    
    
    list_of_NAME_OF_CHAIN.append(NAME_OF_CHAIN)
    list_of_NAME_OF_LIKELIHOOD_CHAIN.append(NAME_OF_LIKELIHOOD_CHAIN)


# where are the dataframes which we use to guess the initial solution
# Ar (Argon)
if str(arc)=='Ar' or arc=='HgAr':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Ar_from_'+str(date_of_input)+'.pkl', 'rb') as f:
        results_of_fit_input_HgAr=pickle.load(f)
        #print(results_of_fit_input_HgAr)
        print('results_of_fit_input_Ar is taken from: '+str(f))
    with open(DATAFRAMES_FOLDER+'finalAr_Feb2020', 'rb') as f:
        finalAr_Feb2020_dataset=pickle.load(f) 

# Ne (Neon)
if str(arc)=='Ne':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Ne_from_'+str(date_of_input)+'.pkl', 'rb') as f:
        results_of_fit_input_Ne=pickle.load(f)
    print('results_of_fit_input_Ne is taken from: '+str(f))
    
    with open(DATAFRAMES_FOLDER + 'finalNe_Feb2020', 'rb') as f:
        finalNe_Feb2020_dataset=pickle.load(f) 

# Kr (Krypton)
if str(arc)=='Kr':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Kr_from_'+str(date_of_input)+'.pkl', 'rb') as f:
        results_of_fit_input_Kr=pickle.load(f)
    print('results_of_fit_input_Kr is taken from: '+str(f))
    with open(DATAFRAMES_FOLDER + 'finalKr_Feb2020', 'rb') as f:
        finalKr_Feb2020_dataset=pickle.load(f)    
    

##############################################    
    
# What are the observations that can be analyzed
# used to associate observation with their input labels, so that the initial parameters guess is correct    
    
# dataset 0, December 2017 data    
#if arc=='HgAr':
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
#elif arc=='Ne':
#    print('Neon?????')
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])+90
    
# F/3.2 data
if dataset==1:
    if arc=='HgAr':
        obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11694,11700,11706,11712,11718,11724,11730,11736])
    elif arc=='Ne':
        obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12349,12343,12337,12331,12325,12319,12313,12307])  
 
# F/2.8 data       
if dataset==2:
    if arc=='HgAr':
        obs_possibilites=np.array([17023,17023+6,17023+12,17023+18,17023+24,17023+30,17023+36,17023+42,17023+48,17023+54,17023+60,17023+66,17023+72,17023+78,17023+84,17023+90,17023+96,17023+48])
    if arc=='Ne':
        obs_possibilites=np.array([16238+6,16238+12,16238+18,16238+24,16238+30,16238+36,16238+42,16238+48,16238+54,16238+60,16238+66,16238+72,16238+78,16238+84,16238+90,16238+96,16238+102,16238+54])
    if arc=='Kr':
         obs_possibilites=np.array([17310+6,17310+12,17310+18,17310+24,17310+30,17310+36,17310+42,17310+48,17310+54,17310+60,17310+66,17310+72,17310+78,17310+84,17310+90,17310+96,17310+102,17310+54])

# F/2.5 data     
if dataset==3:
    if arc=='HgAr':
        obs_possibilites=np.array([19238,19238+6,19238+12,19238+18,19238+24,19238+30,19238+36,19238+42,19238+48,19238+54,19238+60,19238+66,19238+72,19238+78,19238+84,19238+90,19238+96,19238+48])
    elif arc=='Ne':
        obs_possibilites=np.array([19472+6,19472+12,19472+18,19472+24,19472+30,19472+36,19472+42,19472+48,19472+54,19472+60,19472+66,19472+72,19472+78,19472+84,19472+90,19472+96,19472+102,19472+54])    

# F/2.8 July data        
if dataset==4:
    if arc=='HgAr':
        obs_possibilites=np.array([21346+6,21346+12,21346+18,21346+24,21346+30,21346+36,21346+42,21346+48,21346+54,21346+60,21346+66,21346+72,21346+78,21346+84,21346+90,21346+96,21346+102,21346+48])
    if arc=='Ne':
        obs_possibilites=np.array([21550+6,21550+12,21550+18,21550+24,21550+30,21550+36,21550+42,21550+48,21550+54,21550+60,21550+66,21550+72,21550+78,21550+84,21550+90,21550+96,21550+102,21550+54])
    if arc=='Kr':
         obs_possibilites=np.array([21754+6,21754+12,21754+18,21754+24,21754+30,21754+36,21754+42,21754+48,21754+54,21754+60,21754+66,21754+72,21754+78,21754+84,21754+90,21754+96,21754+102,21754+54])
     
if dataset==6:
    if arc=='Ar':
        obs_possibilites=np.array([34341,34341+6,34341+12,34341+18,34341+24,34341+30,34341+36,34341+42,34341+48,\
                                   34341+54,34341+60,34341+66,34341+72,34341+78,34341+84,34341+90,34341+96,21346+48])
    if arc=='Ne':
        obs_possibilites=np.array([34217,34217+6,34217+12,34217+18,34217+24,34217+30,34217+36,34217+42,34217+48,\
                                   34217+54,34217+60,34217+66,34217+72,34217+78,34217+84,34217+90,34217+96,34217+48])
    if arc=='Kr':
         obs_possibilites=np.array([34561,34561+6,34561+12,34561+18,34561+24,34561+30,34561+36,34561+42,34561+48,\
                                    34561+54,34561+60,34561+66,34561+72,34561+78,34561+84,34561+90,34561+96,34561+48])
        
     
 ##############################################    

# associates each observation with the label in the supplied dataframe
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28,0])
label=['m4','m35','m3','m25','m2','m15','m1','m05','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']

list_of_z4Input=[]
for obs in list_of_obs_cleaned:
    z4Input=z4Input_possibilites[obs_possibilites==obs][0]
    list_of_z4Input.append(z4Input)
    
# list of labels that we are passing to the algorithm 
list_of_labelInput=[]
for obs in list_of_obs_cleaned:
    labelInput=label[list(obs_possibilites).index(obs)]
    list_of_labelInput.append(labelInput)
    
# perhaps temporarily
# list_of_defocuses_input_long input in Tokovinin algorith
list_of_defocuses_input_long=list_of_labelInput
    
print('list_of_labelInput: '+str(list_of_labelInput))
    
# list of labels without values near focus (for possible analysis with Tokovinin alrogithm)
list_of_labelInput_without_focus_or_near_focus=deepcopy(list_of_labelInput)
for i in ['m15','m1','m05','0','p05','p1','p15']:
    if i in list_of_labelInput:
        list_of_labelInput_without_focus_or_near_focus.remove(i)
    
print('list_of_labelInput_without_focus_or_near_focus: ' +str(list_of_labelInput_without_focus_or_near_focus))    
    
# positional indices of the defocused data in the whole set of input data (for possible analysis with Tokovinin alrogithm)
index_of_list_of_labelInput_without_focus_or_near_focus=[]
for i in list_of_labelInput_without_focus_or_near_focus:
    index_of_list_of_labelInput_without_focus_or_near_focus.append(list_of_labelInput.index(i))
    
print('index_of_list_of_labelInput_without_focus_or_near_focus: ' +str(index_of_list_of_labelInput_without_focus_or_near_focus))
        

# if we are going to analyzed focused images based on defocused analysis, go there 
if analysis_type=='focus':
    labelInput='0'
else:
    pass

# Input the zmax that you wish to achieve in the analysis
zmax=twentytwo_or_extra
"""
# names for paramters - names if we go up to z11
columns=['z4','z5','z6','z7','z8','z9','z10','z11',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  

# names for paramters - names if we go up to z22
columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
           'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  
"""

columns=['z4','z5','z6','z7','z8','z9','z10','z11',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'wide_0','wide_23','wide_43','misalign',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  

# names for paramters - names if we go up to z22
columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
           'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'wide_0','wide_23','wide_43','misalign',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  

# depening on the arc, select the appropriate dataframe
if arc=="HgAr":
    results_of_fit_input=results_of_fit_input_HgAr
    #finalArc=finalHgAr_Feb2020_dataset
elif arc=="Ne":
    results_of_fit_input=results_of_fit_input_Ne
    finalArc=finalNe_Feb2020_dataset
elif arc=="Kr":
    results_of_fit_input=results_of_fit_input_Kr  
    finalArc=finalKr_Feb2020_dataset
elif arc=="Ar":
    results_of_fit_input=results_of_fit_input_HgAr
    finalArc=finalAr_Feb2020_dataset

else:
    print("what has happened here? Only Argon, HgAr, Neon and Krypton implemented")


#pool = MPIPool()
#if not pool.is_master():
#    pool.wait()
#    sys.exit(0)   
from multiprocessing import Pool
pool=Pool()

# if you are passing only image    
if len(list_of_obs)==1:
        
    # Create an object (called model) - which gives likelihood given the parameters 
    model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
          pupil_parameters=None,use_pupil_parameters=None,\
          save=0,verbosity=0,double_sources=double_sources,zmax=zmax,\
          double_sources_positions_ratios=double_sources_positions_ratios,npix=1536)
else:
    # otherwise, if you are passing multiple images
    # model_multi is only needed to create reasonable parametrizations and could possibly be avoided in future versions?
    model_multi = LN_PFS_multi_same_spot(list_of_sci_images=list_of_sci_images,list_of_var_images=list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                         dithering=1,save=0,verbosity=0,npix=1536,list_of_defocuses=list_of_labelInput,zmax=zmax,double_sources=double_sources,\
                                         double_sources_positions_ratios=double_sources_positions_ratios,test_run=False)           

    Tokovinin_multi_instance_with_pool=Tokovinin_multi(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                        dithering=1,save=0,verbosity=0,npix=1536,zmax=twentytwo_or_extra,\
                                        list_of_defocuses=list_of_defocuses_input_long,double_sources=double_sources,\
                                        double_sources_positions_ratios=double_sources_positions_ratios,fit_for_flux=True,test_run=False, \
                                        num_iter=None,pool=pool)
        
    Tokovinin_multi_instance_without_pool=Tokovinin_multi(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                        dithering=1,save=0,verbosity=0,npix=1536,zmax=twentytwo_or_extra,\
                                        list_of_defocuses=list_of_defocuses_input_long,double_sources=double_sources,\
                                        double_sources_positions_ratios=double_sources_positions_ratios,fit_for_flux=True,test_run=False,\
                                        num_iter=None,pool=None)    





# if you are passing only one image
if len(list_of_obs)==1:
    # results_of_fit_input contains proposal for parameters plus 2 values describing chi2 and chi2_max 
    # need to extract all values except the last two
    try:

        allparameters_proposalp2=results_of_fit_input[labelInput].loc[int(single_number)].values

    except:
        # if you failed with integer index, use str version of the index

        allparameters_proposalp2=results_of_fit_input[labelInput].loc[str(single_number)].values    
      
    # if you are analyzing with Zernike up to 22nd
    if zmax==22: 
        # if you require z22 analysis, but input was created with z11
        if len(allparameters_proposalp2)==33:
            allparameters_proposal=np.concatenate((allparameters_proposalp2[0:8],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],allparameters_proposalp2[8:-2]))
        # if input created with z22 just copy the values
        else:
            allparameters_proposal=allparameters_proposalp2[:len(columns22)]
            
    if zmax==11:
        # if input created with z11, copy the values
        allparameters_proposal=allparameters_proposalp2[:,len(columns)]
        
    if zmax>22:
        #?

        allparameters_proposal=allparameters_proposalp2
        
        
    # protect the parameters from changing and unintentional mistakes
    allparameters_proposal_22=np.copy(allparameters_proposal)
else:
    # you are passing multiple images
    list_of_allparameters=[]
    list_of_defocuses=[]
    # search for the previous avaliable results 
    # add the ones that you found in array_of_allparameters and for which labels they are avaliable in list_of_defocuses
    for label in ['m4','m35','m3','m05','0','p05','p3','p35','p4']:
        
        # check if your single_number is avaliable
        
        
        
        
        
        
        print('adding label '+str(label)+' with single_number '+str(int(single_number) )+' for creation of array_of_allparameters')
        try:
            if int(single_number)>=120:
                list_of_allparameters.append(results_of_fit_input[label].loc[int(37)].values)
                list_of_defocuses.append(label)
            if int(single_number) < 120:
                print(results_of_fit_input[label].index.astype(int))
                # if your single_number is avaliable go ahead
                if int(single_number) in results_of_fit_input[label].index.astype(int):
                    print('checkpoint')
                    if type(results_of_fit_input[label].index[0])==str:
                        list_of_allparameters.append(results_of_fit_input[label].loc[str(single_number)].values)
                        print('results_of_fit_input[label].loc[int(single_number)].values' + str(results_of_fit_input[label].loc[str(single_number)].values))
                    else:
                        #print('results_of_fit_input[label]'+str(results_of_fit_input[label]))
                        list_of_allparameters.append(results_of_fit_input[label].loc[int(single_number)].values)
                        print('results_of_fit_input[label].loc[int(single_number)].values' + str(results_of_fit_input[label].loc[int(single_number)].values))
                    list_of_defocuses.append(label)
                        
                else:
                    # find the closest avaliable, right?
                    x_positions=finalArc.loc[results_of_fit_input[labelInput].index]['xc_effective']
                    y_positions=finalArc.loc[results_of_fit_input[labelInput].index]['yc']
                    
                    position_x_single_number=finalArc['xc_effective'].loc[int(single_number)]
                    position_y_single_number=finalArc['yc'].loc[int(single_number)]
                    
                    distance_of_avaliable_spots=np.abs((x_positions-position_x_single_number)**2+(y_positions-position_y_single_number)**2)
                    single_number_input=distance_of_avaliable_spots[distance_of_avaliable_spots==np.min(distance_of_avaliable_spots)].index[0]
                    if type(results_of_fit_input[label].index[0])==str:
                        list_of_allparameters.append(results_of_fit_input[label].loc[str(single_number_input)].values)
                    else:
                        list_of_allparameters.append(results_of_fit_input[label].loc[int(single_number_input)].values)
                    list_of_defocuses.append(label)
                    print('results_of_fit_input[label].loc[int(single_number)].values' + str(results_of_fit_input[label].loc[int(single_number_input)].values))
                    
                    pass

        except:
            print('not able to add label '+str(label))
            pass
        
        
    array_of_allparameters=np.array(list_of_allparameters)
    
    # based on the information from the previous step (results at list_of_defocuses), generate singular array_of_allparameters at list_of_labelInput positions        
    # has shape 2xN, N=number of parameters
    print('twentytwo_or_extra: '+str(twentytwo_or_extra))
    if analysis_type=='defocus':    
        print('array_of_allparameters.shape: '+str(array_of_allparameters.shape))
    
        array_of_polyfit_1_parameterizations_proposal=model_multi.create_resonable_allparameters_parametrizations(array_of_allparameters=array_of_allparameters,\
                                                                    list_of_defocuses_input=list_of_defocuses,zmax=twentytwo_or_extra,remove_last_n=2)
        
        # lets be explicit that the shape of the array is 2d
        array_of_polyfit_1_parameterizations_proposal_shape_2d=array_of_polyfit_1_parameterizations_proposal
    
    #!!!!! EXTREMLY CAREFUL
    # if you are using the version in December 2020, with Module 0.39 you need to recast some of the parameters
    #if date_of_output[:3]=='May':
    #    
    #    print('modifying wide and misalign parameters')
    #    #array_of_polyfit_1_parameterizations_proposal_shape_2d
    #
    #    pos_wide_0=np.arange(len(columns22))[np.array(columns22)=='wide_0'][0]
    #    pos_wide_23=np.arange(len(columns22))[np.array(columns22)=='wide_23'][0]
    #    pos_wide_43=np.arange(len(columns22))[np.array(columns22)=='wide_43'][0]
    #    pos_misalign=np.arange(len(columns22))[np.array(columns22)=='misalign'][0]
    #
    #    array_of_polyfit_1_parameterizations_proposal_shape_2d[pos_wide_0,1]=0.25
    #   array_of_polyfit_1_parameterizations_proposal_shape_2d[pos_wide_23,1]=0.25
    #   array_of_polyfit_1_parameterizations_proposal_shape_2d[pos_wide_43,1]=0.25
    #   array_of_polyfit_1_parameterizations_proposal_shape_2d[pos_misalign,1]=1
    #else:
    #    print('not modifying wide and misalign parameters')        
    
    
    
    # if you 
    if single_number>=120:
        proposal_number=single_number_original-120
        
        array_of_polyfit_1_parameterizations_proposal_shape_2d=\
            np.load('/tigress/ncaplar/ReducedData/Data_Aug_14/Dataframes_fake/array_of_polyfit_1_parameterizations_proposal_shape_2d_proposal_'+str(proposal_number)+'.npy')
    
    """    
    print('Overriding array_of_polyfit_1_parameterizations_proposal_shape_2d for testing purposes') 
   

    array_of_polyfit_1_parameterizations_proposal_shape_2d=np.array([[   -7.26051272,     0.69807266],
       [    0.23066123,     0.16741276],
       [    0.0618131 ,     0.00773434],
       [   -0.02487276,     0.35311773],
       [    0.06465199,     0.52162136],
       [    0.00562712,     0.11197505],
       [    0.01934347,    -0.24217056],
       [    0.01283784,    -0.06110508],
       [   -0.00078921,    -0.13188456],
       [    0.00479373,    -0.10044155],
       [    0.02467383,    -0.01154937],
       [    0.00379758,     0.03052984],
       [   -0.00053136,    -0.01654971],
       [   -0.01616147,     0.03260598],
       [   -0.00216007,     0.02161721],
       [   -0.00854412,     0.02131408],
       [    0.00023284,    -0.02109201],
       [    0.00437892,     0.01079139],
       [   -0.02521649,     0.02915645],
       [    0.        ,     0.68034106],
       [    0.        ,     0.09909094],
       [    0.        ,    -0.20649468],
       [    0.        ,    -0.01763408],
       [    0.        ,     0.05362782],
       [    0.        ,     0.0568914 ],
       [    0.        ,     0.0000314 ],
       [    0.        ,     0.0000314 ],
       [    0.        ,     0.9767313 ],
       [    0.        ,     0.94836248],
       [    0.        ,     0.05393805],
       [    0.        ,    -0.07711173],
       [    0.        ,     0.98541067],
       [    0.        ,     0.04504055],
       [    0.        ,     0.47515132],
       [    0.        ,     1.01242392],
       [    0.        ,     0.65687713],
       [    0.        , 50892.06684106],
       [    0.        ,     2.34159973],
       [    0.        ,     0.00241179],
       [    0.        ,     0.3847749 ],
       [    0.        ,     1.79539309],
       [    0.        ,     0.99639989],
       [   -0.00151877,    -0.00186343],
       [   -0.00351846,     0.01006906],
       [    0.00043316,     0.00448335],
       [    0.00037921,    -0.00089173],
       [    0.00070346,    -0.00234826],
       [   -0.00039535,     0.00196623],
       [    0.00151173,    -0.0029649 ],
       [    0.00032089,    -0.0020236 ],
       [   -0.00132646,    -0.00689623],
       [    0.00294364,     0.00869508],
       [    0.00238834,     0.00406697],
       [   -0.00277659,     0.00488814],
       [   -0.00056052,    -0.00385096],
       [    0.00123593,    -0.0077173 ],
       [   -0.00041795,     0.0013092 ],
       [   -0.00034322,     0.00064159],
       [    0.00006269,     0.00082814],
       [   -0.00011402,    -0.00039645],
       [   -0.00076829,     0.0013193 ],
       [   -0.00152938,    -0.00319124],
       [   -0.00001484,     0.00019586],
       [    0.00068152,    -0.00369635],
       [    0.00297307,    -0.00839939],
       [    0.0020726 ,    -0.00752959],
       [   -0.00096906,     0.00071716],
       [   -0.00004757,     0.00237883],
       [    0.00049466,    -0.00075045],
       [    0.00024551,    -0.00367132],
       [   -0.00046376,     0.00752054],
       [   -0.00058294,     0.0040566 ],
       [   -0.00029741,    -0.00084313],
       [   -0.00110778,     0.00221074],
       [    0.00023515,     0.00484092],
       [    0.00141415,    -0.00094155]])
        """
    
    
###############################################    
# "lower_limits" and "higher_limits" are only needed to initalize the cosmo_hammer code, but not used in the actual evaluation
# so these are purely dummy values
# can this be deleted?
if zmax==11:
    lower_limits=np.array([z4Input-3,-1,-1,-1,-1,-1,-1,-1,
                 0.5,0.05,-0.8,-0.8,0,-0.5,
                 0,-0.5,0.5,0.5,
                 0.8,-np.pi/2,0.5,
                 0,0,0.85,-0.8,
                 1200,0.5,0,
                 0.2,1.65,0.9])
    
    higher_limits=np.array([z4Input+3,1,1,1,1,1,1,1,
                  1.2,0.2,0.8,0.8,0.2,0.5,
                  3,20,1.5,1.5,
                  1,np.pi/2,1.01,
                  0.05,1,1.15,0.8,
                  120000,3.5,0.5,
                  1.1,1.95,1.1])
    
if zmax>=22:
    lower_limits=np.array([z4Input-3,-1,-1,-1,-1,-1,-1,-1,
                           0,0,0,0,0,0,0,0,0,0,0,
                 0.5,0.05,-0.8,-0.8,0,-0.5,
                 0,0,0,0,
                 0.8,-np.pi/2,0.5,
                 0,0,0.85,-0.8,
                 1200,0.5,0,
                 0.2,1.65,0.9])
    
    higher_limits=np.array([z4Input+3,1,1,1,1,1,1,1,
                            0,0,0,0,0,0,0,0,0,0,0,
                  1.2,0.2,0.8,0.8,0.2,0.5,
                  1,1,1,10,
                  1,np.pi/2,1.01,
                  0.05,1,1.15,0.8,
                  120000,3.5,0.5,
                  1.1,1.95,1.1])    
 ##############################################    
    

# soon deprecated?
# branch out here, depending if you are doing low- or high- Zernike analysis 
if twentytwo_or_extra>=22:  

 
    # initialize pool
    #pool = MPIPool()
    #if not pool.is_master():
    #    pool.wait()
    #    sys.exit(0)   
        
    number_of_extra_zernike=twentytwo_or_extra-22     
    #pool=Pool(processes=36)
    #pool=Pool()
       
    print('Name of machine is '+socket.gethostname())    
        
    
    zmax_input=twentytwo_or_extra
    if multi_var==True:
        for i in range(len(list_of_obs_cleaned)):
            print('Adding image: '+str(i+1)+str('/')+str(len(list_of_obs_cleaned)))              
            
            sci_image=list_of_sci_images[i]
            var_image=list_of_var_images[i]
            #allparameters_proposal_22=list_of_allparameters_proposal[i]
            
            
            print('Spot coordinates are: '+str(single_number))  
            print('Size of input image is: '+str(sci_image.shape)) 
            print('First value of the sci image: '+str(sci_image[0][0]))
            print('First value of the var image: '+str(var_image[0][0]))
            print('Steps: '+str(nsteps)) 
            print('Name: '+str(NAME_OF_CHAIN)) 
            print('Zmax is: '+str(zmax))
            print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())    
            
            
        #allparameters_proposal=allparameters_proposal_22
        # dirty hack to get things running
        if zmax>=22:
            columns=columns22
            
        
        #sys.stdout.flush()
        

    
    ##################################################################################################################### 
    # Determining the scattering_slope before anything
    """
    list_of_masks=create_mask(sci_image_focus_large)
    diagonal_cross=list_of_masks[3]
    res_diagonal_cross_large=create_res_data(sci_image_focus_large,diagonal_cross,custom_cent=True,size_pixel=15)
    distances_large=range(len(res_diagonal_cross_large))
    last_distance=np.array(distances_large[101:])[np.log10(res_diagonal_cross_large[101:])>0.5][-1]
    # if the average is below 0 terminate at that position
    where_are_NaNs=np.isnan(np.log10(np.array(res_diagonal_cross_large[101:])))
    if np.sum(where_are_NaNs)>0:
        first_NaN_position=np.array(distances_large[101:])[where_are_NaNs][0]
    else:
        first_NaN_position=last_distance
    last_distance=np.min([first_NaN_position,last_distance])    -1
    
    z=np.polyfit(np.log10(np.array(distances_large[101:last_distance])),np.log10(np.array(res_diagonal_cross_large)[101:last_distance]),1)
    deduced_scattering_slope=z[0]-0.05
    print("deduced_scattering_slope: "+str(deduced_scattering_slope))
    """
    ##################################################################################################################### 
    # First swarm  
    
    #columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
    #         'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
    #         'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
    #         'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
    #         'x_fiber','y_fiber','effective_radius_illumination',
    #         'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
    #         'grating_lines','scattering_slope','scattering_amplitude',
    #         'pixel_effect','fiber_r','flux']  
    
    # change this array, depening on how do you want to proced 
    # very unsatisfactory solution on how to determine how much parameters should be able to vary
    # zero freedom in radiometric parameters
    # full freedom in wavefront parameters
    # less freedom in all other paramters
    
    

    if analysis_type=='focus':
        # only gloal focus parameters and z4
        stronger_array_01=np.array([1,0,0,0,0,0,0,0,
                                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0,
                                0,0,0,0,
                                0,0,0,
                                0,0,0,0,
                                1,1,1,
                                1,1,1])  
        # only gloal focus parameters and z4-z11
        stronger_array_01=np.array([1,1,1,1,1,1,1,1,
                                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0,
                                0,0,0,0,
                                0,0,0,
                                0,0,0,0,
                                1,1,1,
                                1,1,1]) 
        
    else:
        # if working with defocus, but only one image
        if len(list_of_obs)==1:
            stronger_array_01=np.array([1,1,1,1,1,1,1,1,
                                    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 
                                    0.2,0.2,0.2,0.2,0.2,0.2,
                                    0.0,0.0,0.0,0.0,
                                    0.2,0.2,0.2,
                                    0.2,0.2,0.2,0.2,
                                    0.2,0.2,0.2,
                                    0,0,1])
        else:
            """
            stronger_array_01=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                        0.2,0.2,0.2,0.2,0.2,0.2,
                        0.0,0.0,0.0,0.0,
                        0.2,0.2,0.2,
                        0.2,0.2,0.2,0.2,
                        0.2,0.2,0.2,
                        0.2,0.2,1])
            """            
            # if working with defocus, but many images
            stronger_array_01=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1.2,1.2,1.2,1.2,1.2,1.2,
                        1,1,1.0,1.0,
                        1.2,1.2,1.2,
                        1.2,1.2,1.2,1.2,
                        1.2,1.2,1.2,
                        1.2,1.2,1])            
            
     # should do Tokovinin here, then scatter parameters       
    """
    allparameters_proposal_22=np.array([    0.16492145,    -0.88632798,    -0.51669564,    -0.31938941,
           0.50845528,    -0.4275001 ,    -0.58096021,     0.58544687,
           0.03863879,     0.22954357,    -0.04948961,     0.03281349,
           0.0587434 ,     0.050954  ,     0.02756475,    -0.03203576,
          -0.03294923,     0.02517083,     0.04498642,     0.6982929 ,
           0.10099376,    -0.22868679,     0.27114778,     0.06266987,
           0.10090548,     0.2078081 ,     0.31853494,     0.31105295,
           1.28564277,    -0.0509813 ,    -0.0183548 ,     0.87879873,
           0.03103068,     0.7212447 ,     1.04674646,     0.57226047,
       30096.76438941,     2.31074047,     0.00645553,     0.49203847,
           1.83713253,     0.99250387,    -0.01291785,    -0.01807254,
          -0.00805709,     0.01291273,     0.00101384,     0.00141808,
          -0.01207241,     0.00628245,    -0.00219551,     0.01574115,
           0.00356404,    -0.0226188 ,     0.00086017,     0.01206215,
           0.00444335,    -0.00095454,    -0.00213609,     0.00216633,
           0.01068864,    -0.00586613,     0.00746922,    -0.00992799,
          -0.00369008,    -0.01585207,    -0.00102798,     0.00326984,
          -0.00046356,    -0.00617323,    -0.01062363,     0.00643621,
           0.00136246,     0.00197025,     0.00300083,     0.0008643 ])
    """
    #print(allparameters_proposal_22.shape)

    
    # create inital parameters  
    if len(list_of_obs)==1:
        # if passing only one images, you have to create and pass parameters
        print('allparameters_proposal_22: '+str(allparameters_proposal_22))
        parInit1=create_parInit(allparameters_proposal=allparameters_proposal_22,multi=None,pupil_parameters=None,\
                        allparameters_proposal_err=None,\
                        stronger=stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    
        # the number of walkers is given by options array, specified with the parameter eps at the start
        while len(parInit1)<options[0]:
            parInit1_2=create_parInit(allparameters_proposal=allparameters_proposal_22,multi=None,pupil_parameters=None,\
                                      allparameters_proposal_err=None,\
                                      stronger= stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
            parInit1=np.vstack((parInit1,parInit1_2))
    
    else:
        # if passing multi, you pass the parametrizations
        print('array_of_polyfit_1_parameterizations_proposal_shape_2d: '+str(array_of_polyfit_1_parameterizations_proposal_shape_2d))
        parInit1=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,multi=multi_var,pupil_parameters=None,\
                                allparameters_proposal_err=None,\
                                stronger=stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
            
        # the number of walkers is given by options array, specified with the parameter eps at the start
        while len(parInit1)<options[0]:
            parInit1_2=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,multi=multi_var,pupil_parameters=None,\
                                      allparameters_proposal_err=None,\
                                      stronger= stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
            parInit1=np.vstack((parInit1,parInit1_2))
 
    # standard deviation of parameters (for control only?)
    parInit1_std=[]
    for i in range(parInit1.shape[1]):
        parInit1_std.append(np.std(parInit1[:,i])) 
    parInit1_std=np.array(parInit1_std)
    print('parInit1_std: '+str(parInit1_std))

    
    # number of particles and number of parameters
    particleCount=options[0]  
    paramCount=len(parInit1[0])
    
    
    # initialize the particle
    particle_likelihood=np.array([-9999999])
    particle_position=np.zeros(paramCount)
    particle_velocity=np.zeros(paramCount)
    best_particle_likelihood=particle_likelihood[0]    

    # 

    array_of_particle_position_proposal=parInit1[0:particleCount]
    array_of_particle_velocity_proposal=np.zeros((particleCount,paramCount))

    
    
    list_of_swarm=[]
    list_of_best_particle=[]
    
    list_of_particles=[]
    list_of_likelihood_of_particles=[]
    
    if analysis_type=='defocus':
        now = datetime.now()
        print('Starting work on the initial best result (defocus) at '+str(now))
        time_start_initial=time.time()
        best_result=Tokovinin_multi_instance_with_pool(array_of_polyfit_1_parameterizations_proposal_shape_2d,\
                                                       return_Images=True,num_iter=None,previous_best_result=None,\
                                                           use_only_chi=True)    
        time_end_initial=time.time()
        print('time for the initial evaluation is '+str(time_end_initial-time_start_initial))
    else:
        # if you are infocus
        now = datetime.now()
        print('Starting work on the initial best result (focus) at '+str(now))
        time_start_initial=time.time()
                
        best_result=model(allparameters_proposal_22,return_Image=False,use_only_chi=True)    
        time_end_initial=time.time()
        print('time for the initial evaluation is '+str(time_end_initial-time_start_initial))
        
   

    if analysis_type=='defocus':    
        # changed in version 0.45
        #print('Initial evaluation output is: '+str(best_result[0]))
        # the best likelihood after evaluation is now at stop 1 
        # 0 shows likelihood before evaluation
        print('Initial evaluation output is: '+str(best_result[1]))
    else:
        # 0 is where the likelihood coming from single model
        print('Initial evaluation output is: '+str(best_result[0]))
    print('Finished work on the initial best result') 
    
    array_of_particle_pbest_position=np.zeros((particleCount,paramCount))
    array_of_particle_pbest_likelihood=np.ones((particleCount,1))*-9999999
    
    np.save(RESULT_FOLDER+'initial_best_result_'+str(single_number)+str(arc)+str(eps),best_result)
    
    if analysis_type=='defocus':    
        array_of_multi_background_factors=np.concatenate((np.arange(80,3,-(80-3)/(nsteps-1)),np.array([3])) )
    else:
        array_of_multi_background_factors=np.ones((nsteps))*3

        
    
    for step in range(nsteps):
        
        #np.save(RESULT_FOLDER+'best_result'+str(step),best_result)
        #np.save(RESULT_FOLDER+'array_of_particle_position_proposal'+str(step),array_of_particle_position_proposal)         
  
        now = datetime.now()
        print('############################################################')
        print('Starting step '+str(step)+' out of '+str(nsteps)+' at '+str(now))
        print('array_of_multi_background_factors[step]: '+str(array_of_multi_background_factors[step]))
        time_start_evo_step=time.time()
        print('pool is: '+str(pool))
        # code that moves Zernike parameters via small movements
        # each particle is evaluated individually
        # but the changes of Zernike parameters are assumed from the previous best result
        # reports likelihod, but minimizes absolute different of images/std - model/std
        #out1=pool.map(partial(Tokovinin_multi_instance_without_pool, return_Images=True,\
        #                      previous_best_result=best_result[-1],use_only_chi=True,\
        #                      multi_background_factor=array_of_multi_background_factors[step]), array_of_particle_position_proposal)  
        
        if analysis_type=='defocus':           
            out1=pool.map(partial(Tokovinin_multi_instance_without_pool, return_Images=True,\
                                  previous_best_result=best_result,use_only_chi=True,\
                                  multi_background_factor=array_of_multi_background_factors[step]), array_of_particle_position_proposal)  
            out2=np.array(list(out1))
                
            time_end_evo_step=time.time()

            print('time for the evaluation of initial Tokovinin_multi_instance_without_pool in step '+str(step)+\
            ' is '+str(time_end_evo_step-time_start_evo_step))   
        
        else:
            #list_of_particle_postion_focus_proposal=[]
            #for l in range(len(array_of_particle_position_proposal)):
            #    list_of_minchain=model_multi.create_list_of_allparameters(array_of_particle_position_proposal[0],\
            #                                                          list_of_defocuses=['0'],zmax=56)
            #    list_of_particle_postion_focus_proposal.append(list_of_minchain)
            #array_of_particle_postion_focus_proposal=np.array(list_of_particle_postion_focus_proposal)
            
            out1=pool.map(partial(model, return_Image=True,use_only_chi=True), array_of_particle_position_proposal) 
            out2=np.array(list(out1))

            time_end_evo_step=time.time()

            print('time for the evaluation of initial partial evulation over model in step '+str(step)+\
            ' is '+str(time_end_evo_step-time_start_evo_step))   

        #print('out2.shape '+str(out2.shape))
        #np.save(RESULT_FOLDER+'out2'+str(step),out2)

        ##################
        # create a swarm here  
        # swarm contains each particle - 0. likelihood, 1. position (after computation), 
        # 2. velocity (that brought you here, i.e., before computation)
        # initialized with outputs from Tokovinin_multi_instance_without_pool

        
        swarm_list=[]
        for i in range(particleCount):
            
            if analysis_type=='defocus':  
                # old version - 0, new version, after 0.45 - 1
                likelihood_single_particle=out2[i][1]
                # old version - 3/4, new version, after 0.45 - 8
                position_single_particle=out2[i][8]
            else:
                likelihood_single_particle=out2[i][0]
                position_single_particle=out2[i][2]
            
            if step==0:
                velocity_single_particle=array_of_particle_velocity_proposal[i]
            else:
            # Code analyis tool is giving a warning
            # but, it is ok, because by the time you reach this line in second iteration, swarm will be defined
                velocity_single_particle=swarm[:,2][i]

                
            swarm_list.append([likelihood_single_particle,position_single_particle,velocity_single_particle])
            
        swarm=np.array(swarm_list)
        #np.save(RESULT_FOLDER+'swarm'+str(step),swarm)  

        ##################
        # find the best particle in the swarm           
        swarm_likelihood=swarm[:,0]
    
        index_of_best=np.arange(len(swarm_likelihood))[swarm_likelihood==np.max(swarm_likelihood)][0]
        best_particle_this_step=swarm[index_of_best] 
        #print('best_particle_this_step'+str(best_particle_this_step))
        best_particle_likelihood_this_step=best_particle_this_step[0]
        #np.save(RESULT_FOLDER+'best_particle_this_step'+str(step),best_particle_this_step)  
        print('best_particle_likelihood until now, step '+str(step)+': '+str(best_particle_likelihood))            
        print('proposed best_particle_likelihood_this_step, step '+str(step)+': '+str(best_particle_likelihood_this_step))
        
        if analysis_type=='defocus': 
            
            # until 0.45, if the best particle from the swarm is suspected to be best particle overall we would compute the best_result 
            # if best_particle_likelihood_this_step>best_particle_likelihood:
            # testing in 0.45
            if best_particle_likelihood_this_step>-9999:
    
                best_particle_proposal=best_particle_this_step
                best_particle_proposal_position=best_particle_proposal[1]
                
                # update best_result
                # we are computing if the best solution is actually as good as proposed solution
                print('Starting work on the new proposed best result in step '+str(step))
                time_start_best=time.time()
                best_result=Tokovinin_multi_instance_with_pool(best_particle_proposal_position,\
                                                               return_Images=True,num_iter=step,previous_best_result=None,
                                                               use_only_chi=True,multi_background_factor=array_of_multi_background_factors[step])   
                time_end_best=time.time()
                print('Time for the best evaluation in step '+str(step)+' is '+str(time_end_best-time_start_best))
    
                best_particle_likelihood_this_step=best_result[1]
                best_particle_position_this_step=best_result[8]
                
                print('Best result output is: '+str(best_particle_proposal_position[:5]))
                print('best_particle_likelihood until now: '+str(best_particle_likelihood))            
                print('final best_particle_likelihood_this_step: '+str(best_particle_likelihood_this_step))
    
                # until 0.45 if the particle is better than previous best result, replace the best particle
                
                # test in 0.45 - do it always
                #if best_particle_likelihood_this_step>best_particle_likelihood:
                if best_particle_likelihood_this_step>-9999:
                    best_particle_likelihood=best_particle_likelihood_this_step
                    best_particle_position=best_particle_position_this_step
                    # likelihood of the best particle, its position and velocity is zero
                    best_particle=[best_particle_likelihood,best_particle_position,np.zeros(paramCount)]
            else:
                # in 0.45 this should never happen unless the evaluation fails
                print('Proposed solution is worse than current, so do not evaluate')
        
        else:
            if best_particle_likelihood_this_step>best_particle_likelihood:
                
                best_particle_likelihood_this_step=best_particle_this_step[0]
                best_particle_position_this_step=best_particle_this_step[1]
                
                best_particle_likelihood=best_particle_likelihood_this_step
                best_particle_position=best_particle_position_this_step
                # likelihood of the best particle, its position and velocity is zero
                best_particle=[best_particle_likelihood,best_particle_position,np.zeros(paramCount)]
                
                
                
                
        # save result of the search for the best particle and result
        #np.save(RESULT_FOLDER+'best_particle'+str(step),best_particle)   
        #np.save(RESULT_FOLDER+'best_result'+str(step),best_result)    
    
    
        best_particle_position=best_particle[1]
        list_of_best_particle.append(best_particle)      

        # updating velocity    
        list_of_particle_position_proposal=[]
        for i in range(particleCount):
            

            particle_likelihood=np.array(swarm[i][0])
            particle_position=np.array(swarm[i][1])
            
            
            # if the particle failed (because it went outside global range), reduce its velocity
            #if particle_likelihood<=-9999999:
            #    particle_velocity=np.array(swarm[i][2])/5
            #else:
            #    particle_velocity=np.array(swarm[i][2])  
              
            particle_velocity=np.array(swarm[i][2]) 

            #w = 0.5 + np.random.uniform(0,1,size=paramCount)/2
            w=0.51
            part_vel = w * particle_velocity
            # cog_vel set at 0 at the moment
            #cog_vel = c1 * numpy.random.uniform(0,1,size=paramCount) * (particle.pbest.position - particle.position)   
            cog_vel = c1 * np.random.uniform(0,1,size=paramCount) * (particle_position - particle_position)         
    
            soc_vel = c2 * w * (best_particle_position - particle_position)
            
            
            proposed_particle_velocity = part_vel + cog_vel + soc_vel
            #print('part_vel, cog_vel,soc_vel:' +str([part_vel[0:5],cog_vel[0:5],soc_vel[0:5]]))
            
            # propose the position for next step and check if it is within limits
            # modify if it tries to venture outside limits
            proposed_particle_position = particle_position + proposed_particle_velocity
            
            if multi_var==False or multi_var==None:
                proposed_global_parameters=proposed_particle_position[19:19+23]
                checked_global_parameters=check_global_parameters(proposed_global_parameters)
            
                # warn if the checking algorithm changes one of the parameters
                #if proposed_global_parameters!=checked_global_parameters:
                    #    print('checked_global_parameters are different from the proposed ones ')
            
                new_particle_position=np.copy(proposed_particle_position)
                new_particle_position[19:19+23]=checked_global_parameters                
                
            else:
                proposed_global_parameters=proposed_particle_position[38:38+23]
                checked_global_parameters=check_global_parameters(proposed_global_parameters)
            
                # warn if the checking algorithm changes one of the parameters
                #if proposed_global_parameters!=checked_global_parameters:
                    #    print('checked_global_parameters are different from the proposed ones ')
            
                new_particle_position=np.copy(proposed_particle_position)
                new_particle_position[38:38+23]=checked_global_parameters
            
            particle_velocity=new_particle_position-particle_position            
            
            
            # update velocities here
            swarm[i][2]=particle_velocity
            # create list/array of new proposed positions
            list_of_particle_position_proposal.append(new_particle_position)
    
 
        list_of_swarm.append(swarm)
        array_of_particle_position_proposal=np.array(list_of_particle_position_proposal)
        time_end=time.time()
        print('time for the whole evaluation until step '+str(step)+' is '+str(time_end-time_start))
        
    
    gbests=np.array(list_of_best_particle) 
    swarms=np.array(list_of_swarm)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

               
                
                
                
                
                
                
                
                
                
                
                
                
    print('Managed to reach this point')    
    
    minchain=gbests[-1][1]
    minln=gbests[-1][0]
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][1])     
    res=np.array(res)
    chains=res.reshape(len(swarms),len(swarms[0]),parInit1.shape[1])
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][0])         
    res=np.array(res)
    ln_chains=res.reshape(len(swarms),len(swarms[0]))
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][2])         
    res=np.array(res)
    v_chains=res.reshape(len(swarms),len(swarms[0]),parInit1.shape[1])
    
    res_gbests_position=[]
    res_gbests_fitness=[]
    for i in range(len(swarms)):
            minchain_i=gbests[i][1]
            minln_i=gbests[i][0]
            res_gbests_position.append(minchain_i)
            res_gbests_fitness.append(minln_i)
        
    
    
    #Export this fileparInit1
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'parInit1',parInit1)    
    
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Swarm1',chains)
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'velocity_Swarm1',v_chains)
    #np.save(RESULT_FOLDER+NAME_OF_CHAIN+'list_of_Tokovin_results',list_of_Tokovin_results)    
    
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'res_gbests_position',res_gbests_position)
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'res_gbests_fitness',res_gbests_fitness)
    
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Swarm1',ln_chains) 
       
    
    
    #minchain=chains[np.abs(ln_chains)==np.min(np.abs(ln_chains))][0]
    chi2reduced=2*np.min(np.abs(ln_chains))/(sci_image.shape[0])**2
    minchain_err=[]
    for i in range(len(minchain)):
        minchain_err=np.append(minchain_err,np.std(chains[:,:,i].flatten()))
    
    minchain_err=np.array(minchain_err)
    
    #np.save(RESULT_FOLDER+NAME_OF_CHAIN+'_list_of_swarms_time_1',list_of_swarms_time_1)    
    
    print('Likelihood atm: '+str(np.abs(minln)))
    print('minchain atm: '+str(minchain))
    print('minchain_err: '+str(minchain_err))
    print('Time when first swarm run finished was: '+time.ctime())     
    time_end=time.time()   
    print('Time taken was '+str(time_end-time_start)+' seconds')
    
    list_of_times.append(time_end-time_start)
    sys.stdout.flush()

    
    
    sys.stdout.flush()
    pool.close()
    sys.exit(0)
    
# should update that it goes via date
