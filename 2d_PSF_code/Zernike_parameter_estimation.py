"""
Created on Mon Mar 30 2020

@author: Neven Caplar
@contact: ncaplar@princeton.edu

Mar 31, 2020: ? -> 0.28 added argparser and extra Zernike


"""

#standard library imports
from __future__ import absolute_import, division, print_function
import socket
import time
print(str(socket.gethostname())+': Start time for importing is: '+time.ctime()) 
import sys
import os 
import argparse
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
#from schwimmbad import MultiPool
#import mpi4py

#Local application/library specific imports

#pandas
import pandas

#cosmohammer
import cosmoHammer

#galsim
import galsim

#emcee
import emcee

#Zernike_Module
from Zernike_Module import LNP_PFS,LN_PFS_single,create_parInit,add_pupil_parameters_to_all_parameters,PFSLikelihoodModule,svd_invert

parser = argparse.ArgumentParser(description="Starting args import",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog='Done with import')

############################################################################################################
print('############################################################################################################')  

############################################################################################################
print('Start time is: '+time.ctime())  
time_start=time.time()  

################################################
parser.add_argument("obs", help="name of the observation (actually a number) which we will analyze",type=int,default=argparse.SUPPRESS)
#parser.add_argument('--obs',dest='obs',default=None)
################################################
# name of the spot (again, actually number) which we will analyze
parser.add_argument("spot", help="name of the spot (again, actually number) which we will analyze",type=int)
################################################
# number of steps each walkers will take 
parser.add_argument("nsteps", help="number of steps each walkers will take ",type=int)
################################################
# input argument that controls the paramters of the cosmo_hammer process
# if in doubt, eps=5 is probably a solid option
parser.add_argument("eps", help="input argument that controls the paramters of the cosmo_hammer process; if in doubt, eps=5 is probably a solid option ",type=int)
################################################    
# which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5]   
parser.add_argument("dataset", help="which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5] ",type=int, choices=[0, 1, 2,3,4,5])
################################################    
parser.add_argument("arc", help="which arc lamp is being analyzed (HgAr for Mercury-Argon, Ne for Neon, Kr for Krypton)  ", choices=["HgAr", "Ne", "Kr"])
################################################ 
parser.add_argument("double_sources", help="are there two sources in the image (1==yes, 0==no) ",type=int, choices=[0,1])
################################################ 
parser.add_argument("double_sources_positions_ratios", help="parameters for second source ",action='append')        
################################################        
parser.add_argument("twentytwo_or_extra", help="number of Zernike components (22 or nubmer larger than 22 which leads to extra Zernike analysis)",type=int)             
################################################       
parser.add_argument("date_of_input", help="input date")   
################################################       
parser.add_argument("direct_or_interpolation", help="direct or interpolation ", choices=["direct", "interpolation"])   
################################################   
parser.add_argument("date_of_output", help="date_of_output ", )       
################################################  
 
 # Finished with specifying arguments     
 
################################################  
# Assigning arguments to variables
args = parser.parse_args()
################################################ 
obs=args.obs
print('obs is: '+str(obs))  
################################################ 
single_number= args.spot
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
    options=[390,1.593,1.193]
if eps==7:
    options=[390,4.93,1.193]
if eps==8:
    options=[390,1.193,4.193]
if eps==9:
    options=[190,1.193,1.193]
    nsteps=int(2*nsteps)
if eps==10:
    options=[390,1.893,2.893]
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
    DATA_FOLDER='/tigress/ncaplar/Data/Data_May_28/'

# folder containing the data taken with F/2.8 stop in April and May 2019
# dataset 3
if dataset==3:
    DATA_FOLDER='/tigress/ncaplar/Data/Data_Jun_25/'

# folder containing the data taken with F/2.8 stop in July 2019
# dataset 4 (defocu) and 5 (fine defocus)
if dataset==4 or dataset==5:
    DATA_FOLDER='/tigress/ncaplar/Data/Data_Aug_14/'        

STAMPS_FOLDER=DATA_FOLDER+'Stamps_Cleaned/'
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
################################################  
double_sources=args.double_sources
if double_sources==0:
    double_sources=None
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
 
# Finished with all arguments

################################################  
# import data
# 8600 will also need to be changed - this is to recognize dithered images...

if obs==8600:
    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked_Dithered.npy')
else:
    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
    mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
    sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
    var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')

# If there is no science image, exit 
if int(np.sum(sci_image))==0:
    print('No science image - exiting')
    sys.exit(0)

if np.mean(mask_image)>0:
    print('Cosmics in the image - but we are using 0.21e or higher version of the code that can handle it')
if np.mean(mask_image)>0.1:
    print('but '+str(np.mean(mask_image)*100)+'% of image is masked... when it is more than 10% - exiting')
    sys.exit(0)


# name of the outputs
# should update that it goes via date
NAME_OF_CHAIN='chain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
NAME_OF_LIKELIHOOD_CHAIN='likechain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)

"""
# this was to specify exact pupil parameters
if eps==99:
    pupil_parameters=np.load(RESULT_FOLDER+NAME_OF_PUPIL_RES+'pupil_parameters')
    print("Pupil parameters found!")
else:
    print("Running analysis without specified pupil parameters") 
    pupil_parameters=None

#pupil_parameters=[0.65,0.1,0.,0.,0.08,0,0.99,0.0,1,0.04,1,0]
"""

# where are the dataframe which we use to guess the initial solution

# F/3.2 data
#with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_HgAr_from_Apr15.pkl', 'rb') as f:
#    results_of_fit_many_interpolation_preDecemberrun_HgAr=pickle.load(f)
#with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_Ne_from_Apr15.pkl', 'rb') as f:
#    results_of_fit_many_interpolation_preDecemberrun_Ne=pickle.load(f)

# where are the dataframe which we use to guess the initial solution
# should update that it goes via date input data and direct/interpolation
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_HgAr_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_HgAr=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Ne_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_Ne=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Kr_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_Kr=pickle.load(f)
##############################################    
    
# What are the observations that can be analyzed
# used to associate observation with their input labels, so that the initial parameters guess is correct    
    
# December 2017 data    
#if arc=='HgAr':
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
#elif arc=='Ne':
#    print('Neon?????')
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])+90
    
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
        obs_possibilites=np.array([17023,17023+6,17023+12,17023+18,17023+24,17023+30,17023+36,17023+42,17023+48,17023+48,17023+54,17023+60,17023+66,17023+72,17023+78,17023+84,17023+90,17023+96,17023+48])
    if arc=='Ne':
        # different sequence than for HgAr
        obs_possibilites=np.array([16238+6,16238+12,16238+18,16238+24,16238+30,16238+36,16238+42,16238+48,16238+54,16238+54,16238+60,16238+66,16238+72,16238+78,16238+84,16238+90,16238+96,16238+102,16238+54])
    if arc=='Kr':
         obs_possibilites=np.array([17310+6,17310+12,17310+18,17310+24,17310+30,17310+36,17310+42,17310+48,17310+54,17310+54,17310+60,17310+66,17310+72,17310+78,17310+84,17310+90,17310+96,17310+102,17310+54])

# F/2.5 data     
if dataset==3:
    if arc=='HgAr':
        obs_possibilites=np.array([19238,19238+6,19238+12,19238+18,19238+24,19238+30,19238+36,19238+42,19238+48,19238+48,19238+54,19238+60,19238+66,19238+72,19238+78,19238+84,19238+90,19238+96,19238+48])
    elif arc=='Ne':
    # different sequence than for HgAr
        obs_possibilites=np.array([19472+6,19472+12,19472+18,19472+24,19472+30,19472+36,19472+42,19472+48,19472+54,19472+54,19472+60,19472+66,19472+72,19472+78,19472+84,19472+90,19472+96,19472+102,19472+54])    

# F/2.8 July data        
if dataset==4:
    if arc=='HgAr':
        obs_possibilites=np.array([21346+6,21346+12,21346+18,21346+24,21346+30,21346+36,21346+42,21346+48,21346+54,21346+54,21346+60,21346+66,21346+72,21346+78,21346+84,21346+90,21346+96,21346+102,21346+48])
    if arc=='Ne':
        obs_possibilites=np.array([21550+6,21550+12,21550+18,21550+24,21550+30,21550+36,21550+42,21550+48,21550+54,21550+54,21550+60,21550+66,21550+72,21550+78,21550+84,21550+90,21550+96,21550+102,21550+54])
    if arc=='Kr':
         obs_possibilites=np.array([21754+6,21754+12,21754+18,21754+24,21754+30,21754+36,21754+42,21754+48,21754+54,21754+54,21754+60,21754+66,21754+72,21754+78,21754+84,21754+90,21754+96,21754+102,21754+54])
        
 ##############################################    


# associates each observation with the label in the supplied dataframe
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28,0])
z4Input=z4Input_possibilites[obs_possibilites==obs][0]
label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']
labelInput=label[list(obs_possibilites).index(obs)]

# Input the zmax that you wish to achieve in the analysis
zmax=22
 
columns=['z4','z5','z6','z7','z8','z9','z10','z11',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  

columns22=['z4','z5','z6','z7','z8','z9','z10','z11',
           'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
          'x_fiber','y_fiber','effective_radius_illumination',
          'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
          'grating_lines','scattering_slope','scattering_amplitude',
          'pixel_effect','fiber_r','flux']  

# I BELIEVE THIS CAN BE deleted
# just for the first run when using F/3.2 data, we need to connect the new data and the old data 
# depening on the arc, select the appropriate dataframe
"""
if arc=="HgAr":
    with open(DATAFRAMES_FOLDER + 'finalHgAr_Feb2019.pkl', 'rb') as f:
        finalHgAr_Feb2019=pickle.load(f) 

    single_number_new=int(finalHgAr_Feb2019.loc[int(single_number)]['old_index_approx'])
    single_number=single_number_new
    print('approx. old number is: '+str(single_number))

if arc=="Ne":
    with open(DATAFRAMES_FOLDER + 'finalNe_Feb2019.pkl', 'rb') as f:
        finalNe_Feb2019=pickle.load(f) 

    single_number_new=int(finalNe_Feb2019.loc[int(single_number)]['old_index_approx'])
    single_number=single_number_new
    print('approx. old number is: '+str(single_number))
"""

# depening on the arc, select the appropriate dataframe
if arc=="HgAr":
    results_of_fit_many_interpolation_preDecemberrun=results_of_fit_many_interpolation_preDecemberrun_HgAr
elif arc=="Ne":
    results_of_fit_many_interpolation_preDecemberrun=results_of_fit_many_interpolation_preDecemberrun_Ne  
elif arc=="Kr":
    results_of_fit_many_interpolation_preDecemberrun=results_of_fit_many_interpolation_preDecemberrun_Kr  
else:
    print("what has happened here? Only HgAr, Neon and Krypton implemented")
    
# do analysis with up to 22 or 11 zernike
allparameters_proposalp2=results_of_fit_many_interpolation_preDecemberrun[labelInput].loc[int(single_number)].values
    
if zmax==22: 
    # if input created with z11
    if len(allparameters_proposalp2)==33:
        allparameters_proposal=np.concatenate((allparameters_proposalp2[0:8],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],allparameters_proposalp2[8:-2]))
    # if input created with z22
    else:
        allparameters_proposal=allparameters_proposalp2[:len(columns22)]
if zmax==11:
    # if input created with z11
    allparameters_proposal=allparameters_proposalp2[:,-2]
    
# not sure, but I guess to protect it from chaning?
allparameters_proposal_22=np.copy(allparameters_proposal)

# I BELIEVE THIS CAN BE deleted
# resets fiber values
# not needed any more in the second run
#allparameters_proposal_22[np.array(columns)=='radiometricEffect']=0.2
#allparameters_proposal_22[np.array(columns)=='radiometricExponent']=1
#allparameters_proposal_22[np.array(columns)=='x_fiber']=0
#allparameters_proposal_22[np.array(columns)=='y_fiber']=0
#allparameters_proposal_22[np.array(columns)=='effective_radius_illumination']=0.85
#allparameters_proposal_22[np.array(columns)=='frd_sigma']=0.1
#allparameters_proposal_22[np.array(columns)=='frd_lorentz_factor']=0.5
#allparameters_proposal_22[np.array(columns)=='fiber_r']=1.8


# "lower_limits" and "higher_limits" are only needed to initalize the cosmo_hammer code, but not used in the actual evaluation
# so these are purely dummy values
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
    
if zmax==22:
    lower_limits=np.array([z4Input-3,-1,-1,-1,-1,-1,-1,-1,
                           0,0,0,0,0,0,0,0,0,0,0,
                 0.5,0.05,-0.8,-0.8,0,-0.5,
                 0,-0.5,0.5,0.5,
                 0.8,-np.pi/2,0.5,
                 0,0,0.85,-0.8,
                 1200,0.5,0,
                 0.2,1.65,0.9])
    
    higher_limits=np.array([z4Input+3,1,1,1,1,1,1,1,
                            0,0,0,0,0,0,0,0,0,0,0,
                  1.2,0.2,0.8,0.8,0.2,0.5,
                  3,20,1.5,1.5,
                  1,np.pi/2,1.01,
                  0.05,1,1.15,0.8,
                  120000,3.5,0.5,
                  1.1,1.95,1.1])    
 
    

        
    # Create an object called model - which gives likelihood given the parameters 
    pupil_parameters=None
    # if using dithered, which we only did with December 2017 data
    if obs==8600:
        model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,dithering=2,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax)
    else:
        model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
                              pupil_parameters=pupil_parameters,use_pupil_parameters=None,\
                              save=0,verbosity=0,double_sources=double_sources,zmax=zmax,\
                              double_sources_positions_ratios=double_sources_positions_ratios,npix=1536)
    
    # always returns 0   (did I only use this for multiple temperatures code? - if yes, this can be deleted)
    modelP =LNP_PFS(sci_image,var_image)
    
# branch out here    
if twentytwo_or_extra==22:      
    # initialize pool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)   
    
    #pool=Pool(processes=36)
    #pool=Pool()
       
    print('Name of machine is '+socket.gethostname())    
        
    
    zmax_input=22
    
    print('Spot coordinates are: '+str(single_number))  
    print('Size of input image is: '+str(sci_image.shape)) 
    print('First value of the sci image: '+str(sci_image[0][0]))
    print('First value of the var image: '+str(var_image[0][0]))
    print('Steps: '+str(nsteps)) 
    print('Name: '+str(NAME_OF_CHAIN)) 
    print('Zmax is: '+str(zmax))
    print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())    
    
    print('Testing proposal is: '+str(allparameters_proposal_22))
    time_start_single=time.time()
    print('Likelihood for the testing proposal is: '+str(model(allparameters_proposal_22)))
    time_end_single=time.time()
    print('Time for single calculation is '+str(time_end_single-time_start_single))
    
    allparameters_proposal=allparameters_proposal_22
    # dirty hack to get things running
    if zmax==22:
        columns=columns22
    
    sys.stdout.flush()
    
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
    
    #columns=['z4','z5','z6','z7','z8','z9','z10','z11',
    #         'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
    #         'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
    #         'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
    #         'x_fiber','y_fiber','effective_radius_illumination',
    #         'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
    #         'grating_lines','scattering_slope','scattering_amplitude',
    #         'pixel_effect','fiber_r','flux']  
    
    # change this array, depening on how do you want to proced 
    # zero freedom in radiometric parameters
    # full freedom in wavefront parameters
    # less freedom in all other paramters
    stronger_array_01=np.array([1,1,1,1,1,1,1,1,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 
                            0.2,0.2,0.2,0.2,0.2,0.2,
                            0.0,0.0,0.0,0.0,
                            0.2,0.2,0.2,
                            0.2,0.2,0.2,0.2,
                            0.2,0.2,0.2,
                            0,0,1])
    
    # create inital parameters  
    parInit1=create_parInit(allparameters_proposal=allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=1*stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    # the number of walkers is given by options array
    while len(parInit1)<options[0]:
        parInit1_2=create_parInit(allparameters_proposal=allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=1*
                                  stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
        parInit1=np.vstack((parInit1,parInit1_2))
    
    chain = cosmoHammer.LikelihoodComputationChain()
    chain.addLikelihoodModule(PFSLikelihoodModule(model))
    
    pso = cosmoHammer.ParticleSwarmOptimizer(chain, low=lower_limits, high=higher_limits,particleCount=options[0],pool=pool)
    
    for i in range(len(pso.swarm)):
        pso.swarm[i].position=parInit1[i]
    
    #returns all swarms and the best particle for all iterations
    #swarms, gbests = pso.optimize(maxIter=nsteps,c1=options[1],c2=options[2])
    
    swarms=[]
    gbests=[]
    num_iter = 0
    for swarm in pso.sample(nsteps):
        swarms.append(swarm)
        gbests.append(pso.gbest.copy())
        num_iter += 1
        if pso.isMaster():
            if num_iter % 2 == 0:
                print("First swarm: "+str(100*num_iter/nsteps))
                sys.stdout.flush()
        
    
    minchain=gbests[-1].position
    minln=gbests[-1].fitness
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].position)
            
    res=np.array(res)
    chains=res.reshape(len(swarms),len(swarms[0]),len(lower_limits))
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].fitness)
            
    res=np.array(res)
    ln_chains=res.reshape(len(swarms),len(swarms[0]))
    
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Swarm1',chains)
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Swarm1',ln_chains) 
          
    
    #minchain=chains[np.abs(ln_chains)==np.min(np.abs(ln_chains))][0]
    chi2reduced=2*np.min(np.abs(ln_chains))/(sci_image.shape[0])**2
    minchain_err=[]
    for i in range(len(columns)):
        minchain_err=np.append(minchain_err,np.std(chains[:,:,i].flatten()))
    
    minchain_err=np.array(minchain_err)
    
    
    
    print('Likelihood atm: '+str(np.abs(minln)))
    print('minchain atm: '+str(minchain))
    print('minchain_err: '+str(minchain_err))
    print('Time when first swarm run finished was: '+time.ctime())     
    time_end=time.time()   
    print('Time taken was '+str(time_end-time_start)+' seconds')
    sys.stdout.flush()
    
    ##################################################################################################################### 
    # First emcee  - unfortunately the results are called ``second emcee'' due to historic reasons
    if pupil_parameters is not None:
        minchain=add_pupil_parameters_to_all_parameters(minchain,pupil_parameters)
    
    parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=5*stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    
    # if the generated length of the parameters is larger than size of swarm number of walkers divided by 2, cut the number of walkers
    # number_of_walkers has to be an even number
    number_of_walkers_cut= int(options[0]/2)
    if (number_of_walkers_cut % 2) == 0:
        pass
    else:
        number_of_walkers_cut=number_of_walkers_cut+1
    
    if len(parInit1) > number_of_walkers_cut:      
        parInit1=parInit1[0:number_of_walkers_cut]
    
    
    sampler = emcee.EnsembleSampler(parInit1.shape[0],parInit1.shape[1],model,pool=pool)
    
    for i, result in enumerate(sampler.sample(parInit1, iterations=nsteps)):
        print('First/second emcee run: '+"{0:5.1%}\r".format(float(i) / nsteps)),
        sys.stdout.flush()
    
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee2',sampler.chain)
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee2',sampler.lnprobability) 
          
    # Create minchain and save it
    likechain0=sampler.lnprobability
    chain0=sampler.chain
    minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]  
    #np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)    
    
    print('Likelihood atm: '+str(np.min(np.abs(likechain0))))
    print('minchain atm: '+str(minchain))
    
    
    #chi2reduced=2*np.min(np.abs(likechain0))/(sci_image.shape[0])**2
    
    minchain_err=[]
    for i in range(len(columns)):
        #minchain_err=np.append(minchain_err,np.std(chain0[:,:,i].flatten()))
        minchain_err=np.append(minchain_err,np.std(chain0[:,:,i].flatten()))
    
    minchain_err=np.array(minchain_err)
    #print('chi2reduced: '+str(chi2reduced))
    print('minchain_err: '+str(minchain_err))
    #lower_limits_inter=minchain-minchain_err
    #higher_limits_inter=minchain+minchain_err
    
    #lower_limits_2=np.maximum(lower_limits,lower_limits_inter)
    #higher_limits_2=np.minimum(higher_limits,higher_limits_inter)
    
    #print('lower_limits: '+str(lower_limits_2))
    #print('higher_limits: '+str(higher_limits_2))
    
    print('Time when second emcee run finished was: '+time.ctime())     
    time_end=time.time()   
    print('Time taken was '+str(time_end-time_start)+' seconds')
    
    sys.stdout.flush()
    
    ##################################################################################################################### 
    # Second swarm  
    parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    while len(parInit1)<options[0]:
        parInit1_2=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
        parInit1=np.vstack((parInit1,parInit1_2))
    
    chain = cosmoHammer.LikelihoodComputationChain()
    chain.addLikelihoodModule(PFSLikelihoodModule(model))
    
    pso = cosmoHammer.ParticleSwarmOptimizer(chain, low=lower_limits, high=higher_limits,particleCount=options[0],pool=pool)
    #returns all swarms and the best particle for all iterations
    for i in range(len(pso.swarm)):
        pso.swarm[i].position=parInit1[i]
    
    #returns all swarms and the best particle for all iterations
    #swarms, gbests = pso.optimize(maxIter=4*nsteps,c1=options[1],c2=options[2])
    
    swarms=[]
    gbests=[]
    num_iter = 0
    for swarm in pso.sample(3*nsteps):
        swarms.append(swarm)
        gbests.append(pso.gbest.copy())
        num_iter += 1
        if pso.isMaster():
            if num_iter % 2 == 0:
                print("Last swarm: "+str(100*num_iter/(3*nsteps)))
                sys.stdout.flush()
     
        # if already taken more than 40 steps, and not moved in 5 steps, break
        if num_iter>40:
            
            if gbests[-1].fitness==gbests[-6].fitness:
                break
               
    minchain=gbests[-1].position
    minln=gbests[-1].fitness
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].position)
            
    res=np.array(res)
    chains=res.reshape(len(swarms),len(swarms[0]),len(lower_limits))
    
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].fitness)
            
    res=np.array(res)
    ln_chains=res.reshape(len(swarms),len(swarms[0]))
    
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Swarm2',chains)
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Swarm2',ln_chains) 
    
    #minchain=chains[np.abs(ln_chains)==np.min(np.abs(ln_chains))][0]
    chi2reduced=2*np.min(np.abs(ln_chains))/(sci_image.shape[0])**2
    
    if np.sqrt(chi2reduced)>10:
        chi2reduced_multiplier=10
    else:
        chi2reduced_multiplier=np.sqrt(chi2reduced)    
    
    minchain_err=[]
    for i in range(len(columns)):
        minchain_err=np.append(minchain_err,chi2reduced_multiplier*np.std(chains[:,:,i].flatten()))
    
    minchain_err=np.array(minchain_err)
    
    print('Likelihood atm: '+str(np.abs(gbests[-1].fitness)))
    print('minchain atm: '+str(minchain))
    print('minchain_err: '+str(minchain_err))
    print('Time when last swarm run finished was: '+time.ctime())     
    time_end=time.time()   
    print('Time taken was '+str(time_end-time_start)+' seconds')
    
    sys.stdout.flush()
    ##################################################################################################################### 
    # Second emcee   - unfortunately the results are called ``third emcee'' due to historic reasons
        
    if pupil_parameters is not None:
        minchain=add_pupil_parameters_to_all_parameters(minchain,pupil_parameters)
    
    parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=10*stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    
    # if the generated length of the parameters is larger than size of swarm number of walkers divided by 2, cut the number of walkers
    # number_of_walkers has to be an even number
    number_of_walkers_cut= int(options[0]/2)
    if (number_of_walkers_cut % 2) == 0:
        pass
    else:
        number_of_walkers_cut=number_of_walkers_cut+1
    
    #print(parInit1[0])
    sampler = emcee.EnsembleSampler(parInit1.shape[0],parInit1.shape[1],model,pool=pool)
    for i, result in enumerate(sampler.sample(parInit1, iterations=int(nsteps/2))):
        print('Last emcee run: '+"{0:5.1%}\r".format(float(i) /(nsteps))),
        sys.stdout.flush()
    #sampler.run_mcmc(parInit1, 2*nsteps)  
        
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee3',sampler.chain)
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee3',sampler.lnprobability) 
    
    # Create minchain and save it
    likechain0=sampler.lnprobability
    chain0=sampler.chain
    minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]  
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)   
    
    print('Time when total script finished was: '+time.ctime())     
    time_end=time.time()   
    print('Total time taken was  '+str(time_end-time_start)+' seconds')
    
    print('Likelihood final: '+str(np.min(np.abs(likechain0))))
    print('minchain final: '+str(minchain))
    
    sys.stdout.flush()
    pool.close()
    sys.exit(0)

else:
    
    from multiprocessing import Pool
    
    def model_return(allparameters_proposal):
        return model(allparameters_proposal,return_Image=True)    

    print('Testing proposal is: '+str(allparameters_proposal_22))
    time_start_single=time.time()
    Testing_chi2=(-2)*model(allparameters_proposal_22)/(sci_image.shape[0]*sci_image.shape[1])
    print('Likelihood for the testing proposal is: '+str(Testing_chi2))
    time_end=time.time()
    print('Total time taken was  '+str(time_end-time_start)+' seconds')    
    
    NAME_OF_allparameters='allparameters_proposal_after_iteration_'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    NAME_OF_final_optpsf='final_optpsf'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    
    
    pool=Pool(processes=34)
    print('Staring pool process')
    

    number_of_non_decreses=[0]
    for iteration_number in range(10):
        
        RESULT_DIRECTANDINT_FOLDER='/tigress/ncaplar/Result_DirectAndInt/'
        
        # implement that it is not dif of flux, but sigma too
        mask=sci_image>(np.max(sci_image)*0.1)
        
        # normalized science image
        sci_image_std=sci_image/np.sqrt(var_image)
        # initial SVD treshold
        thresh0 = 0.02
        
        # set number of extra Zernike
        #number_of_extra_zernike=56
        # numbers that make sense are 11,22,37,56,79,106,137,172,211,254
        number_of_extra_zernike=twentytwo_or_extra-22
        
        # initial list of how much to move Zernike coefficents
        list_of_delta_z=[]
        for z_par in range(3,22+number_of_extra_zernike):
            list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))
        #array_of_delta_z=np.array(list_of_delta_z)*np.random.choice([1, -1],len(list_of_delta_z))
        
        # array, randomized delta extra zernike
        array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*0.2+1
        array_of_delta_z=+np.array(list_of_delta_z)*array_of_delta_randomize
        # initialize 
        if iteration_number==0:
            
            # array, randomized delta extra zernike
            array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*0.2+1
            array_of_delta_z=+np.array(list_of_delta_z)*array_of_delta_randomize
            
            
            chi_2_before_iteration=Testing_chi2
            all_wavefront_z_old=np.concatenate((allparameters_proposal[0:19],np.zeros(number_of_extra_zernike)))
            #all_wavefront_z_old=np.concatenate((allparameters_proposal[0:19],allparameters_proposal[42:]))
            thresh=thresh0
            
            input_parameters=[]
            list_of_hashes=[]
            list_of_all_wavefront_z=[]
            
            up_to_z22_start=all_wavefront_z_old[0:19]     
            from_z22_start=all_wavefront_z_old[19:]  
    
            initial_input_parameters=np.concatenate((up_to_z22_start,allparameters_proposal[19:],from_z22_start))
            initial_hash=np.abs(hash(tuple(initial_input_parameters)))
            initial_model_result,image_0,initial_input_parameters=model_return(initial_input_parameters)
            #image_0=np.load(RESULT_DIRECTANDINT_FOLDER+'optPsf_cut_fiber_convolved_downsampled_'+str(initial_hash)+'.npy')
            #print('initial_input_parameters for loop '+str(iteration_number)+' are '+str(initial_input_parameters))
            print('var of initial_input_parameters for loop '+str(iteration_number)+' are '+str(np.var(initial_input_parameters[19:])))
            #print('initial_model_result is '+str(initial_model_result))
            
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_model_result_'+str(iteration_number),initial_model_result)
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/image_0_'+str(iteration_number),image_0)
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_input_parameters_'+str(iteration_number),initial_input_parameters)
    
            for z_par in range(len(all_wavefront_z_old)):
                all_wavefront_z_list=np.copy(all_wavefront_z_old)
                all_wavefront_z_list[z_par]=all_wavefront_z_list[z_par]+array_of_delta_z[z_par]
                list_of_all_wavefront_z.append(all_wavefront_z_list)
                up_to_z22_start=all_wavefront_z_list[0:19]     
                from_z22_start=all_wavefront_z_list[19:]  
                
                allparameters_proposal_int_22_test=np.concatenate((up_to_z22_start,allparameters_proposal[19:42],from_z22_start))
                input_parameters.append(allparameters_proposal_int_22_test)
                list_of_hashes.append(np.abs(hash(tuple(allparameters_proposal_int_22_test))))          
    
        else:
            chi_2_before_iteration=np.copy(chi_2_after_iteration)
            all_wavefront_z_old=np.copy(all_wavefront_z_new)
            if did_chi_2_improve==0:
                thresh=thresh0
            else:
                thresh=thresh*0.5
            
            # array, randomized delta extra zernike
            array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*0.2+1
            array_of_delta_z=+np.array(list_of_delta_z)*array_of_delta_randomize        
        
            input_parameters=[]
            list_of_ex_z=[]
            list_of_hashes=[]
            
            up_to_z22_start=np.copy(all_wavefront_z_old[:19])
            from_z22_start=np.copy(all_wavefront_z_old[19:])
            
            initial_input_parameters=np.concatenate((up_to_z22_start,allparameters_proposal[19:42],from_z22_start))
            initial_hash=np.abs(hash(tuple(initial_input_parameters)))
            initial_model_result,image_0,initial_input_parameters=model_return(initial_input_parameters)
            #image_0=np.load(RESULT_DIRECTANDINT_FOLDER+'optPsf_cut_fiber_convolved_downsampled_'+str(initial_hash)+'.npy')
            #print('input_parameters for loop '+str(iteration_number)+' are '+str(initial_input_parameters))      
            print('var of initial_input_parameters for loop '+str(iteration_number)+' are '+str(np.var(initial_input_parameters[19:])))

            
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_model_result_'+str(iteration_number),initial_model_result)
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/image_0_'+str(iteration_number),image_0)
            np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_input_parameters_'+str(iteration_number),initial_input_parameters)        
            
            
            for z_par in range(len(all_wavefront_z_old)):
                all_wavefront_z_list=np.copy(all_wavefront_z_old)
                all_wavefront_z_list[z_par]=all_wavefront_z_list[z_par]+array_of_delta_z[z_par]
                
                up_to_z22_start=all_wavefront_z_list[0:19]     
                from_z22_start=all_wavefront_z_list[19:]  
                
                list_of_all_wavefront_z.append(all_wavefront_z_list)
                allparameters_proposal_int_22_test=np.concatenate((up_to_z22_start,allparameters_proposal[19:42],from_z22_start))
                input_parameters.append(allparameters_proposal_int_22_test)
                #list_of_hashes.append(np.abs(hash(tuple(allparameters_proposal_int_22_test))))     
    
    
        # standard deviation image
        STD=np.sqrt(var_image )       
    
        # normalized and masked model image before this iteration
        M0=((image_0[mask])/np.sum(image_0[mask])).ravel()
       
       # normalized with STD and masked model image before this iteration
        #M0=((image_0[mask]*STD[mask])/np.sum(image_0[mask]*STD[mask])).ravel()
        
        # normalized and masked science image
        I=((sci_image[mask])/np.sum(sci_image[mask])).ravel()
        #I=((sci_image[mask]*STD[mask])/np.sum(sci_image[mask]*STD[mask])).ravel()
        #print('I[0]: '+str(I[0]))
        #print('M0[0]: '+str(M0[0]))
        #M0_STD=M0/STD
        print('np.sum(np.abs(I-M0)) before iteration '+str(iteration_number)+': '+str(np.sum(np.abs(I-M0))))  
        # used until March 11
        IM_start=np.sum(np.abs(I-M0))
        # used until March 11
        #IM_start_STD=np.sum(np.abs(I_STD-M0_STD))   
        
    
        
        out_ln=[]
        out_images=[]
        out_parameters=[]
        
        #print('len(input_parameters)'+str(len(input_parameters)))
        if __name__ == '__main__':
            print('We are inside of the pool loop number '+str(iteration_number)+' now')
            out1=pool.map(model_return,input_parameters)
            
    
    
            for i in range(len(out1)):
                out_ln.append(out1[i][0])
                out_images.append(out1[i][1])
                out_parameters.append(out1[i][2])
            
            
            
        #print('results after pool loop '+str(iteration_number)+' are '+str((-2)*np.array(out_ln)/(sci_image.shape[0]*sci_image.shape[1])))
        print('min result after pool loop '+str(iteration_number)+' are '+str(np.min((-2)*np.array(out_ln)/(sci_image.shape[0]*sci_image.shape[1]))))                 
         
        
        '''
        optpsf_list=[]
        single_wavefront_parameter_list=[]
        
        #print('len(list_of_hashes):'+str(len(list_of_hashes)))
        for i in range(len(list_of_hashes)):
            single_optpsf_image=np.load(RESULT_DIRECTANDINT_FOLDER+'optPsf_cut_fiber_convolved_downsampled_'+str(list_of_hashes[i])+'.npy')
            optpsf_list.append(single_optpsf_image)
            
            allparameters=np.load(RESULT_DIRECTANDINT_FOLDER+'allparameters_'+str(list_of_hashes[i])+'.npy')
            #print(allparameters)
            #print(len(allparameters))
            single_wavefront_parameter_list.append(np.concatenate((allparameters[:19],allparameters[42:])) )
            
            #extraZernike_parameter_list.append(single_wavefront_parameters)
        '''
        #print('extraZernike_parameter_list[35]'+str(extraZernike_parameter_list[35]))
        #print('Tokovnin_proposal_start'+str(Tokovnin_proposal_start))
        #print('extraZernike_parameter_list[35]-Tokovnin_proposal_start'+str(extraZernike_parameter_list[35]-Tokovnin_proposal_start))
        
        optpsf_list=out_images
        single_wavefront_parameter_list=[]
        for i in range(len(out_parameters)):
           single_wavefront_parameter_list.append(np.concatenate((out_parameters[i][:19],out_parameters[i][42:])) )
       
        '''
        #determine which parameter was moved
        Zernike_which_argument_was_moved=[]
        for i in range(len(single_wavefront_parameter_list)):
                Zernike_which_argument_was_moved.append(np.where((single_wavefront_parameter_list[i]-all_wavefront_z_old)>0))
        #print('extra_Zernike_which_argument_was_moved '+str(extra_Zernike_which_argument_was_moved))
        Zernike_which_argument_was_moved=np.array(Zernike_which_argument_was_moved).astype(int).ravel()
    
        
        Zernike_parameter_list_sorted=np.array(single_wavefront_parameter_list)[Zernike_which_argument_was_moved]
        # array of images sorted as you changed Zernike polynomials in the wavefront
        single_Zernike_parameter_sorted=np.array(optpsf_list)[Zernike_which_argument_was_moved]
        '''
        
        #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_ln_'+str(iteration_number),out_ln)
        #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_images_'+str(iteration_number),out_images)
        #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_parameters_'+str(iteration_number),out_parameters)
        #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/array_of_delta_z_'+str(iteration_number),array_of_delta_z)
        #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/single_wavefront_parameter_list_'+str(iteration_number),single_wavefront_parameter_list)   
        
        # normalize and mask images that have been created in the fitting procedure
        images_normalized=[]
        for i in range(len(optpsf_list)):
            images_normalized.append((optpsf_list[i][mask]/np.sum(optpsf_list[i][mask])).ravel())
        images_normalized=np.array(images_normalized)
        
         # normalize (via STD) and mask images that have been created in the fitting procedure
        #images_normalized=[]
        #for i in range(len(single_Zernike_parameter_sorted)):
        #    images_normalized.append(((single_Zernike_parameter_sorted[i][mask]*STD[mask])/np.sum(single_Zernike_parameter_sorted[i][mask]*STD[mask])).ravel())
        #images_normalized=np.array(images_normalized)
        
        
        #print(len(images_normalized[0]))
        #import starting image
     
        H=np.transpose(np.array((images_normalized-M0))/array_of_delta_z[:,None])    
    
        
        HHt=np.matmul(np.transpose(H),H) 
        print('svd thresh is '+str(thresh))
        invHHt=svd_invert(HHt,thresh)
        invHHtHt=np.matmul(invHHt,np.transpose(H))
        
        first_proposal_Tokovnin=np.matmul(invHHtHt,I-M0)
        #first_proposal_Tokovnin=np.matmul(invHHtHt,I-M0_STD)
       
        Tokovnin_proposal=0.9*first_proposal_Tokovnin
        #print('Tokovnin_proposal '+str(Tokovnin_proposal))
        print('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))
        
        all_wavefront_z_new=np.copy(all_wavefront_z_old)
        #print('len(all_wavefront_z_new)'+str(len(all_wavefront_z_new)))
        #print('len(Tokovnin_proposal)'+str(len(Tokovnin_proposal)))    
        all_wavefront_z_new=all_wavefront_z_new+Tokovnin_proposal
        up_to_z22_end=all_wavefront_z_new[:19]
        from_z22_end=all_wavefront_z_new[19:]
        allparameters_proposal_after_iteration=np.concatenate((up_to_z22_end,allparameters_proposal[19:42],from_z22_end))
        
        likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration=model_return(allparameters_proposal_after_iteration)
        #final_hash=np.abs(hash(tuple(allparameters_proposal_after_iteration)))
        #final_optpsf_image=np.load(RESULT_DIRECTANDINT_FOLDER+'optPsf_cut_fiber_convolved_downsampled_'+str(final_hash)+'.npy')
        
        M_final=(final_optpsf_image[mask]/np.sum(final_optpsf_image[mask])).ravel()
        #M_final=((final_optpsf_image[mask]*STD[mask])/np.sum(final_optpsf_image[mask]*STD[mask])).ravel()   
        
        
        
        IM_final=np.sum(np.abs(I-M_final))
        print('I-M_final after iteration '+str(iteration_number)+': '+str(np.sum(np.abs(I-M_final))))
         
        
        chi_2_after_iteration=(-2)*likelihood_after_iteration/(sci_image.shape[0]*sci_image.shape[1])
        #print('chi_2_after_iteration '+str(iteration_number)+': '+str(chi_2_after_iteration))
        if chi_2_after_iteration/chi_2_before_iteration <1.02 :
            did_chi_2_improve=1
        else:
            did_chi_2_improve=0
            # resetting all parameters
            all_wavefront_z_new=np.copy(all_wavefront_z_old)
            chi_2_after_iteration=chi_2_before_iteration
            all_wavefront_z_new=np.copy(all_wavefront_z_old)
            up_to_z22_end=all_wavefront_z_new[:19]
            from_z22_start=all_wavefront_z_new[19:]
            allparameters_proposal_after_iteration=np.concatenate((up_to_z22_start,allparameters_proposal[19:42],from_z22_start))
            thresh=thresh0
            number_of_non_decreses.append(1)
            print('current value of number_of_non_decreses is: '+str(np.sum(number_of_non_decreses)))
            
            #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/allparameters_proposal_after_iteration_'+str(obs)+str(single_number)+str(eps)+str(arc),allparameters_proposal_after_iteration)
            #np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/final_optpsf_image'+str(obs)+str(single_number)+str(eps)+str(arc),final_optpsf_image)
    
    
            
            if np.sum(number_of_non_decreses)==3:
                # need to put in save here as well, right?
                np.save(NAME_OF_allparameters,allparameters_proposal_after_iteration)
                np.save(NAME_OF_final_optpsf,final_optpsf_image)
                sys.exit(0)
        '''
        if IM_final/IM_start <1.05 :
            did_chi_2_improve=1
        else:
            did_chi_2_improve=0
            # resetting all parameters
            all_wavefront_z_new=np.copy(all_wavefront_z_old)
            chi_2_after_iteration=chi_2_before_iteration
            all_wavefront_z_new=np.copy(all_wavefront_z_old)
            up_to_z22_end=all_wavefront_z_new[:19]
            from_z22_start=all_wavefront_z_new[19:]
            allparameters_proposal_after_iteration=np.concatenate((up_to_z22_start,allparameters_proposal[19:42],from_z22_start))
            thresh=thresh0
            number_of_non_decreses.append(1)
            if np.sum(number_of_non_decreses)==3:
                sys.exit(0)
        '''
        
        
        print('Likelihood for the proposal after iteration number '+str(iteration_number)+' is: '+str((-2)*likelihood_after_iteration/(sci_image.shape[0]*sci_image.shape[1])))
        print('did_chi_2_improve (1 for yes, 0 for no): '+str(did_chi_2_improve))
        
        if iteration_number==9: 
            np.save(NAME_OF_allparameters,allparameters_proposal_after_iteration)
            np.save(NAME_OF_final_optpsf,final_optpsf_image)

    print('Time when total script finished was: '+time.ctime())     
    time_end=time.time()   
    print('Total time taken was  '+str(time_end-time_start)+' seconds')
    

    sys.stdout.flush()
    pool.close()
    sys.exit(0)
    
# should update that it goes via date
