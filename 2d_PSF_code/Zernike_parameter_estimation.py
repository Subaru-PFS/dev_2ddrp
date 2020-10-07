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
#from schwimmbad import MultiPool
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
from Zernike_Module import LN_PFS_single,LN_PFS_multi_same_spot,create_parInit,PFSLikelihoodModule,svd_invert

__version__ = "0.37"

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
parser.add_argument("-dataset", help="which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5] ",type=int, choices=[0, 1, 2,3,4,5])
################################################    
parser.add_argument("-arc", help="which arc lamp is being analyzed (HgAr for Mercury-Argon, Ne for Neon, Kr for Krypton)  ", choices=["HgAr", "Ne", "Kr"])
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
    options=[120,2.793,1.193]
if eps==7:
    options=[480,2.793,1.193]
if eps==8:
    options=[240,2.793,1.593]
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
# Tokovinin function at the moment defined here

def Tokovinin_algorithm_chi(sci_image,var_image,mask_image,allparameters_proposal):
   
    
    model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
                      dithering=1,save=0,zmax=twentytwo_or_extra,verbosity=0,double_sources=None,\
                      double_sources_positions_ratios=double_sources_positions_ratios,fit_for_flux=True,npix=1536,pupil_parameters=None)  


   


    print('Initial testing proposal is: '+str(allparameters_proposal))
    time_start_single=time.time()
    pre_model_result,pre_image,pre_input_parameters,chi_2_before_iteration_array=model(allparameters_proposal,return_Image=True,return_intermediate_images=False)
    chi_2_before_iteration=chi_2_before_iteration_array[2]
    time_end_single=time.time()
    print('Total time taken was  '+str(time_end_single-time_start_single)+' seconds')
    print('pre_model_result is: '+str(pre_model_result))  

    number_of_non_decreses=[0]
     
    for iteration_number in range(5): 
        
        # import values
        if iteration_number==0:

            mean_value_of_background=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*10
            mask=sci_image>(mean_value_of_background)

            # normalized science image
            sci_image_std=sci_image/np.sqrt(var_image)
            # initial SVD treshold
            thresh0 = 0.02

            # set number of extra Zernike
            #number_of_extra_zernike=0
            #twentytwo_or_extra=22
            # numbers that make sense are 11,22,37,56,79,106,137,172,211,254
            #if number_of_extra_zernike is None:
            #    number_of_extra_zernike=0
            #else:
            number_of_extra_zernike=twentytwo_or_extra-22

        # list of how much to move Zernike coefficents
        list_of_delta_z=[]
        for z_par in range(3,22+number_of_extra_zernike):
            list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))

        # array, randomized delta extra zernike
        #array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1.2+1
        #array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1
        #array_of_delta_z=+np.array(list_of_delta_z)*array_of_delta_randomize
        array_of_delta_z=np.array(list_of_delta_z)
        
        
        # initialize 
        # if this is the first iteration of the iterative algorithm
        if iteration_number==0:
            thresh=thresh0
            
            if number_of_extra_zernike==0:
                all_wavefront_z_old=allparameters_proposal[0:19]
            else:
                # if you want more Zernike
                if len(allparameters_proposal)==42:
                    # if you did not pass explicit extra Zernike, start with zeroes
                    all_wavefront_z_old=np.concatenate((allparameters_proposal[0:19],np.zeros(number_of_extra_zernike)))
                else:
                    all_wavefront_z_old=np.concatenate((allparameters_proposal[0:19],allparameters_proposal[19+23:]))
                    
            pass
        # if this is not a first iteration
        else:
            print('array_of_delta_z in '+str(iteration_number)+' '+str(array_of_delta_z))
            # code analysis programs might suggest that there is an error here, but everything is ok
            chi_2_before_iteration=np.copy(chi_2_after_iteration)
            # copy wavefront from the end of the previous iteration
            all_wavefront_z_old=np.copy(all_wavefront_z_new)
            if did_chi_2_improve==1:
                print('did_chi_2_improve: yes')
            else:
                print('did_chi_2_improve: no')                
            if did_chi_2_improve==0:
                thresh=thresh0
            else:
                thresh=thresh*0.5

        
      
        input_parameters=[]
        list_of_all_wavefront_z=[]

        up_to_z22_start=all_wavefront_z_old[0:19]     
        from_z22_start=all_wavefront_z_old[19:]  
        print('from_z22_start: '+str(from_z22_start))

        initial_input_parameters=np.concatenate((up_to_z22_start,pre_input_parameters[19:19+23],from_z22_start))
        print('initial input parameters in iteration '+str(iteration_number)+' is: '+str(initial_input_parameters))
        print('moving input parameters in iteration '+str(iteration_number)+' by: '+str(array_of_delta_z))        
        initial_model_result,image_0,initial_input_parameters,pre_chi2=model(initial_input_parameters,return_Image=True,return_intermediate_images=False)



        # put all new parameters in lists
        for z_par in range(len(all_wavefront_z_old)):
            all_wavefront_z_list=np.copy(all_wavefront_z_old)
            all_wavefront_z_list[z_par]=all_wavefront_z_list[z_par]+array_of_delta_z[z_par]
            list_of_all_wavefront_z.append(all_wavefront_z_list)

            up_to_z22_start=all_wavefront_z_list[0:19]     
            from_z22_start=all_wavefront_z_list[19:]  

            list_of_all_wavefront_z.append(all_wavefront_z_list)
            allparameters_proposal_int_22_test=np.concatenate((up_to_z22_start,pre_input_parameters[19:42],from_z22_start))
            input_parameters.append(allparameters_proposal_int_22_test)
            

        # standard deviation image
        STD=np.sqrt(var_image)      
        
        # model and science image divided by the STD
        image_0_std=image_0/STD
        sci_image_std=sci_image/STD
        

        # normalized and masked model image before this iteration
        M0=((image_0[mask])/np.sum(image_0[mask])).ravel()
        M0_std=((image_0_std[mask])/np.sum(image_0_std[mask])).ravel()
        # normalized and masked science image
        I=((sci_image[mask])/np.sum(sci_image[mask])).ravel()
        I_std=((sci_image_std[mask])/np.sum(sci_image_std[mask])).ravel()
        
        IM_start=np.sum(np.abs(I-M0))
        IM_start_std=np.sum(np.abs(I_std-M0_std))

        #print('np.sum(np.abs(I-M0)) before iteration '+str(iteration_number)+': '+str(IM_start))  
        print('IM_start_std before iteration '+str(iteration_number)+': '+str(IM_start_std))  

        out_ln=[]
        out_images=[]
        out_parameters=[]
        out_chi2=[]


        print('We are inside of the pool loop number '+str(iteration_number)+' now')
        print('len(input_parameters): '+str(len(input_parameters)))
        time_start=time.time()
        
        #a_pool = Pool()
        #out1=a_pool.map(partial(model, return_Image=True), input_parameters)
        out1=pool.map(partial(model, return_Image=True), input_parameters)
        out1=list(out1)
        
        
        #out1=map(partial(model, return_Image=True), input_parameters)
        #out1=list(out1)
        
        #out1=a_pool.map(model,input_parameters,repeat(True))
        for i in range(len(input_parameters)):
            print(i)

            out_ln.append(out1[i][0])
            out_images.append(out1[i][1])
            out_parameters.append(out1[i][2])
            out_chi2.append(out1[i][3])
        

            #for i in range(len(input_parameters)):
            #    print(i)
            #out1=model_return(input_parameters[i])
            #    out_ln.append(out1[0])
            #    out_images.append(out1[1])
            #    out_parameters.append(out1[2])
            #    out_chi2.append(out1[3])
        time_end=time.time()
        print('time_end-time_start '+str(time_end-time_start))
        
        optpsf_list=out_images
        single_wavefront_parameter_list=[]
        for i in range(len(out_parameters)):
            single_wavefront_parameter_list.append(np.concatenate((out_parameters[i][:19],out_parameters[i][42:])) )

        # normalize and mask images that have been created in the fitting procedure
        images_normalized=[]
        for i in range(len(optpsf_list)):
            images_normalized.append((optpsf_list[i][mask]/np.sum(optpsf_list[i][mask])).ravel())
        images_normalized=np.array(images_normalized)
        # same but divided by STD
        images_normalized_std=[]
        for i in range(len(optpsf_list)):      
            optpsf_list_i=optpsf_list[i]
            optpsf_list_i_STD=optpsf_list_i/STD    
            images_normalized_std.append((optpsf_list_i_STD[mask]/np.sum(optpsf_list_i_STD[mask])).ravel())
        images_normalized_std=np.array(images_normalized_std)
        
        
        
        #print('images_normalized.shape: '+str(images_normalized.shape))
        # equation A1 from Tokovinin 2006
        H=np.transpose(np.array((images_normalized-M0))/array_of_delta_z[:,None])    
        H_std=np.transpose(np.array((images_normalized_std-M0_std))/array_of_delta_z[:,None]) 

        HHt=np.matmul(np.transpose(H),H)
        HHt_std=np.matmul(np.transpose(H_std),H_std) 
        #print('svd thresh is '+str(thresh))
        invHHt=svd_invert(HHt,thresh)
        invHHt_std=svd_invert(HHt_std,thresh)
        
        invHHtHt=np.matmul(invHHt,np.transpose(H))
        invHHtHt_std=np.matmul(invHHt_std,np.transpose(H_std))

        first_proposal_Tokovnin=np.matmul(invHHtHt,I-M0)
        first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,I_std-M0_std)
        
        #Tokovnin_proposal=0.9*first_proposal_Tokovnin
        Tokovnin_proposal=0.7*first_proposal_Tokovnin_std
        #print('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))

        all_wavefront_z_new=np.copy(all_wavefront_z_old)
        all_wavefront_z_new=all_wavefront_z_new+Tokovnin_proposal
        up_to_z22_end=all_wavefront_z_new[:19]
        from_z22_end=all_wavefront_z_new[19:]
        allparameters_proposal_after_iteration=np.concatenate((up_to_z22_end,pre_input_parameters[19:42],from_z22_end))
        
        print('allparameters_proposal_after_iteration: '+str(allparameters_proposal_after_iteration))

        likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration,chi2_after_iteration_array=model(allparameters_proposal_after_iteration,return_Image=True)

        M_final=(final_optpsf_image[mask]/np.sum(final_optpsf_image[mask])).ravel()
        IM_final=np.sum(np.abs(I-M_final))
        # STD version 
        final_optpsf_image_std=final_optpsf_image/STD    
        M_final_std=(final_optpsf_image_std[mask]/np.sum(final_optpsf_image_std[mask])).ravel()
        IM_final_std=np.sum(np.abs(I_std-M_final_std))        
        
        #print('I-M_final after iteration '+str(iteration_number)+': '+str(IM_final))
        print('I_std-M_final_std after iteration '+str(iteration_number)+': '+str(IM_final_std))

        chi_2_after_iteration=chi2_after_iteration_array[2]
        #print('chi_2_after_iteration/chi_2_before_iteration '+str(chi_2_after_iteration/chi_2_before_iteration ))
        print('IM_final_std/IM_start_std '+str(IM_final_std/IM_start_std))
        print('#########################################################')
        #if chi_2_after_iteration/chi_2_before_iteration <1.02 :
            
        if IM_final_std/IM_start_std <1.0 :
            #when the quality measure did improve
            did_chi_2_improve=1
            number_of_non_decreses.append(0)
        else:
            #when the quality measure did not improve
            did_chi_2_improve=0
            # resetting all parameters
            all_wavefront_z_new=np.copy(all_wavefront_z_old)
            chi_2_after_iteration=chi_2_before_iteration
            up_to_z22_end=all_wavefront_z_new[:19]
            from_z22_start=all_wavefront_z_new[19:]
            allparameters_proposal_after_iteration=np.concatenate((up_to_z22_start,pre_input_parameters[19:42],from_z22_start))
            thresh=thresh0
            number_of_non_decreses.append(1)
            print('number_of_non_decreses:' + str(number_of_non_decreses))
            print('current value of number_of_non_decreses is: '+str(np.sum(number_of_non_decreses)))
            print('#############################################')
         
        if np.sum(number_of_non_decreses)==2:
            break
            
        if len(allparameters_proposal_after_iteration)==41:
            allparameters_proposal_after_iteration=np.concatenate((allparameters_proposal_after_iteration,np.array([1])))
            
    return likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration,chi2_after_iteration_array        
    


def move_parametrizations_from_2d_shape_to_1d_shape(allparameters_best_parametrization_shape_2d):
    """ 
    change the linear parametrization array in 2d shape to parametrization array in 1d
    
    @param allparameters_best_parametrization_shape_2d        linear parametrization, 2d array
    
    """    
    
    
    if allparameters_best_parametrization_shape_2d.shape[0]>42:
        #  if you are using above Zernike above 22
        print('we are creating new result with Zernike above 22')
        allparameters_best_parametrization_shape_1d=np.concatenate((allparameters_best_parametrization_shape_2d[:19].ravel(),
                                                    allparameters_best_parametrization_shape_2d[19:19+23][:,1],\
                                                        allparameters_best_parametrization_shape_2d[19+23:].ravel()))
        
    else:
        print('we are creating new result with Zernike at 22')
        allparameters_best_parametrization_shape_1d=np.concatenate((allparameters_best_parametrization_shape_2d[:19].ravel(),
                                                    allparameters_best_parametrization_shape_2d[19:-1][:,1]))    
        
    return allparameters_best_parametrization_shape_1d


def check_global_parameters(globalparameters,test_print=None,fit_for_flux=None):
    #When running big fits these are limits which ensure that the code does not wander off in totally non physical region


    globalparameters_output=np.copy(globalparameters)
    # hsc frac
    if globalparameters[0]<0.6 or globalparameters[0]>0.8:
        print('globalparameters[0] outside limits; value: '+str(globalparameters[0])) if test_print == 1 else False 
    if globalparameters[0]<=0.6:
        globalparameters_output=0.6
    if globalparameters[0]>0.8:
        globalparameters_output=0.8

     #strut frac
    if globalparameters[1]<0.07 or globalparameters[1]>0.13:
        print('globalparameters[1] outside limits') if test_print == 1 else False 
    if globalparameters[1]<=0.07:
        globalparameters_output=0.07
    if globalparameters[1]>0.13:
        globalparameters_output=0.13

    #slit_frac < strut frac 
    #if globalparameters[4]<globalparameters[1]:
        #print('globalparameters[1] not smaller than 4 outside limits')
        #return -np.inf

     #dx Focal
    if globalparameters[2]<-0.4 or globalparameters[2]>0.4:
        print('globalparameters[2] outside limits') if test_print == 1 else False 
    if globalparameters[2]<-0.4:
        globalparameters_output[2]=-0.4
    if globalparameters[2]>0.4:
        globalparameters_output[2]=0.4

    # dy Focal
    if globalparameters[3]>0.4:
        print('globalparameters[3] outside limits') if test_print == 1 else False 
        globalparameters_output[3]=0.4
    if globalparameters[3]<-0.4:
        print('globalparameters[3] outside limits') if test_print == 1 else False 
        globalparameters_output[3]=-0.4

    # slitFrac
    if globalparameters[4]<0.05:
        print('globalparameters[4] outside limits') if test_print == 1 else False 
        globalparameters_output[4]=0.05
    if globalparameters[4]>0.09:
        print('globalparameters[4] outside limits') if test_print == 1 else False 
        globalparameters_output[4]=0.09

    # slitFrac_dy
    if globalparameters[5]<-0.5:
        print('globalparameters[5] outside limits') if test_print == 1 else False 
        globalparameters_output[5]=-0.5
    if globalparameters[5]>0.5:
        print('globalparameters[5] outside limits') if test_print == 1 else False 
        globalparameters_output[5]=+0.5

    # radiometricEffect
    if globalparameters[6]<0:
        print('globalparameters[6] outside limits') if test_print == 1 else False 
        globalparameters_output[6]=0
    if globalparameters[6]>1:
        print('globalparameters[6] outside limits') if test_print == 1 else False 
        globalparameters_output[6]=1

    # radiometricExponent
    if globalparameters[7]<0:
        print('globalparameters[7] outside limits') if test_print == 1 else False 
        globalparameters_output[7]=0
    if globalparameters[7]>2:
        print('globalparameters[7] outside limits') if test_print == 1 else False 
        globalparameters_output[7]=2

    # x_ilum
    if globalparameters[8]<0.5:
        print('globalparameters[8] outside limits') if test_print == 1 else False 
        globalparameters_output[8]=0.5
    if globalparameters[8]>1.5:
        print('globalparameters[8] outside limits') if test_print == 1 else False 
        globalparameters_output[8]=1.5

    # y_ilum
    if globalparameters[9]<0.5:
        print('globalparameters[9] outside limits') if test_print == 1 else False 
        globalparameters_output[9]=0.5
    if globalparameters[9]>1.5:
        print('globalparameters[9] outside limits') if test_print == 1 else False 
        globalparameters_output[9]=1.5

    # x_fiber
    if globalparameters[10]<-0.4:
        print('globalparameters[10] outside limits') if test_print == 1 else False 
        globalparameters_output[10]=-0.4
    if globalparameters[10]>0.4:
        print('globalparameters[10] outside limits') if test_print == 1 else False 
        globalparameters_output[10]=0.4

    # y_fiber
    if globalparameters[11]<-0.4:
        print('globalparameters[11] outside limits') if test_print == 1 else False 
        globalparameters_output[11]=-0.4
    if globalparameters[11]>0.4:
        print('globalparameters[11] outside limits') if test_print == 1 else False 
        globalparameters_output[11]=0.4      

    # effective_radius_illumination
    if globalparameters[12]<0.7:
        print('globalparameters[12] outside limits') if test_print == 1 else False 
        globalparameters_output[12]=0.7
    if globalparameters[12]>1.0:
        print('globalparameters[12] outside limits') if test_print == 1 else False 
        globalparameters_output[12]=1

    # frd_sigma
    if globalparameters[13]<0.01:
        print('globalparameters[13] outside limits') if test_print == 1 else False 
        globalparameters_output[13]=0.01
    if globalparameters[13]>.4:
        print('globalparameters[13] outside limits') if test_print == 1 else False 
        globalparameters_output[13]=0.4 

    #frd_lorentz_factor
    if globalparameters[14]<0.01:
        print('globalparameters[14] outside limits') if test_print == 1 else False 
        globalparameters_output[14]=0.01
    if globalparameters[14]>1:
        print('globalparameters[14] outside limits') if test_print == 1 else False 
        globalparameters_output[14]=1 

    # det_vert
    if globalparameters[15]<0.85:
        print('globalparameters[15] outside limits') if test_print == 1 else False 
        globalparameters_output[15]=0.85
    if globalparameters[15]>1.15:
        print('globalparameters[15] outside limits') if test_print == 1 else False 
        globalparameters_output[15]=1.15

    # slitHolder_frac_dx
    if globalparameters[16]<-0.8:
        print('globalparameters[16] outside limits') if test_print == 1 else False 
        globalparameters_output[16]=-0.8
    if globalparameters[16]>0.8:
        print('globalparameters[16] outside limits') if test_print == 1 else False 
        globalparameters_output[16]=0.8 

    # grating_lines
    if globalparameters[17]<1200:
        print('globalparameters[17] outside limits') if test_print == 1 else False 
        globalparameters[17]=1200
    if globalparameters[17]>120000:
        print('globalparameters[17] outside limits') if test_print == 1 else False 
        globalparameters_output[17]=120000 

    # scattering_slope
    if globalparameters[18]<1.5:
        print('globalparameters[18] outside limits') if test_print == 1 else False 
        globalparameters_output[18]=1.5
    if globalparameters[18]>+3.0:
        print('globalparameters[18] outside limits') if test_print == 1 else False 
        globalparameters_output[18]=3 

    # scattering_amplitude
    if globalparameters[19]<0:
        print('globalparameters[19] outside limits') if test_print == 1 else False 
        globalparameters_output[19]=0
    if globalparameters[19]>+0.4:
        print('globalparameters[19] outside limits') if test_print == 1 else False 
        globalparameters_output[19]=0.4          

    # pixel_effect
    if globalparameters[20]<0.35:
        print('globalparameters[20] outside limits') if test_print == 1 else False 
        globalparameters_output[20]=0.35
    if globalparameters[20]>+0.8:
        print('globalparameters[20] outside limits') if test_print == 1 else False 
        globalparameters_output[20]=0.8

    # fiber_r
    if globalparameters[21]<1.78:
        print('globalparameters[21] outside limits') if test_print == 1 else False 
        globalparameters_output[21]=1.78
    if globalparameters[21]>+1.98:
        print('globalparameters[21] outside limits') if test_print == 1 else False 
        globalparameters_output[21] =1.98

    # flux
    if fit_for_flux==True:
        globalparameters_output[22]=1
    else:          
        if globalparameters[22]<0.98:
            print('globalparameters[22] outside limits') if test_print == 1 else False 
            globalparameters_output[22] =0.98
        if globalparameters[22]>1.02:
            print('globalparameters[22] outside limits') if test_print == 1 else False 
            globalparameters_output[22] =1.02

                
    return globalparameters_output
    


################################################  
################################################  
################################################  
################################################  
################################################  

def Tokovinin_algorithm_chi_multi(list_of_sci_images,list_of_var_images,list_of_mask_images,allparameters_parametrization_proposal,\
                                  num_iter=None,move_allparameters=None):
    
    """ 
    Tokovinin algorithm for analyzing multiple images at once
    
    @param list_of_sci_images                         list of science images
    @param list_of_var_images                         list of variance images
    @param list_of_mask_images                        list of mask images
    @param allparameters_parametrization_proposal     1d array contaning parametrization 
    @param num_iter                                   optional parameters only used for naming intermediate outputs
    """              
    # possibly need to refactor so that the algorithm also takes list_of_defocuses
        
    #model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
    #                  dithering=1,save=0,zmax=twentytwo_or_extra,verbosity=0,double_sources=None,\
    #                  double_sources_positions_ratios=double_sources_positions_ratios,fit_for_flux=True,npix=1536,pupil_parameters=None)  
    
    
    #########################################################################################################
    # Create initial modeling as basis for future effort
    
    model_multi=LN_PFS_multi_same_spot(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                       dithering=1,save=0,zmax=56,verbosity=0,double_sources=False,\
                                       double_sources_positions_ratios=double_sources_positions_ratios,npix=1536,fit_for_flux=True,test_run=False)   
    print('****************************')        
    print('Starting Tokovinin procedure')
    print('Initial testing proposal is: '+str(allparameters_parametrization_proposal))
    time_start_single=time.time()
    
    # this uncommented line below will not work
    # proper outputs
    #    initial_model_result,list_of_initial_model_result,list_of_image_0,\
    #                list_of_initial_input_parameters,list_of_pre_chi2=res_multi 
    
    # create list of minchains, one per each image
    list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,list_of_defocuses=list_of_defocuses_input_long,zmax=56)
    # pre_model_result - mean likelihood per images
    # model_results - likelihood per image
    # pre_images - list of created model images
    # pre_input_parameters - list of parameters per image?
    # chi_2_before_iteration_array - list of lists describing quality of fitting    
    pre_model_result,model_results,pre_images,pre_input_parameters,chi_2_before_iteration_array,list_of_psf_positions=model_multi(list_of_minchain,return_Images=True)
    print('list_of_psf_positions at the input stage: '+str(list_of_psf_positions))
    
    if num_iter!=None:
        np.save('/tigress/ncaplar/Results/allparameters_parametrization_proposal',\
                allparameters_parametrization_proposal)   
        np.save('/tigress/ncaplar/Results/pre_images',\
                pre_images)   
        np.save('/tigress/ncaplar/Results/pre_input_parameters',\
                pre_input_parameters)   
        np.save('/tigress/ncaplar/Results/list_of_sci_images',\
                list_of_sci_images)  
        np.save('/tigress/ncaplar/Results/list_of_var_images',\
                list_of_var_images)  
        np.save('/tigress/ncaplar/Results/list_of_mask_images',\
                list_of_mask_images)  
            

    

    # this needs to change - do I ever use this?!?
    chi_2_before_iteration=chi_2_before_iteration_array[2]
    # extract the parameters which will not change in this function, i.e., not-wavefront parameters
    nonwavefront_par=list_of_minchain[0][19:42]
    time_end_single=time.time()
    print('Total time taken was  '+str(time_end_single-time_start_single)+' seconds')
    print('chi_2_pre_input_parameters is: '+str(chi_2_before_iteration_array[2]))  
    
    # import science images and mask them     
    list_of_mean_value_of_background=[]
    list_of_flux_mask=[]
    list_of_sci_image_std=[]
    for i in range(len(list_of_sci_images)):
        sci_image=list_of_sci_images[i]

        mean_value_of_background=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
                                          np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*5
            
        if move_allparameters==True:
            mean_value_of_background=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
                                  np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*3
            

        list_of_mean_value_of_background.append(mean_value_of_background)
        flux_mask=sci_image>(mean_value_of_background)
   
        
        # normalized science image
        var_image=list_of_var_images[i]
        sci_image_std=sci_image/np.sqrt(var_image)
        list_of_sci_image_std.append(sci_image_std)
        list_of_flux_mask.append(flux_mask)
    # perhaps also 
    
    ######################################################################################################### 
    # masked science image
    list_of_I=[]
    list_of_I_std=[]    
    list_of_std_image=[]
    for i in range(len(list_of_sci_images)):
        
        sci_image=list_of_sci_images[i]
        sci_image_std=list_of_sci_image_std[i]
        flux_mask=list_of_flux_mask[i]
        std_image=np.sqrt(list_of_var_images[i][flux_mask]).ravel()
        
        I=sci_image[flux_mask].ravel()        
        #I=((sci_image[flux_mask])/np.sum(sci_image[flux_mask])).ravel()
        I_std=((sci_image_std[flux_mask])/1).ravel()
        #I_std=((sci_image_std[flux_mask])/np.sum(sci_image_std[flux_mask])).ravel()
        
        list_of_I.append(I)
        list_of_std_image.append(std_image)
        list_of_I_std.append(I_std)

    # join all I,I_std from all individual images into one uber I,I_std  
    uber_I=[item for sublist in list_of_I for item in sublist]
    uber_std=[item for sublist in list_of_std_image for item in sublist]
    uber_I_std=[item for sublist in list_of_I_std for item in sublist]    
    
    uber_I=np.array(uber_I)
    uber_std=np.array(uber_std)
    #uber_I_std=np.array(uber_I_std) 
    
    # removing normalization in 0.36
    #uber_I=uber_I/np.sum(uber_I)
    #uber_I_std=uber_I_std/np.sum(uber_I_std)    
    
    if num_iter!=None:
        np.save('/tigress/ncaplar/Results/list_of_sci_images',\
                list_of_sci_images)   
        np.save('/tigress/ncaplar/Results/list_of_flux_mask',\
                list_of_flux_mask)   
        np.save('/tigress/ncaplar/Results/uber_std',\
                uber_std)               
        np.save('/tigress/ncaplar/Results/uber_I',\
                uber_I)   
            
            
        
    #np.save('/tigress/ncaplar/Results/uber_I_std',\
    #        uber_I_std)              
            
    # set number of extra Zernike
    #number_of_extra_zernike=0
    #twentytwo_or_extra=22
    # numbers that make sense are 11,22,37,56,79,106,137,172,211,254
    
    #if number_of_extra_zernike is None:
    #    number_of_extra_zernike=0
    #else:
    number_of_extra_zernike=twentytwo_or_extra-22
    
    
    
    #########################################################################################################
    # Start of the iterative process
    
    number_of_non_decreses=[0]
    
    for iteration_number in range(5): 


        if iteration_number==0:
            
            # initial SVD treshold
            thresh0 = 0.02
            
    

    
        ######################################################################################################### 
        # starting real iterative process here
        # create changes in parametrizations    
            
        # list of how much to move Zernike coefficents
        #list_of_delta_z=[]
        #for z_par in range(3,22+number_of_extra_zernike):
        #    list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))
    
        # list of how much to move Zernike coefficents
        # possibly needs to me modified to be smarther and take into account that every second parameter gets ``amplified'' in defocus
        #list_of_delta_z_parametrizations=[]
        #for z_par in range(0,19*2+2*number_of_extra_zernike):
        #    list_of_delta_z_parametrizations.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))        

        # this should produce reasonable changes in multi analysis
        list_of_delta_z_parametrizations=[]
        for z_par in range(0,19*2+2*number_of_extra_zernike):
            z_par_i=z_par+4
            # if this is the parameter that change
            if np.mod(z_par_i,2)==0:
                list_of_delta_z_parametrizations.append(0.25*0.25/np.sqrt(z_par_i))
            if np.mod(z_par_i,2)==1:
                list_of_delta_z_parametrizations.append(0.25/np.sqrt(z_par_i))
                    

        array_of_delta_global_parametrizations=np.array([0.1,0.02,0.1,0.1,0.1,0.1,
                                        0.3,1,0.1,0.1,
                                        0.15,0.15,0.1,
                                        0.07,0.05,0.05,0.4,
                                        30000,0.5,0.001,
                                        0.05,0.05,0.01])
        array_of_delta_global_parametrizations=array_of_delta_global_parametrizations/10

        # array, randomized delta extra zernike
        #array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1.2+1
        #array_of_delta_parametrizations_randomize=np.random.standard_normal(len(list_of_delta_z_parametrizations))*1
        #array_of_delta_z_parametrizations=+np.array(list_of_delta_z_parametrizations)*array_of_delta_parametrizations_randomize
        
        
        array_of_delta_z_parametrizations=np.array(list_of_delta_z_parametrizations)*(1)
        
        if move_allparameters==True:
            array_of_delta_all_parametrizations=np.concatenate((array_of_delta_z_parametrizations[0:19*2],\
                                                                array_of_delta_global_parametrizations, array_of_delta_z_parametrizations[19*2:]))

        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/array_of_delta_z_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                    array_of_delta_z_parametrizations)        
            np.save('/tigress/ncaplar/Results/array_of_delta_global_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                    array_of_delta_global_parametrizations)      
            if move_allparameters==True:
                np.save('/tigress/ncaplar/Results/array_of_delta_all_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                        array_of_delta_all_parametrizations)                      
                
                
                   
    
        # initialize 
        # if this is the first iteration of the iterative algorithm
        if iteration_number==0:
            thresh=thresh0
            all_global_parametrization_old=allparameters_parametrization_proposal[19*2:19*2+23]
            if number_of_extra_zernike==0:
                all_wavefront_z_parametrization_old=allparameters_parametrization_proposal[0:19*2]
            else:
                # if you want more Zernike
                if len(allparameters_parametrization_proposal)==19*2+23:
                    # if you did not pass explicit extra Zernike, start with zeroes
                    all_wavefront_z_parametrization_old=np.concatenate((allparameters_parametrization_proposal[0:19*2],np.zeros(2*number_of_extra_zernike)))
                else:
                    all_wavefront_z_parametrization_old=np.concatenate((allparameters_parametrization_proposal[0:19*2],allparameters_parametrization_proposal[19*2+23:]))
    
            pass
        # if this is not a first iteration
        else:
            # errors in the typechecker for 10 lines below are fine
            print('array_of_delta_z in '+str(iteration_number)+' '+str(array_of_delta_z_parametrizations))
            # code analysis programs might suggest that there is an error here, but everything is ok
            #chi_2_before_iteration=np.copy(chi_2_after_iteration)
            # copy wavefront from the end of the previous iteration
            
            
            all_wavefront_z_parametrization_old=np.copy(all_wavefront_z_parametrization_new)
            if move_allparameters==True:
                all_global_parametrization_old=np.copy(all_global_parametrization_new)
            if did_chi_2_improve==1:
                print('did_chi_2_improve: yes')
            else:
                print('did_chi_2_improve: no')                
            if did_chi_2_improve==0:
                thresh=thresh0
            else:
                thresh=thresh*0.5
    
        ######################################################################################################### 
        # create a model with input parameters from previous iteration
        
        list_of_all_wavefront_z_parameterization=[]
    
        up_to_z22_parametrization_start=all_wavefront_z_parametrization_old[0:19*2]     
        from_z22_parametrization_start=all_wavefront_z_parametrization_old[19*2:]  
        global_parametrization_start=all_global_parametrization_old
        #print('from_z22_start: '+str(from_z22_start))
        #print('iteration '+str(iteration_number)+' shape of up_to_z22_parametrization_start is: '+str(up_to_z22_parametrization_start.shape))
        if move_allparameters==True:
            initial_input_parameterization=np.concatenate((up_to_z22_parametrization_start,global_parametrization_start,from_z22_parametrization_start))
        else:
            initial_input_parameterization=np.concatenate((up_to_z22_parametrization_start,nonwavefront_par,from_z22_parametrization_start))
        print('initial input parameters in iteration '+str(iteration_number)+' is: '+str(initial_input_parameterization))
        print('moving input parameters in iteration '+str(iteration_number)+' by: '+str(array_of_delta_z_parametrizations))
        if move_allparameters==True:
            print('moving global input parameters in iteration '+str(iteration_number)+' by: '+str(array_of_delta_global_parametrizations))
            
        
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/initial_input_parameterization_'+str(num_iter)+'_'+str(iteration_number),\
                    initial_input_parameterization)                        

        #print('len initial_input_parameterization '+str(len(initial_input_parameterization)))
        
        list_of_minchain=model_multi.create_list_of_allparameters(initial_input_parameterization,list_of_defocuses=list_of_defocuses_input_long,zmax=56)
        #list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,list_of_defocuses=list_of_defocuses_input_long,zmax=56)

        

        res_multi=model_multi(list_of_minchain,return_Images=True)
        #mean_res_of_multi_same_spot_proposal,list_of_single_res_proposal,list_of_single_model_image_proposal,\
        #            list_of_single_allparameters_proposal,list_of_single_chi_results_proposal=res_multi  
        initial_model_result,list_of_initial_model_result,list_of_image_0,\
                    list_of_initial_input_parameters,list_of_pre_chi2,list_of_psf_positions=res_multi      
        
        #initial_model_result,image_0,initial_input_parameters,pre_chi2=model(initial_input_parameters,return_Image=True,return_intermediate_images=False)
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/list_of_initial_model_result_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_initial_model_result)                        
            np.save('/tigress/ncaplar/Results/list_of_image_0_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_image_0)    
            np.save('/tigress/ncaplar/Results/list_of_initial_input_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_initial_input_parameters)   
            np.save('/tigress/ncaplar/Results/list_of_pre_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_pre_chi2)      
            np.save('/tigress/ncaplar/Results/list_of_psf_positions_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_psf_positions)     

        ######################################################################################################### 
        # divided model images by their standard deviations
        
        list_of_image_0_std=[]
        for i in range(len(list_of_image_0)):
            # normalizing by standard deviation image
            STD=np.sqrt(list_of_var_images[i])    
            image_0=list_of_image_0[i]
            list_of_image_0_std.append(image_0/STD)
        
    
        ######################################################################################################### 
        # masked model images at the start of this iteration, before modifying parameters
        
        
        list_of_M0=[]
        list_of_M0_std=[]
        for i in range(len(list_of_image_0_std)):
            
            image_0=list_of_image_0[i]
            image_0_std=list_of_image_0_std[i]
            flux_mask=list_of_flux_mask[i]
            # what is list_of_mask_images?

            M0=image_0[flux_mask].ravel()            
            #M0=((image_0[flux_mask])/np.sum(image_0[flux_mask])).ravel()
            M0_std=((image_0_std[flux_mask])/1).ravel()
            #M0_std=((image_0_std[flux_mask])/np.sum(image_0_std[flux_mask])).ravel()
            
            list_of_M0.append(M0)
            list_of_M0_std.append(M0_std)
    
        # join all M0,M0_std from invidiual images into one uber M0,M0_std
        uber_M0=[item for sublist in list_of_M0 for item in sublist]
        uber_M0_std=[item for sublist in list_of_M0_std for item in sublist]    
        
        uber_M0=np.array(uber_M0)
        uber_M0_std=np.array(uber_M0_std)
        
        #uber_M0=uber_M0/np.sum(uber_M0)
        #uber_M0_std=uber_M0_std/np.sum(uber_M0_std)
    
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/uber_M0_'+str(num_iter)+'_'+str(iteration_number),\
                    uber_M0)             
            np.save('/tigress/ncaplar/Results/uber_M0_std_'+str(num_iter)+'_'+str(iteration_number),\
                    uber_M0_std)     


    
        ######################################################################################################### 
        # difference between model (uber_M0) and science (uber_I) at start of this iteration
        
        # non-std version
        # not used, that is ok, we are at the moment using std version
        IM_start=np.sum(np.abs(np.array(uber_I)-np.array(uber_M0)))        
        # std version 
        IM_start_std=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M0_std)))    
        
        # mean of differences of our images - should we use mean?; probably not... needs to be normalized?
        unitary_IM_start=np.mean(IM_start)  
        #unitary_IM_start_std=np.mean(IM_start_std)  
            
        #print list_of_IM_start_std
        print('np.sum(np.abs(I-M0)) before iteration '+str(iteration_number)+': '+str(unitary_IM_start))  
        #print('np.sum(np.abs(I_std-M0_std)) before iteration '+str(iteration_number)+': '+str(unitary_IM_start_std))  
    
    
                
        ######################################################################################################### 
        # create list of new parametrizations to be tested
        # combine the old wavefront parametrization with the delta_z_parametrization 
        
        # create two lists:
        # 1. one contains only wavefront parametrizations
        # 2. second contains the whole parametrizations
        
        if move_allparameters==True:
            list_of_all_wavefront_z_parameterization=[]
            list_of_input_parameterizations=[]
            for z_par in range(19*2):
                all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
        
                up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
    
                parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                # actually it is parametrization
                list_of_input_parameterizations.append(parametrization_proposal)  

            for g_par in range(23):
                all_global_parametrization_list=np.copy(all_global_parametrization_old)
                all_global_parametrization_list[g_par]=all_global_parametrization_list[g_par]+array_of_delta_global_parametrizations[g_par]
                #list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
        
                up_to_z22_start=all_wavefront_z_parametrization_old[0:19*2]     
                from_z22_start=all_wavefront_z_parametrization_old[19*2:]  
    
                parametrization_proposal=np.concatenate((up_to_z22_start,all_global_parametrization_list,from_z22_start))
                # actually it is parametrization
                list_of_input_parameterizations.append(parametrization_proposal)  

            for z_par in range(19*2,len(all_wavefront_z_parametrization_old)):
                all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
        
                up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
    
                parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                # actually it is parametrization
                list_of_input_parameterizations.append(parametrization_proposal)  

            
        else:
            list_of_all_wavefront_z_parameterization=[]
            list_of_input_parameterizations=[]
            for z_par in range(len(all_wavefront_z_parametrization_old)):
                all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
        
                up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
    
                parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                # actually it is parametrization
                list_of_input_parameterizations.append(parametrization_proposal)    
        
        ######################################################################################################### 
        # Starting testing new set
        # Creating new images
        
        out_ln=[]
        out_ln_ind=[]
        out_images=[]
        out_parameters=[]
        out_chi2=[]
        out_pfs_positions=[]
    
    
        print('We are inside of the pool loop number '+str(iteration_number)+' now')
        # actually it is parametrization
        # list of (56-3)*2 sublists, each one with (56-3)*2 + 23 values
        time_start=time.time()
    
    
        #list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,list_of_defocuses=list_of_defocuses_input_long,zmax=56)
     
        # I need to pass each of 106 parametrization to model_multi BUT
        # model_multi actually takes list of parameters, not a parametrizations
        # I need list that has 106 sublists, each one of those being 9x(53+23)
        uber_list_of_input_parameters=[]
        for i in range(len(list_of_input_parameterizations)):

           list_of_input_parameters=model_multi.create_list_of_allparameters(list_of_input_parameterizations[i],\
                                                                             list_of_defocuses=list_of_defocuses_input_long,zmax=56)
           uber_list_of_input_parameters.append(list_of_input_parameters)
           
        #save the uber_list_of_input_parameters
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/uber_list_of_input_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                    uber_list_of_input_parameters)            
       
        # pass new model_multi that has fixed pos (October 6)   
        model_multi_out=LN_PFS_multi_same_spot(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                             dithering=1,save=0,zmax=56,verbosity=0,double_sources=False,\
                             double_sources_positions_ratios=double_sources_positions_ratios,npix=1536,
                             fit_for_flux=True,test_run=False,list_of_psf_positions=list_of_psf_positions)   
                
        

        

        out1=pool.map(partial(model_multi_out, return_Images=True), uber_list_of_input_parameters)
        # out1=pool.map(partial(model_multi, return_Images=True), uber_list_of_input_parameters)
        out1=list(out1)
    
        #out1=map(partial(model, return_Image=True), input_parameters)
        #out1=list(out1)
    
        # normalization of the preinput run
        pre_input_parameters=np.array(pre_input_parameters)
        print(pre_input_parameters.shape)
        print(pre_input_parameters[0])
        
        array_of_normalizations_pre_input=pre_input_parameters[:,41]
    
    
    
    
    
        #out1=a_pool.map(model,input_parameters,repeat(True))
        for i in range(len(uber_list_of_input_parameters)):
            #print(i)
            
            #    initial_model_result,list_of_initial_model_result,list_of_image_0,\
            #                list_of_initial_input_parameters,list_of_pre_chi2
            
            # outputs are 
            # 0. mean likelihood
            # 1. list of individual res (likelihood)
            # 2. list of science images
            # 3. list of parameters used
            # 4. list of quality measurments
            
            
            out_images_pre_renormalization=np.array(out1[i][2])
            out_parameters_single_move=np.array(out1[i][3])
            array_of_normalizations_out=out_parameters_single_move[:,41]
            
            out_renormalization_parameters=array_of_normalizations_pre_input/array_of_normalizations_out
            
            
            out_ln.append(out1[i][0])
            out_ln_ind.append(out1[i][1])
            out_images.append(out_images_pre_renormalization*out_renormalization_parameters)
            out_parameters.append(out1[i][3])
            out_chi2.append(out1[i][4])
            out_pfs_positions.append(out1[i][5])
            
            # we use these out_images to study the differences due to changing parameters,
            # so we do not want the normalization to affect things (and position of optical center)
            # so we renormalize to that multiplication constants are the same as in the input 
            
            
            
            
        time_end=time.time()
        print('time_end-time_start '+str(time_end-time_start))
    
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/out_images_'+str(num_iter)+'_'+str(iteration_number),\
                    out_images)    
            np.save('/tigress/ncaplar/Results/out_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                    out_parameters)  
            np.save('/tigress/ncaplar/Results/out_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                    out_chi2) 
        
        ######################################################################################################### 
        # Normalize created images
        
        #We created (zmax*2) x N images, where N is the number of defocused images
        # loop over all of (zmax*2) combinations and double-normalize and ravel N images
        # double-normalize = set sum of each image to 1 and then set the sum of all raveled images to 1
        
        list_of_images_normalized_uber=[]
        list_of_images_normalized_std_uber=[]
        # go over zmax*2 images
        for j in range(len(out_images)):
            # two steps for what could have been achived in one, but to ease up transition from previous code 
            out_images_single_parameter_change=out_images[j]
            optpsf_list=out_images_single_parameter_change
            ### breaking here
            # flux image has to correct per image
            # normalize and mask images that have been created in the fitting procedure
            images_normalized=[]
            for i in range(len(optpsf_list)):
                
                flux_mask=list_of_flux_mask[i]
                images_normalized.append((optpsf_list[i][flux_mask]).ravel())                
                #images_normalized.append((optpsf_list[i][flux_mask]/np.sum(optpsf_list[i][flux_mask])).ravel())
                
            images_normalized_flat=[item for sublist in images_normalized for item in sublist]  
            images_normalized_flat=np.array(images_normalized_flat)
            #images_normalized_flat=np.array(images_normalized_flat)/len(optpsf_list)        
            
            # list of (zmax-3)*2 raveled images
            list_of_images_normalized_uber.append(images_normalized_flat)
            
            # same but divided by STD
            images_normalized_std=[]
            for i in range(len(optpsf_list)):   
                # seems that I am a bit more verbose here with my definitions
                optpsf_list_i=optpsf_list[i]
                STD=list_of_sci_image_std[i]
                optpsf_list_i_STD=optpsf_list_i/STD    
                flux_mask=list_of_flux_mask[i]
                #images_normalized_std.append((optpsf_list_i_STD[flux_mask]/np.sum(optpsf_list_i_STD[flux_mask])).ravel())
            
            # join all images together
            #images_normalized_std_flat=[item for sublist in images_normalized_std for item in sublist]  
            # normalize so that the sum is still one
            #images_normalized_std_flat=np.array(images_normalized_std_flat)/len(optpsf_list)
            
            #list_of_images_normalized_std_uber.append(images_normalized_std_flat)
            
        # create uber images_normalized,images_normalized_std    
        # images that have zmax*2 rows and very large number of columns (number of non-masked pixels from all N images)
        uber_images_normalized=np.array(list_of_images_normalized_uber)    
        #uber_images_normalized_std=np.array(list_of_images_normalized_std_uber)          

        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/uber_images_normalized_'+str(num_iter)+'_'+str(iteration_number),\
                    uber_images_normalized)  
        
        #np.save('/tigress/ncaplar/Results/uber_images_normalized_std_'+str(num_iter)+'_'+str(iteration_number),\
        #        uber_images_normalized_std)  


        
        
        #single_wavefront_parameter_list=[]
        #for i in range(len(out_parameters)):
        #    single_wavefront_parameter_list.append(np.concatenate((out_parameters[i][:19],out_parameters[i][42:])) )
    
    
        
        ######################################################################################################### 
        # Core Tokovinin algorithm
    
        
    
        print('images_normalized (uber).shape: '+str(uber_images_normalized.shape))
        # equation A1 from Tokovinin 2006
        # new model minus old model
        if move_allparameters==True:
            H=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_all_parametrizations[:,None])    
            #H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/array_of_delta_z_parametrizations[:,None]) 
            H_std=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_all_parametrizations[:,None])/uber_std.ravel()[:,None]               
        else:                
            H=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_z_parametrizations[:,None])    
            #H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/array_of_delta_z_parametrizations[:,None]) 
            H_std=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_z_parametrizations[:,None])/uber_std.ravel()[:,None]     

        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/H_'+str(num_iter)+'_'+str(iteration_number),\
                    H)  
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/H_std_'+str(num_iter)+'_'+str(iteration_number),\
                    H_std)                  
        
        singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))<0.01]
        non_singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))>0.01]
        
        H=H[:,non_singlular_parameters]
        H_std=H_std[:,non_singlular_parameters]
    
        HHt=np.matmul(np.transpose(H),H)
        HHt_std=np.matmul(np.transpose(H_std),H_std) 
        #print('svd thresh is '+str(thresh))
        #invHHt=svd_invert(HHt,thresh)
        #invHHt_std=svd_invert(HHt_std,thresh)
        invHHt=np.linalg.inv(HHt)        
        invHHt_std=np.linalg.inv(HHt_std)
    
        invHHtHt=np.matmul(invHHt,np.transpose(H))
        invHHtHt_std=np.matmul(invHHt_std,np.transpose(H_std))

    
        # I is uber_I now (science images)
        # M0 is uber_M0 now (set of models before the iteration)
        first_proposal_Tokovnin=np.matmul(invHHtHt,uber_I-uber_M0)
        #first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,uber_I_std-uber_M0_std)
        first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,(uber_I-uber_M0)/uber_std.ravel())    
    
    
        #Tokovnin_proposal=0.7*first_proposal_Tokovnin
        if move_allparameters==True:
            Tokovnin_proposal=np.zeros((129,))
            #Tokovnin_proposal[non_singlular_parameters]=0.7*first_proposal_Tokovnin_std            
            Tokovnin_proposal[non_singlular_parameters]=1*first_proposal_Tokovnin_std     
            
            all_parametrization_new=np.copy(initial_input_parameterization)
            allparameters_parametrization_proposal_after_iteration_before_global_check=all_parametrization_new+Tokovnin_proposal
            # tests if the global parameters would be out of bounds - if yes, reset them to the limit values
            global_parametrization_proposal_after_iteration_before_global_check=\
                allparameters_parametrization_proposal_after_iteration_before_global_check[19*2:19*2+23]
            checked_global_parameters=check_global_parameters(global_parametrization_proposal_after_iteration_before_global_check,test_print=1)
            
            allparameters_parametrization_proposal_after_iteration=np.copy(allparameters_parametrization_proposal_after_iteration_before_global_check)
            allparameters_parametrization_proposal_after_iteration[19*2:19*2+23]=checked_global_parameters

            
            
        else:
            #Tokovnin_proposal=0.7*first_proposal_Tokovnin_std
            Tokovnin_proposal=1*first_proposal_Tokovnin_std

        
        print('Tokovnin_proposal[:5] is: '+str(Tokovnin_proposal[:5]))
        print('Tokovnin_proposal[38:43] is: '+str(Tokovnin_proposal[38:43]))
        #print('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))
        if move_allparameters==True:
            #all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)    
            #all_global_parametrization_new=np.copy(all_global_parametrization_old)
            #all_parametrization_new=np.copy(initial_input_parameterization)
            
            #allparameters_parametrization_proposal_after_iteration=all_parametrization_new+Tokovnin_proposal
            
            up_to_z22_end=allparameters_parametrization_proposal_after_iteration[:19*2]
            from_z22_end=allparameters_parametrization_proposal_after_iteration[19*2+23:]
            all_wavefront_z_parametrization_new=np.concatenate((up_to_z22_end,from_z22_end))
            
            all_global_parametrization_new=allparameters_parametrization_proposal_after_iteration[19*2:19*2+23]
            
            
        else:
            all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
            all_wavefront_z_parametrization_new=all_wavefront_z_parametrization_new+Tokovnin_proposal
            up_to_z22_end=all_wavefront_z_parametrization_new[:19*2]
            from_z22_end=all_wavefront_z_parametrization_new[19*2:]
            allparameters_parametrization_proposal_after_iteration=np.concatenate((up_to_z22_end,nonwavefront_par,from_z22_end))
    
        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/first_proposal_Tokovnin'+str(num_iter)+'_'+str(iteration_number),\
                    first_proposal_Tokovnin) 
            np.save('/tigress/ncaplar/Results/first_proposal_Tokovnin_std'+str(num_iter)+'_'+str(iteration_number),\
                    first_proposal_Tokovnin_std)   
            np.save('/tigress/ncaplar/Results/allparameters_parametrization_proposal_after_iteration_'+str(num_iter)+'_'+str(iteration_number),\
                    allparameters_parametrization_proposal_after_iteration)    
    
        #########################
        # Creating single exposure with new proposed parameters and seeing if there is improvment    
        list_of_parameters_after_iteration=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal_after_iteration,\
                                                                                    list_of_defocuses=list_of_defocuses_input_long,zmax=56)
        res_multi=model_multi(list_of_parameters_after_iteration,return_Images=True)



        final_model_result,list_of_final_model_result,list_of_image_final,\
                    list_of_finalinput_parameters,list_of_after_chi2,list_of_final_psf_positions=res_multi

        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/list_of_final_model_result_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_final_model_result)                        
            np.save('/tigress/ncaplar/Results/list_of_image_final_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_image_final)    
            np.save('/tigress/ncaplar/Results/list_of_finalinput_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_finalinput_parameters)   
            np.save('/tigress/ncaplar/Results/list_of_after_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_after_chi2)        
            np.save('/tigress/ncaplar/Results/list_of_final_psf_positions_'+str(num_iter)+'_'+str(iteration_number),\
                    list_of_final_psf_positions)                       

        print('list_of_final_psf_positions : '+str(list_of_psf_positions))
              
       
        ######################################################################################################### 
        # divided model images by their standard deviations
        
        list_of_image_final_std=[]
        for i in range(len(list_of_image_0)):
            # normalizing by standard deviation image
            STD=np.sqrt(list_of_var_images[i])    
            image_final=list_of_image_final[i]
            list_of_image_final_std.append(image_final/STD)
        
    
        ######################################################################################################### 
        #  masked model images after this iteration
        
        
        list_of_M_final=[]
        list_of_M_final_std=[]
        for i in range(len(list_of_image_final_std)):
            
            image_final=list_of_image_final[i]
            image_final_std=list_of_image_final_std[i]
            flux_mask=list_of_flux_mask[i]
            # what is list_of_mask_images?
            
            #M_final=((image_final[flux_mask])/np.sum(image_final[flux_mask])).ravel()
            M_final=(image_final[flux_mask]).ravel()
            #M_final_std=((image_final_std[flux_mask])/np.sum(image_final_std[flux_mask])).ravel()
            M_final_std=((image_final_std[flux_mask])/1).ravel()
            
            list_of_M_final.append(M_final)
            list_of_M_final_std.append(M_final_std)
    
        # join all M0,M0_std from invidiual images into one uber M0,M0_std
        uber_M_final=[item for sublist in list_of_M_final for item in sublist]
        uber_M_final_std=[item for sublist in list_of_M_final_std for item in sublist]   
       
        uber_M_final=np.array(uber_M_final)
        uber_M_final_std=np.array(uber_M_final_std)
        
        uber_M_final=uber_M_final
        #uber_M_final_std=uber_M_final_std/np.sum(uber_M_final_std)

        if num_iter!=None:
            np.save('/tigress/ncaplar/Results/uber_M_final_'+str(num_iter)+'_'+str(iteration_number),\
                    uber_M_final)                        
            #np.save('/tigress/ncaplar/Results/uber_M_final_std_'+str(num_iter)+'_'+str(iteration_number),\
            #        uber_M_final_std)    

        
        ####
        # Seeing if there is improvment
        
        # non-std version
        # not used, that is ok, we are at the moment using std version
        IM_final=np.sum(np.abs(np.array(uber_I)-np.array(uber_M_final)))        
        # std version 
        IM_final_std=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M_final_std)))     
    
        print('I-M_start before iteration '+str(iteration_number)+': '+str(IM_start))    
        print('I-M_final after iteration '+str(iteration_number)+': '+str(IM_final))
        
        print('I_std-M_start_std after iteration '+str(iteration_number)+': '+str(IM_start_std))        
        print('I_std-M_final_std after iteration '+str(iteration_number)+': '+str(IM_final_std))
        
        print('Likelihood before iteration '+str(iteration_number)+': '+str(initial_model_result))
        print('Likelihood after iteration '+str(iteration_number)+': '+str(final_model_result))
        
        #print('chi_2_after_iteration/chi_2_before_iteration '+str(chi_2_after_iteration/chi_2_before_iteration ))
        print('IM_final/IM_start '+str(IM_final/IM_start))
        print('IM_final_std/IM_start_std '+str(IM_final_std/IM_start_std))
        print('#########################################################')
        #if chi_2_after_iteration/chi_2_before_iteration <1.02 :
    
        ##################
        # If improved take new parameters, if not dont
        
        if IM_final_std/IM_start_std <1.0 :        
        #if IM_final_std/IM_start_std <1.0 :
            #when the quality measure did improve
            did_chi_2_improve=1
            number_of_non_decreses.append(0)
        else:
            #when the quality measure did not improve
            did_chi_2_improve=0
            # resetting all parameters
            if move_allparameters==True:
                all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                all_global_parametrization_new=np.copy(all_global_parametrization_old)
                allparameters_parametrization_proposal_after_iteration=initial_input_parameterization
            else:
                all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                chi_2_after_iteration=chi_2_before_iteration
                up_to_z22_end=all_wavefront_z_parametrization_new[:19*2]
                from_z22_start=all_wavefront_z_parametrization_new[19*2:]
                allparameters_parametrization_proposal_after_iteration=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
            thresh=thresh0
            number_of_non_decreses.append(1)
            print('number_of_non_decreses:' + str(number_of_non_decreses))
            print('current value of number_of_non_decreses is: '+str(np.sum(number_of_non_decreses)))
            print('#############################################')
    
        if np.sum(number_of_non_decreses)==1:
            return final_model_result,list_of_final_model_result,list_of_image_final,\
                allparameters_parametrization_proposal_after_iteration,list_of_finalinput_parameters,list_of_after_chi2,list_of_final_psf_positions
            
            break
    
        #if len(allparameters_proposal_after_iteration)==41:
        #    allparameters_proposal_after_iteration=np.concatenate((allparameters_proposal_after_iteration,np.array([1])))
                
        #return likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration,chi2_after_iteration_array        
        
    return final_model_result,list_of_final_model_result,list_of_image_final,\
            allparameters_parametrization_proposal_after_iteration,list_of_finalinput_parameters,list_of_after_chi2,list_of_final_psf_positions

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
        sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
        var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')   
    
    
        sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
        var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')

    else:
        STAMPS_FOLDER='/tigress/ncaplar/ReducedData/Data_Aug_14/Stamps_cleaned_fake/'
        sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
        mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
        var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy') 
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
            if single_number<120:
                list_of_sci_images_focus.append(sci_image_focus_large)
                list_of_var_images_focus.append(var_image_focus_large)
            # observation which are of good enough quality
            list_of_obs_cleaned.append(obs)
            
print('len of list_of_sci_images: '+str(len(list_of_sci_images)) )           

# If there is no valid images imported, exit 
if list_of_sci_images==[]:
    print('No valid images - exiting')
    sys.exit(0)



# name of the outputs
# should update that it goes via date
list_of_NAME_OF_CHAIN=[]
list_of_NAME_OF_LIKELIHOOD_CHAIN=[]
for obs in list_of_obs_cleaned:
    NAME_OF_CHAIN='chain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    NAME_OF_LIKELIHOOD_CHAIN='likechain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
    
    # if you are passing higher than 120
    if single_number_original>=120:
        NAME_OF_CHAIN='chain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number_original)+str(eps)+str(arc)
        NAME_OF_LIKELIHOOD_CHAIN='likechain'+str(date_of_output)+'20_Single_P_'+str(obs)+str(single_number_original)+str(eps)+str(arc)        
    
    
    list_of_NAME_OF_CHAIN.append(NAME_OF_CHAIN)
    list_of_NAME_OF_LIKELIHOOD_CHAIN.append(NAME_OF_LIKELIHOOD_CHAIN)


# where are the dataframe which we use to guess the initial solution
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_HgAr_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_input_HgAr=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Ne_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_input_Ne=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_'+str(direct_or_interpolation)+'_Kr_from_'+str(date_of_input)+'.pkl', 'rb') as f:
    results_of_fit_input_Kr=pickle.load(f)
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
        obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11748,11694,11700,11706,11712,11718,11724,11730,11736])
    elif arc=='Ne':
        obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12355,12349,12343,12337,12331,12325,12319,12313,12307])  
 
# F/2.8 data       
if dataset==2:
    if arc=='HgAr':
        obs_possibilites=np.array([17023,17023+6,17023+12,17023+18,17023+24,17023+30,17023+36,17023+42,17023+48,17023+48,17023+54,17023+60,17023+66,17023+72,17023+78,17023+84,17023+90,17023+96,17023+48])
    if arc=='Ne':
        obs_possibilites=np.array([16238+6,16238+12,16238+18,16238+24,16238+30,16238+36,16238+42,16238+48,16238+54,16238+54,16238+60,16238+66,16238+72,16238+78,16238+84,16238+90,16238+96,16238+102,16238+54])
    if arc=='Kr':
         obs_possibilites=np.array([17310+6,17310+12,17310+18,17310+24,17310+30,17310+36,17310+42,17310+48,17310+54,17310+54,17310+60,17310+66,17310+72,17310+78,17310+84,17310+90,17310+96,17310+102,17310+54])

# F/2.5 data     
if dataset==3:
    if arc=='HgAr':
        obs_possibilites=np.array([19238,19238+6,19238+12,19238+18,19238+24,19238+30,19238+36,19238+42,19238+48,19238+48,19238+54,19238+60,19238+66,19238+72,19238+78,19238+84,19238+90,19238+96,19238+48])
    elif arc=='Ne':
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
label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']

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
for i in ['m15','m1','m05','0d','0','p05','p1','p15']:
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
    labelInput='0p'
else:
    pass

# Input the zmax that you wish to achieve in the analysis
zmax=twentytwo_or_extra
 
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


# depening on the arc, select the appropriate dataframe
if arc=="HgAr":
    results_of_fit_input=results_of_fit_input_HgAr
elif arc=="Ne":
    results_of_fit_input=results_of_fit_input_Ne
elif arc=="Kr":
    results_of_fit_input=results_of_fit_input_Kr  
else:
    print("what has happened here? Only HgAr, Neon and Krypton implemented")


# if you are passing only image    
if len(list_of_obs)==1:
        
    # Create an object (called model) - which gives likelihood given the parameters 
    model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
          pupil_parameters=None,use_pupil_parameters=None,\
          save=0,verbosity=0,double_sources=double_sources,zmax=zmax,\
          double_sources_positions_ratios=double_sources_positions_ratios,npix=1536)
else:
    # otherwise, if you are passing multiple images
    model_multi = LN_PFS_multi_same_spot(list_of_sci_images=list_of_sci_images,list_of_var_images=list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                         dithering=1,save=0,verbosity=0,npix=1536,list_of_defocuses=list_of_labelInput,zmax=zmax,double_sources=double_sources,\
                                         double_sources_positions_ratios=double_sources_positions_ratios,test_run=False)           

# if you are passing only one image
if len(list_of_obs)==1:
    # results_of_fit_input contains proposal for parameters plus 2 values describing chi2 and chi2_max 
    # need to extract all values except the last two
    allparameters_proposalp2=results_of_fit_input[labelInput].loc[int(single_number)].values
      
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
        
    # protect the parameters from changing and unintentional mistakes
    allparameters_proposal_22=np.copy(allparameters_proposal)
else:
    # you are apssing multiple images
    list_of_allparameters=[]
    list_of_defocuses=[]
    # search for the previous avaliable results 
    # add the ones that you found in array_of_allparameters and for which labels they are avaliable in list_of_defocuses
    for label in ['m4','m35','m3','m05','0','p05','p3','p35','p4']:
        try:
            if single_number>=120:
                list_of_allparameters.append(results_of_fit_input[label].loc[int(37)].values)
                list_of_defocuses.append(label)
            else:
                
            
                list_of_allparameters.append(results_of_fit_input[label].loc[int(single_number)].values)
                list_of_defocuses.append(label)

            
            
            

        except:
            pass
        
    array_of_allparameters=np.array(list_of_allparameters)
    
    # based on the information from the previous step (results at list_of_defocuses), generate singular array_of_allparameters at list_of_labelInput positions        
    # has shape 2xN, N=number of parameters
    
    print('array_of_allparameters.shape: '+str(array_of_allparameters.shape))
    print('twentytwo_or_extra: '+str(twentytwo_or_extra))
    array_of_polyfit_1_parameterizations_proposal=model_multi.create_resonable_allparameters_parametrizations(array_of_allparameters=array_of_allparameters,\
                                                                list_of_defocuses_input=list_of_defocuses,zmax=twentytwo_or_extra,remove_last_n=2)
    
    # lets be explicit that the shape of the array is 2d
    array_of_polyfit_1_parameterizations_proposal_shape_2d=array_of_polyfit_1_parameterizations_proposal
    
    # if you 
    if single_number>=120:
        proposal_number=single_number_original-120
        
        array_of_polyfit_1_parameterizations_proposal_shape_2d=\
            np.load('/tigress/ncaplar/ReducedData/Data_Aug_14/Dataframes_fake/array_of_polyfit_1_parameterizations_proposal_shape_2d_proposal_'+str(proposal_number)+'.npy')
        
    print('Overriding array_of_polyfit_1_parameterizations_proposal_shape_2d for testing purposes') 
   
   
    array_of_polyfit_1_parameterizations_proposal_shape_2d=np.array([[   -7.48584024,     0.73497698],
       [    0.19568978,     0.15682353],
       [    0.03769565,     0.02577918],
       [   -0.03734586,     0.39183789],
       [    0.02550665,     0.556507  ],
       [    0.0072312 ,     0.1261021 ],
       [    0.01097693,    -0.26493479],
       [   -0.01831573,    -0.05247759],
       [    0.01020252,    -0.15314955],
       [    0.00343817,    -0.10475206],
       [    0.02711867,    -0.0137685 ],
       [    0.00445374,     0.0372618 ],
       [    0.01458828,    -0.02237409],
       [   -0.01880653,     0.03841714],
       [   -0.00188257,     0.02112357],
       [   -0.00910924,     0.01975998],
       [   -0.00067647,    -0.02366152],
       [    0.00463751,     0.012873  ],
       [   -0.00621102,     0.02695559],
       [    0.        ,     0.68904049],
       [    0.        ,     0.10290975],
       [    0.        ,    -0.20154677],
       [    0.        ,    -0.01441376],
       [    0.        ,     0.05355728],
       [    0.        ,     0.05813265],
       [    0.        ,     0.0000314 ],
       [    0.        ,     0.0000314 ],
       [    0.        ,     0.9767313 ],
       [    0.        ,     0.94836248],
       [    0.        ,     0.02461952],
       [    0.        ,    -0.05835101],
       [    0.        ,     0.93393462],
       [    0.        ,     0.04867201],
       [    0.        ,     0.49904901],
       [    0.        ,     1.01173008],
       [    0.        ,     0.63182418],
       [    0.        , 51347.90306188],
       [    0.        ,     2.32213302],
       [    0.        ,     0.00238991],
       [    0.        ,     0.37175131],
       [    0.        ,     1.79411345],
       [    0.        ,     0.99627532],
       [    0.00003301,    -0.00155895],
       [   -0.0007012 ,     0.00400087],
       [    0.00093449,     0.00437024],
       [    0.00096567,     0.00066246],
       [    0.00095771,     0.00368164],
       [    0.00029302,    -0.00083184],
       [    0.00196281,    -0.00503305],
       [    0.00055673,     0.00092236],
       [   -0.00170941,    -0.00397606],
       [    0.00222377,     0.01257303],
       [    0.00219571,     0.00159809],
       [   -0.00109161,     0.00276234],
       [   -0.00071055,    -0.00917837],
       [    0.00070316,    -0.00561311],
       [   -0.00074906,     0.00218633],
       [    0.00052506,     0.00031175],
       [   -0.00027124,     0.00124874],
       [    0.00056686,    -0.0004586 ],
       [   -0.00085383,     0.00176925],
       [   -0.00200233,    -0.00099071],
       [   -0.00017492,     0.00005366],
       [    0.00075724,    -0.00381397],
       [    0.0016349 ,    -0.00899893],
       [    0.00057199,    -0.01150359],
       [   -0.00085055,     0.00203238],
       [   -0.00052982,     0.0053546 ],
       [    0.00095456,    -0.00181697],
       [   -0.00055045,    -0.00379142],
       [   -0.00048768,     0.00804741],
       [   -0.00145526,     0.00574891],
       [   -0.0001161 ,    -0.00048279],
       [   -0.00099326,     0.00201623],
       [    0.00012449,     0.0103669 ],
       [    0.00272563,    -0.00187062]])
    
    
    
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
 ##############################################    
    

# soon deprecated?
# branch out here, depending if you are doing low- or high- Zernike analysis 
if twentytwo_or_extra>=22:  

 
    # initialize pool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)   
        
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
        
            
        
        print('Testing proposal is: '+str(array_of_polyfit_1_parameterizations_proposal_shape_2d))
        time_start_single=time.time()
        print('Likelihood for the testing proposal is: '+str(model_multi(array_of_polyfit_1_parameterizations_proposal_shape_2d)))
        time_end_single=time.time()
        print('Time for single calculation is '+str(time_end_single-time_start_single))
            
        #allparameters_proposal=allparameters_proposal_22
        # dirty hack to get things running
        if zmax>=22:
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
        stronger_array_01=np.array([1,0,0,0,0,0,0,0,
                                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0,
                                0,0,0,0,
                                0,0,0,
                                0,0,0,0,
                                1,1,1,
                                1,1,1])  
    else:
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
            
            stronger_array_01=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1.2,1.2,1.2,1.2,1.2,1.2,
                        0.0,0.0,0.0,0.0,
                        1.2,1.2,1.2,
                        1.2,1.2,1.2,1.2,
                        1.2,1.2,1.2,
                        1.2,1.2,1])            
            
     # should do Tokovinin here, then scatter parameters       
            
    print('array_of_polyfit_1_parameterizations_proposal_shape_2d: '+str(array_of_polyfit_1_parameterizations_proposal_shape_2d))
    
    # create inital parameters  
    parInit1=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,multi=multi_var,pupil_parameters=None,\
                            allparameters_proposal_err=None,\
                            stronger=stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    # the number of walkers is given by options array
    while len(parInit1)<options[0]:
        parInit1_2=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,multi=multi_var,pupil_parameters=None,\
                                  allparameters_proposal_err=None,\
                                  stronger= stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
        parInit1=np.vstack((parInit1,parInit1_2))
    
    chain = cosmoHammer.LikelihoodComputationChain()
    if multi_var==True:
        chain.addLikelihoodModule(PFSLikelihoodModule(model_multi))
    else:
        chain.addLikelihoodModule(PFSLikelihoodModule(model))
    
    # number of walkers - as specified in options
    # number of steps - as specified in options
    pso = cosmoHammer.ParticleSwarmOptimizer(chain, low=np.min(parInit1,axis=0), high=np.max(parInit1,axis=0),particleCount=options[0],pool=pool)
    
    for i in range(len(pso.swarm)):
        pso.swarm[i].position=parInit1[i]
    
    #returns all swarms and the best particle for all iterations
    #swarms, gbests = pso.optimize(maxIter=nsteps,c1=options[1],c2=options[2])
    list_of_swarms_time_1=[]
    time_start_swarms_time_1=time.time()
    print('pso.gbest before any nsteps: '+str(pso.gbest))
    swarms=[]
    gbests=[]
    num_iter = 0
    random_seed=42
    for swarm in pso.sample(nsteps):
        print('starting num_iter: '+str(num_iter))
        swarms.append(swarm)
        gbests.append(pso.gbest.copy())
        num_iter += 1
        if pso.isMaster():
            time_end_swarms_time_1=time.time()
            list_of_swarms_time_1.append([num_iter,time_end_swarms_time_1-time_start_swarms_time_1])
            
            print('num_iter % 5: '+str(num_iter % 5))
            #print('pso.swarm[0] in step'+str(num_iter)+' '+str(pso.swarm[0]))
            #print('pso.swarm[1] in step'+str(num_iter)+' '+str(pso.swarm[1]))
            #swarm_positions_sum=[]
            #for i in range(len(pso.swarm)):
            #    swarm_positions_sum.append(np.sum(pso.swarm[i].position))
                 
            print('pso.gbest in step: '+str(num_iter)+' '+str(pso.gbest))    
            print('pso.swarm[0] in step: '+str(num_iter)+' '+str(pso.swarm[0]))  
            print('pso.swarm[1] in step: '+str(num_iter)+' '+str(pso.swarm[1]))  
            #index_of_best_position=swarm_positions_sum.index(np.sum(pso.gbest.position))
            #print('pso.index_of_best_position in step'+str(num_iter)+' '+str(index_of_best_position))
            #print('pso.swarm[index_of_best_position] in step'+str(num_iter)+' '+str(pso.swarm[index_of_best_position]))            
            
            


            print("First swarm: "+str(100*num_iter/(nsteps)))
            sys.stdout.flush()
                
            if num_iter % 5 ==1:
                
                
                
                # if this is first step, take the prepared proposal - but than it would be num_iter=1!!! 
                if num_iter==0:
                    array_of_polyfit_1_parameterizations_proposal_shape_1d=move_parametrizations_from_2d_shape_to_1d_shape(array_of_polyfit_1_parameterizations_proposal_shape_2d)
                else:  
                    array_of_polyfit_1_parameterizations_proposal_shape_1d=pso.gbest.position
            
                list_of_Tokovin_results=[]

                # 0. likelihood_after_iteration - mean out all of the images
                # 1. list_of_likelihoods_after_iteration - likelihood per images
                # 2. list_of_images_final previously named ``final_optpsf_image'' - list of generated images
                # 3. allparameters_parametrization_final - output parametrizations
                # 4. list_of_finalinput_parameters - list of parameters per image
                # 5. list_of_after_chi2 - list of arrays giving quality 
                likelihood_after_iteration,list_of_likelihoods_after_iteration,list_of_images_final,\
                allparameters_parametrization_final,list_of_finalinput_parameters,list_of_after_chi2,list_of_psf_positions=\
                    Tokovinin_algorithm_chi_multi(list_of_sci_images,list_of_var_images,list_of_mask_images,\
                                                  array_of_polyfit_1_parameterizations_proposal_shape_1d,num_iter=num_iter)
            
                list_of_Tokovin_results.append([likelihood_after_iteration,list_of_likelihoods_after_iteration,\
                                                list_of_images_final,allparameters_parametrization_final,
                                                list_of_finalinput_parameters,list_of_after_chi2,list_of_psf_positions])    
                """    
                likelihood_after_iteration,list_of_likelihoods_after_iteration,list_of_images_final,\
                allparameters_parametrization_final,list_of_finalinput_parameters,list_of_after_chi2=\
                    Tokovinin_algorithm_chi_multi(list_of_sci_images,list_of_var_images,list_of_mask_images,\
                                                  allparameters_parametrization_final,num_iter=None,move_allparameters=True)
            
                list_of_Tokovin_results.append([likelihood_after_iteration,list_of_likelihoods_after_iteration,\
                                                list_of_images_final,allparameters_parametrization_final,
                                                list_of_finalinput_parameters,list_of_after_chi2])      
                """  
       
                # replacing the playing above from many images above with this substitution
                allparameters_best_parametrization_shape_1d=allparameters_parametrization_final

                print('result from Tokovinin (pso.gbest.position) in num_iter '+str(num_iter)+' is: '+str(allparameters_best_parametrization_shape_1d))

                # replacing the best position 
                pso.gbest.fitness=likelihood_after_iteration
                pso.gbest.position=allparameters_best_parametrization_shape_1d
                # set up velocity the wavefront up to z22
                pso.gbest.velocity[:19*2]=np.zeros(((19*2),))
                # set up velocity to zero (for all wavefront parameters)
                if zmax>22:
                    pso.gbest.velocity[-(number_of_extra_zernike*2+1):]=np.zeros(((number_of_extra_zernike*2+1),))
                    
                # replacing all the wavefront component in all of the particles                
                for i in range(len(pso.swarm)):
                    #print(allparameters_best_parametrization_shape_1d)
                    pso.swarm[i].position[0:19*2]=allparameters_best_parametrization_shape_1d[0:19*2]     
                    
                    if zmax>22:
                        pso.swarm[i].position[19*2+23:]=allparameters_best_parametrization_shape_1d[19*2+23:]
                        pso.swarm[i].velocity[-(number_of_extra_zernike*2+1):]=np.zeros(((number_of_extra_zernike*2+1),))
                        

               
                
                
                
                
                
                
                
                
                
                
                
                
        
    
    minchain=gbests[-1].position
    minln=gbests[-1].fitness
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].position)     
    res=np.array(res)
    chains=res.reshape(len(swarms),len(swarms[0]),parInit1.shape[1])
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].fitness)         
    res=np.array(res)
    ln_chains=res.reshape(len(swarms),len(swarms[0]))
    
    res=[]
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j].velocity)         
    res=np.array(res)
    v_chains=res.reshape(len(swarms),len(swarms[0]),parInit1.shape[1])
    
    res_gbests_position=[]
    res_gbests_fitness=[]
    for i in range(len(swarms)):
            minchain_i=gbests[i].position
            minln_i=gbests[i].fitness
            res_gbests_position.append(minchain_i)
            res_gbests_fitness.append(minln_i)
        
    
    
    #Export this fileparInit1
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'parInit1',parInit1)    
    
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Swarm1',chains)
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'velocity_Swarm1',v_chains)
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'list_of_Tokovin_results',list_of_Tokovin_results)    
    
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'res_gbests_position',res_gbests_position)
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'res_gbests_fitness',res_gbests_fitness)
    
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Swarm1',ln_chains) 
       
    
    
    #minchain=chains[np.abs(ln_chains)==np.min(np.abs(ln_chains))][0]
    chi2reduced=2*np.min(np.abs(ln_chains))/(sci_image.shape[0])**2
    minchain_err=[]
    for i in range(len(minchain)):
        minchain_err=np.append(minchain_err,np.std(chains[:,:,i].flatten()))
    
    minchain_err=np.array(minchain_err)
    
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'_list_of_swarms_time_1',list_of_swarms_time_1)    
    
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
