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

__version__ = "0.34"

parser = argparse.ArgumentParser(description="Starting args import",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog='Done with import')

############################################################################################################
print('############################################################################################################')  

############################################################################################################
print('Start time is: '+time.ctime())  
time_start=time.time()  

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

# Finished with all arguments
################################################  
# Tokovinin function at the moment here

def Tokovinin_algorithm_chi(sci_image,var_image,mask_image,allparameters_proposal):
   
    
    model = LN_PFS_single(sci_image,var_image,mask_image=mask_image,\
                      dithering=1,save=0,zmax=22,verbosity=0,double_sources=None,\
                      double_sources_positions_ratios=double_sources_positions_ratios,fit_for_flux=True,npix=1536,pupil_parameters=None)  


   


    print('Initial testing proposal is: '+str(allparameters_proposal))
    time_start_single=time.time()
    pre_model_result,pre_image,pre_input_parameters,chi_2_before_iteration_array=model(allparameters_proposal,return_Image=True,return_intermediate_images=False)
    chi_2_before_iteration=chi_2_before_iteration_array[2]
    time_end_single=time.time()
    print('Total time taken was  '+str(time_end_single-time_start_single)+' seconds')
    print('chi_2_pre_input_parameters is: '+str(chi_2_before_iteration_array[2]))  

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
            number_of_extra_zernike=0
            #twentytwo_or_extra=22
            # numbers that make sense are 11,22,37,56,79,106,137,172,211,254
            if number_of_extra_zernike is None:
                number_of_extra_zernike=0
            else:
                number_of_extra_zernike=twentytwo_or_extra-22

        # list of how much to move Zernike coefficents
        list_of_delta_z=[]
        for z_par in range(3,22+number_of_extra_zernike):
            list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))

        # array, randomized delta extra zernike
        #array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1.2+1
        array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1
        array_of_delta_z=+np.array(list_of_delta_z)*array_of_delta_randomize
        
        # initialize 
        # if this is the first iteration of the iterative algorithm
        if iteration_number==0:
            thresh=thresh0
            all_wavefront_z_old=np.concatenate((allparameters_proposal[0:19],np.zeros(number_of_extra_zernike)))
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

        initial_input_parameters=np.concatenate((up_to_z22_start,pre_input_parameters[19:],from_z22_start))
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
        print('np.sum(np.abs(I_std-M0_std)) before iteration '+str(iteration_number)+': '+str(IM_start_std))  

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
        
        
        
        print('images_normalized.shape: '+str(images_normalized.shape))
        # equation A1 from Tokovinin 2006
        H=np.transpose(np.array((images_normalized-M0))/array_of_delta_z[:,None])    
        H_std=np.transpose(np.array((images_normalized_std-M0_std))/array_of_delta_z[:,None]) 

        HHt=np.matmul(np.transpose(H),H)
        HHt_std=np.matmul(np.transpose(H_std),H_std) 
        print('svd thresh is '+str(thresh))
        invHHt=svd_invert(HHt,thresh)
        invHHt_std=svd_invert(HHt_std,thresh)
        
        invHHtHt=np.matmul(invHHt,np.transpose(H))
        invHHtHt_std=np.matmul(invHHt_std,np.transpose(H_std))

        first_proposal_Tokovnin=np.matmul(invHHtHt,I-M0)
        first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,I_std-M0_std)
        
        #Tokovnin_proposal=0.9*first_proposal_Tokovnin
        Tokovnin_proposal=0.7*first_proposal_Tokovnin_std
        print('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))

        all_wavefront_z_new=np.copy(all_wavefront_z_old)
        all_wavefront_z_new=all_wavefront_z_new+Tokovnin_proposal
        up_to_z22_end=all_wavefront_z_new[:19]
        from_z22_end=all_wavefront_z_new[19:]
        allparameters_proposal_after_iteration=np.concatenate((up_to_z22_end,pre_input_parameters[19:42],from_z22_end))

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
            
        if IM_final_std/IM_start_std <1.002 :
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
    sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')
    mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')    
    var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_Stacked.npy')   
    sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
    var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')

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
zmax=22
 
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
                                         double_sources_positions_ratios=double_sources_positions_ratios)           

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

    list_of_allparameters=[]
    list_of_defocuses=[]
    # search for the previous avaliable results 
    # add the ones that you found in array_of_allparameters and for which labels they are avaliable in list_of_defocuses
    for label in ['m4','m35','m3','m05','0','p05','p3','p35','p4']:
        try:
            list_of_allparameters.append(results_of_fit_input[label].loc[int(single_number)].values)
            list_of_defocuses.append(label)
        except:
            pass
        
    array_of_allparameters=np.array(list_of_allparameters)
    
    # based on the information from the previous step (results at list_of_defocuses), generate singular array_of_allparameters at list_of_labelInput positions        
    # has shape 2xN, N=number of parameters
    array_of_polyfit_1_parameterizations_proposal=model_multi.create_resonable_allparameters_parametrizations(array_of_allparameters=array_of_allparameters,\
                                                                list_of_defocuses_input=list_of_defocuses,zmax=zmax)
    #list_of_allparameters_proposal=model_multi.create_list_of_allparameters(array_of_polyfit_1_parameterizations_proposal,list_of_defocuses=['m4','p4'])    

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
 ##############################################    
    

# soon deprecated?
# branch out here, depending if you are doing low- or high- Zernike analysis 
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
            
        print('Testing proposal is: '+str(array_of_polyfit_1_parameterizations_proposal))
        time_start_single=time.time()
        print('Likelihood for the testing proposal is: '+str(model_multi(array_of_polyfit_1_parameterizations_proposal)))
        time_end_single=time.time()
        print('Time for single calculation is '+str(time_end_single-time_start_single))
            
        #allparameters_proposal=allparameters_proposal_22
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
                        0.2,0.2,0.2,0.2,0.2,0.2,
                        0.0,0.0,0.0,0.0,
                        0.2,0.2,0.2,
                        0.2,0.2,0.2,0.2,
                        0.2,0.2,0.2,
                        0.2,0.2,1])            
            
            
            
    
    # create inital parameters  
    parInit1=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal,multi=multi_var,pupil_parameters=None,allparameters_proposal_err=None,\
                            stronger=1*stronger_array_01,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
    # the number of walkers is given by options array
    while len(parInit1)<options[0]:
        parInit1_2=create_parInit(allparameters_proposal=array_of_polyfit_1_parameterizations_proposal,multi=multi_var,pupil_parameters=None,allparameters_proposal_err=None,\
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
    
    swarms=[]
    gbests=[]
    num_iter = 0
    random_seed=42
    for swarm in pso.sample(nsteps):
        swarms.append(swarm)
        gbests.append(pso.gbest.copy())
        num_iter += 1
        if pso.isMaster():
            time_end_swarms_time_1=time.time()
            list_of_swarms_time_1.append(time_end_swarms_time_1-time_start_swarms_time_1)
            
            print('num_iter % 5: '+str(num_iter % 5))
            #print('pso.swarm[0] in step'+str(num_iter)+' '+str(pso.swarm[0]))
            #print('pso.swarm[1] in step'+str(num_iter)+' '+str(pso.swarm[1]))
            swarm_positions_sum=[]
            for i in range(len(pso.swarm)):
                swarm_positions_sum.append(np.sum(pso.swarm[i].position))
                
            #pso.gbest.position[0]=-6   
            print('pso.gbest in step'+str(num_iter)+' '+str(pso.gbest))    
            #index_of_best_position=swarm_positions_sum.index(np.sum(pso.gbest.position))
            #print('pso.index_of_best_position in step'+str(num_iter)+' '+str(index_of_best_position))
            #print('pso.swarm[index_of_best_position] in step'+str(num_iter)+' '+str(pso.swarm[index_of_best_position]))            
            
            


            print("First swarm: "+str(100*num_iter/(nsteps)))
            sys.stdout.flush()
                
            if num_iter % 5 ==1:
                
                # if this is first step, take the preprepered proposal 
                if num_iter==0:
                    array_of_polyfit_1_parameterizations_proposal=array_of_polyfit_1_parameterizations_proposal
                else:  
                    array_of_polyfit_1_parameterizations_proposal=model_multi.move_parametrizations_from_1d_to_2d(pso.gbest.position)
                       
                
                #print('swarm: '+str(len(swarm)))
                #print('swarm[0]: '+str(swarm[0]))    
                list_of_Tokovin_results=[]
                swarm_positions_sum=[]
                for i in range(len(swarm)):
                    swarm_positions_sum.append(np.sum(swarm[i].position))
                 
                # analyze only images which are outside focus with Tokovinin method    
                for index_of_single_image in index_of_list_of_labelInput_without_focus_or_near_focus:
                    
                    print('index_of_single_image: '+str(index_of_single_image))
                    print('len of list_of_sci_images '+str(len(list_of_sci_images)))
                    
                    sci_image=list_of_sci_images[index_of_single_image]
                    var_image=list_of_var_images[index_of_single_image]
                    mask_image=list_of_mask_images[index_of_single_image]
                    
                    
                    list_of_allparameters_proposal=model_multi.create_list_of_allparameters(array_of_polyfit_1_parameterizations_proposal,\
                                                                                            list_of_defocuses=list_of_labelInput)    

                    
                    
                    
                    # need to move between parametrizations and parameters
                    likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration,chi2_after_iteration_array=\
            Tokovinin_algorithm_chi(sci_image,var_image,mask_image,list_of_allparameters_proposal[index_of_single_image])    
                    print('chi2_after_iteration_array:'+str(chi2_after_iteration_array))
                
                    list_of_Tokovin_results.append([likelihood_after_iteration,final_optpsf_image,allparameters_proposal_after_iteration,chi2_after_iteration_array])      

                
                
                
                array_of_Tokovin_results=[]
                for i in range(len(list_of_Tokovin_results)):
                    array_of_Tokovin_results.append(list_of_Tokovin_results[i][2])
                array_of_Tokovin_results=np.array(array_of_Tokovin_results)        
        
                print('array_of_Tokovin_results shape: ' + str(array_of_Tokovin_results.shape))
                # array_of_Tokovin_results has shape 6x42, we need 60x1
                allparameters_best_parametrization_2d=model_multi.create_resonable_allparameters_parametrizations(array_of_Tokovin_results,\
                                                                    list_of_defocuses_input=list_of_labelInput_without_focus_or_near_focus,zmax=22,remove_last_n=0)
                
                if allparameters_best_parametrization_2d.shape[0]>42:
                
                    allparameters_best_parametrization_1d=np.concatenate((allparameters_best_parametrization_2d[:19].ravel(),
                                                                allparameters_best_parametrization_2d[19:19+23][:,1],allparameters_best_parametrization_2d[19+23:].ravel()))
                    
                else:
                    allparameters_best_parametrization_1d=np.concatenate((allparameters_best_parametrization_2d[:19].ravel(),
                                                                allparameters_best_parametrization_2d[19:-1][:,1]))    
                print(allparameters_best_parametrization_1d)    
                #index_of_best_position=swarm_positions_sum.index(np.sum(pso.gbest.position))
                #print('index_of_best_position '+str(index_of_best_position))
                #print('pso.gbest: '+str(pso.gbest.position))
                
                
                # replacing only the best position 
                pso.gbest.position=allparameters_best_parametrization_1d
                #pso.swarm[index_of_best_position].position[0]=allparameters_parametrizations
                #print('swarm[index_of_best_position]'+str(pso.swarm[index_of_best_position]))

                # replacing all the wavefront component in all of the particles                
                for i in range(len(pso.swarm)):
                    pso.swarm[i].position[0:19*2]=allparameters_best_parametrization_1d[0:19*2]     
                    
                    if zmax>22:
                        pso.swarm[i].position[19*2+23:]=allparameters_best_parametrization_1d[19*2+23:]   
                        

               
                
                
                
                
                
                
                
                
                
                
                
                
        
    
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
    
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Swarm1',chains)
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
