"""
Created on Wed Aug 15 10:14:22 2018

@author: Neven Caplar
@contact: ncaplar@princeton.edu
"""

#standard library imports
from __future__ import absolute_import, division, print_function
import socket
import time
print(str(socket.gethostname())+': Start time for importing is: '+time.ctime()) 

import sys


import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

#Related third party imports
from multiprocessing import Pool

 
# MPI imports, depending on configuration
#from schwimmbad import MPIPool
#from schwimmbad import MultiPool
#import mpi4py

#Local application/library specific imports

#galsim
import galsim
galsim.GSParams.maximum_fft_size=12000

#emcee
import emcee

#Zernike_Module
from Zernike_Module import LNP_PFS,LN_PFS_single,create_parInit,create_x

############################################################################################################
print('Start time is: '+time.ctime())  
time_start=time.time()  

#DATA_FOLDER='/tigress/ncaplar/Data/AprData/'
DATA_FOLDER='/tigress/ncaplar/Data/AugData/'
RESULT_FOLDER='/tigress/ncaplar/Results/'

#name of the observation(s) which we will analyze
obs= sys.argv[1]

obs_int_array=np.array([x.strip() for x in obs.split(',')])

obs_int_array=obs_int_array.astype(int)

#name of the spot which we will analyze
xycoordinates=sys.argv[2]
single_number= xycoordinates

# number of steps each walkers will take 
nsteps = int(sys.argv[3])

#  not used, eps if we want to do variance trick ?
eps=int(float(sys.argv[4]))

sci_image_array=[]
var_image_array=[]

for obs in obs_int_array:
    if obs==8600:
        sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned_Dithered.npy')
        var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked_Dithered.npy')
    else:
        sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned.npy')
        var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked.npy')
        
    sci_image_array.append(sci_image)
    var_image_array.append(var_image)
    
for i in range(len(obs_int_array)):
    globals()['model'+str(i)] = LN_PFS_single(sci_image_array[i],var_image_array[i],dithering=1)

NAME_OF_CHAIN='chainSep18_Multi_P_'+str(obs)+str(xycoordinates)+str(eps)
NAME_OF_LIKELIHOOD_CHAIN='likechainSep18_Multi_P_'+str(obs)+str(xycoordinates)+str(eps)


obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28])
defocus_possibilites=np.array([-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0,0.5,1,1.5,2,2.5,3,3.5,4])

#z4Input=z4Input_possibilites[obs_possibilites==obs_int][0]
    
columns=['z4a','z4b','z5a','z5b','z6','z6b','z7a','z7b','z8a','z8b','z9a','z9b','z10a','z10b','z11a','z11b',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
                      'minorAxis','pupilAngle','effective_radius_illumination',
                      'frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']  

allparameters_proposal=np.array([-7,0.00,0.00,-0.0,0.0,0.0,0.00,0.0,0,0.00,0.00,-0.0,0.0,0.0,0.00,0.0,
                                 0.65,0.1,0.0,0.0,0.08,0.0,
                                 0.7,2,-0.2,0.00,
                                 0.99,0.0,1,
                                 0.03,1,0,
                                 50000,50,2.5,10**-1.2,
                                 0.47,1.85,1.0]) 

sorter = np.argsort(obs_possibilites)
defocus_array=defocus_possibilites[sorter[np.searchsorted(obs_possibilites, obs_int_array, sorter=sorter)]]

nT=2
parInit1=create_parInit(allparameters_proposal,multi=1)
parInit2=create_parInit(allparameters_proposal,multi=1)
parInitnT=np.array([parInit1,parInit2])

def multi_model(parameters):
        
    res=[]
    for i in range(len(obs_int_array)):
        parameters_single=create_x(defocus_array[i],parameters)
        #print(parameters_single)
        res.append(globals()['model'+str(i)](parameters_single))
    #print(res)
    return np.sum(res)

#if obs_int==8600:
#    model = LN_PFS_single(sci_image,var_image,dithering=2)
#else:
#    model = LN_PFS_single(sci_image,var_image)
    
modelP =LNP_PFS(sci_image_array[0],var_image_array[0])

  
#pool = MPIPool(loadbalance=True)
#pool=Pool(processes=31)
pool=Pool() 
    
print('Name of machine is '+socket.gethostname())    
    
zmax=11
zmax_input=zmax
print('Spot coordinates are: '+str(xycoordinates))  
print('Size of input image is: '+str(sci_image.shape)) 
#print('First value of the sci image: '+str(sci_image[0][0]))
#print('First value of the var image: '+str(var_image[0][0]))
print('Steps: '+str(nsteps)) 
print('Name: '+str(NAME_OF_CHAIN)) 



print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())    

print('Testing proposal is: '+str(allparameters_proposal))
time_start_single=time.time()
print('Likelihood for the testing proposal is: '+str(multi_model(allparameters_proposal)))
time_end_single=time.time()
print('Time for single calculation is '+str(time_end_single-time_start_single))


##################################################################################################################### 
# First emcee  
sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], multi_model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  
   
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee1',sampler.chain)
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee1',sampler.lnlikelihood) 
   
    
# Create minchain and save it
likechain0=sampler.lnlikelihood[0]
chain0=sampler.chain[0]
minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]  
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)    

print('Time when first emcee run finished was: '+time.ctime())     
time_end=time.time()   
print('Time taken was '+str(time_end-time_start)+' seconds')

print('Likelihood atm: '+str(np.min(np.abs(likechain0))))
print('minchain atm: '+str(minchain))
sys.stdout.flush()
##################################################################################################################### 
# Second emcee  
nT=2
parInit1=create_parInit(minchain,multi=1)
parInit2=create_parInit(minchain,multi=1)
parInitnT=np.array([parInit1,parInit2])

sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], multi_model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  
 
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee2',sampler.chain)
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee2',sampler.lnlikelihood) 
      
    
# Create minchain and save it
likechain0=sampler.lnlikelihood[0]
chain0=sampler.chain[0]
minchain=chain0[np.abs(likechain0)==np.min(np.abs(likechain0))][0]  
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)    

print('Time when second emcee run finished was: '+time.ctime())     
time_end=time.time()   
print('Time taken was '+str(time_end-time_start)+' seconds')

print('Likelihood atm: '+str(np.min(np.abs(likechain0))))
print('minchain atm: '+str(minchain))
sys.stdout.flush()

##################################################################################################################### 

# Third emcee  
nT=2
parInit1=create_parInit(minchain,multi=1)
parInit2=create_parInit(minchain,multi=1)
parInitnT=np.array([parInit1,parInit2])

sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], multi_model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  
    
    
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee3',sampler.chain)
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee3',sampler.lnlikelihood) 

# Create minchain and save it
likechain0=sampler.lnlikelihood[0]
chain0=sampler.chain[0]
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
