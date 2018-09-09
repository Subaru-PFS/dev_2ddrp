#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:14:22 2018

@author: ncaplar@princeton.edu
"""


from __future__ import absolute_import, division, print_function
import socket
import time
from multiprocessing import Pool
from multiprocessing import current_process
print(str(socket.gethostname())+': Start time for importing is: '+time.ctime())  
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import lmfit
import galsim
galsim.GSParams.maximum_fft_size=12000
import numpy as np
np.set_printoptions(suppress=True)
#from schwimmbad import MPIPool
import emcee

import sys
#import math
#from schwimmbad import MultiPool

#from emcee.utils import MPIPool
#import mpi4py

#from scipy.ndimage import gaussian_filter
#import scipy.misc
#import skimage.transform
#from scipy import signal

#import astropy
#import astropy.convolution
#from astropy.convolution import Gaussian2DKernel

#import lsst.afw
#from lsst.afw.cameraGeom import PupilFactory
#from lsst.afw.geom import Angle, degrees
#from lsst.afw import geom
#from lsst.afw.geom import Point2D

import warnings
warnings.filterwarnings("ignore")

np.seterr(divide='ignore', invalid='ignore')


#import Zernike_Module
from Zernike_Module import LNP_PFS,LN_PFS_single,create_parInit

############################################################################################################
# dont know how to make this in the module 
############################################################################################################

def residual(pars):
    #print('computing residual')
    try:
        # unpack parameters:
        #  extract .value attribute for each parameter
        #print("step 1")
        parvals = pars.valuesdict()
        #print("step 2")
        #print(parvals)
        values=np.array(list(parvals.items()))[:,1].astype(np.float32)
        #print("step 3")
        #print([values,current_process()])
        res=-model(values)
        #print([res,current_process()])
        return res
    except (IndexError,ValueError):
        #print('error')
        return np.inf
"""    
def residual(pars):

    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = pars.valuesdict()
    #print(parvals)
    values=np.array(parvals.items())[:,1].astype(float)
    #print(values)
    res=-model(values)
    #print(res)
    return res
""" 

def lmfit_f(x): 
    #print('lmfit')

    minchain=np.load(RESULT_FOLDER+NAME_OF_CHAIN+'minchain.npy' )

    parInit1=create_parInit(minchain)
    # 38 par, 8 is walker_mult from create_parInit
    parInitnT2D=parInit1.flatten().reshape(len(minchain)*8,len(minchain))

    
    columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'radiometricEffect','radiometricExponent',
                          'x_ilum','y_ilum','minorAxis','pupilAngle',
                          'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']  


    paramsp = lmfit.Parameters()
    for i in range(len(columns)):
        if columns[i][0]=='z':
            min_bound=-np.inf
            max_bound=np.inf
        if columns[i]=='hscFrac':
            min_bound=0.5
            max_bound=1.2
        if columns[i]=='strutFrac':
            min_bound=0.02
            max_bound=0.2
        if columns[i]=='dxFocal':
            min_bound=-0.8
            max_bound=0.8
        if columns[i]=='dyFocal':
            min_bound=-0.8
            max_bound=0.8           
        if columns[i]=='slitFrac':
            min_bound=0
            max_bound=0.2
        if columns[i]=='slitFrac_dy':
            min_bound=-0.5
            max_bound=0.5
        if columns[i]=='radiometricEffect':
            min_bound=0.0
            max_bound=3  
        if columns[i]=='radiometricExponent':
            min_bound=-0.5
            max_bound=20
        if columns[i]=='x_ilum':
            min_bound=-0.8
            max_bound=0.8
        if columns[i]=='y_ilum':
            min_bound=-0.8
            max_bound=0.8
        if columns[i]=='minorAxis':
            min_bound=0.8
            max_bound=1.0
        if columns[i]=='pupilAngle':
            min_bound=-np.pi/2
            max_bound=+np.pi/2            
        if columns[i]=='grating_lines':
            min_bound=1200
            max_bound=120000
        if columns[i]=='scattering_radius':
            min_bound=1
            max_bound=1200
        if columns[i]=='scattering_slope':
            min_bound=0.5
            max_bound=3.5
        if columns[i]=='scattering_amplitude':
            min_bound=0
            max_bound=10   
        if columns[i]=='pixel_effect':
            min_bound=0.2
            max_bound=1
        if columns[i]=='fiber_r':
            min_bound=1.4
            max_bound=2.4
        if columns[i]=='flux':
            min_bound=0.9
            max_bound=1.1
        paramsp.add(columns[i], value=parInitnT2D[x][i],min=min_bound,max=max_bound)    
        #paramsp.add(columns[i], value=parInitnT2D[x][i])    
 
    #print(paramsp)    
    res_iter=[]
    time_start=time.time() 
    #print(time_start)
    def iter_cb_f(params,iter,resid):
        # return true breaks the iteration
        if time.time() -time_start<2*5*nsteps*1.1:
            if iter <6*nsteps:
                res=[]
                #print( time.time())
                for key in fitter.params:
                    res.append(params[key].value)
    
                res=np.array(res)
                res=np.concatenate((res,[resid]),axis=0)
                res_iter.append(res)
                #print([iter,resid,res,socket.gethostname()])
                #print([iter,resid,res,current_process()])
                
                
                #if resid>0:
                #    print(res)
                if resid==np.inf:
                    return True
                #else:
                #    return None
                    
                return None
            else:
                return True
        else:
            return True
    
    fitter=lmfit.Minimizer(residual,paramsp,iter_cb=iter_cb_f)
    try:
        result=fitter.minimize(method='Powell')
    
        res=[]
        for key in fitter.params:
            res.append(result.params[key].value)

        #append chisqr (has not physical connection as nubmer of data pointsimport mpi4py is "wrong")
        res.append(result.residual)
        res=np.array(res)
        #np.save('/Users/nevencaplar/Documents/PFS/Testing/Methods/'+str(x),res)
        return res
    
    except TypeError:
        res_iter=np.array(res_iter)
        print(res_iter)
        minchain=res_iter[np.abs(res_iter[:,-1])==np.min(np.abs(res_iter[:,-1]))][0]      
        return minchain    
    
############################################################################################################


print('Start time is: '+time.ctime())  
time_start=time.time()  

#DATA_FOLDER='/tigress/ncaplar/Data/AprData/'
DATA_FOLDER='/tigress/ncaplar/Data/AugData/'
RESULT_FOLDER='/tigress/ncaplar/Results/'

#name of the observation which we will analyze
obs= sys.argv[1]
obs_int=int(obs)
#name of the spot which we will analyze
xycoordinates=sys.argv[2]
single_number= xycoordinates

# number of steps each walkers will take 
nsteps = int(sys.argv[3])

#  not used, eps if we want to do variance trick ?
eps=int(float(sys.argv[4]))
if obs_int==8600:
    sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned_Dithered.npy')
    var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked_Dithered.npy')
else:
    sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned.npy')
    var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked.npy')


NAME_OF_CHAIN='chainSep7_Single_P_'+str(obs)+str(xycoordinates)+str(eps)
NAME_OF_LIKELIHOOD_CHAIN='likechainSep7_Single_P_'+str(obs)+str(xycoordinates)+str(eps)

#NAME_OF_FLATCHAIN='flatchainAug15_512_P_'+str(obs)+str(xycoordinates)+str(eps)
#NAME_OF_LIKELIHOOD_FLATCHAIN='likeflatchainAug15_512_P_'+str(obs)+str(xycoordinates)+str(eps)
#NAME_OF_PROBABILITY_CHAIN='probchainAug15_512_P_'+str(obs)+str(xycoordinates)+str(eps)



# Telescope size
#diam_sic=Exit_pupil_size(-693.5)

# This is 'normal' orientation of the system


obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28])
z4Input=z4Input_possibilites[obs_possibilites==obs_int][0]
    
columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent',
                      'x_ilum','y_ilum','minorAxis','pupilAngle',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']  

allparameters_proposal=np.array([z4Input,0.00,0.00,-0.0,0.0,0.0,0.00,0.0,
                                 0.65,0.1,0.0,0.0,0.08,0.0,
                                 0.7,2,
                                 -0.2,0.00,0.99,0.0,
                                 50000,50,2.5,10**-1.2,
                                 0.47,1.85,1.0]) 

'''
allparameters_proposal=np.array([0,0.4645,-0.0534,-0.158,-0.06,0.004,-0.148,-0.045,
                                 0.766,0.14,0.035,0.012,0.002,0.495,
                                 0.673,2.142,
                                 0.148,0.03968,0.846,0.0447,
                                 69042,39.42,2.637,0.069511,
                                 0.490,0.9939]) 
'''
nT=2
parInit1=create_parInit(allparameters_proposal)
parInit2=create_parInit(allparameters_proposal)
parInitnT=np.array([parInit1,parInit2])
# 38 par, 4 temperatures, 10 is walker_mult from create_parInit
#parInitnT2D=parInitnT.flatten().reshape(38*10*4,38)
if obs_int==8600:
    model = LN_PFS_single(sci_image,var_image,dithering=2)
else:
    model = LN_PFS_single(sci_image,var_image)
    
modelP =LNP_PFS(sci_image,var_image)

  
#pool = MPIPool(loadbalance=True)
pool=Pool()


    
    
print('Name of machine is '+socket.gethostname())    
    
zmax=11
zmax_input=zmax
print('Spot coordinates are: '+str(xycoordinates))  
print('Size of input image is: '+str(sci_image.shape)) 
print('First value of the sci image: '+str(sci_image[0][0]))
print('First value of the var image: '+str(var_image[0][0]))
print('Steps: '+str(nsteps)) 
print('Name: '+str(NAME_OF_CHAIN)) 



print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())    

print('Testing proposal is: '+str(allparameters_proposal))
time_start_single=time.time()
print('Likelihood for the testing proposal is: '+str(model(allparameters_proposal)))
time_end_single=time.time()
print('Time for single calculation is '+str(time_end_single-time_start_single))


##################################################################################################################### 
# First emcee  
sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  

#for i, result in enumerate(sampler.sample(parInitnT, iterations=nsteps)):
#    print('z22: '+"{0:5.1%}\r".format(float(i+1) / nsteps)+' time '+str(time.time())),    
 
    
    
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee1',sampler.chain)
#np.save(RESULT_FOLDER+NAME_OF_FLATCHAIN,sampler.flatchain)
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN,sampler.lnprobability) 
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_FLATCHAIN,sampler.flatlnprobability) 
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee1',sampler.lnlikelihood) 
#np.save(RESULT_FOLDER+NAME_OF_PROBABILITY_CHAIN,sampler.lnprobability)      
    
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

#####################################################################################################################   
 # First lmfit

#print(range(int(parInitnT.shape[1])))
res_lmfit=[]
for i in pool.map(lmfit_f,range(int(parInitnT.shape[1]))):
    #print(i)
    res_lmfit.append([i])  

res_lmfit=np.array(res_lmfit)
#np.save(RESULT_FOLDER+NAME_OF_CHAIN+'res_lmfit',res_lmfit)
#print('res_lmfit[:,:,-1]'+str(np.abs(res_lmfit[:,:,-1])))
minchain=res_lmfit[np.abs(res_lmfit[:,:,-1])==np.min(np.abs(res_lmfit[:,:,-1]))][0]
like_lmfit=minchain[-1]
minchain=minchain[0:len(minchain)-1]
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)

print('Time when first lmfit finished was: '+time.ctime())     
time_end=time.time()   
print('Time taken to this point was '+str(time_end-time_start)+' seconds')

print('Likelihood atm: '+str(like_lmfit))
print('minchain atm: '+str(minchain))
##################################################################################################################### 
# Second emcee  
nT=2
parInit1=create_parInit(minchain)
parInit2=create_parInit(minchain)
parInitnT=np.array([parInit1,parInit2])

sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  

#for i, result in enumerate(sampler.sample(parInitnT, iterations=nsteps)):
#    print('z22: '+"{0:5.1%}\r".format(float(i+1) / nsteps)+' time '+str(time.time())),    
    
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee2',sampler.chain)
#np.save(RESULT_FOLDER+NAME_OF_FLATCHAIN,sampler.flatchain)
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN,sampler.lnprobability) 
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_FLATCHAIN,sampler.flatlnprobability) 
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee2',sampler.lnlikelihood) 
#np.save(RESULT_FOLDER+NAME_OF_PROBABILITY_CHAIN,sampler.lnprobability)      
    
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
##################################################################################################################### 
# Second lmfit  

res_lmfit=[]
for i in pool.map(lmfit_f,range(int(parInitnT.shape[1]))):
    #print(i)
    res_lmfit.append([i])  

res_lmfit=np.array(res_lmfit)
minchain=res_lmfit[np.abs(res_lmfit[:,:,-1])==np.min(np.abs(res_lmfit[:,:,-1]))][0]
like_lmfit=minchain[-1]
minchain=minchain[0:len(minchain)-1]



#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'minchain',minchain)

print('Time when second lmfit finished was: '+time.ctime())     
time_end=time.time()   
print('Time taken was '+str(time_end-time_start)+' seconds')


print('Likelihood atm: '+str(like_lmfit))
print('minchain atm: '+str(minchain))

##################################################################################################################### 
# Third emcee  
nT=2
parInit1=create_parInit(minchain)
parInit2=create_parInit(minchain)
parInitnT=np.array([parInit1,parInit2])

sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], model,modelP,
                pool=pool)

sampler.run_mcmc(parInitnT, nsteps)  

#for i, result in enumerate(sampler.sample(parInitnT, iterations=1*nsteps)):
#    print('z22: '+"{0:5.1%}\r".format(float(i+1) /( 1*nsteps))+' time '+str(time.time())),    
    
#Export this file
np.save(RESULT_FOLDER+NAME_OF_CHAIN+'Emcee3',sampler.chain)
#np.save(RESULT_FOLDER+NAME_OF_FLATCHAIN,sampler.flatchain)
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN,sampler.lnprobability) 
#np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_FLATCHAIN,sampler.flatlnprobability) 
np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN+'Emcee3',sampler.lnlikelihood) 
#np.save(RESULT_FOLDER+NAME_OF_PROBABILITY_CHAIN,sampler.lnprobability)  

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

pool.close()
sys.exit(0)
