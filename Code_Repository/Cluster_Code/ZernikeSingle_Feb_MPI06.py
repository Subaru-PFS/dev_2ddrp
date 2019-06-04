"""
Created on Thu Apr 11 2019

@author: Neven Caplar
@contact: ncaplar@princeton.edu

Fourth analysis of the new data taken at LAM (with 0.21b version of the code)
- this is using results_of_fit_many_interpolation_HgAr_from_Apr10 and results_of_fit_many_interpolation_Ne_from_Apr10
- fits Zernike components+illumination parameters+postprocessingparamters

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
from Zernike_Module import LNP_PFS,LN_PFS_single,create_parInit,add_pupil_parameters_to_all_parameters,PFSLikelihoodModule,create_mask,create_res_data

############################################################################################################
print('Start time is: '+time.ctime())  
time_start=time.time()  

########################
# folder contaning the data from December 2017
#DATA_FOLDER='/tigress/ncaplar/Data/Dec18Data/'
########################

# folder contaning the data from February 2019
DATA_FOLDER='/tigress/ncaplar/Data/Feb5Data/'

STAMPS_FOLDER=DATA_FOLDER+'Stamps_Cleaned/'
DATAFRAMES_FOLDER=DATA_FOLDER+'Dataframes/'
RESULT_FOLDER='/tigress/ncaplar/Results/'

#name of the observation (actually number) which we will analyze
obs= sys.argv[1]
obs_int=int(obs)

########################
# not used yet?
#focus_numbers=np.array([8597,8594,8591,8588,8585,8582,8579,8576])
#if obs_int in focus_numbers:
#    obs_int=8603
########################

print('obs is: '+str(obs_int))  
print('obs_int is: '+str(obs_int))  

# name of the spot (again, actually number) which we will analyze
xycoordinates=sys.argv[2]
single_number= xycoordinates
print('single_number is: '+str(single_number)) 

# number of steps each walkers will take 
nsteps = int(sys.argv[3])
print('nsteps is: '+str(nsteps)) 

# input argument that controls the paramters of the cosmo_hammer process
eps=int(float(sys.argv[4]))
print('eps is: '+str(eps))  

if eps==1:
    #particle count, c1 parameter (individual), c2 parameter (global)
    options=[400,1.193,1.193]
if eps==2:
    options=[800,1.193,1.193]
    nsteps=int(nsteps/2)
if eps==3:
    options=[400,1.593,1.193]
if eps==4:
    options=[400,0.993,1.193]
if eps==5:
    options=[400,2.793,1.593]
if eps==6:
    options=[400,1.593,1.193]
if eps==7:
    options=[400,4.93,1.193]
if eps==8:
    options=[400,1.193,4.193]
if eps==9:
    options=[200,1.193,1.193]
    nsteps=int(2*nsteps)
if eps==10:
    options=[400,1.893,2.893]
    
################################################    
# which arc lamp is being analyzed ("HgAr" or "Ne" for Neon)    
#arc=sys.argv[5]
#print('arc lamp is: '+str(arc))
#if arc=="HgAr":
#    single_number_focus=8603
#elif arc=="Ne":
#    single_number_focus=8693    
#else:
#    print("Not recognized arc-line")
################################################ 

################################################    
# which arc lamp is being analyzed ("HgAr" or "Ne" for Neon)    
arc=sys.argv[5]
print('arc lamp is: '+str(arc))
if str(arc)=="HgAr":
    single_number_focus=11748
elif str(arc)=="Ne":
    single_number_focus=12355 
else:
    print("Not recognized arc-line")
################################################ 
    
# import data
# 8600 will also need to be changed
if obs_int==8600:
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
# For this run - if there are cosmic, do not analyse
if int(np.sum(mask_image))>1:
    print('Cosmics in the image - exiting')
    sys.exit(0) 


# name of the outputs
NAME_OF_CHAIN='chainApr16_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)
NAME_OF_LIKELIHOOD_CHAIN='likechainApr16_Single_P_'+str(obs)+str(single_number)+str(eps)+str(arc)

"""
if eps==99:
    pupil_parameters=np.load(RESULT_FOLDER+NAME_OF_PUPIL_RES+'pupil_parameters')
    print("Pupil parameters found!")
else:
    print("Running analysis without specified pupil parameters") 
    pupil_parameters=None

#pupil_parameters=[0.65,0.1,0.,0.,0.08,0,0.99,0.0,1,0.04,1,0]
"""

# where are the dataframe which we use to guess the initial solution
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_HgAr_from_Apr15.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_HgAr=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_Ne_from_Apr15.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_Ne=pickle.load(f)
    
##############################################    
#if arc=='HgAr':
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
#elif arc=='Ne':
#    print('Neon?????')
#    obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])+90
##############################################    

# What are the observations that can be analyzed
# used to associate observation with their input labels, so that the initial parameters guess is correct
if arc=='HgAr':
    obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11748,11694,11700,11706,11712,11718,11724,11730,11736])
elif arc=='Ne':
    # different sequence than for HgAr
    obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12355,12349,12343,12337,12331,12325,12319,12313,12307])


# associates each observation with the label in the supplied dataframe
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28])
z4Input=z4Input_possibilites[obs_possibilites==obs_int][0]
label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']
labelInput=label[list(obs_possibilites).index(obs_int)]

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

# just for the first run run, we need to connect the new data and the old data 
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
else:
    print("what has happened here?")
    
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


"""
# old code, pending deletion
if pupil_parameters is None:
    allparameters_proposal=np.array([z4Input,0.00,0.00,-0.0,0.0,0.0,0.00,0.0,
                                     0.8,0.1,0.0,0.0,0.08,0.0,
                                     0.7,2,-0.2,0.00,
                                     0.99,0.0,1,
                                     0.02,1,0,
                                     50000,20,2.5,10**-1.2,
                                     0.47,1.85,1.0]) 
else:
    allparameters_proposal_short=np.array([z4Input,0.00,0.00,-0.0,0.0,0.0,0.00,0.0,
                                     0.7,2,-0.2,0.00,
                                     50000,20,2.5,10**-1.2,
                                     0.47,1.85,1.0]) 
    allparameters_proposal=add_pupil_parameters_to_all_parameters(allparameters_proposal_short,pupil_parameters)
"""

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
"""
# relic of old code with multiple temperatures, pending deletion
#nT=2
#parInit1=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInit2=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInitnT=np.array([parInit1,parInit2])
 """
 
# Create an object called model - which gives likelihood given the parameters 
pupil_parameters=None
if obs_int==8600:
    model = LN_PFS_single(sci_image,var_image,dithering=2,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax)
else:
    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax)

# always returns 0   (did I only use this for multiple temperatures code?)
modelP =LNP_PFS(sci_image,var_image)
  
# initialize pool
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)   

#pool=Pool(processes=36)
#pool=Pool()
   
print('Name of machine is '+socket.gethostname())    
    

zmax_input=22

print('Spot coordinates are: '+str(xycoordinates))  
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

# constrained parameters describing the pupil and the illumination
stronger_array_06=np.array([1,1,1,1,1,1,1,1,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 
                        0.0,0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,
                        1.0,1.0,1.0,
                        1.0,1.0,1])

# creat inital parameters  
parInit1=create_parInit(allparameters_proposal=allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=2*stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
while len(parInit1)<options[0]:
    parInit1_2=create_parInit(allparameters_proposal=allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=2*
                              stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
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
# First emcee  
"""

if pupil_parameters is not None:
    minchain=add_pupil_parameters_to_all_parameters(minchain,pupil_parameters)

nT=2
parInit1=create_parInit(minchain,None,pupil_parameters)
parInit2=create_parInit(minchain,None,pupil_parameters)
parInitnT=np.array([parInit1,parInit2])


sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], model,modelP,
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
"""
##################################################################################################################### 
# First/Second emcee  
if pupil_parameters is not None:
    minchain=add_pupil_parameters_to_all_parameters(minchain,pupil_parameters)

parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=10*stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)

sampler = emcee.EnsembleSampler(parInit1.shape[0],parInit1.shape[1],model,pool=pool)

for i, result in enumerate(sampler.sample(parInit1, iterations=nsteps)):
    print('First/second emcee run: '+"{0:5.1%}\r".format(float(i) / nsteps)),
    sys.stdout.flush()

#sampler.run_mcmc(parInit1, nsteps)  
 
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
parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
while len(parInit1)<options[0]:
    parInit1_2=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)
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
minchain_err=[]
for i in range(len(columns)):
    minchain_err=np.append(minchain_err,np.sqrt(chi2reduced)*np.std(chains[:,:,i].flatten()))

minchain_err=np.array(minchain_err)

print('Likelihood atm: '+str(np.abs(gbests[-1].fitness)))
print('minchain atm: '+str(minchain))
print('minchain_err: '+str(minchain_err))
print('Time when last swarm run finished was: '+time.ctime())     
time_end=time.time()   
print('Time taken was '+str(time_end-time_start)+' seconds')

sys.stdout.flush()
##################################################################################################################### 
# Third/Second emcee  
    
if pupil_parameters is not None:
    minchain=add_pupil_parameters_to_all_parameters(minchain,pupil_parameters)

parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=minchain_err,stronger=10*stronger_array_06,use_optPSF=None,deduced_scattering_slope=None,zmax=zmax_input)

#It would be better to do
# parInit1=create_parInit(allparameters_proposal=minchain,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=2,use_optPSF=None,deduced_scattering_slope=None,zmax=22)
# as area covered is so small right now

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

