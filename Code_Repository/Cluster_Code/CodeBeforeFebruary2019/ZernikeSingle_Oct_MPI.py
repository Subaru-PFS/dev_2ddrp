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
galsim.GSParams.maximum_fft_size=12000

#emcee
import emcee

#Zernike_Module
from Zernike_Module import LNP_PFS,LN_PFS_single,create_parInit,add_pupil_parameters_to_all_parameters,PFSLikelihoodModule

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
print('eps is: '+str(eps))  

if eps==1:
    #particle count, cl, c2
    options=[400,1.193,1.193]
if eps==2:
    options=[800,1.193,1.193]
    nsteps=int(nsteps/2)
if eps==3:
    options=[400,1.593,1.193]
if eps==4:
    options=[400,0.993,1.193]
if eps==5:
    options=[400,1.193,1.393]
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



if obs_int==8600:
    sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned_Dithered.npy')
    var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked_Dithered.npy')
else:
    sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'Stacked_Cleaned.npy')
    var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'Stacked.npy')


NAME_OF_CHAIN='chainOct12_Single_P_'+str(obs)+str(xycoordinates)+str(eps)
NAME_OF_LIKELIHOOD_CHAIN='likechainOct12_Single_P_'+str(obs)+str(xycoordinates)+str(eps)
NAME_OF_PUPIL_RES='pupilOct12_Single_P_'+str(obs)

if eps==99:
    pupil_parameters=np.load(RESULT_FOLDER+NAME_OF_PUPIL_RES+'pupil_parameters')
    print("Pupil parameters found!")
else:
    print("Running analysis without specified pupil parameters") 
    pupil_parameters=None

#pupil_parameters=[0.65,0.1,0.,0.,0.08,0,0.99,0.0,1,0.04,1,0]


with open(DATA_FOLDER + 'results_of_fit_many_interpolation_preOctoberrun.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preOctoberrun=pickle.load(f)




obs_possibilites=np.array([8552,8555,8558,8561,8564,8567,8570,8573,8603,8600,8606,8609,8612,8615,8618,8621,8624,8627])
z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28])
z4Input=z4Input_possibilites[obs_possibilites==obs_int][0]
label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']
labelInput=label[list(obs_possibilites).index(obs_int)]
 
columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
                      'x_fiber','y_fiber','effective_radius_illumination',
                      'frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']  



allparameters_proposal=results_of_fit_many_interpolation_preOctoberrun[labelInput].loc[int(single_number)].values[:len(columns)]

"""
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

lower_limits=np.array([z4Input-3,-1,-1,-1,-1,-1,-1,-1,
             0.5,0.05,-0.8,-0.8,0,-0.5,
             0,-0.5,-0.8,-0.8,
             0.8,-np.pi/2,0.5,0,0.85,-0.8,
             1200,1,0.5,0,
             0.2,1.65,0.9])

higher_limits=np.array([z4Input+3,1,1,1,1,1,1,1,
              1.2,0.2,0.8,0.8,0.2,0.5,
              3,20,0.8,0.8,
              1,np.pi/2,1.01,0.05,1.15,0.8,
               120000,50,3.5,0.5,
               1.1,1.95,1.1])


#nT=2
#parInit1=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInit2=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInitnT=np.array([parInit1,parInit2])


if obs_int==8600:
    model = LN_PFS_single(sci_image,var_image,dithering=2,pupil_parameters=pupil_parameters,use_pupil_parameters=None)
else:
    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,use_pupil_parameters=None)
    
modelP =LNP_PFS(sci_image,var_image)

  
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)   

#pool=Pool(processes=36)
#pool=Pool()
  
    
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

sys.stdout.flush()
##################################################################################################################### 
# First swarm  

parInit1=create_parInit(allparameters_proposal,None,pupil_parameters)
while len(parInit1)<options[0]:
    parInit1_2=create_parInit(allparameters_proposal,None,pupil_parameters)
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

parInit1=create_parInit(minchain,None,pupil_parameters,minchain_err)

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
parInit1=create_parInit(minchain,None,pupil_parameters,minchain_err)
while len(parInit1)<options[0]:
    parInit1_2=create_parInit(minchain,None,pupil_parameters,minchain_err)
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
for swarm in pso.sample(4*nsteps):
    swarms.append(swarm)
    gbests.append(pso.gbest.copy())
    num_iter += 1
    if pso.isMaster():
        if num_iter % 2 == 0:
            print("Last swarm: "+str(100*num_iter/(4*nsteps)))
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

parInit1=create_parInit(minchain,None,pupil_parameters,None,stronger=2)

sampler = emcee.EnsembleSampler(parInit1.shape[0],parInit1.shape[1],model,pool=pool)
for i, result in enumerate(sampler.sample(parInit1, iterations=nsteps)):
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

