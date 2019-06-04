"""
Created on Tue Mar 26 17:29:00 2019

@author: Neven Caplar
ncaplar@princeton.edu
www.ncaplar.com
 
v0.01 - investgation on a single object, from z11 to z22
v0.02 - using for the test on single object, arc=HgAr, singlespot=102
v0.03 - change to accept input at z22
        -using for the test on single object, arc=Ne, singlespot=69


# run by e.g., python /home/ncaplar/Code/Direct_and_Interpolation_Analysis_Extra_Zernike_Apr2019.py 12403 69 6 5 Ne
# run by e.g., python /home/ncaplar/Code/Direct_and_Interpolation_Analysis_Extra_Zernike_Apr2019.py 12397 69 6 5 Ne
# run by e.g., python /home/ncaplar/Code/Direct_and_Interpolation_Analysis_Extra_Zernike_Apr2019.py 12313 69 6 5 Ne
# run by e.g., python /home/ncaplar/Code/Direct_and_Interpolation_Analysis_Extra_Zernike_Apr2019.py 12307 69 6 5 Ne
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
from multiprocessing import Pool
from multiprocessing import current_process

# MPI imports, depending on configuration
#from schwimmbad import MPIPool
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

#DATA_FOLDER='/tigress/ncaplar/Data/AprData/'
DATA_FOLDER='/tigress/ncaplar/Data/Feb5Data/'
STAMPS_FOLDER=DATA_FOLDER+'Stamps_Cleaned/'
DATAFRAMES_FOLDER=DATA_FOLDER+'Dataframes/'
RESULT_FOLDER='/tigress/ncaplar/Results/'

#name of the observation which we will analyze
obs= sys.argv[1]





#name of the spot which we will analyze
xycoordinates=sys.argv[2]
single_number= xycoordinates
print('single_number is: '+str(single_number)) 

# number of steps each walkers will take 
nsteps = int(sys.argv[3])
print('nsteps is: '+str(nsteps)) 
#  parameters to control the cosmo_hammer process
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
    
# which arc lamp is being analyzed ("HgAr" or "Ne" for Neon)    
arc=sys.argv[5]
print('arc lamp is: '+str(arc))
if str(arc)=="HgAr":
    single_number_focus=11748
elif str(arc)=="Ne":
    single_number_focus=12355 
else:
    print("Not recognized arc-line")


obs_int=int(obs)
"""
Do I need this

if arc=='HgAr':
    focus_numbers=np.array([8597,8594,8591,8588,8585,8582,8579,8576])
    if obs_int in focus_numbers:
        obs_int=8603
elif arc=='Ne':
    focus_numbers=np.array([8597,8594,8591,8588,8585,8582,8579,8576])+90
    if obs_int in focus_numbers:
        obs_int=8603+90
"""

#import data
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


"""
if eps==99:
    pupil_parameters=np.load(RESULT_FOLDER+NAME_OF_PUPIL_RES+'pupil_parameters')
    print("Pupil parameters found!")
else:
    print("Running analysis without specified pupil parameters") 
    pupil_parameters=None

#pupil_parameters=[0.65,0.1,0.,0.,0.08,0,0.99,0.0,1,0.04,1,0]
"""

with open(DATAFRAMES_FOLDER + 'results_of_fit_many_direct_HgAr_from_Mar22.pkl', 'rb') as f:
    results_of_fit_many_direct_preDecemberrun_HgAr=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_direct_HgAr_from_Mar22.pkl', 'rb') as f:
    results_of_fit_many_direct_preDecemberrun_Ne=pickle.load(f)

# where are the dataframe which we use to guess the initial solution
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_HgAr_from_Mar22.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_HgAr=pickle.load(f)
with open(DATAFRAMES_FOLDER + 'results_of_fit_many_interpolation_Ne_from_Mar22.pkl', 'rb') as f:
    results_of_fit_many_interpolation_preDecemberrun_Ne=pickle.load(f)
    
if arc=='HgAr':
    obs_possibilites=np.array([11796,11790,11784,11778,11772,11766,11760,11754,11748,11748,11694,11700,11706,11712,11718,11724,11730,11736])
elif arc=='Ne':
    obs_possibilites=np.array([12403,12397,12391,12385,12379,12373,12367,12361,12355,12355,12349,12343,12337,12331,12325,12319,12313,12307])



z4Input_possibilites=np.array([28,24.5,21,17.5,14,10.5,7,3.5,0,0,-3.5,-7,-10.5,-14,-17.5,-21,-24.5,-28])
z4Input=z4Input_possibilites[obs_possibilites==obs_int][0]
label=['m4','m35','m3','m25','m2','m15','m1','m05','0d','0','p05','p1','p15','p2','p25','p3','p35','p4','0p']
labelInput=label[list(obs_possibilites).index(obs_int)]


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

if arc=="HgAr":
    results_of_fit_many_direct_preDecemberrun=results_of_fit_many_direct_preDecemberrun_HgAr
elif arc=="Ne":
    results_of_fit_many_direct_preDecemberrun=results_of_fit_many_direct_preDecemberrun_Ne  
else:
    print("what has happened here?")

if arc=="HgAr":
    results_of_fit_many_interpolation_preDecemberrun=results_of_fit_many_interpolation_preDecemberrun_HgAr
elif arc=="Ne":
    results_of_fit_many_interpolation_preDecemberrun=results_of_fit_many_interpolation_preDecemberrun_Ne  
else:
    print("what has happened here?")
 
if labelInput=='m4':    
    full_proposal=np.array([   29.68697185,      0.05519361,     -1.18704926,      0.59558734,
            0.20766521,     -0.04107686,     -0.40601122,     -0.31837946,
           -0.03452826,      0.10607307,     -0.00273766,      0.04665575,
            0.05073057,      0.08869098,     -0.01153578,     -0.00024623,
           -0.05495544,      0.04594823,      0.03418093,      0.68506276,
            0.08014573,     -0.01484252,     -0.06009328,      0.06072556,
            0.01065128,      0.00000783,      0.00000782,      0.97303762,
            0.9483367 ,     -0.04332539,     -0.07515384,      0.80155356,
            0.05408019,      0.9998427 ,      1.02643908,      0.03309604,
       119746.23432356,      2.41597486,      0.00363292,      0.46227394,
            1.82079416,      0.99496788])
if labelInput=='m35':    
    full_proposal=np.array([   25.97388577,      0.06991468,     -1.12601434,      0.59892774,
            0.19237114,     -0.04084678,     -0.39677593,     -0.3251182 ,
           -0.03296496,      0.09489769,     -0.00170918,      0.05030816,
            0.05397042,      0.07477867,     -0.01722757,     -0.001806  ,
           -0.05189408,      0.04358025,      0.0184269 ,      0.68506276,
            0.08014574,     -0.01484252,     -0.06009328,      0.06072556,
            0.01065127,      0.00000782,      0.00000782,      0.97303761,
            0.94833669,     -0.04332539,     -0.07515384,      0.80155356,
            0.05408019,      0.9998427 ,      1.02643908,      0.03309604,
       119230.92170004,      2.49944585,      0.00307517,      0.45785109,
            1.80276317,      0.99562514])
if labelInput=='p35':    
    full_proposal=np.array([  -26.01862402,     -0.10001975,     -0.21083343,      0.38095728,
            0.02476478,     -0.07085732,     -0.20478432,     -0.53516736,
            0.0131455 ,     -0.04047305,      0.05171454,     -0.01329873,
            0.02951348,     -0.0108068 ,     -0.01663108,     -0.02098131,
            0.02344343,      0.01295711,     -0.03596844,      0.68506272,
            0.08014577,     -0.01484248,     -0.06009324,      0.06072552,
            0.01065119,      0.00000778,      0.00000778,      0.97303753,
            0.94833663,     -0.04332539,     -0.07515384,      0.80155356,
            0.05408019,      0.9998427 ,      1.02643912,      0.03309602,
       119915.796026  ,      2.45731164,      0.00400156,      0.3885899 ,
            1.81225669,      0.99503363])
if labelInput=='p4':    
    full_proposal=np.array([  -29.67933852,     -0.08466306,     -0.16493224,      0.38942196,
            0.01029832,     -0.07510596,     -0.18950122,     -0.54515127,
            0.00574793,     -0.04988694,      0.05129696,     -0.01271377,
            0.03053679,     -0.00580168,     -0.01710942,     -0.01710089,
            0.02928522,      0.01783932,     -0.03311391,      0.68506272,
            0.08014577,     -0.01484248,     -0.06009324,      0.06072552,
            0.01065119,      0.00000777,      0.00000778,      0.97303753,
            0.94833662,     -0.04332539,     -0.07515384,      0.80155356,
            0.05408019,      0.9998427 ,      1.02643912,      0.03309601,
       119736.81445291,      2.31563859,      0.00254817,      0.41388718,
            1.80818686,      0.99460485])    
    
'''   
full_proposal=np.array([   25.46927509,    -0.02124209,    -1.04992786,     0.69853391,
          -0.0641416 ,     0.02247082,    -0.29192157,    -0.18292153,
           0.69976007,     0.08438785,     0.00028714,    -0.09023436,
           0.06347543,     0.01088842,     0.00008689,     0.51837526,
           0.9760724 ,     0.83827641,    -0.02061162,    -0.06103453,
           0.82536987,     0.08903461,     0.69209871,     1.04051888,
          -0.04201478, 90682.83733571,     2.58159004,     0.00378112,
           0.48956584,     1.81188317,     0.99545856,0,0])
'''       
    
# do analysis with up to 22 or 11 zernike - need to modify for this code
zmax_input_results_of_fit_many=22
if zmax_input_results_of_fit_many==22:    
    # this is how it should be, but I am hacking to take values from above for this test    
    #allparameters_proposal=results_of_fit_many_direct_preDecemberrun[labelInput].loc[int(single_number)].values[:len(columns22)]
    allparameters_proposal=np.concatenate((full_proposal[0:8],full_proposal[8:]))   
    
else:
    full_proposal=results_of_fit_many_direct_preDecemberrun[labelInput].loc[int(single_number)].values
    # select z4-z11, skip to z22 and join everything else
    allparameters_proposal=np.concatenate((full_proposal[0:8],[0,0,0,0,0,0,0,0,0,0,0],full_proposal[8:-2]))

#allparameters_proposal=np.concatenate((full_proposal[0:8],[0,0,0,0,0,0,0,0,0,0,0],full_proposal[8:-2]))

'''
allparameters_proposal=np.array([   25.46927509,    -0.02124209,    -1.04992786,     0.69853391,
          -0.0641416 ,     0.02247082,    -0.29192157,    -0.18292153,
           0.02162654,     0.04928138,    -0.00505438,    -0.00438297,
          -0.00598119,     0.0099617 ,     0.00672375,     0.01452023,
          -0.01101103,     0.01580502,    -0.00427035,     0.69976007,
           0.08438785,     0.00028714,    -0.09023436,     0.06347543,
           0.01088842,     0.00008689,     0.51837526,     0.9760724 ,
           0.83827641,    -0.02061162,    -0.06103453,     0.82536987,
           0.08903461,     0.69209871,     1.04051888,    -0.04201478,
       90682.83733571,     2.58159004,     0.00378112,     0.48956584,
           1.81188317,     0.99545856])
'''
    

# not sure, but I guess to protect it from chaning?
allparameters_proposal_direct_22=np.copy(allparameters_proposal)      
"""
# have to implement this 
if zmax==22:    
    allparameters_proposal_int=results_of_fit_many_interpolation_preDecemberrun[labelInput].loc[int(single_number)].values[:len(columns22)]
else:
    allparameters_proposal_int=results_of_fit_many_interpolation_preDecemberrun[labelInput].loc[int(single_number)].values[:len(columns)]    
    
    
allparameters_proposal_int_22=np.copy(allparameters_proposal_int)
"""

#allparameters_proposal_22=np.insert(allparameters_proposal_22,8,[0,0,0,0,0,0,0,0,0,0,0])


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

# "lower_limits" and "higher_limits" are only needed to initalize the cosmo_hammer code, but not used in the actual evaluation
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

#nT=2
#parInit1=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInit2=create_parInit(allparameters_proposal,None,pupil_parameters)
#parInitnT=np.array([parInit1,parInit2])
    
pupil_parameters=None
if obs_int==8600:
    model = LN_PFS_single(sci_image,var_image,dithering=2,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax)
else:
    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax,save=1)
    
modelP =LNP_PFS(sci_image,var_image)
  
def model_return(allparameters_proposal):
    return model(allparameters_proposal,return_Image=True)
#pool=Pool()
   
print('Name of machine is '+socket.gethostname())    
    
zmax_input=zmax
print('Spot coordinates are: '+str(xycoordinates))  
print('Size of input image is: '+str(sci_image.shape)) 
print('First value of the sci image: '+str(sci_image[0][0]))
print('First value of the var image: '+str(var_image[0][0]))


print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())    

print('Testing proposal is: '+str(allparameters_proposal_direct_22))
time_start_single=time.time()
Testing_chi2=(-2)*model(allparameters_proposal_direct_22)/(sci_image.shape[0]*sci_image.shape[1])
print('Likelihood for the testing proposal is: '+str(Testing_chi2))
time_end=time.time()
print('Total time taken was  '+str(time_end-time_start)+' seconds')

#implementing parts of the analysis seen at https://github.com/tribeiro/donut/blob/master/donut/don11.pyr

from scipy.special import gamma,jv
from scipy.linalg import svd,inv

def svd_invert(matrix,threshold):
    '''
    :param matrix:
    :param threshold:
    :return:SCD-inverted matrix
    '''
    # print 'MATRIX:',matrix
    u,ws,v = svd(matrix,full_matrices=True)

    #invw = inv(np.identity(len(ws))*ws)
    #return ws

    ww = np.max(ws)
    n = len(ws)
    invw = np.identity(n)
    ncount = 0

    for i in range(n):
        if ws[i] < ww*threshold:
            # log.info('SVD_INVERT: Value %i=%.2e rejected (threshold=%.2e).'%(i,ws[i],ww*threshold))
            invw[i][i]= 0.
            ncount+=1
        else:
            # print 'WS[%4i] %15.9f'%(i,ws[i])
            invw[i][i] = 1./ws[i]

    # log.info('%i singular values rejected in inversion'%ncount)

    inv_matrix = np.dot(u , np.dot( invw, v))

    return inv_matrix

pool=Pool(processes=36)
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
    number_of_extra_zernike=254-22
    
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
        print('initial_input_parameters for loop '+str(iteration_number)+' are '+str(initial_input_parameters))
        #print('initial_model_result is '+str(initial_model_result))
        
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_model_result_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),initial_model_result)
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/image_0_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),image_0)
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_input_parameters_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),initial_input_parameters)

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
        print('input_parameters for loop '+str(iteration_number)+' are '+str(initial_input_parameters))      
        
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_model_result_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),initial_model_result)
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/image_0_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),image_0)
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/initial_input_parameters_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),initial_input_parameters)        
        
        
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
    print('I[0]'+str(I[0]))
    print('M0[0]'+str(M0[0]))
    #M0_STD=M0/STD
    print('I-M0 before iteration '+str(iteration_number)+': '+str(np.sum(np.abs(I-M0))))  
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
        
        
        
    print('results after pool loop '+str(iteration_number)+' are '+str((-2)*np.array(out_ln)/(sci_image.shape[0]*sci_image.shape[1])))
        
     
    
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
    
    np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_ln_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),out_ln)
    np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_images_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),out_images)
    np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/out_parameters_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),out_parameters)
    np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/array_of_delta_z_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),array_of_delta_z)
    np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/single_wavefront_parameter_list_'+str(iteration_number)+str(obs)+str(single_number)+str(eps)+str(arc),single_wavefront_parameter_list)   
    
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
    print('Tokovnin_proposal '+str(Tokovnin_proposal))
    
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
    print('did_chi_2_improve: '+str(did_chi_2_improve))
    
    if iteration_number==9: 
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/allparameters_proposal_after_iteration_'+str(obs)+str(single_number)+str(eps)+str(arc),allparameters_proposal_after_iteration)
        np.save('/tigress/ncaplar/Result_DirectAndInt_FinalResults/final_optpsf_image'+str(obs)+str(single_number)+str(eps)+str(arc),final_optpsf_image)

    
    
#res=np.zeros((8,))
#res[:]=pool.map(my_fuction,[1,2,3,4,5,6,7,8])
#print(res)
