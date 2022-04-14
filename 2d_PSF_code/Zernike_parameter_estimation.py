# standard library imports
from __future__ import absolute_import, division, print_function
from Zernike_Module import LN_PFS_multi_same_spot, create_parInit, Tokovinin_multi,\
    Zernike_estimation_preparation, check_global_parameters
import Zernike_Module
import socket
import time
import sys
import os
import argparse
from datetime import datetime
from copy import deepcopy


import numpy as np

import warnings


# import pickle

# Related third party imports
import multiprocessing
from multiprocessing import Pool

# from multiprocessing import current_process

# MPI imports, depending on configuration
# from schwimmbad import MPIPool
# from schwimmbad import MultiPool
# import mpi4py
from functools import partial

# Local application/library specific imports

# pandas
# import pandas

# cosmohammer
# import cosmoHammer

# galsim
# import galsim

# emcee
# import emcee

# Here we check that version of Zernike_Module being used is as expected.
# User has to manualy the version number in order to ensure
# that the verions used is the one the user expects.
# This is somewhat extreme, but potentially saves a lot of trouble
assert Zernike_Module.__version__ == '0.51e', "Zernike_Module version is not as expected"

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
np.set_printoptions(suppress=True)

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

print('multiprocessing.cpu_count()' + str(multiprocessing.cpu_count()))
print(str(socket.gethostname()) + ': Start time for importing is: ' + time.ctime())

"""
Created on Mon Mar 30 2020

Code and scripts used for parameter estimation for Zernike analysis

Versions:
Mar 31, 2020: ? -> 0.28 added argparser and extra Zernike
Apr 07, 2020: 0.28 -> 0.28b added /tigress/ncaplar/Result_DirectAndInt_FinalResults/' to the output path of
                            the final product
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
Jul 14, 2020: 0.32b -> 0.32c changed /tigress/ncaplar/Data/ to /tigress/ncaplar/ReducedData/ and
                             Stamps_Cleaned to Stamps_cleaned
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
May 20, 2021: 0.41 -> 0.41b updated eps=8 parameter and set w=0.81
May 27, 2021: 0.41b -> 0.42 introduced wavelength in the estimation algorithm
Jun 15, 2021: 0.42 -> 0.42b dataset 7, only Neon included
Jun 28, 2021: 0.42b -> 0.42c wavelength reintroduced
Jul 01, 2021: 0.42c -> 0.43 fake image index increased to accomodate extra 21 fibers
Jul 01, 2021: 0.43 -> 0.43a introduced final(arc) for 10+21 fibers
Jul 02, 2021: 0.43a -> 0.43b small changes to multiprocessing
Jul 11, 2021: 0.43b -> 0.43c str to int, in index when catching nearby points for initial solution
Jul 28, 2021: 0.43c -> 0.43d added support for numpy.str_ in index
Oct 12, 2021: 0.43d -> 0.44 initial implementation of mutli_spot version
Oct 25, 2021: 0.44 -> 0.45 changed soc_vel to be different for each par
Nov 18, 2021: 0.45 -> 0.45a changed RESULT_FOLDER to be on gpfs
Dec 06, 2021: 0.45a -> 0.46 go via label and multi arc possibility
Dec 09, 2021: 0.46 -> 0.46a introduced fixed_single_spot
Mar 09, 2022: 0.46a -> 0.46b fixed_par test
Mar 12, 2022: 0.46b -> 0.46c twentytwo_or_extra parameter equal to paramCount
Mar 17, 2022: 0.46c -> 0.46d changed soc_vel application
Apr 14, 2022: 0.46d -> 0.46e changed ``is'' to ``==''' in if analysis_type_fiber == 'fiber_par':
Apr 14, 2022: 0.46e -> 0.46f removed saving intermediate files

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""


__version__ = "0.46f"

parser = argparse.ArgumentParser(description="Starting args import",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog='Done with import')

##########################################################################
print('#####################################################################################################')

##########################################################################
print('Start time is: ' + time.ctime())
time_start = time.time()

print('version of Zernike_parameter_estimation is: ' + str(__version__))

################################################
# nargs '+' specifies that the algorithm can accept one or more arguments
parser.add_argument(
    "-label",
    help="name of the observation (actually a number/numbers) which we will analyze",
    nargs='+',
    type=str,
    choices=['m4', 'm35', 'm3', 'm25', 'm2', 'm15', 'm1', 'm05', '0',
             'p05', 'p1', 'p15', 'p2', 'p25', 'p3', 'p35', 'p4', '0p'])
################################################
# name of the spot (again, actually number) which we will analyze
parser.add_argument(
    "-spot",
    help="name of the spot (again, actually a number) which we will analyze",
    nargs='+',
    type=int,
    default=argparse.SUPPRESS)
################################################
# number of steps each walker will take
parser.add_argument(
    "-nsteps",
    help="number of steps each walker will take ",
    type=int)
################################################
# input argument that controls the paramters of the cosmo_hammer process
# if in doubt, eps=8 is probably a solid option
parser.add_argument(
    "-eps",
    help="input argument that controls the paramters of the cosmo_hammer process;\
                    if in doubt, eps=8 is probably a solid option ",
    type=int)
################################################
# which dataset is being analyzed [numerical value of 0,1,2,3,4, 5, 6, 7, 8]
parser.add_argument("-dataset", help="which dataset is being analyzed\
                    [numerical value between 0 and 8] ", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
################################################
parser.add_argument(
    "-arc",
    help="which arc lamp is being analyzed (HgAr for Mercury-Argon, \
                    Ne for Neon, Kr for Krypton)  ",
    nargs='+',
    type=str,
    choices=["HgAr", "Ar", "Ne", "Kr", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
################################################
parser.add_argument(
    "-double_sources",
    help="are there two sources in the image (True, False) ",
    default='False',
    type=str,
    choices=[
        'True',
         'False'])
################################################
parser.add_argument(
    "-double_sources_positions_ratios",
    help="parameters for second source ",
    action='append')
################################################
parser.add_argument("-twentytwo_or_extra", help="number of Zernike components\
                    (22 or number larger than 22 which leads to extra Zernike analysis)", type=int)
################################################
parser.add_argument("-date_of_input", help="input date")
################################################
parser.add_argument(
    "-direct_or_interpolation",
    help="direct or interpolation ",
    choices=[
        "direct",
         "int"])
################################################
parser.add_argument("-date_of_output", help="date_of_output ")
################################################
parser.add_argument("-analysis_type", help="fixed_single_spot? ")
################################################
parser.add_argument("-analysis_type_fiber", help="Are fiber parameters fixed?")

# Finished with specifying arguments

################################################
# Assigning arguments to variables
args = parser.parse_args()
################################################
# put all passed observations in a list

list_of_labelInput = args.label
len_of_list_of_labelInput = len(list_of_labelInput)
# len_of_list_of_obs = len(list_of_obs)

# if you passed only one value, somehow put in a list and make sure that
# the code runs
# print('all values in the obs_list is/are: ' + str(list_of_obs))
# print('number of images analyzed is: ' + str(len_of_list_of_obs))

if len_of_list_of_labelInput > 1:
    multi_var = True
else:
    multi_var = False

# obs variable is assigned to the first number in the list
# obs_init = list_of_obs[0]
# print('obs_init is: ' + str(obs_init))
################################################

list_of_spots = args.spot
len_of_list_of_spots = len(list_of_spots)


print('all values in the list_of_spots is/are: ' + str(list_of_spots))
print('number of spots analyzed is: ' + str(len_of_list_of_spots))

if len_of_list_of_spots > 1:
    multi_spots = True
else:
    multi_spots = False

# if only one spots, use single_number variable
# if multi_spots is False:
#     single_number = int(args.spot)
#     print('spot number (single_number) is: ' + str(single_number))

################################################
nsteps = args.nsteps
print('nsteps is: ' + str(nsteps))
################################################
eps = args.eps
print('eps parameter is: ' + str(eps))
"""
if eps == 1:
    # particle count, c1 parameter (individual), c2 parameter (global)
    options = [390, 1.193, 1.193]
if eps == 2:
    options = [790, 1.193, 1.193]
    nsteps = int(nsteps / 2)
if eps == 3:
    options = [390, 1.593, 1.193]
if eps == 4:
    options = [390, 0.993, 1.193]
if eps == 5:
    options = [480, 2.793, 1.593]
if eps == 6:
    options = [480, 2.793, 1.193]
if eps == 7:
    options = [16, 2.793, 1.193]
if eps == 8:
    options = [480, 2.793, 0.193]
if eps == 9:
    options = [190, 1.193, 1.193]
    nsteps = int(2 * nsteps)
if eps == 10:
    options = [390, 1.893, 2.893]

c1 = options[1]
c2 = options[2]
"""
################################################
dataset = args.dataset

print('Dataset analyzed is: ' + str(dataset))
if dataset == 0 or dataset == 1:
    print('ehm.... old data, not analyzed')

# folder contaning the data from December 2017
# dataset 0
# DATA_FOLDER='/tigress/ncaplar/Data/Dec18Data/'

# folder contaning the data from February 2019
# dataset 1
# DATA_FOLDER='/tigress/ncaplar/Data/Feb5Data/'

# folder containing the data taken with F/2.8 stop in April and May 2019
# dataset 2
if dataset == 2:
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_28/'

# folder containing the data taken with F/2.8 stop in April and May 2019
# dataset 3
if dataset == 3:
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_Jun_25/'

# folder containing the data taken with F/2.8 stop in July 2019
# dataset 4 (defocu) and 5 (fine defocus)
if dataset == 4 or dataset == 5:
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_Aug_14/'

# folder contaning the data taken with F/2.8 stop in November 2020 on Subaru
if dataset == 6:
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_Nov_20/'

# folder contaning the data taken with F/2.8 stop in June 2021, at LAM, on SM2
if dataset == 7:
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_21_2021/'

# folder contaning the data taken with F/2.8 stop in June 2021, at Subaru
# (21 fibers)
if dataset == 8:
    if 'subaru' in socket.gethostname():
        DATA_FOLDER = '/work/ncaplar/ReducedData/Data_May_25_2021/'
    else:
        DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_25_2021/'


STAMPS_FOLDER = DATA_FOLDER + 'Stamps_cleaned/'
DATAFRAMES_FOLDER = DATA_FOLDER + 'Dataframes/'
if 'subaru' in socket.gethostname():
    RESULT_FOLDER = '/work/ncaplar/Results/'
else:
    RESULT_FOLDER = '/tigress/ncaplar/Results/'
# RESULT_FOLDER = '/scratch/gpfs/ncaplar/Results/'

################################################
arc = args.arc
list_of_arc = args.arc
print('arc lamp is: ' + str(arc))

# inputs for Dec 2017 data
# dataset=0

# if arc=="HgAr":
#    single_number_focus=8603
# elif arc=="Ne":
#    single_number_focus=8693
# else:
#    print("Not recognized arc-line")

# inputs for Februaruy 2019 data, F/3.2 stop
# dataset=1

# if str(arc)=="HgAr":
#    single_number_focus=11748
# elif str(arc)=="Ne":
#    single_number_focus=12355
# else:
#    print("Not recognized arc-line")
"""
# inputs for Februaruy 2019 data, F/2.8 stop
# dataset=2
if dataset == 2:
    if str(arc) == "HgAr":
        single_number_focus = 17017 + 54
    elif str(arc) == "Ne":
        single_number_focus = 16292
    elif str(arc) == "Kr":
        single_number_focus = 17310 + 54
    else:
        print("Not recognized arc-line")

# inputs for April/May 2019 data, F/2.5 stop
# dataset=3
if dataset == 3:
    if str(arc) == "HgAr":
        single_number_focus = 19238 + 54
    elif str(arc) == "Ne":
        single_number_focus = 19472 + 54
    else:
        print("Not recognized arc-line")

# defocused data from July 2019
# dataset=4
if dataset == 4:
    if str(arc) == "HgAr":
        single_number_focus = 21346 + 54
    elif str(arc) == "Ne":
        single_number_focus = 21550 + 54
    elif str(arc) == "Kr":
        single_number_focus = 21754 + 54
    else:
        print("Not recognized arc-line")

# defocused data from November 2020
# dataset = 6
if dataset == 6:
    if str(arc) == "Ar":
        single_number_focus = 34341 + 48
    elif str(arc) == "Ne":
        single_number_focus = 34217 + 48
    elif str(arc) == "Kr":
        single_number_focus = 34561 + 48
    else:
        print('Arc line specified is ' + str(arc))
        print("Not recognized arc-line")

# defocused data from June 2021, at LAM, at SM2, corrected
if dataset == 7:
    # if str(arc) == "Ar":
    #    single_number_focus=34341+48
    if str(arc) == "Ne":
        single_number_focus = 27677
    # elif str(arc) == "Kr":
    #    single_number_focus = 34561+48
    else:
        print('Arc line specified is ' + str(arc))
        print("Not recognized arc-line")

# defocused data from June 2021, at LAM, at SM2, corrected
if dataset == 8:
    if str(arc) == "Ar":
        single_number_focus = 51581
    elif str(arc) == "Ne":
        single_number_focus = 59751
    elif str(arc) == "Kr":
        single_number_focus = 52181
    else:
        print('Arc line specified is ' + str(arc))
        print("Not recognized arc-line")

"""
################################################
print('double_sources argument: ' + str(args.double_sources))


def str2bool(v):
    """
    Small function that take a string and gives back boolean value

    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


double_sources = str2bool(args.double_sources)

print('double sources argument is: ' + str(double_sources))
################################################
double_sources_positions_ratios = eval(args.double_sources_positions_ratios[0])
print('double source parameters are: ' + str(double_sources_positions_ratios))
################################################
twentytwo_or_extra = args.twentytwo_or_extra
print('22_or_extra parameter is: ' + str(twentytwo_or_extra))

################################################
parser.add_argument("date_of_input", help="input date")
date_of_input = args.date_of_input
print('date of input database: ' + str(date_of_input))
################################################
parser.add_argument("direct_or_interpolation", help="direct or interpolation ",
                    choices=["direct", "interpolation"])
direct_or_interpolation = args.direct_or_interpolation
print('direct or interpolation: ' + str(direct_or_interpolation))
################################################
parser.add_argument("date_of_output", help="date_of_output ")
date_of_output = args.date_of_output
print('date of output: ' + str(date_of_output))
################################################
parser.add_argument("analysis_type", help="analysis_type ")
analysis_type = args.analysis_type
print('analysis_type: ' + str(analysis_type))
################################################
parser.add_argument("analysis_type_fiber", help="analysis_type_fiber")
analysis_type_fiber = args.analysis_type_fiber
print('analysis_type_fiber: ' + str(analysis_type_fiber))
################################################

Zernike_estimation_preparation_instance = \
    Zernike_estimation_preparation(list_of_labelInput=list_of_labelInput,
                                   list_of_spots=list_of_spots, dataset=dataset,
                                   list_of_arc=list_of_arc, eps=eps, nsteps=nsteps,
                                   analysis_type=analysis_type,analysis_type_fiber=analysis_type_fiber)

particleCount, c1, c2 = Zernike_estimation_preparation_instance.return_auxiliary_info()

################################################
################################################
################################################
################################################
################################################
################################################
# (1.) import data
################################################

array_of_sci_images_multi_spot, array_of_var_images_multi_spot,\
    array_of_mask_images_multi_spot, array_of_obs_cleaned_multi_spot = \
    Zernike_estimation_preparation_instance.get_sci_var_mask_data()

################################################
# (2.) create output names
################################################

# name of the outputs
# create list with names for each spot

list_of_NAME_OF_CHAIN, list_of_NAME_OF_LIKELIHOOD_CHAIN =\
    Zernike_estimation_preparation_instance.create_output_names(date_of_output)

################################################
# (3.) import dataframes
################################################

# where are the dataframes located
# these files give auxiliary information which enables us to connect spot number with other properties
# such as the position on the detector, wavelength, etc...
# Ar (Argon)
"""
if str(arc) == 'Ar' or arc == 'HgAr':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation) + '_Ar_from_' +
              str(date_of_input) + '.pkl', 'rb') as f:
        results_of_fit_input_HgAr = pickle.load(f)
        print('results_of_fit_input_Ar is taken from: ' + str(f))
    # if before considering all fibers
    if dataset < 8:
        with open(DATAFRAMES_FOLDER + 'finalAr_Feb2020', 'rb') as f:
            finalAr_Feb2020_dataset = pickle.load(f)
    else:
        with open(DATAFRAMES_FOLDER + 'finalAr_Jul2021.pkl', 'rb') as f:
            finalAr_Feb2020_dataset = pickle.load(f)


# Ne (Neon)
if str(arc) == 'Ne':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation) + '_Ne_from_' +
              str(date_of_input) + '.pkl', 'rb') as f:
        results_of_fit_input_Ne = pickle.load(f)
    print('results_of_fit_input_Ne is taken from: ' + str(f))
    if dataset < 8:
        with open(DATAFRAMES_FOLDER + 'finalNe_Feb2020', 'rb') as f:
            finalNe_Feb2020_dataset = pickle.load(f)
    else:
        with open(DATAFRAMES_FOLDER + 'finalNe_Jul2021.pkl', 'rb') as f:
            finalNe_Feb2020_dataset = pickle.load(f)

# Kr (Krypton)
if str(arc) == 'Kr':
    with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation) + '_Kr_from_' +
              str(date_of_input) + '.pkl', 'rb') as f:
        results_of_fit_input_Kr = pickle.load(f)
    print('results_of_fit_input_Kr is taken from: ' + str(f))
    if dataset < 8:
        with open(DATAFRAMES_FOLDER + 'finalKr_Feb2020', 'rb') as f:
            finalKr_Feb2020_dataset = pickle.load(f)
    else:
        with open(DATAFRAMES_FOLDER + 'finalKr_Jul2021.pkl', 'rb') as f:
            finalKr_Feb2020_dataset = pickle.load(f)
"""


# TODO: do we have to change this for multi spot?

# list_of_z4Input = []
# for obs in list_of_obs_cleaned:
#    z4Input = z4Input_possibilites[obs_possibilites == obs][0]
#    list_of_z4Input.append(z4Input)

# list of labels that we are passing to the algorithm, that are `clean`
# list_of_labelInput = []
# for obs in list_of_obs_cleaned:
#    labelInput = label[list(obs_possibilites).index(obs)]
#    list_of_labelInput.append(labelInput)

# TODO: evaluate if all of these different inputs are needed
# list_of_defocuses_input_long input in Tokovinin algorithm
list_of_defocuses_input_long = list_of_labelInput

print('list_of_labelInput: ' + str(list_of_labelInput))

# list of labels without values near focus (for possible analysis with
# Tokovinin alrogithm)
list_of_labelInput_without_focus_or_near_focus = deepcopy(list_of_labelInput)
for i in ['m15', 'm1', 'm05', '0', 'p05', 'p1', 'p15']:
    if i in list_of_labelInput:
        list_of_labelInput_without_focus_or_near_focus.remove(i)

print('list_of_labelInput_without_focus_or_near_focus: ' +
      str(list_of_labelInput_without_focus_or_near_focus))

# positional indices of the defocused data in the whole set of input data
# (for possible analysis with Tokovinin alrogithm)
index_of_list_of_labelInput_without_focus_or_near_focus = []
for i in list_of_labelInput_without_focus_or_near_focus:
    index_of_list_of_labelInput_without_focus_or_near_focus.append(
        list_of_labelInput.index(i))

print('index_of_list_of_labelInput_without_focus_or_near_focus: ' +
      str(index_of_list_of_labelInput_without_focus_or_near_focus))


# Input the zmax that you wish to achieve in the analysis
zmax = twentytwo_or_extra

# depening on the arc, select the appropriate dataframe
# change here to account for 21 fiber data
"""
if arc == "HgAr":
    results_of_fit_input = results_of_fit_input_HgAr
    # finalArc = finalHgAr_Feb2020_dataset
elif arc == "Ne":
    results_of_fit_input = results_of_fit_input_Ne
    finalArc = finalNe_Feb2020_dataset
elif arc == "Kr":
    results_of_fit_input = results_of_fit_input_Kr
    finalArc = finalKr_Feb2020_dataset
elif arc == "Ar":
    results_of_fit_input = results_of_fit_input_HgAr
    finalArc = finalAr_Feb2020_dataset

else:
    print("what has happened here? Only Argon, HgAr, Neon and Krypton implemented")
"""

array_of_wavelengths = Zernike_estimation_preparation_instance.create_array_of_wavelengths()

# pool = MPIPool()
# if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

print('print check just before the pool of workers is created')
if 'subaru' in socket.gethostname():
    pool = Pool(32)
else:
    #pool = Pool(processes=multiprocessing.cpu_count())
    pool = Pool(processes=20)
print('print check just after the pool of workers is created')


################################################
# (5.) Create Tokovinin algorithm instances
################################################

list_of_Tokovinin_multi_instance_with_pool = []
for s in range(len(list_of_spots)):
    list_of_Tokovinin_multi_instance_with_pool.append(Tokovinin_multi(
        array_of_sci_images_multi_spot[s],
        array_of_var_images_multi_spot[s],
        list_of_mask_images=array_of_mask_images_multi_spot[s],
        wavelength=array_of_wavelengths[s],
        dithering=1,
        save=0,
        verbosity=0,
        npix=1536,
        zmax=twentytwo_or_extra,
        list_of_defocuses=list_of_defocuses_input_long,
        double_sources=double_sources,
        double_sources_positions_ratios=double_sources_positions_ratios,
        fit_for_flux=True,
        test_run=False,
        num_iter=None,
        pool=pool))

list_of_Tokovinin_multi_instance_without_pool = []
for s in range(len(list_of_spots)):
    list_of_Tokovinin_multi_instance_without_pool.append(Tokovinin_multi(
        array_of_sci_images_multi_spot[s],
        array_of_var_images_multi_spot[s],
        list_of_mask_images=array_of_mask_images_multi_spot[s],
        wavelength=array_of_wavelengths[s],
        dithering=1,
        save=0,
        verbosity=0,
        npix=1536,
        zmax=twentytwo_or_extra,
        list_of_defocuses=list_of_defocuses_input_long,
        double_sources=double_sources,
        double_sources_positions_ratios=double_sources_positions_ratios,
        fit_for_flux=True,
        test_run=False,
        num_iter=None,
        pool=None))

################################################
# (6.) Create parametrization proposals
################################################

array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d =\
    Zernike_estimation_preparation_instance.create_parametrization_proposals(date_of_input,
                                                                             direct_or_interpolation,
                                                                             twentytwo_or_extra)


########
# "lower_limits" and "higher_limits" are only needed to initalize the cosmo_hammer code,
# but not used in the actual evaluation.
# As such these are purely dummy values.
# TODO: find a way in which this can be deleted?
"""
lower_limits = np.array([z4Input - 3, -1, -1, -1, -1, -1, -1, -1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0.5, 0.05, -0.8, -0.8, 0, -0.5,
                         0, 0, 0, 0,
                         0.8, -np.pi / 2, 0.5,
                         0, 0, 0.85, -0.8,
                         1200, 0.5, 0,
                         0.2, 1.65, 0.9])

higher_limits = np.array([z4Input + 3, 1, 1, 1, 1, 1, 1, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1.2, 0.2, 0.8, 0.8, 0.2, 0.5,
                          1, 1, 1, 10,
                          1, np.pi / 2, 1.01,
                          0.05, 1, 1.15, 0.8,
                          120000, 3.5, 0.5,
                          1.1, 1.95, 1.1])
"""
########

################################################
# (7.) Sanity check of inputs
################################################

# initialize pool
# pool = MPIPool()
# if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

"""
number_of_extra_zernike = twentytwo_or_extra - 22
# pool = Pool(processes = 36)
# pool = Pool()

print('Name of machine is ' + socket.gethostname())

zmax_input = twentytwo_or_extra

for i in range(len(list_of_obs_cleaned)):
    print('Adding image: ' + str(i + 1) +
          str('/') + str(len(list_of_obs_cleaned)))
    for s in range(len(list_of_spots)):

        list_of_sci_images = array_of_sci_images_multi_spot[s]
        list_of_var_images = array_of_var_images_multi_spot[s]
        single_number = list_of_spots[s]

        sci_image = list_of_sci_images[i]
        var_image = list_of_var_images[i]
        # allparameters_proposal_22 = list_of_allparameters_proposal[i]

        print('Spot coordinates are: ' + str(single_number))
        print('Size of input image is: ' + str(sci_image.shape))
        print('First value of the sci image: ' + str(sci_image[0][0]))
        print('First value of the var image: ' + str(var_image[0][0]))
    print('Steps: ' + str(nsteps))
    print('Name: ' + str(NAME_OF_CHAIN))
    print('Zmax is: ' + str(zmax))
    print(str(socket.gethostname()) +
          ' - Starting calculation at ' + time.ctime())

"""
################################################
# (8.) Create init parameters for particles
################################################

# Hardwired on March 10, 2022 to study global par 
# list_of_array_of_particle_position_proposal, list_of_array_of_particle_velocity_proposal,\
#    list_of_best_particle_likelihood, paramCount =\
#    Zernike_estimation_preparation_instance.create_init_parameters_for_particles(zmax_input=twentytwo_or_extra, 
#    analysis_type='fiber_par')

# Modified April 1
list_of_array_of_particle_position_proposal, list_of_array_of_particle_velocity_proposal,\
    list_of_best_particle_likelihood, paramCount =\
    Zernike_estimation_preparation_instance.create_init_parameters_for_particles(zmax_input=twentytwo_or_extra, 
    analysis_type=analysis_type, analysis_type_fiber=analysis_type_fiber)
    
if analysis_type_fiber == 'fiber_par':
    up_to_which_z_var = False
else:
    up_to_which_z_var = None

print('up_to_which_z_var '+str(up_to_which_z_var))


list_of_swarm = []
list_of_best_particle = []
list_of_particles = []
list_of_likelihood_of_particles = []

################################################
# (9.) Run init Tokovinin
################################################
now = datetime.now()
print('Starting work on the initial best result (defocus) at ' + str(now))

time_start_initial = time.time()
list_of_best_result = []
for s in range(len(list_of_spots)):
    list_of_best_result.append(list_of_Tokovinin_multi_instance_with_pool[s](
        array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s],
        return_Images=True,
        num_iter='initial',
        previous_best_result=None,
        use_only_chi=True,
        multi_background_factor=3,
        up_to_which_z = up_to_which_z_var))

time_end_initial = time.time()
print('Time for the initial evaluation is ' +
      str(time_end_initial - time_start_initial))


# the best likelihood after evaluation, at index 1 in the output
# (index 0 shows the likelihood before the evaluation)
for s in range(len(list_of_spots)):
    print('Initial evaluation output for spot ' +
          str(s) + ' is: ' + str(list_of_best_result[s][1]))

print('Finished work on the initial best result')

array_of_particle_pbest_position = np.zeros((particleCount, paramCount))
array_of_particle_pbest_likelihood = np.ones((particleCount, 1)) * -9999999

# TODO check if this is -1
list_of_obs_cleaned = Zernike_estimation_preparation_instance.create_list_of_obs_from_list_of_label()
obs = list_of_obs_cleaned[-1]
for s in range(len(list_of_spots)):
    single_number_original = list_of_spots[s]
    np.save(RESULT_FOLDER + 'initial_best_result_' + str(date_of_output) + '_Single_P_' +
            str(obs) + str(single_number_original) + str(eps) + str(arc), list_of_best_result[s])

# this allows to have multi_background_factor to be different in each step
# at the moment set up to be 3 in each step
array_of_multi_background_factors = np.ones((nsteps)) * 3

################################################
# Start stepping
################################################

# arrays in which we collect final results
array_of_gbests = np.empty((len(list_of_spots), nsteps), dtype=object)
array_of_swarms = np.empty((len(list_of_spots), nsteps), dtype=object)

for step in range(nsteps):

    ################################################
    # (10.) Run Tokovinin
    ################################################

    now = datetime.now()
    print('############################################################')
    print('Starting step ' + str(step) + ' out of ' + str(nsteps-1) + ' at ' + str(now))
    print('Variable array_of_multi_background_factors[step]: ' + str(array_of_multi_background_factors[step]))
    time_start_evo_step = time.time()
    print('Pool is at ' + str(pool))

    # code that moves Zernike parameters via small movements
    # each particle is evaluated individually
    # but the changes of Zernike parameters are assumed from the previous best result
    # if this is first step, the best result is from dedicated initial run
    # reports likelihod, but minimizes absolute different of images/std - model/std
    # The output is saved in poorly named list_of_out2

    # list_of_array_of_particle_position_proposal carries particle positions for this step
    # this list needs to be updated at the end of the step

    # list_of_best_results carries the best result from the previous step
    # this list needs to be updated at the end of the step

    list_of_out2 = []
    for s in range(len(list_of_spots)):
        out1 = pool.map(partial(
            list_of_Tokovinin_multi_instance_without_pool[s],
            return_Images=True,
            previous_best_result=list_of_best_result[s],
            use_only_chi=True,
            multi_background_factor=array_of_multi_background_factors[step],
            up_to_which_z = up_to_which_z_var),
            list_of_array_of_particle_position_proposal[s])
        out2 = np.array(list(out1))
        list_of_out2.append(out2)
        print('Checkpoint - finished creating out2 for '+str(list_of_spots[s]))

    time_end_evo_step = time.time()

    print('Time for the evaluation of initial Tokovinin_multi_instance_without_pool in step ' +
          str(step) + ' is ' + str(time_end_evo_step - time_start_evo_step))

    # save ?
    # np.save(RESULT_FOLDER + 'list_of_out2_' + str(step), list_of_out2)


    ################################################
    # (11.) Create swarm
    ################################################

    ##################
    # create a swarm here
    # input are, for each spot:
    # 1. out2
    #    out2[i][1] - to get likelihood
    #    out2[i][8] = to get position
    # 2. previous swarm
    # swarm contains each particle - 0. likelihood, 1. position (after computation),
    # 2. velocity (that brought you here, i.e., before computation)
    # initialized with outputs from Tokovinin_multi_instance_without_pool

    list_of_swarms = []

    for s in range(len(list_of_spots)):
        out2 = list_of_out2[s]

        swarm_list = []

        list_of_swarms_single_spot = []

        for i in range(particleCount):

            likelihood_single_particle = out2[i][1]
            position_single_particle = out2[i][8]

            if step == 0:
                array_of_particle_velocity_proposal = list_of_array_of_particle_velocity_proposal[s]
                velocity_single_particle = array_of_particle_velocity_proposal[i]
            else:
                swarm = list_of_swarm[s]
                velocity_single_particle = swarm[:, 2][i] # noqa

            swarm_list.append(
                [likelihood_single_particle, position_single_particle, velocity_single_particle])

        swarm = np.array(swarm_list)
        # collate the information about the swarms

        list_of_swarms_single_spot.append(swarm)
        # same as list_of_swarms.append(swarm)
        list_of_swarms.append(list_of_swarms_single_spot[0])

    ##################
    # find the best particle
    # the method:
    # arrange the particels in all of the spots
    # find the particle, which on average has highest ordering
    # for all of the spots
    # np.save(RESULT_FOLDER + 'list_of_swarms' + '_' + str(step), list_of_swarms)

    list_of_quality_of_particle = []
    for s in range(len(list_of_spots)):
        # select the swarm for a single spot
        swarm = list_of_swarms[s]
        # select only the quality measures
        swarm_likelihood = swarm[:, 0]

        # prepare an array which we will fill with the information about ordering of the
        # quality of the result
        quality_of_particle = np.empty(len(swarm_likelihood))
        # populate the array with the ordering of the particles
        quality_of_particle[np.arange(len(swarm_likelihood))[np.argsort(swarm_likelihood)]] = \
            np.arange(len(swarm_likelihood))
        list_of_quality_of_particle.append(quality_of_particle)

    # find the index of the best particle from the quality odering for each spot
    array_of_quality_of_particle = np.array(list_of_quality_of_particle)
    average_of_quality_of_particle = np.mean(array_of_quality_of_particle, axis=0)
    index_of_best = \
        np.where(average_of_quality_of_particle == np.max(average_of_quality_of_particle))[0][0]
    print('Particle with best index in step ' + str(step) + ': ' + str(index_of_best))

    # given the index of the best particles, extract this information
    list_of_best_particle_this_step = []
    list_of_best_particle_likelihood_this_step = []
    for s in range(len(list_of_spots)):
        swarm = list_of_swarms[s]
        best_particle_this_step = swarm[index_of_best]
        best_particle_likelihood_this_step = best_particle_this_step[0]
        # print('best_particle_likelihood until now for spot ' + str(s) + ', step ' +
        #       str(step) + ': ' + str(best_particle_likelihood))
        print('Proposed best_particle_likelihood_this_step for spot ' + str(s) + ', step ' +
              str(step) + ': ' + str(best_particle_likelihood_this_step))

        # list_of_best_particle_this_step_single_spot = []
        # list_of_best_particle_likelihood_this_step_single_spot = []

        # list_of_best_particle_this_step_single_spot.append(best_particle_this_step)
        # list_of_best_particle_likelihood_this_step_single_spot.append(best_particle_likelihood_this_step)

        # information about best particle for all spots here
        list_of_best_particle_this_step.append(best_particle_this_step)

    # np.save(RESULT_FOLDER + 'list_of_best_particle_this_step_' + str(step), list_of_best_particle_this_step)

    ################################################
    # (14.) run dedicated Tokovinin
    ################################################
    list_of_best_results = []
    list_of_best_particle_likelihood = []
    list_of_best_particle_position = []
    for s in range(len(list_of_spots)):

        list_of_best_particle_for_single_spot = []
        list_of_best_particle_likelihood_for_single_spot = []
        list_of_best_particle_position_for_single_spot = []
        if best_particle_likelihood_this_step > -9999:

            best_particle_proposal = list_of_best_particle_this_step[s]
            best_particle_proposal_position = best_particle_proposal[1]

            # update best_result
            # we are computing if the best solution is actually as good as
            # proposed solution
            # TODO: can be skipped if evaluating only pupil parameters
            print(
                'Starting work on the new proposed best result in step ' +
                str(step) + ' for spot ' + str(s))
            time_start_best = time.time()
            
            fiber_par = True
            # attempt to evade the evaluation if not needed
            if fiber_par is True:
                out2 = list_of_out2[s]
                best_result = out2[index_of_best]

            else:
                best_result = list_of_Tokovinin_multi_instance_with_pool[s](
                    best_particle_proposal_position,
                    return_Images=True,
                    num_iter=step,
                    previous_best_result=None,
                    use_only_chi=True,
                    multi_background_factor=array_of_multi_background_factors[step],
                    up_to_which_z = up_to_which_z_var)

            time_end_best = time.time()

            # updating the list with best results, to be carried over for the next step
            list_of_best_results.append(best_result)
            print('Time for the best evaluation for spot ' + str(s) + ' in step ' +
                  str(step) + ' is ' + str(time_end_best - time_start_best))

            best_particle_likelihood_this_step = best_result[1]
            best_particle_position_this_step = best_result[8]

            print('Best result proposed output for spot (first 5 par) ' + str(s) + ' is: ' +
                  str(best_particle_proposal_position[:5]))
            print('Best result proposed output for spot (fiber_par) ' + str(s) + ' is: ' +
                  str(best_particle_proposal_position[19*2+10:19*2+15]))
            print('Best result output after final Tokovinin for spot (first 5 par) ' + str(s) + ' is: ' +
                  str(best_particle_position_this_step[:5]))
            print('Best result proposed output for spot (fiber_par) ' + str(s) + ' is: ' +
                  str(best_particle_position_this_step[19*2+10:19*2+15]))
            # print(
            #    'best_particle_likelihood for spot ' + str(s) + ' until now: ' +
            #    str(best_particle_likelihood))
            print(
                'Final best_particle_likelihood_this_step for spot ' + str(s) + ': ' +
                str(best_particle_likelihood_this_step))

            # if best_particle_likelihood_this_step>best_particle_likelihood:
            if best_particle_likelihood_this_step > -9999:
                best_particle_likelihood = best_particle_likelihood_this_step
                best_particle_position = best_particle_position_this_step
                # likelihood of the best particle, its position and
                # velocity is zero
                best_particle = [
                    best_particle_likelihood,
                    best_particle_position,
                    np.zeros(paramCount)]
        else:
            # in 0.45 and later versions this should never happen unless the evaluation fails
            print('Proposed solution is worse than current, so do not evaluate')

        best_particle_position = best_particle[1]

        list_of_best_particle_for_single_spot.append(best_particle)
        list_of_best_particle_position_for_single_spot.append(best_particle_position)
        list_of_best_particle_likelihood_for_single_spot.append(best_particle_likelihood)

        list_of_best_particle.append(list_of_best_particle_for_single_spot)
        list_of_best_particle_position.append(list_of_best_particle_position_for_single_spot)
        list_of_best_particle_likelihood_for_single_spot.\
            append(list_of_best_particle_likelihood_for_single_spot)

        # update the array in which we collect all the best solutions for post-analysis
        # same as array_of_gbests[s][step] = best_particle
        array_of_gbests[s][step] = list_of_best_particle_for_single_spot[0]

        # save result of the search for the best particle and result
        # np.save(RESULT_FOLDER+'best_particle_spot_' + str(s) + '_step_'+str(step), best_particle)
        # np.save(RESULT_FOLDER+'list_of_best_results_' + str(s) + '_step_'+str(step), list_of_best_results)
        # np.save(RESULT_FOLDER+'list_of_swarms'+str(step),list_of_swarms)

    ################################################
    # (15.) Suggest position for the next step
    ################################################

    list_of_gbests = []
    list_of_swarm = []
    list_of_particle_position_proposal = []

    list_of_array_of_particle_position_proposal = []

    array_of_velocity_modifiers = np.random.random_sample(size=(len(list_of_spots), paramCount))*(2)
    array_of_velocity_modifiers = np.random.random_sample(size=(paramCount))*(2)
    
    for s in range(len(list_of_spots)):
        swarm = list_of_swarms[s]
        best_particle_position = list_of_best_particle_position[s][0]
        # np.save(RESULT_FOLDER + 'best_particle_position_spot_' + str(s) + '_step_' + str(step),
        #         best_particle_position)
        # updating velocity
        list_of_particle_position_proposal_single_spot = []
        for i in range(particleCount):

            particle_likelihood = np.array(swarm[i][0])
            particle_position = np.array(swarm[i][1])
            # if this is the best particle set all of its velocities to 0
            if i == index_of_best:
                particle_velocity = np.zeros(swarm[i][2].shape)
            else:
                particle_velocity = np.array(swarm[i][2])

            # velocities are set here
            w = 0.81
            part_vel = w * particle_velocity
            # cog_vel is at the moment set so it is always zero
            # in the original algorithm one of the `particle_positions` should be replaced
            # by the best position that that particular particle managed to achieve
            # during the whole procedure
            cog_vel = c1 * \
                np.random.uniform(0, 1, size=paramCount) * (particle_position - particle_position)
            # change in 0.45
            #soc_vel = c2 * array_of_velocity_modifiers[s] * (best_particle_position - particle_position)
            soc_vel = c2 * array_of_velocity_modifiers * (best_particle_position - particle_position)
            proposed_particle_velocity = part_vel + cog_vel + soc_vel

            # propose the position for next step and check if it is within limits
            # modify if it tries to venture outside limits
            proposed_particle_position = particle_position + proposed_particle_velocity

            proposed_global_parameters = proposed_particle_position[38:38 + 23]
            checked_global_parameters = check_global_parameters(proposed_global_parameters)

            # warn if the checking algorithm changes one of the parameters
            # if proposed_global_parameters != checked_global_parameters:
            #    print('checked_global_parameters are different from the proposed ones ')

            new_particle_position = np.copy(proposed_particle_position)
            new_particle_position[38:38 + 23] = checked_global_parameters

            particle_velocity = new_particle_position - particle_position

            # update velocities here
            swarm[i][2] = particle_velocity
            # create list/array of new proposed positions
            list_of_particle_position_proposal_single_spot.append(new_particle_position)

            # TODO: update that z-parameters are the same as in the best particle???

        # objects which are needed to go to the next step
        # list having array of particle positions for the next step
        list_of_array_of_particle_position_proposal.append(np.array(
            list_of_particle_position_proposal_single_spot))
        # this list is needed to properly calculate velocities
        list_of_swarm.append(swarm)

        array_of_swarms[s][step] = swarm

    time_end = time.time()
    print('Time for the whole evaluation until step ' +
          str(step) + ' is ' + str(time_end - time_start))

    gbests = np.array(list_of_best_particle)
    swarms = np.array(list_of_swarm)

    list_of_gbests.append(gbests)
    list_of_swarms.append(swarms)

################################################
# (16.) Compile results and finish
################################################

# np.save(RESULT_FOLDER + 'array_of_swarms', array_of_swarms)
# np.save(RESULT_FOLDER + 'array_of_gbests', array_of_gbests)

print('Managed to reach this point')
for s in range(len(list_of_spots)):
    gbests = array_of_gbests[s]
    swarms = array_of_swarms[s]

    minchain = gbests[-1][1]
    minln = gbests[-1][0]

    res = []
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][1])
    res = np.array(res)
    # reshape so that in has format of :
    # 0. nsteps (same as len(swarms))
    # 1. number of particles (same as len(swarms[0]))
    # 2. number of parameters (same as parInit1.shape[1])
    # chains = res.reshape(len(swarms), len(swarms[0]), parInit1.shape[1])

    chains = res.reshape(len(swarms), len(swarms[0]), len(minchain))

    res = []
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][0])
    res = np.array(res)
    ln_chains = res.reshape(len(swarms), len(swarms[0]))

    res = []
    for i in range(len(swarms)):
        for j in range(len(swarms[0])):
            res.append(swarms[i][j][2])
    res = np.array(res)

    # v_chains = res.reshape(len(swarms), len(swarms[0]), parInit1.shape[1])
    v_chains = res.reshape(len(swarms), len(swarms[0]), len(minchain))

    res_gbests_position = []
    res_gbests_fitness = []
    for i in range(len(swarms)):
        minchain_i = gbests[i][1]
        minln_i = gbests[i][0]
        res_gbests_position.append(minchain_i)
        res_gbests_fitness.append(minln_i)

    # Export
    NAME_OF_CHAIN = list_of_NAME_OF_CHAIN[s]
    NAME_OF_LIKELIHOOD_CHAIN = list_of_NAME_OF_LIKELIHOOD_CHAIN[s]

    # parInit1 = list_of_parInit1[s]
    # np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'parInit1', parInit1)
    np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'Swarm1', chains)
    np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'velocity_Swarm1', v_chains)

    np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'res_gbests_position', res_gbests_position)
    np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'res_gbests_fitness', res_gbests_fitness)
    np.save(RESULT_FOLDER + NAME_OF_LIKELIHOOD_CHAIN + 'Swarm1', ln_chains)

    # estimate of minchain error, that I do not actually use
    # keeping for now, until deleting soon (comment added October 12, 2021)
    # chi2reduced = 2 * np.min(np.abs(ln_chains)) / (sci_image.shape[0])**2
    # minchain_err = []
    # for i in range(len(minchain)):
    #    minchain_err = np.append(
    #        minchain_err, np.std(chains[:, :, i].flatten()))
    # minchain_err = np.array(minchain_err)

    print('Variable `Likelihood` at the moment: ' + str(np.abs(minln)))
    print('Variable `minchain` at the moment: ' + str(minchain))
    # print('minchain_err: ' + str(minchain_err))
    print('Time when first swarm run finished was: ' + time.ctime())
    time_end = time.time()
    print('Time taken was ' + str(time_end - time_start) + ' seconds')

# list_of_times.append(time_end - time_start)

sys.stdout.flush()
pool.close()
sys.exit(0) # noqa: W292