# standard library imports
from __future__ import absolute_import, division, print_function
from Zernike_Module import LN_PFS_multi_same_spot, create_parInit, Tokovinin_multi, check_global_parameters
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


import pickle

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

# here we check that version of Zernike_Module being used is as expected
# User has to manualy the version number in order to ensure
# that the verions used is the one the user expects
# This is somewhat extreme, but potentially saves a lot of trouble
assert Zernike_Module.__version__ == '0.49', "Zernike_Module version is not as expected"

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

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""


__version__ = "0.45"

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
    "-obs",
    help="name of the observation (actually a number/numbers) which we will analyze",
    nargs='+',
    type=int,
    default=argparse.SUPPRESS)
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
# if in doubt, eps=5 is probably a solid option
parser.add_argument(
    "-eps",
    help="input argument that controls the paramters of the cosmo_hammer process;\
                    if in doubt, eps=5 is probably a solid option ",
    type=int)
################################################
# which dataset is being analyzed [numerical value of 0,1,2,3,4 or 5]
parser.add_argument("-dataset", help="which dataset is being analyzed\
                    [numerical value between 0 and 8] ", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
################################################
parser.add_argument(
    "-arc",
    help="which arc lamp is being analyzed (HgAr for Mercury-Argon, \
                    Ne for Neon, Kr for Krypton)  ",
    choices=[
        "HgAr",
        "Ar",
        "Ne",
         "Kr"])
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
         "interpolation"])
################################################
parser.add_argument("-date_of_output", help="date_of_output ")
################################################
parser.add_argument("-analysis_type", help="defocus or focus? ")
################################################

# Finished with specifying arguments

################################################
# Assigning arguments to variables
args = parser.parse_args()
################################################
# put all passed observations in a list
list_of_obs = args.obs
len_of_list_of_obs = len(list_of_obs)

# if you passed only one value, somehow put in a list and make sure that
# the code runs
print('all values in the obs_list is/are: ' + str(list_of_obs))
print('number of images analyzed is: ' + str(len_of_list_of_obs))

if len_of_list_of_obs > 1:
    multi_var = True
else:
    multi_var = False

# obs variable is assigned to the first number in the list
obs_init = list_of_obs[0]
print('obs_init is: ' + str(obs_init))
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
if multi_spots is False:
    single_number = int(args.spot)
    print('spot number (single_number) is: ' + str(single_number))

################################################
nsteps = args.nsteps
print('nsteps is: ' + str(nsteps))
################################################
eps = args.eps
print('eps parameter is: ' + str(eps))

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
    options = [39, 2.793, 1.193]
if eps == 8:
    options = [480, 2.793, 0.193]
if eps == 9:
    options = [190, 1.193, 1.193]
    nsteps = int(2 * nsteps)
if eps == 10:
    options = [390, 1.893, 2.893]

c1 = options[1]
c2 = options[2]
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
    DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_25_2021/'


STAMPS_FOLDER = DATA_FOLDER + 'Stamps_cleaned/'
DATAFRAMES_FOLDER = DATA_FOLDER + 'Dataframes/'
RESULT_FOLDER = '/tigress/ncaplar/Results/'

################################################
arc = args.arc

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


################################################
################################################
################################################
################################################
################################################
################################################
# (1.) import data
################################################
list_of_sci_images_multi_spot = []
list_of_mask_images_multi_spot = []
list_of_var_images_multi_spot = []
list_of_obs_cleaned_multi_spot = []

for s in range(len(list_of_spots)):
    single_number = list_of_spots[s]

    list_of_sci_images = []
    list_of_mask_images = []
    list_of_var_images = []
    list_of_obs_cleaned = []
    # list_of_times = []

    # loading images for the analysis
    for obs in list_of_obs:
        try:
            sci_image = np.load(
                STAMPS_FOLDER +
                'sci' +
                str(obs) +
                str(single_number) +
                str(arc) +
                '_Stacked.npy')
            mask_image = np.load(
                STAMPS_FOLDER +
                'mask' +
                str(obs) +
                str(single_number) +
                str(arc) +
                '_Stacked.npy')
            var_image = np.load(
                STAMPS_FOLDER +
                'var' +
                str(obs) +
                str(single_number) +
                str(arc) +
                '_Stacked.npy')
            print(
                'sci_image loaded from: ' +
                STAMPS_FOLDER +
                'sci' +
                str(obs) +
                str(single_number) +
                str(arc) +
                '_Stacked.npy')
        except Exception:
            # change to that code does not fail and hang if the image is not found
            # this will lead to pass statment in next step because
            # np.sum(sci_image) = 0
            print('sci_image not found')
            sci_image = np.zeros((20, 20))
            var_image = np.zeros((20, 20))
            mask_image = np.zeros((20, 20))
            print('not able to load image at: ' + str(STAMPS_FOLDER + 'sci' +
                  str(obs) + str(single_number) + str(arc) + '_Stacked.npy'))

        # If there is no science image, do not add images
        if int(np.sum(sci_image)) == 0:
            print('No science image - passing')
            pass
        else:
            # do not analyze images where a large fraction of the image is masked
            if np.mean(mask_image) > 0.1:
                print(str(np.mean(mask_image) * 100) +
                      '% of image is masked... \
                      when it is more than 10% - exiting')
                pass
            else:
                # the images ahs been found successfully
                print('adding images for obs: ' + str(obs))
                list_of_sci_images.append(sci_image)
                list_of_mask_images.append(mask_image)
                list_of_var_images.append(var_image)

                # observation which are of good enough quality to be analyzed get added here
                list_of_obs_cleaned.append(obs)

    print('for spot ' + str(list_of_spots[s]) + ' len of list_of_sci_images: ' +
          str(len(list_of_sci_images)))
    print('len of accepted images ' + str(len(list_of_obs_cleaned)) +
          ' / len of asked images ' + str(len(list_of_obs)))

    # If there is no valid images imported, exit
    if list_of_sci_images == []:
        print('No valid images - exiting')
        sys.exit(0)

    # if you were able only to import only a fraction of images
    # if this fraction is too low - exit
    if (len(list_of_obs_cleaned) / len(list_of_obs)) < 0.6:
        print('Fraction of images imported is too low - exiting')
        sys.exit(0)

    list_of_sci_images_multi_spot.append(list_of_sci_images)
    list_of_mask_images_multi_spot.append(list_of_mask_images)
    list_of_var_images_multi_spot.append(list_of_var_images)
    list_of_obs_cleaned_multi_spot.append(list_of_obs_cleaned)

array_of_sci_images_multi_spot = np.array(list_of_sci_images_multi_spot)
array_of_mask_images_multi_spot = np.array(list_of_mask_images_multi_spot)
array_of_var_images_multi_spot = np.array(list_of_var_images_multi_spot)
array_of_obs_cleaned_multi_spot = np.array(list_of_obs_cleaned_multi_spot)

################################################
# (2.) create output names
################################################

# string to describe which spots are being analyzed
# this was initial idea on how to name outputs in multispot scenario
# delete if not used
# single_number_str = ''
# for i in list_of_spots:
#     single_number_str += str(i) + '_'
# single_number_str = single_number_str[:-1]

# name of the outputs
# create list with names for each spot
list_of_NAME_OF_CHAIN = []
list_of_NAME_OF_LIKELIHOOD_CHAIN = []

for s in range(len(list_of_spots)):
    # to be consistent with preivous versions of the code, use the last obs avalible in the name
    obs_for_naming = list_of_obs_cleaned[-1]
    # give it invidual name here, just to make srue that by accident we do not
    # overload the variable and cause errors downstream
    single_number_str = list_of_spots[s]
    NAME_OF_CHAIN = 'chain' + str(date_of_output) + '_Single_P_' + \
        str(obs_for_naming) + str(single_number_str) + str(eps) + str(arc)
    NAME_OF_LIKELIHOOD_CHAIN = 'likechain' + str(date_of_output) + '_Single_P_' + str(obs_for_naming) +\
        str(single_number_str) + str(eps) + str(arc)

    list_of_NAME_OF_CHAIN.append(NAME_OF_CHAIN)
    list_of_NAME_OF_LIKELIHOOD_CHAIN.append(NAME_OF_LIKELIHOOD_CHAIN)

################################################
# (3.) import dataframes
################################################

# where are the dataframes located
# these files give auxiliary information which enables us to connect spot number with other properties
# such as the position on the detector, wavelength, etc...
# Ar (Argon)
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


################################################
# (3.) import obs_pos & connect
################################################

# What are the observations that can be analyzed
# This information is used to associate observation with their input labels (see definition of `label` below)
# This is so that the initial parameters guess is correct

# dataset 0, December 2017 data - possibly deprecated
"""
if arc == 'HgAr':
    obs_possibilites = np.array([8552, 8555, 8558, 8561, 8564, 8567, 8570, 8573,
                                 8603, 8600, 8606, 8609, 8612, 8615, 8618, 8621, 8624, 8627])
elif arc == 'Ne':
    print('Neon?????')
    obs_possibilites = np.array([8552, 8555, 8558, 8561, 8564, 8567, 8570, 8573,
                                 8603, 8600, 8606, 8609, 8612, 8615, 8618, 8621, 8624, 8627])+90
"""
# F/3.2 data
if dataset == 1:
    if arc == 'HgAr':
        obs_possibilites = np.array([11796,
                                     11790,
                                     11784,
                                     11778,
                                     11772,
                                     11766,
                                     11760,
                                     11754,
                                     11748,
                                     11694,
                                     11700,
                                     11706,
                                     11712,
                                     11718,
                                     11724,
                                     11730,
                                     11736])
    elif arc == 'Ne':
        obs_possibilites = np.array([12403,
                                     12397,
                                     12391,
                                     12385,
                                     12379,
                                     12373,
                                     12367,
                                     12361,
                                     12355,
                                     12349,
                                     12343,
                                     12337,
                                     12331,
                                     12325,
                                     12319,
                                     12313,
                                     12307])

# F/2.8 data
if dataset == 2:
    if arc == 'HgAr':
        obs_possibilites = np.array([17023,
                                     17023 + 6,
                                     17023 + 12,
                                     17023 + 18,
                                     17023 + 24,
                                     17023 + 30,
                                     17023 + 36,
                                     17023 + 42,
                                     17023 + 48,
                                     17023 + 54,
                                     17023 + 60,
                                     17023 + 66,
                                     17023 + 72,
                                     17023 + 78,
                                     17023 + 84,
                                     17023 + 90,
                                     17023 + 96,
                                     17023 + 48])
    if arc == 'Ne':
        obs_possibilites = np.array([16238 +
                                     6, 16238 +
                                     12, 16238 +
                                     18, 16238 +
                                     24, 16238 +
                                     30, 16238 +
                                     36, 16238 +
                                     42, 16238 +
                                     48, 16238 +
                                     54, 16238 +
                                     60, 16238 +
                                     66, 16238 +
                                     72, 16238 +
                                     78, 16238 +
                                     84, 16238 +
                                     90, 16238 +
                                     96, 16238 +
                                     102, 16238 +
                                     54])
    if arc == 'Kr':
        obs_possibilites = np.array([17310 +
                                     6, 17310 +
                                     12, 17310 +
                                     18, 17310 +
                                     24, 17310 +
                                     30, 17310 +
                                     36, 17310 +
                                     42, 17310 +
                                     48, 17310 +
                                     54, 17310 +
                                     60, 17310 +
                                     66, 17310 +
                                     72, 17310 +
                                     78, 17310 +
                                     84, 17310 +
                                     90, 17310 +
                                     96, 17310 +
                                     102, 17310 +
                                     54])

# F/2.5 data
if dataset == 3:
    if arc == 'HgAr':
        obs_possibilites = np.array([19238,
                                     19238 + 6,
                                     19238 + 12,
                                     19238 + 18,
                                     19238 + 24,
                                     19238 + 30,
                                     19238 + 36,
                                     19238 + 42,
                                     19238 + 48,
                                     19238 + 54,
                                     19238 + 60,
                                     19238 + 66,
                                     19238 + 72,
                                     19238 + 78,
                                     19238 + 84,
                                     19238 + 90,
                                     19238 + 96,
                                     19238 + 48])
    elif arc == 'Ne':
        obs_possibilites = np.array([19472 +
                                     6, 19472 +
                                     12, 19472 +
                                     18, 19472 +
                                     24, 19472 +
                                     30, 19472 +
                                     36, 19472 +
                                     42, 19472 +
                                     48, 19472 +
                                     54, 19472 +
                                     60, 19472 +
                                     66, 19472 +
                                     72, 19472 +
                                     78, 19472 +
                                     84, 19472 +
                                     90, 19472 +
                                     96, 19472 +
                                     102, 19472 +
                                     54])

# F/2.8 July data
if dataset == 4:
    if arc == 'HgAr':
        obs_possibilites = np.array([21346 +
                                     6, 21346 +
                                     12, 21346 +
                                     18, 21346 +
                                     24, 21346 +
                                     30, 21346 +
                                     36, 21346 +
                                     42, 21346 +
                                     48, 21346 +
                                     54, 21346 +
                                     60, 21346 +
                                     66, 21346 +
                                     72, 21346 +
                                     78, 21346 +
                                     84, 21346 +
                                     90, 21346 +
                                     96, 21346 +
                                     102, 21346 +
                                     48])
    if arc == 'Ne':
        obs_possibilites = np.array([21550 +
                                     6, 21550 +
                                     12, 21550 +
                                     18, 21550 +
                                     24, 21550 +
                                     30, 21550 +
                                     36, 21550 +
                                     42, 21550 +
                                     48, 21550 +
                                     54, 21550 +
                                     60, 21550 +
                                     66, 21550 +
                                     72, 21550 +
                                     78, 21550 +
                                     84, 21550 +
                                     90, 21550 +
                                     96, 21550 +
                                     102, 21550 +
                                     54])
    if arc == 'Kr':
        obs_possibilites = np.array([21754 +
                                     6, 21754 +
                                     12, 21754 +
                                     18, 21754 +
                                     24, 21754 +
                                     30, 21754 +
                                     36, 21754 +
                                     42, 21754 +
                                     48, 21754 +
                                     54, 21754 +
                                     60, 21754 +
                                     66, 21754 +
                                     72, 21754 +
                                     78, 21754 +
                                     84, 21754 +
                                     90, 21754 +
                                     96, 21754 +
                                     102, 21754 +
                                     54])

# F/2.8 data, Subaru
if dataset == 6:
    if arc == 'Ar':
        obs_possibilites = np.array([34341,
                                     34341 + 6,
                                     34341 + 12,
                                     34341 + 18,
                                     34341 + 24,
                                     34341 + 30,
                                     34341 + 36,
                                     34341 + 42,
                                     34341 + 48,
                                     34341 + 54,
                                     34341 + 60,
                                     34341 + 66,
                                     34341 + 72,
                                     34341 + 78,
                                     34341 + 84,
                                     34341 + 90,
                                     34341 + 96,
                                     21346 + 48])
    if arc == 'Ne':
        obs_possibilites = np.array([34217,
                                     34217 + 6,
                                     34217 + 12,
                                     34217 + 18,
                                     34217 + 24,
                                     34217 + 30,
                                     34217 + 36,
                                     34217 + 42,
                                     34217 + 48,
                                     34217 + 54,
                                     34217 + 60,
                                     34217 + 66,
                                     34217 + 72,
                                     34217 + 78,
                                     34217 + 84,
                                     34217 + 90,
                                     34217 + 96,
                                     34217 + 48])
    if arc == 'Kr':
        obs_possibilites = np.array([34561,
                                     34561 + 6,
                                     34561 + 12,
                                     34561 + 18,
                                     34561 + 24,
                                     34561 + 30,
                                     34561 + 36,
                                     34561 + 42,
                                     34561 + 48,
                                     34561 + 54,
                                     34561 + 60,
                                     34561 + 66,
                                     34561 + 72,
                                     34561 + 78,
                                     34561 + 84,
                                     34561 + 90,
                                     34561 + 96,
                                     34561 + 48])

# SM2 test data
if dataset == 7:
    if arc == 'Ar':
        obs_possibilites = np.array([27779, -
                                     999, 27683, -
                                     999, -
                                     999, -
                                     999, -
                                     999, -
                                     999, 27767, -
                                     999, -
                                     999, -
                                     999, -
                                     999, -
                                     999, 27698, -
                                     999, 27773, -
                                     999])
    if arc == 'Ne':
        obs_possibilites = np.array([27713, -
                                     999, 27683, -
                                     999, -
                                     999, -
                                     999, -
                                     999, -
                                     999, 27677, -
                                     999, -
                                     999, -
                                     999, -
                                     999, -
                                     999, 27698, -
                                     999, 27719, -
                                     999])
    # Krypton data not taken
    # if arc == 'Kr':
    #     obs_possibilites = np.array([34561, 34561+6, 34561+12, 34561+18, 34561+24, 34561+30,
    #                                 34561+36, 34561+42, 34561+48,
    #                                 34561+54, 34561+60, 34561+66, 34561+72,
    # 34561+78, 34561+84, 34561+90, 34561+96, 34561+48])

# 21 fibers data from May/Jun 2021, taken at Subaru
if dataset == 8:
    if arc == 'Ar':
        obs_possibilites = np.array([51485,
                                     51485 + 12,
                                     51485 + 2 * 12,
                                     51485 + 3 * 12,
                                     51485 + 4 * 12,
                                     51485 + 5 * 12,
                                     51485 + 6 * 12,
                                     51485 + 7 * 12,
                                     51485 + 8 * 12,
                                     51485 + 9 * 12,
                                     51485 + 10 * 12,
                                     51485 + 11 * 12,
                                     51485 + 12 * 12,
                                     51485 + 13 * 12,
                                     51485 + 14 * 12,
                                     51485 + 15 * 12,
                                     51485 + 16 * 12,
                                     51485 + 8 * 12])
    if arc == 'Ne':
        obs_possibilites = np.array([59655,
                                     59655 + 12,
                                     59655 + 2 * 12,
                                     59655 + 3 * 12,
                                     59655 + 4 * 12,
                                     59655 + 5 * 12,
                                     59655 + 6 * 12,
                                     59655 + 7 * 12,
                                     59655 + 8 * 12,
                                     59655 + 9 * 12,
                                     59655 + 10 * 12,
                                     59655 + 11 * 12,
                                     59655 + 12 * 12,
                                     59655 + 13 * 12,
                                     59655 + 14 * 12,
                                     59655 + 15 * 12,
                                     59655 + 16 * 12,
                                     59655 + 8 * 12])
    if arc == 'Kr':
        obs_possibilites = np.array([52085,
                                     52085 + 12,
                                     52085 + 2 * 12,
                                     52085 + 3 * 12,
                                     52085 + 4 * 12,
                                     52085 + 5 * 12,
                                     52085 + 6 * 12,
                                     52085 + 7 * 12,
                                     52085 + 8 * 12,
                                     52085 + 9 * 12,
                                     52085 + 10 * 12,
                                     52085 + 11 * 12,
                                     52085 + 12 * 12,
                                     52085 + 13 * 12,
                                     52085 + 14 * 12,
                                     52085 + 15 * 12,
                                     52085 + 16 * 12,
                                     52085 + 8 * 12])


##############################################

# associates each observation with the label describing movement of the hexapod and rough estimate of z4
z4Input_possibilites = np.array([28, 24.5, 21, 17.5, 14, 10.5, 7, 3.5, 0,
                                 -3.5, -7, -10.5, -14, -17.5, -21, -24.5, -28, 0])
label = ['m4', 'm35', 'm3', 'm25', 'm2', 'm15', 'm1', 'm05', '0',
         'p05', 'p1', 'p15', 'p2', 'p25', 'p3', 'p35', 'p4', '0p']


# TODO: do we have to change this for multi spot?

list_of_z4Input = []
for obs in list_of_obs_cleaned:
    z4Input = z4Input_possibilites[obs_possibilites == obs][0]
    list_of_z4Input.append(z4Input)

# list of labels that we are passing to the algorithm, that are `clean`
list_of_labelInput = []
for obs in list_of_obs_cleaned:
    labelInput = label[list(obs_possibilites).index(obs)]
    list_of_labelInput.append(labelInput)

# TODO: evaluate if all of these different inputs are needed
# list_of_defocuses_input_long input in Tokovinin algorith
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


# names for paramters - names if we go up to z22
columns22 = [
    'z4',
    'z5',
    'z6',
    'z7',
    'z8',
    'z9',
    'z10',
    'z11',
    'z12',
    'z13',
    'z14',
    'z15',
    'z16',
    'z17',
    'z18',
    'z19',
    'z20',
    'z21',
    'z22',
    'hscFrac',
    'strutFrac',
    'dxFocal',
    'dyFocal',
    'slitFrac',
    'slitFrac_dy',
    'wide_0',
    'wide_23',
    'wide_43',
    'misalign',
    'x_fiber',
    'y_fiber',
    'effective_radius_illumination',
    'frd_sigma',
    'frd_lorentz_factor',
    'det_vert',
    'slitHolder_frac_dx',
    'grating_lines',
    'scattering_slope',
    'scattering_amplitude',
    'pixel_effect',
    'fiber_r',
    'flux']

# depening on the arc, select the appropriate dataframe
# change here to account for 21 fiber data
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


list_of_wavelengths = []
for s in range(len(list_of_spots)):
    single_number = list_of_spots[s]
    wavelength = float(finalArc.iloc[int(single_number)]['wavelength'])
    print("wavelength used for spot "+str(s)+" [nm] is: " + str(wavelength))
    list_of_wavelengths.append(wavelength)
array_of_wavelengths = np.array(list_of_wavelengths)

# pool = MPIPool()
# if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

print('print check just before the pool of workers is created')
# pool = Pool()
pool = Pool(processes=32)
print('print check just after the pool of workers is created')


################################################
# (5.) Create Tokovinin algorithm instances
################################################


# model_multi is only needed to create reasonable parametrizations and
# could possibly be avoided in future versions?
model_multi = LN_PFS_multi_same_spot(
    list_of_sci_images=list_of_sci_images,
    list_of_var_images=list_of_var_images,
    list_of_mask_images=list_of_mask_images,
    wavelength=wavelength,
    dithering=1,
    save=0,
    verbosity=0,
    npix=1536,
    list_of_defocuses=list_of_labelInput,
    zmax=zmax,
    double_sources=double_sources,
    double_sources_positions_ratios=double_sources_positions_ratios,
    test_run=False)

# TODO: at the moment, just explicity initialize two Tokovinin multi instances,
# but this needs to be solved explicitly

# for the spot with index 0
list_of_sci_images_0 = array_of_sci_images_multi_spot[0]
list_of_var_images_0 = array_of_var_images_multi_spot[0]
list_of_mask_images_0 = array_of_mask_images_multi_spot[0]
wavelength_0 = array_of_wavelengths[0]

Tokovinin_multi_instance_with_pool_0 = Tokovinin_multi(
    list_of_sci_images_0,
    list_of_var_images_0,
    list_of_mask_images=list_of_mask_images_0,
    wavelength=wavelength_0,
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
    pool=pool)
Tokovinin_multi_instance_without_pool_0 = Tokovinin_multi(
    list_of_sci_images_0,
    list_of_var_images_0,
    list_of_mask_images=list_of_mask_images_0,
    wavelength=wavelength_0,
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
    pool=None)

# for the spot with index 1
list_of_sci_images_1 = array_of_sci_images_multi_spot[1]
list_of_var_images_1 = array_of_var_images_multi_spot[1]
list_of_mask_images_1 = array_of_mask_images_multi_spot[1]
wavelength_1 = array_of_wavelengths[1]

Tokovinin_multi_instance_with_pool_1 = Tokovinin_multi(
    list_of_sci_images_1,
    list_of_var_images_1,
    list_of_mask_images=list_of_mask_images_1,
    wavelength=wavelength_1,
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
    pool=pool)
Tokovinin_multi_instance_without_pool_1 = Tokovinin_multi(
    list_of_sci_images_1,
    list_of_var_images_1,
    list_of_mask_images=list_of_mask_images_1,
    wavelength=wavelength_1,
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
    pool=None)

################################################
# (6.) Create parametrization proposals
################################################

list_of_array_of_polyfit_1_parameterizations_proposal_shape_2d = []

for s in range(len(list_of_spots)):

    single_number = list_of_spots[s]

    # you are passing multiple images, so allparameters and defocuses need to be passed into a list
    list_of_allparameters = []
    list_of_defocuses = []
    # search for the previous avaliable results
    # add the ones that you found in array_of_allparameters and for which
    # labels are avaliable in list_of_defocuses
    for label in ['m4', 'm35', 'm3', 'm05', '0', 'p05', 'p3', 'p35', 'p4']:

        # check if your single_number is avaliable

        print('adding label ' +
              str(label) +
              ' with single_number ' +
              str(int(single_number)) +
              ' for creation of array_of_allparameters')
        try:
            if int(single_number) < 999:
                print(results_of_fit_input[label].index.astype(int))
                # if your single_number is avaliable go ahead
                if int(single_number) in results_of_fit_input[label].index.astype(
                        int):
                    print('Solution for this spot is avaliable')
                    if isinstance(results_of_fit_input[label].index[0], str) or str(
                            type(results_of_fit_input[label].index[0])) == "<class 'numpy.str_'>":
                        list_of_allparameters.append(
                            results_of_fit_input[label].loc[str(single_number)].values)
                        print('results_of_fit_input[' + str(label) + '].loc[' +
                              str(int(single_number)) + '].values' + str(
                            results_of_fit_input[label].loc[str(single_number)].values))
                    else:
                        # print('results_of_fit_input[label]'+str(results_of_fit_input[label]))
                        list_of_allparameters.append(
                            results_of_fit_input[label].loc[int(single_number)].values)
                        print('results_of_fit_input[' + str(label) + '].loc[' +
                              str(int(single_number)) + '].values' + str(
                            results_of_fit_input[label].loc[int(single_number)].values))
                    list_of_defocuses.append(label)

                else:
                    # if the previous solution is not avaliable,
                    # find the closest avaliable, right?
                    print(
                        'Solution for this spot is not avaliable, reconstructing from nearby spot')

                    # positions of all avaliable spots
                    x_positions = finalArc.loc[results_of_fit_input[label].index.astype(
                        int)]['xc_effective']
                    y_positions = finalArc.loc[results_of_fit_input[label].index.astype(
                        int)]['yc']
                    print('checkpoint 1')
                    print(label)
                    print(results_of_fit_input[labelInput].index)
                    # position of the input spot
                    position_x_single_number = finalArc['xc_effective'].loc[int(
                        single_number)]
                    position_y_single_number = finalArc['yc'].loc[int(
                        single_number)]
                    print('checkpoint 2')
                    print(position_x_single_number)
                    distance_of_avaliable_spots = np.abs(
                        (x_positions - position_x_single_number)**2 +
                        (y_positions - position_y_single_number)**2)
                    single_number_input =\
                        distance_of_avaliable_spots[distance_of_avaliable_spots ==
                                                    np.min(distance_of_avaliable_spots)].index[0]
                    print(
                        'Nearest spot avaliable is: ' +
                        str(single_number_input))
                    if isinstance(results_of_fit_input[label].index[0], str) or str(
                            type(results_of_fit_input[label].index[0])) == "<class 'numpy.str_'>":
                        list_of_allparameters.append(
                            results_of_fit_input[label].loc[str(single_number_input)].values)
                    else:
                        list_of_allparameters.append(
                            results_of_fit_input[label].loc[int(single_number_input)].values)
                    list_of_defocuses.append(label)
                    print('results_of_fit_input[' + str(label) + '].loc[' +
                          str(int(single_number_input)) + '].values' + str(
                        results_of_fit_input[label].loc[int(single_number_input)].values))

                    pass

        except BaseException:
            print('not able to add label ' + str(label))
            pass

    array_of_allparameters = np.array(list_of_allparameters)

    # based on the information from the previous step (results at list_of_defocuses),
    # generate singular array_of_allparameters at list_of_labelInput positions
    # has shape 2xN, N = number of parameters
    print('Variable twentytwo_or_extra: ' + str(twentytwo_or_extra))
    if analysis_type == 'defocus':
        print('Variable array_of_allparameters.shape: ' +
              str(array_of_allparameters.shape))

        array_of_polyfit_1_parameterizations_proposal =\
            model_multi.create_resonable_allparameters_parametrizations(
                array_of_allparameters=array_of_allparameters,
                list_of_defocuses_input=list_of_defocuses,
                zmax=twentytwo_or_extra,
                remove_last_n=2)

        # lets be explicit that the shape of the array is 2d
        array_of_polyfit_1_parameterizations_proposal_shape_2d =\
            array_of_polyfit_1_parameterizations_proposal

    list_of_array_of_polyfit_1_parameterizations_proposal_shape_2d.append(
        array_of_polyfit_1_parameterizations_proposal_shape_2d)

# array contaning the arrays with parametrizations for all spots
array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d = np.array(
    list_of_array_of_polyfit_1_parameterizations_proposal_shape_2d)


########
# "lower_limits" and "higher_limits" are only needed to initalize the cosmo_hammer code,
# but not used in the actual evaluation
# so these are purely dummy values
# TODO: find a way in which this can be deleted?

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
########

################################################
# (7.) Sanity check of inputs
################################################

# initialize pool
# pool = MPIPool()
# if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

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


################################################
# (8.) Create init parameters for particles
################################################

# TODO: ellimate either columns or columns22 variable
columns = columns22

#############
# First swarm

# if working with defocus, but many images
# z4, z5, z6, z7, z8, z9, , z11
# z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22
# hscFrac, strutFrac, dxFocal, dyFocal, slitFrac, slitFrac_dy
# wide_0, wide_23, wide_43, misalign
# x_fiber, y_fiber, effective_radius_illumination, frd_sigma, frd_lorentz_factor, scattering_amplitude, \
# pixel_effect, fiber_r
# det_vert, slitHolder_frac_dx, grating_lines, scattering_slope
stronger_array_01 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                              1, 1, 1, 1,
                              1.2, 1.2, 1.2, 1.2, 1.2,
                              1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1])

# we are passing the parametrizations, which need to be translated to parameters for each image
# from Zernike_Module we imported the function `check_global_parameters'
list_of_global_parameters = []
for s in range(len(list_of_spots)):
    array_of_polyfit_1_parameterizations_proposal_shape_2d =\
        array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s]

    global_parameters = array_of_polyfit_1_parameterizations_proposal_shape_2d[:, 1][19:19 + 23]

    list_of_global_parameters.append(global_parameters)

# create global parameters which are the same for all spots
# modify global parameters which should be the same for all of the spots in the fiber
# these parama
array_of_global_parameters = np.array(list_of_global_parameters)
global_parameters = np.mean(array_of_global_parameters, axis=0)
checked_global_parameters = check_global_parameters(global_parameters)


# return the unified global parameters to each spot, for the parameters which should be unified
# index of global parameters that we are setting to be same
# 'hscFrac', 'strutFrac', 'dxFocal', 'slitFrac', 'misalign',
# 'x_fiber', 'y_fiber', 'effective_radius_illumination', 'frd_sigma',
# 'frd_lorentz_factor', 'slitHolder_frac_dx', 'fiber_r'
unified_index = [0, 1, 2, 4, 9, 10, 11, 12, 13, 14, 16, 21]
for s in range(len(list_of_spots)):
    for i in unified_index:
        array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s][:, 1][19:19 + 23][i] =\
            checked_global_parameters[i]

# list contaning parInit1 for each spot
list_of_parInit1 = []
for s in range(len(list_of_spots)):

    array_of_polyfit_1_parameterizations_proposal_shape_2d =\
        array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s]
    print(
        'array_of_polyfit_1_parameterizations_proposal_shape_2d: ' +
        str(array_of_polyfit_1_parameterizations_proposal_shape_2d))
    parInit1 = create_parInit(
        allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,
        multi=multi_var,
        pupil_parameters=None,
        allparameters_proposal_err=None,
        stronger=stronger_array_01,
        use_optPSF=None,
        deduced_scattering_slope=None,
        zmax=zmax_input)

    # the number of walkers is given by options array, specified with
    # the parameter eps at the start
    while len(parInit1) < options[0]:
        parInit1_2 = create_parInit(
            allparameters_proposal=array_of_polyfit_1_parameterizations_proposal_shape_2d,
            multi=multi_var,
            pupil_parameters=None,
            allparameters_proposal_err=None,
            stronger=stronger_array_01,
            use_optPSF=None,
            deduced_scattering_slope=None,
            zmax=zmax_input)
        parInit1 = np.vstack((parInit1, parInit1_2))

    list_of_parInit1.append(parInit1)

# standard deviation of parameters (for control only?)
# One array is enough, because we can use it for both spots (right?)
parInit1_std = []
for i in range(parInit1.shape[1]):
    parInit1_std.append(np.std(parInit1[:, i]))
parInit1_std = np.array(parInit1_std)
print('parInit1_std: ' + str(parInit1_std))

# number of particles and number of parameters
particleCount = options[0]
paramCount = len(parInit1[0])

# initialize the particles
particle_likelihood = np.array([-9999999])
particle_position = np.zeros(paramCount)
particle_velocity = np.zeros(paramCount)
best_particle_likelihood = particle_likelihood[0]

#
list_of_array_of_particle_position_proposal = []
list_of_array_of_particle_velocity_proposal = []
list_of_best_particle_likelihood = []
for s in range(len(list_of_spots)):
    parInit1 = list_of_parInit1[s]
    array_of_particle_position_proposal = parInit1[0:particleCount]
    array_of_particle_velocity_proposal = np.zeros(
        (particleCount, paramCount))

    list_of_array_of_particle_position_proposal.append(
        array_of_particle_position_proposal)
    list_of_array_of_particle_velocity_proposal.append(
        array_of_particle_velocity_proposal)
    list_of_best_particle_likelihood.append(best_particle_likelihood)

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
array_of_polyfit_1_parameterizations_proposal_shape_2d_0 =\
    array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[0]
best_result_0 = Tokovinin_multi_instance_with_pool_0(
    array_of_polyfit_1_parameterizations_proposal_shape_2d_0,
    return_Images=True,
    num_iter='initial',
    previous_best_result=None,
    use_only_chi=True)

array_of_polyfit_1_parameterizations_proposal_shape_2d_1 =\
    array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[1]
best_result_1 = Tokovinin_multi_instance_with_pool_1(
    array_of_polyfit_1_parameterizations_proposal_shape_2d_1,
    return_Images=True,
    num_iter='initial',
    previous_best_result=None,
    use_only_chi=True)

list_of_best_results = [best_result_0, best_result_1]

time_end_initial = time.time()
print('Time for the initial evaluation is ' +
      str(time_end_initial - time_start_initial))


# the best likelihood after evaluation, at index 1 in the output
# (index 0 shows the likelihood before the evaluation)
s = 0
print('Initial evaluation output for spot ' +
      str(s) + ' is: ' + str(best_result_0[1]))
s = 1
print('Initial evaluation output for spot ' +
      str(s) + ' is: ' + str(best_result_1[1]))

print('Finished work on the initial best result')

array_of_particle_pbest_position = np.zeros((particleCount, paramCount))
array_of_particle_pbest_likelihood = np.ones((particleCount, 1)) * -9999999

single_number_original = list_of_spots[0]
np.save(RESULT_FOLDER + 'initial_best_result_' + str(date_of_output) + '_Single_P_' +
        str(obs) + str(single_number_original) + str(eps) + str(arc), best_result_0)
single_number_original = list_of_spots[1]
np.save(RESULT_FOLDER + 'initial_best_result_' + str(date_of_output) + '_Single_P_' +
        str(obs) + str(single_number_original) + str(eps) + str(arc), best_result_1)

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
    # reports likelihod, but minimizes absolute different of images/std - model/std

    # list_of_array_of_particle_position_proposal carries particle positions for this step
    # this list needs to be updated at the end of the step

    # list_of_best_results carries the best result from the previous step
    # this list needs to be update at the end of the step

    best_result_0 = list_of_best_results[0]
    array_of_particle_position_proposal_0 = list_of_array_of_particle_position_proposal[0]
    out1_0 = pool.map(
        partial(
            Tokovinin_multi_instance_without_pool_0,
            return_Images=True,
            previous_best_result=best_result_0,
            use_only_chi=True,
            multi_background_factor=array_of_multi_background_factors[step]),
        array_of_particle_position_proposal_0)
    out2_0 = np.array(list(out1_0))

    print('Checkpoint - finished creating out2_0, moving onto out2_1')
    best_result_1 = list_of_best_results[1]
    array_of_particle_position_proposal_1 = list_of_array_of_particle_position_proposal[1]
    out1_1 = pool.map(
        partial(
            Tokovinin_multi_instance_without_pool_1,
            return_Images=True,
            previous_best_result=best_result_1,
            use_only_chi=True,
            multi_background_factor=array_of_multi_background_factors[step]),
        array_of_particle_position_proposal_1)
    out2_1 = np.array(list(out1_1))

    time_end_evo_step = time.time()

    print('Time for the evaluation of initial Tokovinin_multi_instance_without_pool in step ' +
          str(step) + ' is ' + str(time_end_evo_step - time_start_evo_step))

    # TODO: for s in ?
    list_of_out2 = [out2_0, out2_1]

    np.save(RESULT_FOLDER + 'list_of_out2_' + str(step), list_of_out2)
    # save ?

    ################################################
    # (11.) Create swarm
    ################################################

    ##################
    # create a swarm here
    # input are, for each spot:
    # 1. out2
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
    # arrange the particlese in all spots
    # find the particle, which on average has highest ordering
    # for all of the spots
    np.save(RESULT_FOLDER + 'list_of_swarms' + '_' + str(step), list_of_swarms)

    list_of_quality_of_particle = []
    for s in range(len(list_of_spots)):
        # select the swarm for a single spot
        swarm = list_of_swarms[s]
        # select only the quality measures
        swarm_likelihood = swarm[:, 0]

        # prepare an array which we will fill the information about ordering of the
        # quality of the result
        quality_of_particle = np.empty(len(swarm_likelihood))
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

    np.save(RESULT_FOLDER + 'list_of_best_particle_this_step_' + str(step), list_of_best_particle_this_step)

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
            print(
                'Starting work on the new proposed best result in step ' +
                str(step) + ' for spot ' + str(s))
            time_start_best = time.time()

            if s == 0:
                best_result = Tokovinin_multi_instance_with_pool_0(
                    best_particle_proposal_position,
                    return_Images=True,
                    num_iter=step,
                    previous_best_result=None,
                    use_only_chi=True,
                    multi_background_factor=array_of_multi_background_factors[step])

            if s == 1:
                best_result = Tokovinin_multi_instance_with_pool_1(
                    best_particle_proposal_position,
                    return_Images=True,
                    num_iter=step,
                    previous_best_result=None,
                    use_only_chi=True,
                    multi_background_factor=array_of_multi_background_factors[step])
            time_end_best = time.time()

            # updating the list with best results, to be carried over for the next step
            list_of_best_results.append(best_result)
            print('Time for the best evaluation for spot ' + str(s) + ' in step ' +
                  str(step) + ' is ' + str(time_end_best - time_start_best))

            best_particle_likelihood_this_step = best_result[1]
            best_particle_position_this_step = best_result[8]

            print('Best result proposed output for spot ' + str(s) + ' is: ' +
                  str(best_particle_proposal_position[:5]))
            print('Best result output after final Tokovinin for spot ' + str(s) + ' is: ' +
                  str(best_particle_position_this_step[:5]))
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
        np.save(RESULT_FOLDER+'best_particle_spot_' + str(s) + '_step_'+str(step), best_particle)
        np.save(RESULT_FOLDER+'list_of_best_results_' + str(s) + '_step_'+str(step), list_of_best_results)
        # np.save(RESULT_FOLDER+'list_of_swarms'+str(step),list_of_swarms)

    ################################################
    # (15.) Suggest position for the next step
    ################################################

    list_of_gbests = []
    list_of_swarm = []
    list_of_particle_position_proposal = []

    list_of_array_of_particle_position_proposal = []

    array_of_velocity_modifiers = np.random.random_sample(size=(particleCount, paramCount))*(2)

    for s in range(len(list_of_spots)):
        swarm = list_of_swarms[s]
        best_particle_position = list_of_best_particle_position[s][0]
        np.save(RESULT_FOLDER + 'best_particle_position_spot_' + str(s) + '_step_' + str(step),
                best_particle_position)
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
            soc_vel = c2 * array_of_velocity_modifiers[s] * (best_particle_position - particle_position)
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
        # needed to properly calculate velocities
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

np.save(RESULT_FOLDER + 'array_of_swarms', array_of_swarms)
np.save(RESULT_FOLDER + 'array_of_gbests', array_of_gbests)

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
    chains = res.reshape(len(swarms), len(swarms[0]), parInit1.shape[1])

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
    v_chains = res.reshape(len(swarms), len(swarms[0]), parInit1.shape[1])

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

    parInit1 = list_of_parInit1[s]
    np.save(RESULT_FOLDER + NAME_OF_CHAIN + 'parInit1', parInit1)
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