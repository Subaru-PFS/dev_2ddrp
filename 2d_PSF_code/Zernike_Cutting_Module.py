#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 09:42:10 2020

Module and stand-alone code for the creation of poststamp images

Versions:
Jun 08, 2020: 0.28e initial version, version number from Zernike_Module
Nov 27, 2020: 0.29 implmented cutting of prestacked images
Dec 01, 2020: 0.29a if final cut outside range, still create the image
Dec 16, 2020: 0.29b subtract the general background level, deduced from [3176:4176,3333:3433] part of image
Dec 23, 2020: 0.29c find_centroid_of_flux can tolerate nan values now
May 25, 2021: 0.30 modified to be able to do May 2021 images; basic support for blue
Jul 13, 2021: 0.31 modifed Argon to have correct secondary info
May 10, 2022: 0.32 remove last spot in Neon in blue
May 19, 2022: 0.32 -> 0.32a small fix in HgAr blue
May 20, 2022: 0.32a -> 0.32b added HgAr fully
May 23, 2022: 0.32b -> 0.33 subtraction is self-contained now

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""

########################################

import os
import numpy as np
from astropy import stats
from astropy.io import fits
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm


import lsst
from lsst.daf.persistence import Butler
import lsst.daf.persistence as dafPersist
from lsst.afw.math import stringToInterpStyle, makeInterpolate

# automatic creation of finalArc from detectorMap

########################################

__all__ = ['Zernike_cutting', 'create_Zernike_info_df',
           'find_centroid_of_flux', 'find_nearest', 'running_mean']

__version__ = "0.33a"

############################################################
# name your directory where you want to have files!
# for example on my local computer
# PSF_DIRECTORY_CUTS='/Users/nevencaplar/Documents/PFS/ReducedData/Data_Nov_20_2020/Stamps_cleaned/'
# PSF_DIRECTORY_CUTS = '/Volumes/Saturn_USA/PFS/ReducedData/Data_May_21/Stamps_cleaned/'
# On tiger
# PSF_DIRECTORY_CUTS = '/tigress/ncaplar/ReducedData/Data_May_2022/'
# DATAFRAMES FOLDER?
# AUXILIARY DATA FOLDER?

########################################

class Zernike_cutting(object):
    """Class that helps with cutting the poststamps data
    """

    def __init__(self, DATA_FOLDER, PSF_DIRECTORY_CUTS, run_0, number_of_images, SUB_LAM, Zernike_info_df,
                 subtract_continuum=False, verbosity=0, save=True,
                 use_previous_full_stacked_images=True, use_median_rejected_stacked_images=False,
                 exposure_arc=None, exposure_defocus_explicit=None, detector=None):
        """Init class

        Parameters
        ----------
        DATA_FOLDER: `str`
            where is the calExp data
        PSF_DIRECTORY_CUTS: `str`
            where will the poststamps be located
        run_0: `int`
            first visit number of the set of images
        number_of_images: `int`
            number of images at same defocus in the set of images
        SUB_LAM: `str`
            SUB or LAM; Subaru or LAM
        Zernike_info_df: `pandas.df`
            info about arc positions
        subtract_continuum: `bool`
            subtracting continuum for HgAr data
        verbosity: `int`
            print verbosity messages
        save: `bool`
            ?
        use_previous_full_stacked_images: `bool`
            ?
        use_median_rejected_stacked_images: `bool`
            ?
        exposure_arc: `str`
            which arc are we considering
        exposure_defocus_explicit: `float`
            ?
        detector: `str`
            red or blue    
        """ 
        self.DATA_FOLDER = DATA_FOLDER
        self.PSF_DIRECTORY_CUTS = PSF_DIRECTORY_CUTS
        self.run_0 = run_0
        self.number_of_images = number_of_images
        
        self.SUB_LAM = SUB_LAM
        if SUB_LAM == 'SUB':
            SUB_LAM_letter = 'S'
        elif SUB_LAM == 'LAM':
            SUB_LAM_letter = 'L'
        self.SUB_LAM_letter = SUB_LAM_letter
        
        self.Zernike_info_df = Zernike_info_df
        self.subtract_continuum = subtract_continuum
        self.verbosity = verbosity
        if save is None:
            save = False
        self.save = save

        self.use_previous_full_stacked_images = use_previous_full_stacked_images
        self.use_median_rejected_stacked_images = use_median_rejected_stacked_images
        self.exposure_arc = exposure_arc
        self.exposure_defocus = exposure_defocus_explicit
        self.exposure_defocus_explicit = exposure_defocus_explicit
        if detector is None:
            self.detector = 'red'
        else:
            self.detector = detector

    def removal_of_continuum(self, scidata_cut, maskdata_cut, vardata_cut, y_median, good, yposmin, yposmax):
        """! removing continuum, primarly from HgAr images - returns science, mask and variance image

        @param scidata_cut             science data, large post-stamp
        @param maskdata_cut            mask data, large post-stamp
        @param vardata_cut             variance data, large post-stamp
        @param y_median                amplitude correction, derived via median?
        @param good                    which pixels are good, i.e., where to apply correction?
        @param yposmin,\
        yposmax,                       explicit positions where to cut is goign to happen
        """

        #################
        # removal of continuum - estimating
        #################

        # estimating local background
        sci_upper_cut = scidata_cut[len(scidata_cut[:, 0])-70:len(scidata_cut[:, 0])-5]
        sci_lower_cut = scidata_cut[5:70]

        overall_median_lower_left = stats.sigma_clipped_stats(sci_lower_cut[:, :25].ravel(),
                                                              sigma_lower=100, sigma_upper=2, iters=6)[1]
        overall_median_lower_right = stats.sigma_clipped_stats(sci_lower_cut[:, :-25:].ravel(),
                                                               sigma_lower=100, sigma_upper=2, iters=6)[1]
        overall_median_upper_left = stats.sigma_clipped_stats(sci_upper_cut[:, :25].ravel(),
                                                              sigma_lower=100, sigma_upper=2, iters=6)[1]
        overall_median_upper_right = stats.sigma_clipped_stats(sci_upper_cut[:, :-25:].ravel(),
                                                               sigma_lower=100, sigma_upper=2, iters=6)[1]
        list_of_medians = [overall_median_lower_left, overall_median_lower_right,
                           overall_median_upper_left, overall_median_upper_right]
        overall_median = np.array(list_of_medians)[np.abs(list_of_medians) ==
                                                   np.min(np.abs(list_of_medians))][0]
        if self.verbosity == 1:
            print('overall_median is: '+str(overall_median))

        # scidata_cut_median_subtracted=scidata_cut-overall_median
        scidata_cut_median_subtracted = scidata_cut
        scidata_cut_original_median_subtracted = np.copy(scidata_cut_median_subtracted)

        # shape of the continuum in the test region, from a single exposure
        # single_fiber_extracted_dimensions[1] is where the fiber (i.e., y_median) starts

        y_median_single_spot = y_median[yposmin:yposmax]
        good_single_spot = good[yposmin:yposmax]

        y_median_single_spot_good = y_median_single_spot[good_single_spot]
        scidata_cut_original_median_subtracted_summed =\
            np.sum(scidata_cut_original_median_subtracted, axis=1)
        scidata_cut_original_median_subtracted_summed_good =\
            scidata_cut_original_median_subtracted_summed[good_single_spot]

        stack_to_extract_normalization_factor =\
            np.median(scidata_cut_original_median_subtracted_summed_good/y_median_single_spot_good)
        if self.verbosity == 1:
            print('stack_to_extract_normalization_factor: '+str(stack_to_extract_normalization_factor))

        y_median_single_spot_renormalized = y_median_single_spot*stack_to_extract_normalization_factor

        #################
        # removal of continuum - actual removal
        #################
        scidata_cut_large_removed = np.copy(scidata_cut_median_subtracted)
        maskdata_cut_large_removed = maskdata_cut
        vardata_cut_large_removed = vardata_cut

        select_central = np.concatenate((np.zeros((1, int(len(good_single_spot)/2-40))), np.ones((1, 80)),
                                         np.zeros((1, int(len(good_single_spot)/2-40)))), axis=1)[0]

        try:
            for j in range(scidata_cut_large_removed.shape[1]):
                good_single_spot_centred = (good_single_spot*select_central).astype('bool')
                if np.sum(good_single_spot_centred) > 25:
                    y_median_single_spot_renormalized =\
                        y_median_single_spot *\
                        np.median(scidata_cut_large_removed[:, j][good_single_spot_centred]) /\
                        np.median(y_median_single_spot[good_single_spot_centred])
                else:
                    y_median_single_spot_renormalized = y_median_single_spot *\
                        np.median(scidata_cut_large_removed[:, j][good_single_spot]) /\
                        np.median(y_median_single_spot[good_single_spot])
                scidata_cut_large_removed[:, j] = scidata_cut_large_removed[:, j] -\
                    y_median_single_spot_renormalized
        except:
            print('No subtraction!!!')
            pass

        return scidata_cut_large_removed, maskdata_cut_large_removed, vardata_cut_large_removed

    def stack_images_and_gather_information(self, run):
        """_summary_

        Args:
            run (_type_): _description_

        Returns:
            _type_: _description_
        """
        # if run number is 4 digit number, prepand one zero in front of run
        # I have not explored all possible cases of this behaviour,
        # possibly breaks for 1,2,3, or 6 digit numbers?
        if run < 10000:
            run_string = '0'+str(run)
        else:
            run_string = str(run)

        # if you are using prestacked images, run simplified procedure
        if self.use_median_rejected_stacked_images is True:
            # data=fits.open(self.DATA_FOLDER+'median_rejected_stacked_image_'+run_string+'.fits')
            if self.detector == 'red':
                data = fits.open(self.DATA_FOLDER+'mean_rejected_stacked_image_'+run_string+'.fits')
            else:
                data = fits.open(self.DATA_FOLDER+'mean_rejected_stacked_image_'+run_string+'_b.fits')
        else:
            if run < 10000:
                if self.detector == 'red':
                    data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                     '/calExp-' + self.SUB_LAM_letter+'A0' + run_string+'r1.fits')
                else:
                    data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                     '/calExp-' + self.SUB_LAM_letter+'A0' + run_string+'b1.fits')
            else:
                if self.detector == 'red':
                    data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                     '/calExp-' + self.SUB_LAM_letter+'A0' + run_string+'r1.fits')
                else:
                    data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                     '/calExp-' + self.SUB_LAM_lette + 'A0' + run_string+'b1.fits')
        exposure_arc = self.exposure_arc
        # establish which arc is being used
        if exposure_arc is None:
            if data[0].header['W_AITNEO'] is True:
                exposure_arc = 'Ne'
            if data[0].header['W_AITHGA'] is True:
                exposure_arc = 'HgAr'
            if data[0].header['W_AITKRY'] is True:
                exposure_arc = 'Kr'
            if data[0].header['W_AITXEN'] is True:
                exposure_arc = 'Xe'
            if data[0].header['W_AITARG'] is True:
                exposure_arc = 'Ar'

        print('exposure_arc '+str(exposure_arc))

        # establish the defocus of the data
        try:
            exposure_defocus = np.round(data[0].header['W_ENFCAX'], 2)
        except:
            exposure_defocus = None

        if self.use_median_rejected_stacked_images is True:
            scidata = data[1].data
            maskdata = data[2].data
            vardata = data[3].data

            maskdata[maskdata > 1] = 1

            # careful - manually selecting area from where to determine ``low level''
            if self.detector == 'red':
                empty_part_of_scidata = np.nan_to_num(scidata[3176:4176, 3333:3433]) 
            else:               
                empty_part_of_scidata = np.nan_to_num(scidata[1000:1400, 3450:3550])
            background_level = np.median(empty_part_of_scidata)
            scidata = scidata - background_level
            if self.verbosity == 1:
                print('subtracted ' + str(background_level) + ' from the whole image')
            if self.detector == 'blue':
                print('mean_after_scidata ' + str(np.mean(scidata[1000:1400, 3450:3550])))
            else:
                print('mean_after_scidata ' + str(np.mean(scidata[3176:4176, 3333:3433])))               

        else:
            # if you already this and use use_previous_full_stacked_images=False you can import only
            # one image to read auxiliary data, but no time-consuming stacking
            if self.use_previous_full_stacked_images is True and\
                os.path.exists(self.PSF_DIRECTORY_CUTS + "scidata" +
                               str(run) + exposure_arc+'_Stacked.npy') is True:
                if self.verbosity is True:
                    print('Using previously created full images,\
                            because they exist and use_previous_full_stacked_images==True')
                    print('Loading from: ' + self.PSF_DIRECTORY_CUTS + "scidata" +
                          str(run) + exposure_arc + '_Stacked.npy')
                scidata = np.load(self.PSF_DIRECTORY_CUTS + "scidata" +
                                  str(run) + exposure_arc + '_Stacked.npy')
                maskdata = np.load(self.PSF_DIRECTORY_CUTS + "maskdata" +
                                   str(run) + exposure_arc + '_Stacked.npy')
                vardata = np.load(self.PSF_DIRECTORY_CUTS + "vardata" +
                                  str(run) + exposure_arc + '_Stacked.npy')
            else:
                # prepare empty arrays that will store the data
                scidata = np.zeros_like(data[1].data)
                maskdata = np.zeros_like(scidata)
                vardata = np.zeros_like(scidata)

                # prepare arrays and lists which will be filled
                list_of_sci_data = []
                list_of_var_data = []
                list_of_mask_data = []

                list_of_exposure_defocus = []
                list_of_exposure_arc = []

                # for each image, gather information and stack it
                for run_i in tqdm(range(self.number_of_images)):
                    if run+run_i < 10000:
                        run_string = '0'+str(run+run_i)
                    else:
                        run_string = str(run+run_i)

                    # import fits file
                    if self.detector == 'red':
                        data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                         '/calExp-SA0' + run_string+'r1.fits')
                    else:
                        data = fits.open(self.DATA_FOLDER + 'v0' + run_string +
                                         '/calExp-SA0' + run_string+'b1.fits')

                    # establish the defocus of the data
                    exposure_defocus = np.round(data[0].header['W_ENFCAX'], 2)

                    # establish which arc is being used
                    if data[0].header['W_AITNEO'] is True:
                        exposure_arc = 'Ne'
                    if data[0].header['W_AITHGA'] is True:
                        exposure_arc = 'HgAr'
                    if data[0].header['W_AITKRY'] is True:
                        exposure_arc = 'Kr'
                    if data[0].header['W_AITXEN'] is True:
                        exposure_arc = 'Xe'
                    if data[0].header['W_AITARG'] is True:
                        exposure_arc = 'Ar'

                    # add the information to the list
                    list_of_exposure_defocus.append(exposure_defocus)
                    list_of_exposure_arc.append(exposure_arc)

                    # background=background_estimate_sigma_clip_fit_function(exposure_defocus)
                    # if run_i==0:
                    #    print('background estimate is: '+str(background))
                    # scidata_single=data[1].data-background
                    
                    # separate scidata, maskdata and var data
                    scidata_single = data[1].data
                    maskdata_single = data[2].data
                    if self.verbosity == 1:
                        print('np.sum(maskdata_single) for image iteration ' + str(run_i) +
                              ' is: ' + str(np.sum(maskdata_single)))
                    vardata_single = data[3].data

                    # stack the images
                    scidata = scidata + scidata_single
                    maskdata = maskdata + maskdata_single
                    vardata = vardata + vardata_single

                    # add the data to individual list, if needed for debugging and analysis
                    list_of_sci_data.append(scidata_single)
                    maskdata_single[np.isin(maskdata_single, [0])] = 0
                    maskdata_single[~np.isin(maskdata_single, [0])] = 1
                    list_of_mask_data.append(maskdata_single)
                    list_of_var_data.append(vardata_single)

                array_of_sci_data = np.array(list_of_sci_data)
                #
                maskdata[np.isin(maskdata, [0])] = 0
                maskdata[~np.isin(maskdata, [0])] = 1

            # careful - manually selecting area from where to determine ``low level''
            if self.detector == 'red':
                empty_part_of_scidata = np.nan_to_num(scidata[3176:4176, 3333:3433]) 
            else:               
                empty_part_of_scidata = np.nan_to_num(scidata[1000:1400, 3450:3550])
            background_level = np.median(empty_part_of_scidata)
            scidata = scidata-background_level

            if self.save is True:
                if not os.path.exists(self.PSF_DIRECTORY_CUTS):
                    if self.verbosity == 1:
                        print("Creating PSF_DIRECTORY_CUTS directory at " + str(self.PSF_DIRECTORY_CUTS))
                        os.makedirs(self.PSF_DIRECTORY_CUTS)
                if self.verbosity == 1:
                    print('saving scidata, maskdata, vardata in ' + str(self.PSF_DIRECTORY_CUTS))
                np.save(self.PSF_DIRECTORY_CUTS + "array_of_sci_data" +
                        str(run)+exposure_arc + '_Stacked.npy', array_of_sci_data)
                np.save(self.PSF_DIRECTORY_CUTS + "scidata"+str(run) + exposure_arc + '_Stacked.npy', scidata)
                np.save(self.PSF_DIRECTORY_CUTS + "maskdata"+str(run) + exposure_arc + '_Stacked.npy', maskdata)
                np.save(self.PSF_DIRECTORY_CUTS + "vardata"+str(run) + exposure_arc + '_Stacked.npy', vardata)

        return scidata, maskdata, vardata, exposure_arc, exposure_defocus

    def cut_initial_image(self, scidata, maskdata, vardata, xinit, yinit, size_of_stamp,
                          xposmin=None, xposmax=None, yposmin=None, yposmax=None):
        """!given the initially centered image

        @param full_data        full science data image
        @param xinit            x position of the initial guess
        @param yinit            y position of the initial guess
        @param size_of_stamp    size of the stamp that we wish to create
        """
        # if you do not pass explicit coordinates for the cut, do the full fitting procedure
        if yposmin is None:
            # Iteration 0
            # size_of_stamp_big=2*size_of_stamp

            xposmin = int(int(xinit)-size_of_stamp/2)
            xposmax = int(int(xinit)+size_of_stamp/2)
            yposmin = int(int(yinit)-size_of_stamp/2)
            yposmax = int(int(yinit)+size_of_stamp/2)
            print('iteration nr. 0: '+str([yposmin, yposmax, xposmin, xposmax]))
            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]
            print('reached scidata_cut')

            # mprint('np.any(scidata_cut>300, axis = 1)'+str(np.any(scidata_cut>300, axis = 1)))
            # print('np.any(scidata_cut>300, axis = 0)'+str(np.any(scidata_cut>300, axis = 0)))
            med_pos = (int(np.round(np.median(np.where(np.any(scidata_cut > 300, axis=1))))),
                       int(np.round(np.median(np.where(np.any(scidata_cut > 300, axis=0))))))

            # Iteration 1
            new_xinit = xinit + med_pos[1]-size_of_stamp/2
            new_yinit = yinit + med_pos[0]-size_of_stamp/2
            xposmin = int(new_xinit-size_of_stamp/2)
            xposmax = int(new_xinit + size_of_stamp/2)
            yposmin = int(new_yinit-size_of_stamp/2)
            yposmax = int(new_yinit + size_of_stamp/2)
            print('iteration nr. 1: ' + str([yposmin, yposmax, xposmin, xposmax]))

            # import pdb
            # pdb.set_trace()
            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]

            # iteration 2
            centroid_of_flux_model = find_centroid_of_flux(scidata_cut)
            # print(centroid_of_flux_model)

            # centroid_of_flux_model_int = np.array(map(int,map(np.round,centroid_of_flux_model)))
            dif_centroid_of_flux_model_int_from_center = np.array(centroid_of_flux_model) -\
                np.array([int(size_of_stamp/2), int(size_of_stamp/2)])

            xposmin = int(np.round(xposmin + dif_centroid_of_flux_model_int_from_center[0]))
            xposmax = int(np.round(xposmax + dif_centroid_of_flux_model_int_from_center[0]))
            yposmin = int(np.round(yposmin + dif_centroid_of_flux_model_int_from_center[1]))
            yposmax = int(np.round(yposmax + dif_centroid_of_flux_model_int_from_center[1]))

            print('iteration nr. 2: ' + str([yposmin, yposmax, xposmin, xposmax]))
            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]

            # iteration 3
            centroid_of_flux_model = find_centroid_of_flux(scidata_cut)
            # print(centroid_of_flux_model)
            # centroid_of_flux_model_int = np.array(map(int,map(np.round,centroid_of_flux_model)))
            dif_centroid_of_flux_model_int_from_center = np.array(centroid_of_flux_model) -\
                np.array([int(size_of_stamp/2), int(size_of_stamp/2)])

            xposmin = int(np.round(xposmin + dif_centroid_of_flux_model_int_from_center[0]))
            xposmax = int(np.round(xposmax + dif_centroid_of_flux_model_int_from_center[0]))
            yposmin = int(np.round(yposmin + dif_centroid_of_flux_model_int_from_center[1]))
            yposmax = int(np.round(yposmax + dif_centroid_of_flux_model_int_from_center[1]))

            print('iteration nr. 3: ' + str([yposmin, yposmax, xposmin, xposmax]))
            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]

            center_pos_new = [xposmin + size_of_stamp/2, yposmin + size_of_stamp/2]

            # take the outut and get positions of where to cut
            xposmin = int(np.round(center_pos_new[0]-70-10))
            xposmax = int(np.round(center_pos_new[0] + 70 + 10))
            yposmin = int(np.round(center_pos_new[1]-70-45))
            yposmax = int(np.round(center_pos_new[1] + 70 + 45))

            # if you are on the edge, stop at the edge
            # stop one pixel beforhand, just in case if you do some sort of dithering
            if xposmin < 1:
                xposmin = 1
            if xposmax > 4095:
                xposmax = 4095
                
            if yposmax > 4176:
                yposmin = yposmin + (yposmax - 4176)
                yposmax = 4176

            if self.verbosity == 1:
                print('initial cut for image_index ' + str(self.image_index) +
                      ' - xposmin,xposmax,yposmin,yposmax: ' + str([xposmin, xposmax, yposmin, yposmax]))

            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]
            maskdata_cut = maskdata[yposmin:yposmax, xposmin:xposmax]
            vardata_cut = vardata[yposmin:yposmax, xposmin:xposmax]

            return scidata_cut, maskdata_cut, vardata_cut, xposmin, xposmax, yposmin, yposmax
        else:
            # if you passed explicit values, just use those
            scidata_cut = scidata[yposmin:yposmax, xposmin:xposmax]
            maskdata_cut = maskdata[yposmin:yposmax, xposmin:xposmax]
            vardata_cut = vardata[yposmin:yposmax, xposmin:xposmax]

            # do not return the input coordinates, to avoid possible confusion
            return scidata_cut, maskdata_cut, vardata_cut

    def cut_final_image(self, scidata_cut_large_removed, maskdata_cut_large_removed,
                        vardata_cut_large_removed, size_of_stamp, mean_kernel_size,
                        y_lower_limit_for_final_cut=None, y_upper_limit_for_final_cut=None, 
                        x_lower_limit_for_final_cut=None, x_upper_limit_for_final_cut=None):
        """! final step in cutting the image - returns science, mask and variance cuts


        @param scidata_cut_large_removed             science data with median correction already applied
        @param maskdata_cut_large_removed            mask to cover science data
        @param vardata_cut_large_removed             vardata
        @param size_of_stamp                         size of stamp that we wish to create
        @param mean_kernel_size                      size of averaging kernel, in pixel,
                                                     when estimating the centering
        @param y_lower_limit_for_final_cut,\
        y_upper_limit_for_final_cut,\
        x_lower_limit_for_final_cut,\
        x_upper_limit_for_final_cut                  explicit positions where to cut, overriding the centering
        """
        # assert that if one y_x_upper_lower values is None that all of them are None

        x_min_for_scidata_cut_large_removed = int(scidata_cut_large_removed.shape[1]/2-size_of_stamp/2-20)
        x_max_for_scidata_cut_large_removed = int(scidata_cut_large_removed.shape[1]/2 + size_of_stamp/2 + 20)
        y_min_for_scidata_cut_large_removed = int(scidata_cut_large_removed.shape[0]/2-size_of_stamp/2-5)
        y_max_for_scidata_cut_large_removed = int(scidata_cut_large_removed.shape[0]/2 + size_of_stamp/2 + 5)

        scidata_cut_large_removed_prestep =\
            scidata_cut_large_removed[y_min_for_scidata_cut_large_removed:y_max_for_scidata_cut_large_removed,
                                      x_min_for_scidata_cut_large_removed:x_max_for_scidata_cut_large_removed]
        maskdata_cut_large_removed_prestep =\
            maskdata_cut_large_removed[y_min_for_scidata_cut_large_removed:
                                       y_max_for_scidata_cut_large_removed,
                                       x_min_for_scidata_cut_large_removed:
                                       x_max_for_scidata_cut_large_removed]
        vardata_cut_large_removed_prestep =\
            vardata_cut_large_removed[y_min_for_scidata_cut_large_removed:y_max_for_scidata_cut_large_removed,
                                      x_min_for_scidata_cut_large_removed:x_max_for_scidata_cut_large_removed]

        if y_lower_limit_for_final_cut is None:

            size_of_stamp = int(size_of_stamp)
            y_profile = np.sum(scidata_cut_large_removed_prestep *
                               np.abs((1-maskdata_cut_large_removed_prestep)), axis=1)
            y_profile_running_mean = running_mean(y_profile, mean_kernel_size)
            y_pos_for_final_cut = int(np.argmax(y_profile_running_mean) + mean_kernel_size/2)

            # need to implement for the other size of the image as well
            if y_pos_for_final_cut < size_of_stamp/2:
                y_pos_for_final_cut = int(size_of_stamp/2)

            x_profile = np.sum(scidata_cut_large_removed_prestep *
                               np.abs((1-maskdata_cut_large_removed_prestep)), axis=0)
            x_profile_running_mean = running_mean(x_profile, mean_kernel_size)
            x_pos_for_final_cut = int(np.argmax(x_profile_running_mean) + mean_kernel_size/2)

            if x_pos_for_final_cut < size_of_stamp/2:
                x_pos_for_final_cut = int(size_of_stamp/2)

            y_lower_limit_for_final_cut = int(y_pos_for_final_cut-size_of_stamp/2)
            y_upper_limit_for_final_cut = int(y_pos_for_final_cut + size_of_stamp/2)
            x_lower_limit_for_final_cut = int(x_pos_for_final_cut-size_of_stamp/2)
            x_upper_limit_for_final_cut = int(x_pos_for_final_cut + size_of_stamp/2)

            print('y_x_lower_upper positions in the final cut: ' +
                  str([y_lower_limit_for_final_cut, y_upper_limit_for_final_cut,
                       x_lower_limit_for_final_cut, x_upper_limit_for_final_cut]))

            scidata_final_cut =\
                scidata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,
                                                  x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            maskdata_final_cut =\
                maskdata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,
                                                   x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            vardata_final_cut =\
                vardata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,
                                                  x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            return scidata_final_cut, maskdata_final_cut, vardata_final_cut,\
                y_lower_limit_for_final_cut, y_upper_limit_for_final_cut,\
                x_lower_limit_for_final_cut, x_upper_limit_for_final_cut
        else:
            scidata_final_cut =\
                scidata_cut_large_removed_prestep[y_lower_limit_for_final_cut:
                                                  y_upper_limit_for_final_cut,
                                                  x_lower_limit_for_final_cut:
                                                  x_upper_limit_for_final_cut]
            maskdata_final_cut =\
                maskdata_cut_large_removed_prestep[y_lower_limit_for_final_cut:
                                                   y_upper_limit_for_final_cut,
                                                   x_lower_limit_for_final_cut:
                                                   x_upper_limit_for_final_cut]
            vardata_final_cut =\
                vardata_cut_large_removed_prestep[y_lower_limit_for_final_cut:
                                                  y_upper_limit_for_final_cut,
                                                  x_lower_limit_for_final_cut:
                                                  x_upper_limit_for_final_cut]
            # if explicit values passed, return only cut images
            return scidata_final_cut, maskdata_final_cut, vardata_final_cut

    def create_poststamps(self, exposure_defocus_explicit=None, run_explicit=None, dm = None):
        """! master function that creates poststamps

         """

        if run_explicit is None:
            run_output = self.run_0
        else:
            run_output = run_explicit
        # first part is to stack the images and gather information
        # import initial image and gather information


        scidata00, maskdata00, vardata00, exposure_arc, exposure_defocus =\
            self.stack_images_and_gather_information(run_output)
        print(exposure_defocus)

        print('mean value of empty slice in scidata ' + str(np.mean(scidata00[3176:4176, 3333:3433])))

        if exposure_defocus is None:
            exposure_defocus = self.exposure_defocus_explicit

        if self.verbosity == 1:
            print('Successfully stacked images')
            print('exposure_arc is ' + str(exposure_arc))
            print('exposure_defocus is ' + str(exposure_defocus))

        Zernike_info_df = self.Zernike_info_df

        # values of defocus that we analyse, and sizes of stamps
        # does not support fine_defocus!!!
        defocus_values = np.array([-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5,
                                   0.5, +1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        sizes_of_stamps = np.array([60, 60, 50, 40, 30, 30, 24, 20, 20, 20, 24, 30, 30, 40, 50, 60, 60, 70])

        # Size of stamp that we wish to create
        # set it here if in focus
        if np.abs(exposure_defocus) == 0:
            size_of_stamp = 20
        # if in defocus take it from the list
        else:
            size_of_stamp = sizes_of_stamps[defocus_values == exposure_defocus]

        self.size_of_stamp = size_of_stamp
        if self.verbosity == 1:
            print('size_of_stamp is ' + str(size_of_stamp))
        # list of fiber_IDs - for the standard 10 fiber configuration
        # will need updating to handle more flexible configurations
        list_of_fiber_IDs = list(np.unique(Zernike_info_df['fiber'].values))

        # now run the cutting procedure on the stack, for each initial position
        # for image_index in tqdm(range(len(Zernike_info_df))):
        for image_index in tqdm(Zernike_info_df.index):
            self.image_index = image_index
            try:
                # central position as deduced from focused images/detectormap
                center_pos = [int(np.array(Zernike_info_df.loc[image_index, ['xc', 'yc']].values)[0]),
                              int(np.array(Zernike_info_df.loc[image_index, ['xc', 'yc']].values)[1])]
                if self.verbosity == 1:
                    print('exposure_defocus is ' + str(exposure_defocus) +
                          ': Central position of the spot ' + str(image_index) + ' is: ' + str(center_pos))

                # intial cut, pure centering and before removing continuum
                scidata_cut, maskdata_cut, vardata_cut, xposmin, xposmax, yposmin, yposmax =\
                    self.cut_initial_image(scidata00, maskdata00, vardata00,
                                           center_pos[0], center_pos[1], size_of_stamp)
                print('Initial cut completed ')

                ############################
                # removal of continuum
                ############################

                if self.subtract_continuum is True:
                    print('subtracting_continuum - this must be HgAr!')
                    # which fiber is being analyzed
                    which_fiber = Zernike_info_df.loc[image_index]['fiber']
                    
                    vw_blue = 550
                    vw_red = 800
                    if self.detector == 'red':
                        vw_central = vw_red
                    else:
                        vw_central = vw_blue
                    x_pos_of_fiber = dm.findPoint(which_fiber,vw_central)[0] 

                    # cut out the large rectangle from which to subtract continuum
                    single_fiber = scidata00[0:4176, int(x_pos_of_fiber - size_of_stamp/2) :
                        int(x_pos_of_fiber + size_of_stamp/2) ]
                    # make box sum over the whole rectangle
                    data = np.sum(single_fiber,axis=1)

                    # get fits and where the arcs are
                    y_median, good = determine_cont(data, iterations=4, numKnots=36, rejection=3)

                    # apply the removal algorithm
                    scidata_cut_large_removed, maskdata_cut_large_removed, vardata_cut_large_removed =\
                        self.removal_of_continuum(scidata_cut, maskdata_cut, vardata_cut,
                                                  y_median, good, yposmin, yposmax)

                    # overload the definitions and assings the data with continuum
                    # removed to the previous data names
                    scidata_cut, maskdata_cut, vardata_cut = scidata_cut_large_removed,\
                        maskdata_cut_large_removed, vardata_cut_large_removed

                ############################
                # cutting final image
                ############################

                # mean kernel size which should change as a function of defocus
                # mean_kernel_size empirically should be around 5 in the focus
                # and around the size of the full image for fully defocused ( + -4 mm of the slit) images
                mean_kernel_size = int(40*np.abs(exposure_defocus/4) + 5)
                size_of_stamp = int(size_of_stamp)

                # apply the final cut and centering algorithm
                # this allows to slightly shift image to better capture faint areas of a donut
                scidata_final_cut, maskdata_final_cut, vardata_final_cut,\
                    y_lower_limit_for_final_cut, y_upper_limit_for_final_cut,\
                    x_lower_limit_for_final_cut, x_upper_limit_for_final_cut =\
                    self.cut_final_image(scidata_cut, maskdata_cut, vardata_cut,
                                         size_of_stamp, mean_kernel_size)


                if run_explicit is None:
                    run_output = self.run_0
                else:
                    run_output = run_explicit

                # save the output
                # need to implement custom date
                if self.verbosity == 1:
                    print('image with index ' + str(self.image_index) + ' seems successful')
                    if self.detector == 'red':
                        print('saving at: ' + self.PSF_DIRECTORY_CUTS + "sci" + str(run_output) +
                              str(image_index) + exposure_arc + '_Stacked.npy')
                    else:
                        print('saving at: ' + self.PSF_DIRECTORY_CUTS + "sci" + str(run_output) +
                              str(image_index) + exposure_arc + '_b_Stacked.npy')
                    print('######################################################')

                if self.save is True:
                    if self.detector == 'red':
                        np.save(self.PSF_DIRECTORY_CUTS + "sci" + str(run_output) + str(image_index) +
                                exposure_arc + '_Stacked.npy', scidata_final_cut)
                        np.save(self.PSF_DIRECTORY_CUTS + "mask" + str(run_output) + str(image_index) +
                                exposure_arc + '_Stacked.npy', maskdata_final_cut)
                        np.save(self.PSF_DIRECTORY_CUTS + "var" + str(run_output) + str(image_index) +
                                exposure_arc + '_Stacked.npy', vardata_final_cut)
                    else:
                        np.save(self.PSF_DIRECTORY_CUTS + "sci" + str(run_output) + str(image_index) +
                                exposure_arc + '_b_Stacked.npy', scidata_final_cut)
                        np.save(self.PSF_DIRECTORY_CUTS + "mask" + str(run_output) + str(image_index) +
                                exposure_arc + '_b_Stacked.npy', maskdata_final_cut)
                        np.save(self.PSF_DIRECTORY_CUTS + "var" + str(run_output) + str(image_index) +
                                exposure_arc + '_b_Stacked.npy', vardata_final_cut)

            except Exception as e:
                if self.verbosity == 1:
                    print(e)
                    print('image with index ' + str(self.image_index) + ' failed')
                    print('######################################################')
                pass

####################################
# Free standing code
####################################


def create_Zernike_info_df(pfsDetectorMap, arc, DATA_FOLDER, obs, pfsConfig=10, detector='red'):
    """! cretes a pandas dataframe that holds information about the poststamps that we wish to extract
    from the image for the psf analysis

    @param pfsDetectorMap          pfsDetectorMap, created via pipeline
    @param arc                     which arc is being used (HgAr, Ar, Ne, Kr)
    @param DATA_FOLDER             data folder in which to express the final
    @param obs                     observation number - only used for the export name

    # not implemented yet
    @param pfsConfig               pfsConfig file to deduce which fibers are being used
                                   at the moment it assumes 10 fibers configuration
    """

    if pfsConfig == 10:
        fibers_10 = np.flip(np.array([2, 63, 192, 255, 339, 401, 464, 525, 587, 650]))

    if pfsConfig == 21:
        fibers_10 = np.flip(np.array([12, 32, 60, 110, 111, 161, 210, 223, 259, 289, 341, 360, 400,
                                      418, 449, 497, 518, 545, 593, 621, 643]))

    if detector == 'red':
        if arc == 'HgAr':
            wavelengths_arc = ['690.9346', '696.7261', '706.8989', '727.47876', 
                               '738.6248', '763.74286', '795.0522', '826.6699',
                               '852.4029', '912.5693', '922.7301', '966.0642']
            close_arc = ['0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1']

            second_offset = [-19.25, 0, 16.93, 0, -13.14, -38.59, 0, 18.83, -14.26, 0, 22.44, 0]
            second_flux = [0.0034, 0, 0.268, 0, 0.011, 0.008, 0, 0.0043, 0.015, 0, 0.022, 0]
            second2_offset = [-40.7, 0, 28.33, 0, 0, 0, 0, 0, 0, 0, -13.14, 0]
            second2_flux = [0.006, 0, 0.088, 0, 0, 0, 0, 0, 0, 0, 0.011, 0]
        if arc == 'Ar':
            wavelengths_arc = ['690.9346', '696.7261', '706.8989', '727.47876',
                               '738.6248', '763.74286', '795.0522', '826.6699',
                               '852.4029', '912.5693', '922.7301', '966.0642']
            close_arc = ['0', '1', '0', '1', '0', '1', '1', '1', '1', '1', '0', '1']

            second_offset = [-19.25, 0, -39.597, 0, -13.14, 0, 0, 0, 0, 0, -35.264, 0]
            second_flux = [0.0034, 0, 0.0143, 0, 0.011, 0, 0, 0.0, 0, 0, 0.0124, 0]
            second2_offset = [-40.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            second2_flux = [0.006, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if arc == 'Ne':
            wavelengths_arc = ['650.84186', '653.4774', '660.0668', '668.01764',
                               '671.9268', '693.13116', '717.59015', '724.72437', '744.1276']
            close_arc = ['0', '-1', '1', '0', '-1', '1', '1', '1', '0']

            second_offset = [30.12, 0, 0, 44.24, 0, 0, 0, 0, 38.57]
            second_flux = [0.406, 0, 0, 0.67, 0, 0, 0, 0, 0.012]
            second2_offset = [0, 0, 0, -29.97, 0, 0, 0, 0, 57.44]
            second2_flux = [0, 0, 0, 0.003, 0, 0, 0, 0, 0.162]
        if arc == 'Kr':
            wavelengths_arc = [785.48, 819.0, 877.68, 892.87]
            close_arc = [0, 0, 0, 1]

            second_offset = [8, 20, -15, 0]
            second_flux = [0.01, 0.004, 0.004, 0]
            second2_offset = [0, 0, 10, 0]
            second2_flux = [0, 0, 0.003, 0]
    if detector == 'blue':
        if arc == 'Ne':
            wavelengths_arc = ['603.16666', '621.90013', '626.82285', '650.83255']
            close_arc = ['1', '1', '1', '0']

            second_offset = [0, 0, 0, 30.12]
            second_flux = [0, 0, 0, 0.406]
            second2_offset = [0, 0, 0, 0]
            second2_flux = [0, 0, 0, 0]
        if arc == 'HgAr':
            wavelengths_arc = ['404.77080', '435.95600', '546.22680']
            close_arc = ['0', '1', '1']
            
            second_offset = [30, 0, 0]
            second_flux = [0.12, 0, 0]
            second2_offset = [0, 0, 0]
            second2_flux = [0, 0, 0]
            
    list_of_actual_fibers = []
    list_of_yc = []
    list_of_xc = []
    list_of_xc_eff = []
    list_of_wavelength = []
    list_of_close = []
    list_of_lamp = []
    list_of_second_offset = []
    list_of_second_ratio = []
    list_of_second2_offset = []
    list_of_second2_ratio = []

    x_pfsDetectorMap = np.array(pfsDetectorMap[3].data)
    y_pfsDetectorMap = np.array(pfsDetectorMap[4].data)

    for i in range(len(fibers_10)*len(wavelengths_arc)):

        # fiber index, sequenetial as they appear on the detector, i.e., starting at 0 and going +1
        index_of_single_fiber_sequential = int(np.floor(i/len(wavelengths_arc)))
        # actual number of the fiber (i.e., going up to 650)
        actual_fiber = int(fibers_10[index_of_single_fiber_sequential])
        # index of a fiber, as they go through the detectorMap (i.e., going up to 599)
        index_of_single_fiber = np.where(pfsDetectorMap[1].data == actual_fiber)[0][0]

        # add the information about the actual fiber in the dataframe
        # Zernike_info_df.loc[i]['fiber']=actual_fiber
        list_of_actual_fibers.append(actual_fiber)

        # select the parts of the detectorMap which correspond to that particular fiber
        x_pfsDetectorMap_single_fiber = x_pfsDetectorMap[x_pfsDetectorMap['index'] == index_of_single_fiber]
        y_pfsDetectorMap_single_fiber = y_pfsDetectorMap[y_pfsDetectorMap['index'] == index_of_single_fiber]

        # first find the y position
        # Zernike_info_df.loc[i]['yc']=int(x_pfsDetectorMap_single_fiber['knot']\
        # [np.where(y_pfsDetectorMap_single_fiber['value']==\
        #   find_nearest(y_pfsDetectorMap_single_fiber['value'],
        # float(wavelengths_HgAr[np.mod(i,len(fibers_10))])))[0][0]])

        y_value =\
            int(y_pfsDetectorMap_single_fiber
                ['knot'][np.where(y_pfsDetectorMap_single_fiber['value'] ==
                                  find_nearest(y_pfsDetectorMap_single_fiber['value'],
                                               float(wavelengths_arc[np.mod(i,
                                                                            len(wavelengths_arc))])))[0][0]])
        x_value = int(x_pfsDetectorMap_single_fiber
                      ['value'][np.where(x_pfsDetectorMap_single_fiber['knot'] == y_value)])

        if x_value > 2048:
            x_value_eff = x_value + 69
        else:
            x_value_eff = x_value

        list_of_yc.append(y_value)

        # from y position, deduce x position
        # Zernike_info_df.loc[i]['xc']=int(x_pfsDetectorMap_single_fiber['value']
        # [np.where(x_pfsDetectorMap_single_fiber['knot']==Zernike_info_df.loc[i]['yc'])])
        list_of_xc.append(x_value)

        list_of_xc_eff.append(x_value_eff)

        # Zernike_info_df.loc[i]['wavelength'] = wavelengths_HgAr[np.mod(i,len(fibers_10))]
        list_of_wavelength.append(wavelengths_arc[np.mod(i, len(wavelengths_arc))])
        # Zernike_info_df.loc[i]['close']=close_HgAr[np.mod(i,len(fibers_10))]
        list_of_close.append(close_arc[np.mod(i, len(wavelengths_arc))])
        # Zernike_info_df.loc[i]['lamp']='HgAr'
        list_of_lamp.append(arc)

        # Zernike_info_df.loc[i]['second_offset']=second_offset[np.mod(i,len(fibers_10))]
        list_of_second_offset.append(second_offset[np.mod(i, len(wavelengths_arc))])
        # Zernike_info_df.loc[i]['second_flux']=second_flux[np.mod(i,len(fibers_10))]
        list_of_second_ratio.append(second_flux[np.mod(i, len(wavelengths_arc))])
        # Zernike_info_df.loc[i]['second2_offset']=second2_offset[np.mod(i,len(fibers_10))]
        list_of_second2_offset.append(second2_offset[np.mod(i, len(wavelengths_arc))])
        # Zernike_info_df.loc[i]['second2_flux']=second2_flux[np.mod(i,len(fibers_10))]
        list_of_second2_ratio.append(second2_flux[np.mod(i, len(wavelengths_arc))])

    # for i in range(len(fibers_10)*len(wavelengths_HgAr)):
    #    if Zernike_info_df.loc[i]['xc']<2048:
    #        Zernike_info_df.loc[i]['xc_effective']=Zernike_info_df.loc[i]['xc']
    #    else:
    #        Zernike_info_df.loc[i]['xc_effective']=Zernike_info_df.loc[i]['xc'] + 69

    Zernike_info_df = pd.DataFrame({'fiber': list_of_actual_fibers, 'xc': list_of_xc,
                                    'yc': list_of_yc, 'wavelength': list_of_wavelength,
                                    'close': list_of_close, 'lamp': list_of_lamp,
                                    'xc_effective': list_of_xc_eff, 'second_offset': list_of_second_offset,
                                    'second_ratio': list_of_second_ratio,
                                    'second2_offset': list_of_second2_offset,
                                    'second2_ratio': list_of_second2_ratio})

    if pfsConfig == 21:
        Zernike_info_df.index = Zernike_info_df.index + (10*len(wavelengths_arc))

    if not os.path.exists(DATA_FOLDER + "Dataframes/"):
        os.makedirs(DATA_FOLDER + "Dataframes/")
    Zernike_info_df.to_pickle(DATA_FOLDER + "Dataframes/Zernike_info_df_" + str(arc) + '_' + str(obs))


def find_centroid_of_flux(image):
    """
    function giving the tuple of the position of weighted average of the flux in a square image

    @param image    poststamp image for which to find center
    """

    x_center = []
    y_center = []

    # if there are nan values (most likely cosmics), replace them with max value in the rest of the image
    # for this purpose
    max_value_image = np.max(image[~np.isnan(image)])
    image[np.isnan(image)] = max_value_image

    I_x = []
    for i in range(len(image)):
        I_x.append([i, np.sum(image[:, i])])

    I_x = np.array(I_x)

    I_y = []
    for i in range(len(image)):
        I_y.append([i, np.sum(image[i])])

    I_y = np.array(I_y)

    x_center = (np.sum(I_x[:, 0]*I_x[:, 1])/np.sum(I_x[:, 1]))
    y_center = (np.sum(I_y[:, 0]*I_y[:, 1])/np.sum(I_y[:, 1]))

    return(x_center, y_center)


def find_nearest(array, value):
    """
    out of array of values, returns the index of the closest value

    @param array    array in which to search for a value
    @param value    value that we search for
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def running_mean(mylist, N):
    """
    return the list with a running mean acroos the array, with a given kernel size

    @param mylist   array,list over which to average
    @param value    length of the averaging kernel
    """

    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

def binData(xx, yy, good, numBins):
    """Bin arrays
    Parameters
    ----------
    xx, yy : `numpy.ndarray`
        Arrays to bin.
    good : `numpy.ndarray`, boolean
        Boolean array indicating which points are good.
    numBins : `int`
        Number of bins.
    Returns
    -------
    xBinned, yBinned : `numpy.ndarray`
        Binned data.
    """
    edges = (np.linspace(0, len(xx), numBins + 1) + 0.5).astype(int)
    xBinned = np.empty(numBins)
    yBinned = np.empty(numBins)
    for ii, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        select = good[low:high]
        xBinned[ii] = np.median(xx[low:high][select])
        yBinned[ii] = np.median(yy[low:high][select])
    return xBinned, yBinned


def fitContinuumImpl(values, good, numKnots, showPlot=0):
    indices = np.arange(len(values), dtype=np.float)
    knots, binned = binData(indices, values, good, numKnots)
    interp = makeInterpolate(knots, binned, stringToInterpStyle('AKIMA_SPLINE'))
    fit = np.array(interp.interpolate(indices))
    if showPlot==1:
        fig, ax = plt.subplots(figsize=(30,8))

        # Show good points as black, rejected points as red, but with a continuous line
        # https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
   
        from matplotlib.collections import LineCollection
        cmap, norm = matplotlib.colors.from_levels_and_colors([0.0, 0.5, 2.0], ['red', 'black'])
        points = np.array([indices, values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lines = LineCollection(segments, cmap=cmap, norm=norm)
        lines.set_array(good.astype(int))
        ax.add_collection(lines)
        """
        # Plot binned data
        xBinned, yBinned = binData(indices, values, good, lsstDebug.Info(__name__).plotBins or 1000)
        ax.plot(xBinned, yBinned, 'k-')
        """
        ax.plot(indices, fit, 'b-')
        ax.plot(knots, binned, 'bo')
        ax.plot(indices, (values-fit)*good, 'orange')
        ax.set_ylim(0.7*fit.min(), 1.3*fit.max())
        plt.show()
        plt.clf()
            
    return fit

def determine_cont(data, iterations=4, numKnots=36, rejection=3):
    num=len(data)
    good = np.ones(num, dtype=bool)
    oldGood = good    
    for ii in range(iterations):

        fit = fitContinuumImpl(data, good, numKnots, showPlot=0)
        diff = data - fit
        lq, median, uq = np.percentile(diff[good], [25.0, 50.0, 75.0])

        # Estimating the stddev only from the lower side
        stdev = 0.741*(uq - lq)
        good = good*(np.isfinite(diff) & ((diff <= rejection*stdev)))
        # print(ii, rejection,sum(good), rejection*stdev)
        if np.all(good == oldGood):
            break

        oldGood = good

    fit= fitContinuumImpl(data, good, numKnots, showPlot=0)
    fit_defocus=fit    
    return fit_defocus, good
