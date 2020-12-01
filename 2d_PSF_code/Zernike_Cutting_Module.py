#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 09:42:10 2020

Module and stand-alone code for the creation of poststamp images 

    

Versions:
Jun 08, 2020: 0.28e initial version, version number from Zernike_Module
Nov 27, 2020: 0.29 implmented cutting of prestacked images


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

# add 

# analysis of dithering
# automatic creation of finalArc from detectorMap

########################################

__all__ = ['Zernike_cutting','create_Zernike_info_df', 'create_dither_plot','find_centroid_of_flux','find_nearest','running_mean']

__version__ = "0.29"

############################################################
# name your directory where you want to have files!
PSF_DIRECTORY_CUTS='/Users/nevencaplar/Documents/PFS/ReducedData/Data_Nov_20_2020/Stamps_cleaned/'
# DATAFRAMES FOLDER?
# AUXILIARY DATA FOLDER?

########################################

class Zernike_cutting(object):
    """
    Class that deals with cutting the PSF data
    
    Documentation here
    
    
    
    
    
    
    
    
    
    """

    def __init__(self, DATA_FOLDER,run_0,number_of_images,SUB_LAM,Zernike_info_df,
                 subtract_continuum=False,dither=None,verbosity=0,save=True,\
                 use_previous_full_stacked_images=True,use_median_rejected_stacked_images=False,exposure_arc=None):
        self.DATA_FOLDER=DATA_FOLDER
        self.run_0=run_0
        self.number_of_images=number_of_images
        self.SUB_LAM=SUB_LAM
        
        if SUB_LAM=='SUB':
           SUB_LAM_letter='S'
        elif SUB_LAM=='LAM':
           SUB_LAM_letter='L'   
        self.SUB_LAM_letter=SUB_LAM_letter
        self.Zernike_info_df=Zernike_info_df
        
        self.subtract_continuum=subtract_continuum
        self.use_previous_full_stacked_images=use_previous_full_stacked_images
        self.dither=dither
            

        self.verbosity=verbosity
        if save is None:
            save=False
            self.save=save
        else:
            self.save=save
            
        self.use_median_rejected_stacked_images=use_median_rejected_stacked_images
        self.exposure_arc=exposure_arc


    
    def removal_of_continuum(self,scidata_cut,maskdata_cut,vardata_cut,y_median,good,yposmin,yposmax):

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
        sci_upper_cut=scidata_cut[len(scidata_cut[:,0])-70:len(scidata_cut[:,0])-5]
        sci_lower_cut=scidata_cut[5:70]
    
        overall_median_lower_left=stats.sigma_clipped_stats(sci_lower_cut[:,:25].ravel(), sigma_lower=100,sigma_upper=2, iters=6)[1]  
        overall_median_lower_right=stats.sigma_clipped_stats(sci_lower_cut[:,:-25:].ravel(), sigma_lower=100,sigma_upper=2, iters=6)[1]  
        overall_median_upper_left=stats.sigma_clipped_stats(sci_upper_cut[:,:25].ravel(), sigma_lower=100,sigma_upper=2, iters=6)[1]  
        overall_median_upper_right=stats.sigma_clipped_stats(sci_upper_cut[:,:-25:].ravel(), sigma_lower=100,sigma_upper=2, iters=6)[1]  
        list_of_medians=[overall_median_lower_left,overall_median_lower_right,overall_median_upper_left,overall_median_upper_right]
        overall_median=np.array(list_of_medians)[np.abs(list_of_medians)==np.min(np.abs(list_of_medians))][0]
        if self.verbosity==1:
            print('overall_median is: '+str(overall_median))
    
        #scidata_cut_median_subtracted=scidata_cut-overall_median
        scidata_cut_median_subtracted=scidata_cut
        scidata_cut_original_median_subtracted=np.copy(scidata_cut_median_subtracted)
    
    
        # shape of the continuum in the test region, from a single exposure
        # single_fiber_extracted_dimensions[1] is where the fiber (i.e., y_median) starts
        
        y_median_single_spot=y_median[yposmin:yposmax]
        good_single_spot=good[yposmin:yposmax]
    
        y_median_single_spot_good=y_median_single_spot[good_single_spot]
        scidata_cut_original_median_subtracted_summed=np.sum(scidata_cut_original_median_subtracted,axis=1)
        scidata_cut_original_median_subtracted_summed_good=scidata_cut_original_median_subtracted_summed[good_single_spot]
    
        stack_to_extract_normalization_factor=np.median(scidata_cut_original_median_subtracted_summed_good/y_median_single_spot_good)
        if self.verbosity==1:
            print('stack_to_extract_normalization_factor: '+str(stack_to_extract_normalization_factor))
    
        y_median_single_spot_renormalized=y_median_single_spot*stack_to_extract_normalization_factor
    
        # we probably want to save these 
        #plt.plot(y_median_single_spot_renormalized,color='red',ls=':')
        #plt.plot(scidata_cut_original_median_subtracted_summed,color='black',ls=':')
        #plt.plot(y_median_single_spot_renormalized*good_single_spot,color='red')
        #plt.plot(scidata_cut_original_median_subtracted_summed*good_single_spot,color='black')
        #plt.ylim(0.9*np.min(y_median_single_spot_renormalized),1.1*np.max(y_median_single_spot_renormalized))
    
        #################
        # removal of continuum - actual removal
        #################
        scidata_cut_large_removed=np.copy(scidata_cut_median_subtracted)
        maskdata_cut_large_removed=maskdata_cut
        vardata_cut_large_removed=vardata_cut
    
        select_central=np.concatenate((np.zeros((1,int(len(good_single_spot)/2-40))),np.ones((1,80)),np.zeros((1,int(len(good_single_spot)/2-40)))),axis=1)[0]
    
        try:
            for j in range(scidata_cut_large_removed.shape[1]):  
                good_single_spot_centred=(good_single_spot*select_central).astype('bool')
                if np.sum(good_single_spot_centred)>25:
                    y_median_single_spot_renormalized=y_median_single_spot*np.median(scidata_cut_large_removed[:,j][good_single_spot_centred])/np.median(y_median_single_spot[good_single_spot_centred])
                else:
                    y_median_single_spot_renormalized=y_median_single_spot*np.median(scidata_cut_large_removed[:,j][good_single_spot])/np.median(y_median_single_spot[good_single_spot])
                scidata_cut_large_removed[:,j]=scidata_cut_large_removed[:,j]-y_median_single_spot_renormalized            
                #scidata_cut_large_removed[0:int(scidata_cut.shape[0]/1),j]=scidata_cut_large_removed[0:int(scidata_cut.shape[0]/1),j]-fiber_profile_central_renormalized[j]*y_median_single_spot_renormalized
        except:
            print('No subtraction!!!')
            pass
    
        return scidata_cut_large_removed,maskdata_cut_large_removed,vardata_cut_large_removed
    
    def combine_dither_images(self,scidata00_final_cut,scidatapp_final_cut,scidatap0_final_cut,scidata0p_final_cut,\
                              maskdata00_final_cut,maskdatapp_final_cut,maskdatap0_final_cut,maskdata0p_final_cut,\
                              vardata00_final_cut,vardatapp_final_cut,vardatap0_final_cut,vardata0p_final_cut):
        
        mean_per_image=np.sum([scidata00_final_cut,scidatapp_final_cut,scidatap0_final_cut,scidata0p_final_cut])/4
        scidata00_final_cut_renormalized=scidata00_final_cut*mean_per_image/np.sum(scidata00_final_cut)
        scidata0p_final_cut_renormalized=scidata0p_final_cut*mean_per_image/np.sum(scidata0p_final_cut)
        scidatap0_final_cut_renormalized=scidatap0_final_cut*mean_per_image/np.sum(scidatap0_final_cut)
        scidatapp_final_cut_renormalized=scidatapp_final_cut*mean_per_image/np.sum(scidatapp_final_cut)

        vardata00_final_cut_renormalized=vardata00_final_cut*mean_per_image/np.sum(scidata00_final_cut)
        vardata0p_final_cut_renormalized=vardata0p_final_cut*mean_per_image/np.sum(scidata0p_final_cut)
        vardatap0_final_cut_renormalized=vardatap0_final_cut*mean_per_image/np.sum(scidatap0_final_cut)
        vardatapp_final_cut_renormalized=vardatapp_final_cut*mean_per_image/np.sum(scidatapp_final_cut)


        sci_image_dithered_pp=np.zeros((self.size_of_stamp*2,self.size_of_stamp*2))
        sci_image_dithered_pp[::2,1::2] = scidata0p_final_cut_renormalized
        sci_image_dithered_pp[1::2,::2] = scidatap0_final_cut_renormalized
        sci_image_dithered_pp[::2,::2] = scidatapp_final_cut_renormalized
        sci_image_dithered_pp[1::2,1::2] = scidata00_final_cut_renormalized

        mask_image_dithered_pp=np.zeros((self.size_of_stamp*2,self.size_of_stamp*2))
        mask_image_dithered_pp[::2,1::2] = maskdata0p_final_cut
        mask_image_dithered_pp[1::2,::2] = maskdatap0_final_cut
        mask_image_dithered_pp[::2,::2] = maskdatapp_final_cut
        mask_image_dithered_pp[1::2,1::2] = maskdata00_final_cut

        var_image_dithered_pp=np.zeros((self.size_of_stamp*2,self.size_of_stamp*2))
        var_image_dithered_pp[::2,1::2] = vardata0p_final_cut_renormalized
        var_image_dithered_pp[1::2,::2] = vardatap0_final_cut_renormalized
        var_image_dithered_pp[::2,::2] = vardatapp_final_cut_renormalized
        var_image_dithered_pp[1::2,1::2] = vardata00_final_cut_renormalized
        
        return sci_image_dithered_pp,mask_image_dithered_pp,var_image_dithered_pp
    
    

    def stack_images_and_gather_information(self,run):
        
        
    
        
        
        
        
        
        
        #if run number is 4 digit number, prepand one zero in front of run
        # I have not explored all possible cases of this behaviour, possibly breaks for 1,2,3, or 6 digit numbers?
        if run<10000:
            run_string='0'+str(run)
        else:
            run_string=str(run)
            
        # if you are using prestacked images, run simplified procedure 
        if self.use_median_rejected_stacked_images==True:
            data=fits.open(self.DATA_FOLDER+'median_rejected_stacked_image_'+run_string+'.fits')    
        else:

            if run<10000:
                data=fits.open(self.DATA_FOLDER+'v0'+run_string+'/calExp-'+self.SUB_LAM_letter+'A0'+run_string+'r1.fits')
            else:
                data=fits.open(self.DATA_FOLDER+'v0'+run_string+'/calExp-'+self.SUB_LAM_letter+'A0'+run_string+'r1.fits')       
        exposure_arc=self.exposure_arc
        # establish which arc is being used
        if exposure_arc is None:
            if data[0].header['W_AITNEO']==True:
                exposure_arc='Ne'
            if data[0].header['W_AITHGA']==True:
                exposure_arc='HgAr'
            if data[0].header['W_AITKRY']==True:
                exposure_arc='Kr'
            if data[0].header['W_AITXEN']==True:
                exposure_arc='Xe'
            if data[0].header['W_AITARG']==True:
                exposure_arc='Ar'
            
        print('exposure_arc '+str(exposure_arc))
            
        # establish the defocus of the data
        try:
            exposure_defocus=np.round(data[0].header['W_ENFCAX'],2)
        except:
            exposure_defocus=None
        
        if self.use_median_rejected_stacked_images==True:
            scidata=data[1].data
            maskdata=data[2].data
            vardata=data[3].data
        else:            
    
    
    
            # if you already this and use use_previous_full_stacked_images=False you can import only 
            #one image to read auxiliary data, but no time-consuming stacking
            if self.use_previous_full_stacked_images==True and os.path.exists(PSF_DIRECTORY_CUTS+"scidata"+str(run)+exposure_arc+'_Stacked.npy')==True:
                    if self.verbosity==True:
                        print('Using previously created full images, because they exist and use_previous_full_stacked_images==True')
                        print('Loading from: '+PSF_DIRECTORY_CUTS+"scidata"+str(run)+exposure_arc+'_Stacked.npy')
                    scidata=np.load(PSF_DIRECTORY_CUTS+"scidata"+str(run)+exposure_arc+'_Stacked.npy')
                    maskdata=np.load(PSF_DIRECTORY_CUTS+"maskdata"+str(run)+exposure_arc+'_Stacked.npy')
                    vardata=np.load(PSF_DIRECTORY_CUTS+"vardata"+str(run)+exposure_arc+'_Stacked.npy')       
            else:
                # prepare empty arrays that will store the data 
                scidata=np.zeros_like(data[1].data)
                maskdata=np.zeros_like(scidata)
                vardata=np.zeros_like(scidata)  
                
                # prepare arrays and lists which will be filled
                list_of_sci_data=[]
                list_of_var_data=[]
                list_of_mask_data=[]   
                
                list_of_exposure_defocus=[]
                list_of_exposure_arc=[]
                
                # for each image, gather information and stack it
                for run_i in tqdm(range(self.number_of_images)):
                    
                    if run+run_i<10000:
                        run_string='0'+str(run+run_i)
                    else:
                        run_string=str(run+run_i)
                    
                    # import fits file
                    data=fits.open(self.DATA_FOLDER+'v0'+run_string+'/calExp-SA0'+run_string+'r1.fits')
                    
                    # establish the defocus of the data
                    exposure_defocus=np.round(data[0].header['W_ENFCAX'],2)
                    
                    # establish which arc is being used
                    if data[0].header['W_AITNEO']==True:
                        exposure_arc='Ne'
                    if data[0].header['W_AITHGA']==True:
                        exposure_arc='HgAr'
                    if data[0].header['W_AITKRY']==True:
                        exposure_arc='Kr'
                    if data[0].header['W_AITXEN']==True:
                        exposure_arc='Xe'
                    if data[0].header['W_AITARG']==True:
                        exposure_arc='Ar'
                    
                    # add the information to the list
                    list_of_exposure_defocus.append(exposure_defocus)
                    list_of_exposure_arc.append(exposure_arc)
                    
                    
                    #background=background_estimate_sigma_clip_fit_function(exposure_defocus)
                    #if run_i==0:
                    #    print('background estimate is: '+str(background))
                
                    #scidata_single=data[1].data-background
                    
                    
                    # separate scidata, maskdata and var data
                    scidata_single=data[1].data
                    maskdata_single=data[2].data
                    if self.verbosity==1:
                        print('np.sum(maskdata_single) for image iteration '+str(run_i)+' is: '+str(np.sum(maskdata_single)))
                    vardata_single=data[3].data
                
                    # stack the images
                    scidata=scidata+scidata_single
                    maskdata=maskdata+maskdata_single
                    vardata=vardata+vardata_single
                
                    # add the data to individual list, if needed for debugging and analysis          
                    list_of_sci_data.append(scidata_single)
                    maskdata_single[np.isin(maskdata_single,[0])]=0
                    maskdata_single[~np.isin(maskdata_single,[0])]=1
                    list_of_mask_data.append(maskdata_single) 
                    list_of_var_data.append(vardata_single)
               
                array_of_sci_data=np.array(list_of_sci_data)
                #
                maskdata[np.isin(maskdata,[0])]=0
                maskdata[~np.isin(maskdata,[0])]=1
            
            if self.save==True:
                if not os.path.exists(PSF_DIRECTORY_CUTS):
                    if self.verbosity==1:
                        print("Creating PSF_DIRECTORY_CUTS directory at "+str(PSF_DIRECTORY_CUTS))
                        os.makedirs(PSF_DIRECTORY_CUTS)

                np.save(PSF_DIRECTORY_CUTS+"array_of_sci_data"+str(run)+exposure_arc+'_Stacked.npy',array_of_sci_data)                
                np.save(PSF_DIRECTORY_CUTS+"scidata"+str(run)+exposure_arc+'_Stacked.npy',scidata)
                np.save(PSF_DIRECTORY_CUTS+"maskdata"+str(run)+exposure_arc+'_Stacked.npy',maskdata)
                np.save(PSF_DIRECTORY_CUTS+"vardata"+str(run)+exposure_arc+'_Stacked.npy',vardata)        

        return scidata,maskdata,vardata,exposure_arc,exposure_defocus

    
    def cut_initial_image(self,scidata,maskdata,vardata,xinit,yinit,size_of_stamp,xposmin=None,xposmax=None,yposmin=None,yposmax=None):
        """!given the initially centered image
        
        @param full_data        full science data image 
        @param xinit            x position of the initial guess 
        @param yinit            y position of the initial guess 
        @param size_of_stamp    size of the stamp that we wish to create
        """         
         
        #print('checkpoint initial')

        # if you do not pass explicit coordinates for the cut, do the full fitting procedure 
        if yposmin==None:
            
            
            # Iteration 0
            #size_of_stamp_big=2*size_of_stamp

            xposmin=int(int(xinit)-size_of_stamp/2)
            xposmax=int(int(xinit)+size_of_stamp/2)
            yposmin=int(int(yinit)-size_of_stamp/2)
            yposmax=int(int(yinit)+size_of_stamp/2)
            print('iteration nr. 0: '+str([yposmin,yposmax,xposmin,xposmax]))
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            print('reached scidata_cut')
            
            #print('np.any(scidata_cut>300, axis=1)'+str(np.any(scidata_cut>300, axis=1)))
            #print('np.any(scidata_cut>300, axis=0)'+str(np.any(scidata_cut>300, axis=0)))
            med_pos=(int(np.round(np.median(np.where(np.any(scidata_cut>300, axis=1))))),
                            int(np.round(np.median(np.where(np.any(scidata_cut>300, axis=0))))))
            
            # Iteration 1
            new_xinit=xinit+med_pos[1]-size_of_stamp/2
            new_yinit=yinit+med_pos[0]-size_of_stamp/2
            xposmin=int(new_xinit-size_of_stamp/2)
            xposmax=int(new_xinit+size_of_stamp/2)
            yposmin=int(new_yinit-size_of_stamp/2)
            yposmax=int(new_yinit+size_of_stamp/2)
            print('iteration nr. 1: '+str([yposmin,yposmax,xposmin,xposmax]))
            
            #import pdb
            #pdb.set_trace()
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            
            
            # iteration 2
            centroid_of_flux_model=find_centroid_of_flux(scidata_cut)
            # print(centroid_of_flux_model)
        
            #centroid_of_flux_model_int=np.array(map(int,map(np.round,centroid_of_flux_model)))
            dif_centroid_of_flux_model_int_from_center=np.array(centroid_of_flux_model)-np.array([int(size_of_stamp/2),int(size_of_stamp/2)])
        
            xposmin=int(np.round(xposmin+dif_centroid_of_flux_model_int_from_center[0]))
            xposmax=int(np.round(xposmax+dif_centroid_of_flux_model_int_from_center[0]))
            yposmin=int(np.round(yposmin+dif_centroid_of_flux_model_int_from_center[1]))
            yposmax=int(np.round(yposmax+dif_centroid_of_flux_model_int_from_center[1]))
        
        
            print('iteration nr. 2: '+str([yposmin,yposmax,xposmin,xposmax]))
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            
            # iteration 3
            centroid_of_flux_model=find_centroid_of_flux(scidata_cut)
            # print(centroid_of_flux_model)
        
            #centroid_of_flux_model_int=np.array(map(int,map(np.round,centroid_of_flux_model)))
            dif_centroid_of_flux_model_int_from_center=np.array(centroid_of_flux_model)-np.array([int(size_of_stamp/2),int(size_of_stamp/2)])
        
            xposmin=int(np.round(xposmin+dif_centroid_of_flux_model_int_from_center[0]))
            xposmax=int(np.round(xposmax+dif_centroid_of_flux_model_int_from_center[0]))
            yposmin=int(np.round(yposmin+dif_centroid_of_flux_model_int_from_center[1]))
            yposmax=int(np.round(yposmax+dif_centroid_of_flux_model_int_from_center[1]))
        
            print('iteration nr. 3: '+str([yposmin,yposmax,xposmin,xposmax]))
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            
            center_pos_new=[xposmin+size_of_stamp/2,yposmin+size_of_stamp/2]
    
            # take the outut and get positions of where to cut
            xposmin=int(np.round(center_pos_new[0]-70-10))
            xposmax=int(np.round(center_pos_new[0]+70+10))
            yposmin=int(np.round(center_pos_new[1]-70-45))
            yposmax=int(np.round(center_pos_new[1]+70+45))
    
            
            # if you are on the edge, stop at the edge
            # stop one pixel beforhand, just in case if you do some sort of dithering
            if xposmin<1:
                xposmin=1
            if xposmax>4095:
                xposmax=4095
                
       
            if self.verbosity==1:            
                print('initial cut for image_index '+str(self.image_index)+' - xposmin,xposmax,yposmin,yposmax: '+str([xposmin,xposmax,yposmin,yposmax]))                  
    
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            maskdata_cut=maskdata[yposmin:yposmax,xposmin:xposmax]
            vardata_cut=vardata[yposmin:yposmax,xposmin:xposmax]
            
                    
            return scidata_cut,maskdata_cut,vardata_cut,xposmin,xposmax,yposmin,yposmax
        else:
            # if you passed explicit values, just use those
            scidata_cut=scidata[yposmin:yposmax,xposmin:xposmax]
            maskdata_cut=maskdata[yposmin:yposmax,xposmin:xposmax]
            vardata_cut=vardata[yposmin:yposmax,xposmin:xposmax]
            
            # do not return the input coordinates, to avoid possible confusion
            return scidata_cut,maskdata_cut,vardata_cut
            

    def cut_final_image(self,scidata_cut_large_removed,maskdata_cut_large_removed,vardata_cut_large_removed,size_of_stamp,mean_kernel_size,\
                        y_lower_limit_for_final_cut=None,y_upper_limit_for_final_cut=None,x_lower_limit_for_final_cut=None,x_upper_limit_for_final_cut=None):
    
        """! final step in cutting the image - returns science, mask and variance cuts  
        
        
        @param scidata_cut_large_removed             science data with median correction already applied 
        @param maskdata_cut_large_removed            mask to cover science data 
        @param vardata_cut_large_removed             vardata 
        @param size_of_stamp                         size of stamp that we wish to create
        @param mean_kernel_size                      size of averaging kernel, in pixel, when estimating the cetnering
        @param y_lower_limit_for_final_cut,\
        y_upper_limit_for_final_cut,\
        x_lower_limit_for_final_cut,\
        x_upper_limit_for_final_cut                  explicit positions where to cut, overriding the centering             
        """       
            
        #assert that if one y_x_upper_lower values is None that all of them are None 
         
        x_min_for_scidata_cut_large_removed=int(scidata_cut_large_removed.shape[1]/2-size_of_stamp/2-20)
        x_max_for_scidata_cut_large_removed=int(scidata_cut_large_removed.shape[1]/2+size_of_stamp/2+20)
        y_min_for_scidata_cut_large_removed=int(scidata_cut_large_removed.shape[0]/2-size_of_stamp/2-5)
        y_max_for_scidata_cut_large_removed=int(scidata_cut_large_removed.shape[0]/2+size_of_stamp/2+5)
    
        scidata_cut_large_removed_prestep=scidata_cut_large_removed[y_min_for_scidata_cut_large_removed:y_max_for_scidata_cut_large_removed,x_min_for_scidata_cut_large_removed:x_max_for_scidata_cut_large_removed]
        maskdata_cut_large_removed_prestep=maskdata_cut_large_removed[y_min_for_scidata_cut_large_removed:y_max_for_scidata_cut_large_removed,x_min_for_scidata_cut_large_removed:x_max_for_scidata_cut_large_removed]   
        vardata_cut_large_removed_prestep=vardata_cut_large_removed[y_min_for_scidata_cut_large_removed:y_max_for_scidata_cut_large_removed,x_min_for_scidata_cut_large_removed:x_max_for_scidata_cut_large_removed]
    
        #y_pos_for_final_cut=np.round(np.median(np.where(np.max(scidata_cut_large_removed_prestep*np.abs((1-maskdata_cut_large_removed_prestep)),axis=1)>300)))
        #x_pos_for_final_cut=np.round(np.median(np.where(np.max(scidata_cut_large_removed_prestep*np.abs((1-maskdata_cut_large_removed_prestep)),axis=0)>300)))
        
        if y_lower_limit_for_final_cut is None:
    
            size_of_stamp=int(size_of_stamp)
            y_profile=np.sum(scidata_cut_large_removed_prestep*np.abs((1-maskdata_cut_large_removed_prestep)),axis=1)
            y_profile_running_mean=running_mean(y_profile, mean_kernel_size)
            y_pos_for_final_cut=int(np.argmax(y_profile_running_mean)+mean_kernel_size/2)
    
            x_profile=np.sum(scidata_cut_large_removed_prestep*np.abs((1-maskdata_cut_large_removed_prestep)),axis=0)
            x_profile_running_mean=running_mean(x_profile, mean_kernel_size)
            x_pos_for_final_cut=int(np.argmax(x_profile_running_mean)+mean_kernel_size/2)
    
            y_lower_limit_for_final_cut=int(y_pos_for_final_cut-size_of_stamp/2)
            y_upper_limit_for_final_cut=int(y_pos_for_final_cut+size_of_stamp/2)
            x_lower_limit_for_final_cut=int(x_pos_for_final_cut-size_of_stamp/2)
            x_upper_limit_for_final_cut=int(x_pos_for_final_cut+size_of_stamp/2)
    
            print('y_x_lower_upper positions in the final cut: '+str([y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,\
                                                                      x_lower_limit_for_final_cut,x_upper_limit_for_final_cut]))
    
            scidata_final_cut=scidata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,\
                                                                x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            maskdata_final_cut=maskdata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,\
                                                                  x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            vardata_final_cut=vardata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,\
                                                                x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            return scidata_final_cut,maskdata_final_cut,vardata_final_cut,\
            y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut
        else:

            scidata_final_cut=scidata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            maskdata_final_cut=maskdata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            vardata_final_cut=vardata_cut_large_removed_prestep[y_lower_limit_for_final_cut:y_upper_limit_for_final_cut,x_lower_limit_for_final_cut:x_upper_limit_for_final_cut]
            # if explicit values passed, return only cut images
            return scidata_final_cut,maskdata_final_cut,vardata_final_cut
                
            
    def create_poststamps(self,exposure_defocus_explicit=None,run_explicit=None):            
        """! master function that creates poststamps

         """   

        if run_explicit==None:
            run_output=self.run_0
        else:
            run_output=run_explicit        
        # first part is to stack the images and gather information
        # import initial image and gather information 
        # if there is no dithering, do it just once       
        if self.dither==None or self.dither==1:
            scidata00,maskdata00,vardata00,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(run_output)
        
        # if there is dithering, do it for every position of the hexapod               
        if self.dither==4:
            scidata00,maskdata00,vardata00,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0)
            scidata0p,maskdata0p,vardata0p,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+12)
            scidatap0,maskdatap0,vardatap0,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+6)
            scidatapp,maskdatapp,vardatapp,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+18)
        if self.dither==9:
            scidatamm,maskdatamm,vardatamm,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0)
            scidatam0,maskdatam0,vardatam0,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+6)
            scidatamp,maskdatamp,vardatamp,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+12)
            
            scidata0m,maskdata0m,vardata0m,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+18)            
            scidata00,maskdata00,vardata00,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+24)
            scidata0p,maskdata0p,vardata0p,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+30)

            scidatapm,maskdatapm,vardatapm,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+36)
            scidatap0,maskdatap0,vardatap0,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+42)
            scidatapp,maskdatapp,vardatapp,exposure_arc,exposure_defocus=self.stack_images_and_gather_information(self.run_0+48)                                                                                                                  
    
        if exposure_defocus is None:
            exposure_defocus=exposure_defocus_explicit
    
        if self.verbosity==1:
            print('Successfully stacked images')
            print('exposure_arc is '+str(exposure_arc))
            print('exposure_defocus is '+str(exposure_defocus))
            print('dithering is '+str(self.dither))
        
        Zernike_info_df=self.Zernike_info_df
        
        # load Zernike_info_df files, contaning information about the lines we want to cut
        #if exposure_arc=='HgAr':
        #    Zernike_info_df=np.load('/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Dataframes/finalHgAr_May2019.pkl',allow_pickle=True)        
        # no special dataframe for Argon only (yet?)
        #if exposure_arc=='Ar':
        #    Zernike_info_df=np.load('/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Dataframes/finalHgAr_May2019.pkl',allow_pickle=True)
        #if exposure_arc=='Ne':
        #    Zernike_info_df=np.load('/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Dataframes/finalNe_May2019.pkl',allow_pickle=True) 
        #if exposure_arc=='Kr':
        #    Zernike_info_df=np.load('/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Dataframes/finalKr_May2019.pkl',allow_pickle=True)      
            
        
        # values of defocus that we analyse, and sizes of stamps
        # does not support fine_defocus!!!
        defocus_values=np.array([-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,+1,1.5,2,2.5,3,3.5,4,4.5])
        sizes_of_stamps=np.array([60, 60, 50, 40, 30, 30, 24, 20, 20, 20, 24, 30, 30, 40, 50, 60,60, 70])    
        
        # Size of stamp that we wish to create 
        # set it here if in focus
        if np.abs(exposure_defocus)==0:
            size_of_stamp=20
        # if in defocus take it from the list 
        else:
            size_of_stamp=sizes_of_stamps[defocus_values==exposure_defocus]
           
        self.size_of_stamp=size_of_stamp   
        if self.verbosity==1:
            print('size_of_stamp is '+str(size_of_stamp))        
        #list of fiber_IDs - for the standard 10 fiber configuration   
        # will need updating to handle more flexible configurations
        list_of_fiber_IDs=list(np.unique(Zernike_info_df['fiber'].values))
        
        if self.dither is not None:
            pos_4_overview=[]
           
        # now run the cutting procedure on the stack, for each initial position   
        for image_index in tqdm(range(len(Zernike_info_df))):
            self.image_index=image_index
            try:
                # central position as deduced from focused images/detectormap
                center_pos=[int(np.array(Zernike_info_df.loc[image_index,['xc','yc']].values)[0]),int(np.array(Zernike_info_df.loc[image_index,['xc','yc']].values)[1])]
                if self.verbosity==1:
                    print('exposure_defocus is '+str(exposure_defocus)+': Central position of the spot '+str(image_index)+' is: '+str(center_pos))
    
                # old code, before transforming
                # run centering algorithm - initial estimate of the centered position
                #center_pos_new=self.centering_algorithm(scidata,center_pos[0],center_pos[1],size_of_stamp)
                #print('checkpoint 1')
                #print(scidata00.shape)
                #print(maskdata00.shape)
                #print(vardata00.shape)
                # intial cut, pure centering and before removing continuum
                scidata_cut,maskdata_cut,vardata_cut,xposmin,xposmax,yposmin,yposmax=self.cut_initial_image(scidata00,maskdata00,vardata00,center_pos[0],center_pos[1],size_of_stamp)
                #print('checkpoint 2')
                # if dithering, applying the same x and y coordinates to the dithered images
                if self.dither==4:
                    #scidata00_cut,maskdata00_cut,vardata00_cut=scidata_cut,maskdata_cut,vardata_cut
                   
                    scidata0p_cut,maskdata0p_cut,vardata0p_cut=\
                    self.cut_initial_image(scidata0p,maskdata0p,vardata0p,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)
                    scidatap0_cut,maskdatap0_cut,vardatap0_cut=\
                    self.cut_initial_image(scidatap0,maskdatap0,vardatap0,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)
                    scidatapp_cut,maskdatapp_cut,vardatapp_cut=\
                    self.cut_initial_image(scidatapp,maskdatapp,vardatapp,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)

                if self.dither==9:
                    #scidata00_cut,maskdata00_cut,vardata00_cut=scidata_cut,maskdata_cut,vardata_cut

                    scidatamm_cut,maskdatamm_cut,vardatamm_cut=\
                    self.cut_initial_image(scidatamm,maskdatamm,vardatamm,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax) 
                    scidatam0_cut,maskdatam0_cut,vardatam0_cut,exposure_arc,exposure_defocus=\
                    self.cut_initial_image(scidatam0,maskdatam0,vardatam0,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax) 
                    scidatamp_cut,maskdatamp_cut,vardatamp_cut=\
                    self.cut_initial_image(scidatamp,maskdatamp,vardatamp,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax) 
                    
                    scidata0m_cut,maskdata0m_cut,vardata0m_cut=\
                    self.cut_initial_image(scidata0m,maskdata0m,vardata0m,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)               
                    scidata0p_cut,maskdata0p_cut,vardata0p_cut=\
                    self.cut_initial_image(scidata0p,maskdata0p,vardata0p,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)   
        
                    scidatapm_cut,maskdatapm_cut,vardatapm_cut=\
                    self.cut_initial_image(scidatapm,maskdatapm,vardatapm,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)   
                    scidatap0_cut,maskdatap0_cut,vardatap0_cut=\
                    self.cut_initial_image(scidatap0,maskdatap0,vardatap0,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)                   
                    scidatapp_cut,maskdatapp_cut,vardataoo_cut=\
                    self.cut_initial_image(scidatapp,maskdatapp,vardatapp,center_pos[0],center_pos[1],size_of_stamp,xposmin,xposmax,yposmin,yposmax)


                
                # assigned, but unused - initial cut before anything, for debugging 
                # scidata_cut_original=np.copy(scidata_cut)
    
                
                ############################
                # removal of continuum 
                ############################     
                
                if self.subtract_continuum==True:
                    
                    #which fiber is being analyzed
                    which_fiber_starting_at_0=list_of_fiber_IDs.index(Zernike_info_df.loc[image_index]['fiber'])
        
                    # load appropriate fits
                    # where was this created?
                    array_of_fit_defocus=np.load('/Users/nevencaplar/Documents/PFS/Testing/Fibers/array_of_fit_defocus_'+str(which_fiber_starting_at_0)+'_f25.npy')
                    array_of_good=np.load('/Users/nevencaplar/Documents/PFS/Testing/Fibers/array_of_good_'+str(which_fiber_starting_at_0)+'_f25.npy')            
                    
                    # index_defocus is a number that goes between 0 and 18 - gives index of the defocused image
                    # 0 corresponds to defocus of -4.5 mm
                    # 18 corresponds to defocus of +4.5 mm 
                    index_defocus=np.where(np.arange(-4.5,4.6,0.5)==self.exposure_defocus)[0][0]
                    # get appropriate fit, depening on the defocus
                    y_median=array_of_fit_defocus[index_defocus]
                    good=array_of_good[index_defocus] 
        
                    # apply the removal algorithm
                    scidata_cut_large_removed,maskdata_cut_large_removed,vardata_cut_large_removed=\
                    self.removal_of_continuum(scidata_cut,maskdata_cut,vardata_cut,y_median,good,yposmin,yposmax)
                
                    # overload the definitions and assings the data with continuum removed to the previous data names
                    scidata_cut,maskdata_cut,vardata_cut=scidata_cut_large_removed,maskdata_cut_large_removed,vardata_cut_large_removed
                    
                    # add the support for dithering
                    if self.dither==4:
                        print('not implemented yet!')
                    if self.dither==9:
                        print('not implemented yet!')      
                        
                ############################ 
                # cutting final image
                ############################ 
                
                
                # mean kernel size which should change as a function of defocus
                # mean_kernel_size empirically should be around 5 in the focus
                # and around the size of the full image for fully defocused (+-4 mm of the slit) images
                mean_kernel_size=int(40*np.abs(exposure_defocus/4)+5)
                size_of_stamp=int(size_of_stamp)
    
                # apply the final cut and centering algorithm
                # this allows to slightly shift image to better capture faint areas of a donut
                scidata_final_cut,maskdata_final_cut,vardata_final_cut,\
                y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut=\
                self.cut_final_image(scidata_cut,maskdata_cut,vardata_cut,size_of_stamp,mean_kernel_size)
                
                if self.dither==4:
                    
                    scidata00_final_cut,maskdata00_final_cut,vardata00_final_cut=scidata_final_cut,maskdata_final_cut,vardata_final_cut
                    
                    scidata0p_final_cut,maskdata0p_final_cut,vardata0p_final_cut=\
                    self.cut_final_image(scidata0p_cut,maskdata0p_cut,vardata0p_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatap0_final_cut,maskdatap0_final_cut,vardatap0_final_cut=\
                    self.cut_final_image(scidatap0_cut,maskdatap0_cut,vardatap0_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatapp_final_cut,maskdatapp_final_cut,vardatapp_final_cut=\
                    self.cut_final_image(scidatapp_cut,maskdatapp_cut,vardatapp_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)  
                    
                    
                    #self.create_dither_report(self.dither)
                    
                    #sci_image_dithered_pp,mask_image_dithered_pp,var_image_dithered_pp=\
                    scidata_final_cut,maskdata_final_cut,vardata_final_cut=\
                        self.combine_dither_images(scidata00_final_cut,scidatapp_final_cut,scidatap0_final_cut,scidata0p_final_cut,\
                        maskdata00_final_cut,maskdatapp_final_cut,maskdatap0_final_cut,maskdata0p_final_cut,\
                        vardata00_final_cut,vardatapp_final_cut,vardatap0_final_cut,vardata0p_final_cut)
                    
                    pos00=np.array(find_centroid_of_flux(scidata00_final_cut))
                    pos0p=np.array(find_centroid_of_flux(scidata0p_final_cut))-pos00
                    posp0=np.array(find_centroid_of_flux(scidatap0_final_cut))-pos00
                    pospp=np.array(find_centroid_of_flux(scidatapp_final_cut))-pos00
                    pos_4=np.array([pos00,pos0p,posp0,pospp])
                    
                    pos_4_overview.append(pos_4)
                    
                    
                if self.dither==9:
                    
                    # expand to all possible combinations, at the moment only one in the upper-right corner
                    scidata00_final_cut,maskdata00_final_cut,vardata00_final_cut=scidata_final_cut,maskdata_final_cut,vardata_final_cut
                    
                    scidata0p_final_cut,maskdata0p_final_cut,vardata0p_final_cut=\
                    self.cut_final_image(scidata0p_cut,maskdata0p_cut,vardata0p_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatap0_final_cut,maskdatap0_final_cut,vardatap0_final_cut=\
                    self.cut_final_image(scidatap0_cut,maskdatap0_cut,vardatap0_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatapp_final_cut,maskdatapp_final_cut,vardatapp_final_cut=\
                    self.cut_final_image(scidatapp_cut,maskdatapp_cut,vardatapp_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)  
                    
                    scidata0m_final_cut,maskdata0m_final_cut,vardata0m_final_cut=\
                    self.cut_final_image(scidata0m_cut,maskdata0m_cut,vardata0m_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatam0_final_cut,maskdatam0_final_cut,vardatam0_final_cut=\
                    self.cut_final_image(scidatam0_cut,maskdatam0_cut,vardatam0_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatamm_final_cut,maskdatamm_final_cut,vardatamm_final_cut=\
                    self.cut_final_image(scidatamm_cut,maskdatamm_cut,vardatamm_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)  
                    
                    scidatapm_final_cut,maskdatapm_final_cut,vardatapm_final_cut=\
                    self.cut_final_image(scidatapm_cut,maskdatapm_cut,vardatapm_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)
                    scidatamp_final_cut,maskdatamp_final_cut,vardatamp_final_cut=\
                    self.cut_final_image(scidatamp_cut,maskdatamp_cut,vardatamp_cut,size_of_stamp,mean_kernel_size,\
                                    y_lower_limit_for_final_cut,y_upper_limit_for_final_cut,x_lower_limit_for_final_cut,x_upper_limit_for_final_cut)

                    
                    pos00=np.array(find_centroid_of_flux(scidata00_final_cut))
                    
                    pos0p=np.array(find_centroid_of_flux(scidata0p_final_cut))-pos00
                    posp0=np.array(find_centroid_of_flux(scidatap0_final_cut))-pos00
                    pospp=np.array(find_centroid_of_flux(scidatapp_final_cut))-pos00
                    
                    pos0m=np.array(find_centroid_of_flux(scidata0m_final_cut))-pos00
                    posm0=np.array(find_centroid_of_flux(scidatam0_final_cut))-pos00
                    posmm=np.array(find_centroid_of_flux(scidatamm_final_cut))-pos00
                    
                    pospm=np.array(find_centroid_of_flux(scidatapm_final_cut))-pos00
                    posmp=np.array(find_centroid_of_flux(scidatamp_final_cut))-pos00
                    
                    pos_4=np.array([pos00,pos0p,posp0,pospp,pos0m,posm0,posmm,pospm,posmp])
                    
                    pos_4_overview.append(pos_4)
                    
                    scidata_final_cut,maskdata_final_cut,vardata_final_cut=\
                        self.combine_dither_images(scidata00_final_cut,scidatapp_final_cut,scidatap0_final_cut,scidata0p_final_cut,\
                        maskdata00_final_cut,maskdatapp_final_cut,maskdatap0_final_cut,maskdata0p_final_cut,\
                        vardata00_final_cut,vardatapp_final_cut,vardatap0_final_cut,vardata0p_final_cut)
                    
 
                if run_explicit==None:
                    run_output=self.run_0
                else:
                    run_output=run_explicit
                
                
                # save the output
                # need to implement custom date
                if self.verbosity==1:
                    print('image with index '+str(self.image_index)+' seems successful')
                    print('saving at: '+PSF_DIRECTORY_CUTS+"sci"+str(run_output)+str(image_index)+exposure_arc+'_Stacked.npy')
                    print('######################################################')
                
                if self.save==True:

                    np.save(PSF_DIRECTORY_CUTS+"sci"+str(run_output)+str(image_index)+exposure_arc+'_Stacked.npy',scidata_final_cut)
                    np.save(PSF_DIRECTORY_CUTS+"mask"+str(run_output)+str(image_index)+exposure_arc+'_Stacked.npy',maskdata_final_cut)
                    np.save(PSF_DIRECTORY_CUTS+"var"+str(run_output)+str(image_index)+exposure_arc+'_Stacked.npy',vardata_final_cut)    
                    

                        
                        
                    
            except Exception as e:
                if self.verbosity==1:
                    print(e)
                    print('image with index '+str(self.image_index)+' failed')
                    print('######################################################')
                pass


        if self.dither is not None and self.dither!=1:
            pos_4_overview=np.array(pos_4_overview)
            if self.save==True:
                np.save(PSF_DIRECTORY_CUTS+"sci"+str(self.run_0)+'_pos_4_overview.npy',pos_4_overview)

                if self.verbosity==1:
                    print('pos_4_overview saved at '+PSF_DIRECTORY_CUTS+"sci"+str(self.run_0)+'_pos_4_overview.npy') 
                    
            pos_4_overview_quantiles=[]
            #print(pos_4_overview.shape)
            for i in [1,2,3]:
                pos_4_overview_quantiles.append(np.concatenate(([np.quantile(pos_4_overview[:,i][:,0],[0.5,0.16,0.84]),np.quantile(pos_4_overview[:,i][:,1],[0.5,0.16,0.84])])))
                
            pos_4_overview_quantiles=np.array(pos_4_overview_quantiles)
            
            # transforming x quantiles in x errors 
            pos_4_overview_quantiles[:,1]=(-pos_4_overview_quantiles[:,0]+pos_4_overview_quantiles[:,1])*(-1)
            pos_4_overview_quantiles[:,2]=-pos_4_overview_quantiles[:,0]+pos_4_overview_quantiles[:,2]
            
            # transforming y quantiles in y errors 
            pos_4_overview_quantiles[:,4]=(-pos_4_overview_quantiles[:,3]+pos_4_overview_quantiles[:,4])*(-1)
            pos_4_overview_quantiles[:,5]=-pos_4_overview_quantiles[:,3]+pos_4_overview_quantiles[:,5]
            
            
            
            if self.save==True:
                np.save(PSF_DIRECTORY_CUTS+"sci"+str(self.run_0)+'_pos_4_overview_quantiles.npy',pos_4_overview_quantiles)

                if self.verbosity==1:
                    print('pos_4_overview_quantiles saved at '+PSF_DIRECTORY_CUTS+"sci"+str(self.run_0)+'_pos_4_overview_quantiles.npy') 
                    
                    
        """   
        # Example code for plotting the result below!

                 
        plt.figure(figsize=(11,11))
        
        plt.errorbar(x=pos_4_overview_quantiles[:,0],y=pos_4_overview_quantiles[:,3],\
                     xerr=np.transpose(pos_4_overview_quantiles[:,1:3]),yerr=np.transpose(pos_4_overview_quantiles[:,4:6]),\
                     elinewidth=3,ls='',marker='o',capsize=3,label='label here',color='black')
        
        
        plt.axes().set_aspect('equal', 'datalim')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35))
        
        plt.axhline(-0.5,color='gray',ls='--')
        plt.axhline(0,color='gray',ls='--')
        plt.axhline(+0.5,color='gray',ls='--')
        
        plt.axvline(-0.5,color='gray',ls='--')
        plt.axvline(0,color='gray',ls='--')
        plt.axvline(0.5,color='gray',ls='--')
        
        
        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
        plt.xlabel('Delta x pixels')
        plt.ylabel('Delta y pixels')
        """                    
      
       # plot the result out of many pos_4             

####################################
# Free standing code
####################################

def create_Zernike_info_df(pfsDetectorMap,arc,DATA_FOLDER,obs,pfsConfig=None):
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
    
    
   
    if pfsConfig is None:
        fibers_10=np.flip(np.array([  2,  63, 192, 255, 339, 401, 464, 525, 587, 650])  ) 
        
        
    if arc=='HgAr' or arc=='Ar':
        wavelengths_arc=['690.9346', '696.7261', '706.8989', '727.47876',\
                           '738.6248','763.74286', '795.0522', '826.6699',\
                           '852.4029', '912.5693','922.7301', '966.0642']
        close_arc=['0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1']
        
        second_offset=[-19.25, 0, 16.93, 0, -13.14, -38.59, 0, 18.83, -14.26, 0, 22.44, 0]
        second_flux=[0.0034, 0, 0.268, 0, 0.011, 0.008, 0, 0.0043, 0.015, 0, 0.022, 0]
        second2_offset=[-40.7, 0, 28.33, 0, 0, 0, 0, 0, 0, 0, -13.14, 0]
        second2_flux=[0.006, 0, 0.088, 0, 0, 0, 0, 0, 0, 0, 0.011, 0]
    if arc=='Ne':
        wavelengths_arc=['650.84186' ,'653.4774' ,'660.0668' ,'668.01764',\
                         '671.9268', '693.13116','717.59015', '724.72437' ,'744.1276']
        close_arc=['0', '-1', '1', '0', '-1' ,'1' ,'1' ,'1' ,'0']
        
        second_offset=[30.12, 0 ,0 ,44.24, 0, 0 ,0 ,0, 38.57]
        second_flux=[0.406, 0 ,0 ,0.67, 0, 0 ,0, 0, 0.012]
        second2_offset=[0, 0 ,0 ,-29.97, 0, 0, 0, 0, 57.44]
        second2_flux=[0, 0 ,0 ,0.003, 0, 0, 0, 0 ,0.162]
    if arc=='Kr':
        wavelengths_arc=[785.48, 819.0, 877.68, 892.87]
        close_arc=[0, 0, 0, 1]
        
        second_offset=[8, 20, -15, 0]
        second_flux=[0.01, 0.004, 0.004, 0]
        second2_offset=[0, 0, 10 ,0]
        second2_flux=[0 ,0 ,0.003, 0]



        
    list_of_actual_fibers=[]
    list_of_yc=[]
    list_of_xc=[]
    list_of_xc_eff=[]
    list_of_wavelength=[]
    list_of_close=[]
    list_of_lamp=[]
    list_of_second_offset=[]
    list_of_second_ratio=[]
    list_of_second2_offset=[]
    list_of_second2_ratio=[]
    
    x_pfsDetectorMap=np.array(pfsDetectorMap[3].data)
    y_pfsDetectorMap=np.array(pfsDetectorMap[4].data)
    
    for i in range(len(fibers_10)*len(wavelengths_arc)):
        
        # fiber index, sequenetial as they appear on the detector, i.e., starting at 0 and going +1
        index_of_single_fiber_sequential=int(np.floor(i/len(wavelengths_arc)))
        # actual number of the fiber (i.e., going up to 650)
        actual_fiber=int(fibers_10[index_of_single_fiber_sequential])
        # index of a fiber, as they go through the detectorMap (i.e., going up to 599)
        index_of_single_fiber=np.where(pfsDetectorMap[1].data==actual_fiber)[0][0]
    
        
        # add the information about the actual fiber in the dataframe
        #Zernike_info_df.loc[i]['fiber']=actual_fiber
        list_of_actual_fibers.append(actual_fiber)
        

        
        
        # select the parts of the detectorMap which correspond to that particular fiber
        x_pfsDetectorMap_single_fiber=x_pfsDetectorMap[x_pfsDetectorMap['index']==index_of_single_fiber]
        y_pfsDetectorMap_single_fiber=y_pfsDetectorMap[y_pfsDetectorMap['index']==index_of_single_fiber]
        
        # first find the y position
        #Zernike_info_df.loc[i]['yc']=int(x_pfsDetectorMap_single_fiber['knot'][np.where(y_pfsDetectorMap_single_fiber['value']==\
        #                                                                                find_nearest(y_pfsDetectorMap_single_fiber['value'],float(wavelengths_HgAr[np.mod(i,len(fibers_10))])))[0][0]])
        

        y_value=int(y_pfsDetectorMap_single_fiber['knot'][np.where(y_pfsDetectorMap_single_fiber['value']==\
                find_nearest(y_pfsDetectorMap_single_fiber['value'],float(wavelengths_arc[np.mod(i,len(wavelengths_arc))])))[0][0]])
        x_value=int(x_pfsDetectorMap_single_fiber['value'][np.where(x_pfsDetectorMap_single_fiber['knot']==y_value)])
        
        if x_value>2048:
            x_value_eff=x_value+69
        else:
            x_value_eff=x_value
        
        list_of_yc.append(y_value)
        
        # from y position, deduce x position
        #Zernike_info_df.loc[i]['xc']=int(x_pfsDetectorMap_single_fiber['value'][np.where(x_pfsDetectorMap_single_fiber['knot']==Zernike_info_df.loc[i]['yc'])])
        list_of_xc.append(x_value)
        
        list_of_xc_eff.append(x_value_eff)
        
        #Zernike_info_df.loc[i]['wavelength']=wavelengths_HgAr[np.mod(i,len(fibers_10))]
        list_of_wavelength.append(wavelengths_arc[np.mod(i,len(wavelengths_arc))])
        #Zernike_info_df.loc[i]['close']=close_HgAr[np.mod(i,len(fibers_10))]
        list_of_close.append(close_arc[np.mod(i,len(wavelengths_arc))])
        #Zernike_info_df.loc[i]['lamp']='HgAr'
        list_of_lamp.append(arc)
        
        #Zernike_info_df.loc[i]['second_offset']=second_offset[np.mod(i,len(fibers_10))]
        list_of_second_offset.append(second_offset[np.mod(i,len(wavelengths_arc))])
        #Zernike_info_df.loc[i]['second_flux']=second_flux[np.mod(i,len(fibers_10))]
        list_of_second_ratio.append(second_flux[np.mod(i,len(wavelengths_arc))])
        #Zernike_info_df.loc[i]['second2_offset']=second2_offset[np.mod(i,len(fibers_10))]
        list_of_second2_offset.append(second2_offset[np.mod(i,len(wavelengths_arc))])
        #Zernike_info_df.loc[i]['second2_flux']=second2_flux[np.mod(i,len(fibers_10))]
        list_of_second2_ratio.append(second2_flux[np.mod(i,len(wavelengths_arc))])    
        
        
    
    #for i in range(len(fibers_10)*len(wavelengths_HgAr)):
    #    if Zernike_info_df.loc[i]['xc']<2048:
    #        Zernike_info_df.loc[i]['xc_effective']=Zernike_info_df.loc[i]['xc']
    #    else:
    #        Zernike_info_df.loc[i]['xc_effective']=Zernike_info_df.loc[i]['xc']+69
    
    
    Zernike_info_df=pd.DataFrame({'fiber':list_of_actual_fibers,'xc':list_of_xc,'yc':list_of_yc,'wavelength':list_of_wavelength,'close':list_of_close,\
                                         'lamp':list_of_lamp,'xc_effective':list_of_xc_eff,'second_offset':list_of_second_offset,'second_ratio':list_of_second_ratio,\
                                          'second2_offset':list_of_second2_offset, 'second2_ratio':list_of_second2_ratio})
    
    if not os.path.exists(DATA_FOLDER+"Dataframes/"):
        os.makedirs(DATA_FOLDER+"Dataframes/")
    Zernike_info_df.to_pickle(DATA_FOLDER+"Dataframes/Zernike_info_df_"+str(arc)+'_'+str(obs))              
    
    # For example, load with:
    # np.load('/Users/nevencaplar/Documents/PFS/ReducedData/Data_Jun_7_2020/Dataframes/Zernike_info_df_HgAr_021604',allow_pickle=True)
    
def create_dither_plot(list_of_pos_4_quantiles,list_of_labels):

    assert len(list_of_pos_4_quantiles)<10, print('Up to 10 inputs only')
    
    
    plt.figure(figsize=(11,11))
    for i in range(len(list_of_pos_4_quantiles)):
        pos_4_quantiles_i=list_of_pos_4_quantiles[i]
        label_i=list_of_labels[i]
        plt.errorbar(x=pos_4_quantiles_i[:,0],y=pos_4_quantiles_i[:,3],\
                     xerr=np.transpose(pos_4_quantiles_i[:,1:3]),yerr=np.transpose(pos_4_quantiles_i[:,4:6]),\
                     elinewidth=3,ls='',marker='o',capsize=2,label=label_i)
        

    
    plt.axes().set_aspect('equal', 'datalim')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
    
    plt.axhline(-0.5,color='gray',ls='--')
    plt.axhline(0,color='gray',ls='--')
    plt.axhline(+0.5,color='gray',ls='--')
    
    plt.axvline(-0.5,color='gray',ls='--')
    plt.axvline(0,color='gray',ls='--')
    plt.axvline(0.5,color='gray',ls='--')
    
    
    plt.xlim(-0.75,0.75)
    plt.ylim(-0.75,0.75)
    plt.xlabel('Delta x pixels')
    plt.ylabel('Delta y pixels')    

    
    
def find_centroid_of_flux(image):
    """
    function giving the tuple of the position of weighted average of the flux in a square image
    
    @param image    poststamp image for which to find center
    """
    
    
    x_center=[]
    y_center=[]

    I_x=[]
    for i in range(len(image)):
        I_x.append([i,np.sum(image[:,i])])

    I_x=np.array(I_x)

    I_y=[]
    for i in range(len(image)):
        I_y.append([i,np.sum(image[i])])

    I_y=np.array(I_y)


    x_center=(np.sum(I_x[:,0]*I_x[:,1])/np.sum(I_x[:,1]))
    y_center=(np.sum(I_y[:,0]*I_y[:,1])/np.sum(I_y[:,1]))

    return(x_center,y_center)

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
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves
