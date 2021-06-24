"""
First created on Mon Aug 13 10:01:03 2018

Main code for the creation of the image for Zernike analysis;
Other moduls avaliable are:
    Zernike_Cutting_Module
    Zernike_Analysis_Module


Versions:
Oct 31, 2018; 0.1 -> 0.11 fixed FRD effect
Nov 1, 2018; 0.11 -> 0.12 added correct edges to the detector; fixed wrong behavior for misaligment 
Nov 2, 2018; 0.12 -> 0.13 added lorentzian wings to the illumination of the pupil
Nov 3, 2018; 0.13 -> 0.13b fixed edges of detector when det_vert is not 1
Nov 12, 2018; 0.13b -> 0.13c changed parameter describing hexagonal effect "f" from 0.1 to 0.2
Nov 12, 2018; 0.13c -> 0.14 changed illumination description modifying entrance -> exit pupil illumination
Nov 29, 2018; 0.14 -> 0.14b added fixed scattering slope, deduced from large image in focus
Dec 16, 2018; 0.14b -> 0.14c allparameters_proposal_err from list to array
Dec 18, 2018; 0.14c -> 0.14d strutFrac upper limit to 0.13 in create_parInit
Dec 23, 2018; 0.14d -> 0.15 refactoring so that x_ilum and y_ilum is one
Dec 26, 2018; 0.15 -> 0.15b when in focus, create exactly 10x oversampling
Dec 31, 2018; 0.15b -> 0.16 major rewrite of downsampling algorithm
Jan 8, 2019; 0.16 -> 0.17 added support for zmax=22
Jan 14, 2019; 0.17 -> 0.18 fixed bug with dowsampling algorithm - I was just taking central values
Jan 15, 2019; 0.18 -> 0.19 added simple algorithm to interpolate between 1/10 pixels in the best position
Feb 15, 2019; 0.19 -> 0.20 updated analysis for the new data
Feb 21, 2019; 0.20 -> 0.20b test parameter for showing globalparamers outside their limits
Feb 22, 2019; 0.20 -> 0.21 added support for Zernike higher than 22
Feb 22, 2019; 0.21 -> 0.21b added support for return image along side likelihood
Apr 17, 2019; 0.21b -> 0.21c changed defintion of residuals from (model-data) to (data-model)
Jun 4, 2019; 0.21c -> 0.21d slight cleaning of the code, no functional changes
Jun 26, 2019; 0.21d -> 0.21e included variable ``dataset'', which denots which data we are using in the analysis
Jul 29, 2019; 0.21e -> 0.21f changed the spread of paramters when drawing initial solutions, based on data
Sep 11, 2019; 0.21f -> 0.21g globalparameters_flat_6<1 to globalparameters_flat_6<=1
Oct 10, 2019: 0.21g -> 0.21h scattered_light_kernel saving option
Oct 31, 2019: 0.21h -> 0.22 (re)introduced small amount of apodization (PIPE2D-463)
Oct 31, 2019: 0.22 -> 0.22b introduced verbosity
Nov 07, 2019: 0.22b -> 0.22c nan values can pass through find_single_realization_min_cut
Nov 08, 2019: 0.22c -> 0.22d changes to resizing and centering 
Nov 13, 2019: 0.22d -> 0.23 major changes to centering - chief ray in the center of oversampled image
Nov 15, 2019: 0.23 -> 0.24 change likelihood definition
Dec 16, 2019: 0.24 -> 0.24a added iluminaton with z4,z11,z22=0
Jan 14, 2020: 0.24a -> 0.24b added verbosity in find_single_realization_min_cut function
Jan 31, 2020: 0.24b -> 0.25 added support for data contaning spots from two wavelengths
Feb 11, 2020: 0.25 -> 0.26 proper bilinear interpolation of the spots
Feb 17, 2020: 0.26 -> 0.26a increased speed when save parameter=0
Feb 18, 2020: 0.26a -> 0.26b mask image going through subpixel interpolation
Feb 19, 2020: 0.26b -> 0.26c normalization of sci image takes into account mask
Mar 1, 2020: 0.26c -> 0.27 apodization scales with the size of input images
Mar 4, 2020: 0.27 -> 0.28 (re-)introduced custom size of pupil image
Mar 6, 2020: 0.28 -> 0.28b refactored cut_square function (making it much faster)
Mar 8, 2020: 0.28b -> 0.28c set limit in grating factor to 120000 in generating code
Apr 1, 2020: 0.28c -> 0.28d svd_invert function
May 6, 2020: 0.28d -> 0.28e clarified and expanded comments in postprocessing part
Jun 28, 2020: 0.28e -> 0.29 added multi analysis
Jul 02, 2020: 0.29 -> 0.30 added internal fitting for flux
Jul 02, 2020: 0.30 -> 0.30a lnlike_Neven_multi_same_spot can accept both 1d and 2d input 
Jul 07, 2020: 0.30a -> 0.30b added threading time information
Jul 09, 2020: 0.30b -> 0.30c expwf_grid changed to complex64 from complex128
Jul 09, 2020: 0.30c -> 0.30d changed all float64 to float32
Jul 16, 2020: 0.30d -> 0.31 moved all fft to scipy.signal.fftconvolve
Jul 20, 2020: 0.31 -> 0.32 introduced renormalization_of_var_sum for multi_var analysis
Jul 26, 2020: 0.32 -> 0.32a only changed last value of allparameters if len()==42
Aug 10, 2020: 0.32a -> 0.33 added extra Zernike to parInit
Aug 12, 2020: 0.33 -> 0.33a changed iters to 6 in fluxfit
Sep 08, 2020: 0.33a -> 0.33b added test_run to help with debugging
Oct 05, 2020: 0.33b -> 0.33c trying to always output flux multiplier when fit_for_flux
Oct 06, 2020: 0.33c -> 0.34 added posibility to specify position of created psf
Oct 13, 2020: 0.34 -> 0.34b added finishing step of centering, done with Nelder-Mead
Oct 22, 2020: 0.34b -> 0.35 added class that does Tokovinin multi analysis
Nov 03, 2020: 0.35 -> 0.35a create parInit up to z=22, with larger parametrization
Nov 05, 2020: 0.35a -> 0.35b return same value if Tokovinin does not work
Nov 16, 2020: 0.35b -> 0.35c modified movement of parameters
Nov 17, 2020: 0.35c -> 0.35d small fixes in check_global_parameters with paramters 0 and 1
Nov 19, 2020: 0.35d -> 0.36 realized that vertical strut is different than others - first, simplest implementation
Nov 19, 2020: 0.36 -> 0.36a modified parInit movements for multi (mostly reduced)
Dec 05, 2020: 0.36a -> 0.37 misalignment and variable strut size
Dec 13, 2020: 0.37 -> 0.37a changed weights in multi_same_spot
Jan 17, 2021: 0.37a -> 0.37b accept True as input for simulation00
Jan 25, 2021: 0.37b -> 0.37c fixed fillCrop function in PsfPosition, slice limits need to be integers
Jan 26, 2021: 0.37c -> 0.38 PIPE2D-701, fixed width of struts implementation
Jan 28, 2021: 0.38 -> 0.39 added flux mask in chi**2 calculation
Jan 28, 2021: 0.39 -> 0.39b lowered allowed values for pixel_effect and fiber_r
Feb 08, 2021: 0.39b -> 0.4 fixed bilinear interpolation for secondary, x and y confusion
Feb 25, 2021: 0.4 -> 0.40a added directory for work on Tiger
Mar 05, 2021: 0.40a -> 0.41 introduced create_custom_var function 
Mar 08, 2021: 0.41 -> 0.41a added suport for saving intermediate images to tiger 
Mar 24, 2021: 0.41a -> 0.41b added support for masked images in find_centroid_of_flux 
Mar 26, 2021: 0.41b -> 0.41c added create_custom_var function as a separate function
Mar 26, 2021: 0.41c -> 0.41d semi-implemented custom variance function in Tokovinin algorithm
Mar 26, 2021: 0.41d -> 0.41e model_multi_out has correct input parameters now
Apr 01, 2021: 0.41e -> 0.42 changed bug/feature in checking wide_43 and wide_42 parameters
Apr 02, 2021: 0.42 -> 0.43 changed width of slit shadow and slit holder shadow
Apr 04, 2021: 0.43 -> 0.44 implemented f_multiplier_factor 
Apr 04, 2021: 0.44 -> 0.44a implemented possibility for using np.abs(chi) as likelihood
Apr 08, 2021: 0.44a -> 0.44b propagated change from 0.44a to Tokovinin algorithm
Apr 12, 2021: 0.44b -> 0.44c modified renormalization factors for abs(chi) value
Apr 13, 2021: 0.44c -> 0.44d fixed bug in the estimate of mean_value_of_background
Apr 14, 2021: 0.44d -> 0.44e mean_value_of_background estimated from sci or var data
Apr 22, 2021: 0.44e -> 0.44f introduced multi_background_factor
Apr 27, 2021: 0.44f -> 0.45 Tokovinin now works much quicker with multi_background_factor (create_simplified_H updated)
Apr 29, 2021: 0.45 -> 0.45a many changes in order to run create_simplified_H efficently
May 07, 2021: 0.45a -> 0.45b if Premodel analysis failed, return 15 values
May 08, 2021: 0.45b -> 0.45c changed that images of same size do not crash out_images creation
May 14, 2021: 0.45c -> 0.45d create_parInit, changed from <> to <= and >=
May 18, 2021: 0.45d -> 0.45e testing focus constrain in Tokovinin
May 19, 2021: 0.45e -> 0.45f expanded verbosity messages in Tokovinin algorithm
May 19, 2021: 0.45f -> 0.45g testing [8., 8., 8., 8., 1., 8., 8., 8., 8.] renormalization
May 20, 2021: 0.45g -> 0.45h do not use multi_background for image in or near focus
May 27, 2021: 0.45h -> 0.45i reordered variables in LN_PFS_single, in preparation for wv analysis
May 27, 2021: 0.45i -> 0.46 changed oversampling to be always 10
Jun 08, 2021: 0.46 -> 0.46a changed to Psf_position to be able to take only_chi and center of flux
Jun 08, 2021: 0.46a -> 0.46b changed normalization so that in focus it is indentical as in pipeline
Jun 15, 2021: 0.46b -> 0.46c change limit on the initial cut of the oversampled image, in order to handle bluer data
Jun 19, 2021: 0.46c -> 0.46d changed skimage.transform.resize to resize, to avoid skimage.transform not avaliable in LSST
Jun 20, 2021: 0.46d -> 0.46e changed scipy.signal to signal, and require that optPsf_cut_downsampled_scattered size is int / no change to unit test

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""
########################################
#standard library imports
from __future__ import absolute_import, division, print_function
import os
import time
#import sys
import math
import socket
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
#print(np.__config__)
from multiprocessing import current_process
from functools import lru_cache
import threading
import platform
import traceback


#import pyfftw
#import pandas as pd
import scipy
from scipy.special import erf

########################################
# Related third party imports
# none at the moment

########################################
# Local application/library specific imports
# galsim
import galsim
galsim.GSParams.maximum_fft_size=12000

# astropy
import astropy
import astropy.convolution
from astropy.convolution import Gaussian2DKernel

# scipy and skimage
import scipy.misc
import scipy.fftpack
#import skimage.transform
#import scipy.optimize as optimize
from scipy.ndimage.filters import gaussian_filter
from scipy import signal

#lmfit
import lmfit

#matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# needed for resizing routines
from typing import Tuple, Iterable

# for svd_invert function
from scipy.linalg import svd

# for distributing image creation in Tokovinin algorithm
from functools import partial
########################################

__all__ = ['PupilFactory', 'Pupil','ZernikeFitter_PFS','LN_PFS_multi_same_spot','LN_PFS_single','LNP_PFS',\
           'find_centroid_of_flux','create_parInit',\
           'Zernike_Analysis','PFSPupilFactory','custom_fftconvolve','stepK','maxK',\
           'sky_scale','sky_size','remove_pupil_parameters_from_all_parameters',\
           'resize','_interval_overlap','svd_invert','Tokovinin_multi','find_centroid_of_flux','create_custom_var']

__version__ = "0.46e"




############################################################

# name your directory where you want to have files!
if socket.gethostname()=='IapetusUSA':
    PSF_DIRECTORY='/Users/nevencaplar/Documents/PFS/'
else:
    PSF_DIRECTORY='/tigress/ncaplar/PFS/'

#PSF_DIRECTORY='/Users/nevencaplar/Documents/PFS/'
############################################################   

TESTING_FOLDER=PSF_DIRECTORY+'Testing/'
TESTING_PUPIL_IMAGES_FOLDER=TESTING_FOLDER+'Pupil_Images/'
TESTING_WAVEFRONT_IMAGES_FOLDER=TESTING_FOLDER+'Wavefront_Images/'
TESTING_FINAL_IMAGES_FOLDER=TESTING_FOLDER+'Final_Images/'

class Pupil(object):
    """!Pupil obscuration function.
    """

    def __init__(self, illuminated, size, scale):
        """!Construct a Pupil

        @param[in] illuminated  2D numpy array indicating which parts of
                                the pupil plane are illuminated.
        @param[in] size         Size of pupil plane array in meters.  Note
                                that this may be larger than the actual
                                diameter of the illuminated pupil to
                                accommodate zero-padding.
        @param[in] scale        Sampling interval of pupil plane array in
                                meters.
        """
        self.illuminated = illuminated
        self.size = size
        self.scale = scale

class PupilFactory(object):
    """!Pupil obscuration function factory for use with Fourier optics.
    """

    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,\
                 x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,verbosity=None,
                 wide_0=0,wide_23=0,wide_43=0,misalign=0):
        """!Construct a PupilFactory.

        @params[in] others
        @param[in] npix       Constructed Pupils will be npix x npix.
        """
        self.verbosity=verbosity
        if self.verbosity==1:
            print('Entering PupilFactory class')

        
        self.pupilSize = pupilSize
        self.npix = npix
        self.input_angle=input_angle
        self.hscFrac=hscFrac
        self.strutFrac=strutFrac
        self.pupilScale = pupilSize/npix
        self.slitFrac=slitFrac
        self.slitFrac_dy=slitFrac_dy
        self.effective_ilum_radius=effective_ilum_radius
        self.frd_sigma=frd_sigma
        self.frd_lorentz_factor=frd_lorentz_factor
        self.det_vert=det_vert
        
        self.wide_0=wide_0
        self.wide_23=wide_23
        self.wide_43=wide_43
        self.misalign=misalign
        
        u = (np.arange(npix, dtype=np.float32) - (npix - 1)/2) * self.pupilScale
        self.u, self.v = np.meshgrid(u, u)


    @staticmethod
    def _pointLineDistance(p0, p1, p2):
        """Compute the right-angle distance between the points given by `p0`
        and the line that passes through `p1` and `p2`.

        @param[in] p0  2-tuple of numpy arrays (x,y coords)
        @param[in] p1  2-tuple of scalars (x,y coords)
        @param[in] p2  2-tuple of scalars (x,y coords)
        @returns       numpy array of distances; shape congruent to p0[0]
        """
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        dy21 = y2 - y1
        dx21 = x2 - x1
        return np.abs(dy21*x0 - dx21*y0 + x2*y1 - y2*x1)/np.hypot(dy21, dx21)

    def _fullPupil(self):
        """Make a fully-illuminated Pupil.

        @returns Pupil
        """      
        illuminated = np.ones(self.u.shape, dtype=np.float32)
        return Pupil(illuminated, self.pupilSize, self.pupilScale)       

    def _cutCircleInterior(self, pupil, p0, r):
        """Cut out the interior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          Circular region radius
        """
        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        pupil.illuminated[r2 < r**2] = False
        
    def _cutCircleExterior(self, pupil, p0, r):
        """Cut out the exterior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0     2-tuple indicating region center
        @param[in] r      Circular region radius
        """
        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        pupil.illuminated[r2 > r**2] = False
    
    def _cutEllipseExterior(self, pupil, p0, r, b, thetarot):
        """Cut out the exterior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0     2-tuple indicating region center
        @param[in] r      Ellipse region radius = major axis
        @param[in] b      Ellipse region radius = minor axis
        @param[in] thetarot   Ellipse region rotation
        """

        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        theta=np.arctan(self.u/self.v)+thetarot
        
        pupil.illuminated[r2 > r**2*b**2/(b**2*(np.cos(theta))**2+r**2*(np.sin(theta))**2)] = False


    def _cutSquare_slow(self,pupil, p0, r,angle,det_vert):
        """Cut out the interior of a circular region from a Pupil.
        
        Is this even used (April 3, 2021)?
        

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half length of the square side
        @param[in] angle      angle that the camera is rotated
        @param[in] det_vert
        """
        pupil_illuminated_only1=np.ones_like(pupil.illuminated,dtype=np.float32)
        
        time_start_single_square=time.time()
        
        ###########################################################
        # Central square
        if det_vert is None:
            det_vert=1
        
        print('r'+str(r))
        
        x21 = -r/2*det_vert*1
        x22 = +r/2*det_vert*1
        y21 = -r/2*1
        y22 = +r/2*1

        angleRad = angle
        print('p0'+str(p0))
        print('angleRad'+str(angleRad))
        
        pupil_illuminated_only1[np.logical_and(((self.u-p0[0])*np.cos(-angle)+(self.v-p0[1])*np.sin(-angleRad)<x22) & \
                          ((self.u-p0[0])*np.cos(-angleRad)+(self.v-p0[1])*np.sin(-angleRad)>x21),\
                          ((self.v-p0[1])*np.cos(-angleRad)-(self.u-p0[0])*np.sin(-angleRad)<y22) & \
                          ((self.v-p0[1])*np.cos(-angleRad)-(self.u-p0[0])*np.sin(-angleRad)>y21))] = False    
        
        

    
        f=0.2
        ###########################################################
        # Lower right corner
        x21 = -r/2
        x22 = +r/2
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        
        
        angleRad21=-np.pi/4 
        triangle21=[[p0[0]+x22,p0[1]+y21],[p0[0]+x22,p0[1]+y21-y21*f],[p0[0]+x22-x22*f,p0[1]+y21]]

        p21=triangle21[0]
        y22=(triangle21[1][1]-triangle21[0][1])/np.sqrt(2)
        y21=0
        x21=(triangle21[2][0]-triangle21[0][0])/np.sqrt(2)
        x22=-(triangle21[2][0]-triangle21[0][0])/np.sqrt(2)

        pupil_illuminated_only1[np.logical_and(((self.u-p21[0])*np.cos(-angleRad21)+(self.v-p21[1])*np.sin(-angleRad21)<x22) & \
                  ((self.u-p21[0])*np.cos(-angleRad21)+(self.v-p21[1])*np.sin(-angleRad21)>x21),\
                  ((self.v-p21[1])*np.cos(-angleRad21)-(self.u-p21[0])*np.sin(-angleRad21)<y22) & \
                  ((self.v-p21[1])*np.cos(-angleRad21)-(self.u-p21[0])*np.sin(-angleRad21)>y21))  ] = True
    
        ###########################################################
        # Upper left corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=-np.pi/4   
        triangle12=[[p0[0]+x21,p0[1]+y22],[p0[0]+x21,p0[1]+y22-y22*f],[p0[0]+x21-x21*f,p0[1]+y22]]
 
        p21=triangle12[0]
        y22=0
        y21=(triangle12[1][1]-triangle12[0][1])/np.sqrt(2)
        x21=-(triangle12[2][0]-triangle12[0][0])/np.sqrt(2)
        x22=+(triangle12[2][0]-triangle12[0][0])/np.sqrt(2)

        pupil_illuminated_only1[np.logical_and(((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)<x22) & \
                  ((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)>x21),\
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)<y22) & \
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)>y21))  ] = True   
        ###########################################################
        # Upper right corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=np.pi/4   
        triangle22=[[p0[0]+x22,p0[1]+y22],[p0[0]+x22,p0[1]+y22-y22*f],[p0[0]+x22-x22*f,p0[1]+y22]]
 
        p21=triangle22[0]
        y22=-0
        y21=+(triangle22[1][1]-triangle22[0][1])/np.sqrt(2)
        x21=+(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)
        x22=-(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)

        pupil_illuminated_only1[np.logical_and(((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)<x22) & \
                  ((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)>x21),\
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)<y22) & \
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)>y21))  ] = True  

        ###########################################################
        # Lower right corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=np.pi/4   
        triangle11=[[p0[0]+x21,p0[1]+y21],[p0[0]+x21,p0[1]+y21-y21*f],[p0[0]+x21-x21*f,p0[1]+y21]]
 
        p21=triangle11[0]
        y22=-(triangle22[1][1]-triangle22[0][1])/np.sqrt(2)
        y21=0
        x21=+(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)
        x22=-(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)

        pupil_illuminated_only1[np.logical_and(((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)<x22) & \
                  ((self.u-p21[0])*np.cos(-angleRad12)+(self.v-p21[1])*np.sin(-angleRad12)>x21),\
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)<y22) & \
                  ((self.v-p21[1])*np.cos(-angleRad12)-(self.u-p21[0])*np.sin(-angleRad12)>y21))  ] = True  
        
        
        pupil.illuminated=pupil.illuminated*pupil_illuminated_only1
        time_end_single_square=time.time()
        
        if self.verbosity==1:
            print('Time for cutting out the square is '+str(time_end_single_square-time_start_single_square))    
  
    def _cutSquare(self,pupil, p0, r,angle,det_vert):
        """Cut out the interior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half lenght of the length of square side
        @param[in] angle      angle that the camera is rotated
        @param[in] det_vert   multiplicative factor that distorts the square into a rectangle
        
        
        """
        pupil_illuminated_only1=np.ones_like(pupil.illuminated,dtype=np.float32)
        
        time_start_single_square=time.time()
        
        ###########################################################
        # Central square
        if det_vert is None:
            det_vert=1
        
        x21 = -r/2*det_vert*1
        x22 = +r/2*det_vert*1
        y21 = -r/2*1
        y22 = +r/2*1
        i_max=self.npix/2-0.5
        i_min=-i_max
        
        i_y_max=int(np.round((x22+p0[1])/self.pupilScale - (i_min)))
        i_y_min=int(np.round((x21+p0[1])/self.pupilScale - (i_min)))
        i_x_max=int(np.round((y22+p0[0])/self.pupilScale - (i_min)))
        i_x_min=int(np.round((y21+p0[0])/self.pupilScale - (i_min)))
        

        assert angle==np.pi/2
        angleRad = angle

        camX_value_for_f_multiplier=p0[0]
        camY_value_for_f_multiplier=p0[1]
        
        #print(camX_value_for_f_multiplier,camY_value_for_f_multiplier)
        camY_Max=0.02
        f_multiplier_factor=(-camX_value_for_f_multiplier*100/3)*(np.abs(camY_value_for_f_multiplier)/camY_Max)+1
        #f_multiplier_factor=1
        if self.verbosity==1:
            print('f_multiplier_factor for size of detector triangle is: '+str(f_multiplier_factor))
        

        pupil_illuminated_only0_in_only1=np.zeros((i_y_max-i_y_min,i_x_max-i_x_min))


        
        u0=self.u[i_y_min:i_y_max,i_x_min:i_x_max]
        v0=self.v[i_y_min:i_y_max,i_x_min:i_x_max]
   
        # factor that is controling how big is the triangle in the corner of the detector?
        f=0.2
        f_multiplier=f_multiplier_factor/1
        
        ###########################################################
        # Lower right corner
        x21 = -r/2
        x22 = +r/2
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        
        f_lr=np.copy(f)*(1/f_multiplier)

        angleRad21=-np.pi/4 
        triangle21=[[p0[0]+x22,p0[1]+y21],[p0[0]+x22,p0[1]+y21-y21*f_lr],[p0[0]+x22-x22*f_lr,p0[1]+y21]]

        p21=triangle21[0]
        y22=(triangle21[1][1]-triangle21[0][1])/np.sqrt(2)
        y21=0
        x21=(triangle21[2][0]-triangle21[0][0])/np.sqrt(2)
        x22=-(triangle21[2][0]-triangle21[0][0])/np.sqrt(2)

        #print('lr'+str([x21,x22,y21,y22]))
        
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Results/pupil_illuminated_only0_in_only1',pupil_illuminated_only0_in_only1)
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Results/v0',v0)
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Results/u0',u0)
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Results/p21',p21)
         
        pupil_illuminated_only0_in_only1[((v0-p21[1])*np.cos(-angleRad21)-(u0-p21[0])*np.sin(-angleRad21)<y22)  ] = True
    
        ###########################################################
        # Upper left corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=-np.pi/4   
        f_ul=np.copy(f)*(1/f_multiplier)
        
        triangle12=[[p0[0]+x21,p0[1]+y22],[p0[0]+x21,p0[1]+y22-y22*f_ul],[p0[0]+x21-x21*f_ul,p0[1]+y22]]
 
        p21=triangle12[0]
        y22=0
        y21=(triangle12[1][1]-triangle12[0][1])/np.sqrt(2)
        x21=-(triangle12[2][0]-triangle12[0][0])/np.sqrt(2)
        x22=+(triangle12[2][0]-triangle12[0][0])/np.sqrt(2)

        #print('ul'+str([x21,x22,y21,y22]))

        pupil_illuminated_only0_in_only1[ ((v0-p21[1])*np.cos(-angleRad21)-(u0-p21[0])*np.sin(-angleRad21)>y21)] = True
        
        ###########################################################
        # Upper right corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=np.pi/4   
        f_ur=np.copy(f)*f_multiplier
        
        triangle22=[[p0[0]+x22,p0[1]+y22],[p0[0]+x22,p0[1]+y22-y22*f_ur],[p0[0]+x22-x22*f_ur,p0[1]+y22]]
 
        p21=triangle22[0]
        y22=-0
        y21=+(triangle22[1][1]-triangle22[0][1])/np.sqrt(2)
        x21=+(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)
        x22=-(triangle22[2][0]-triangle22[0][0])/np.sqrt(2)

        #print('ur'+str([x21,x22,y21,y22]))

        pupil_illuminated_only0_in_only1[((u0-p21[0])*np.cos(-angleRad21)+(v0-p21[1])*np.sin(-angleRad21)>x21) ] = True  

        ###########################################################
        # Lower left corner
        x21 = -r/2*1
        x22 = +r/2*1
        y21 = -r/2*det_vert
        y22 = +r/2*det_vert
        angleRad12=np.pi/4   
        f_ll=np.copy(f)*f_multiplier
        
        triangle11=[[p0[0]+x21,p0[1]+y21],[p0[0]+x21,p0[1]+y21-y21*f_ll],[p0[0]+x21-x21*f_ll,p0[1]+y21]]
 
        p21=triangle11[0]
        y22=-(triangle11[1][1]-triangle11[0][1])/np.sqrt(2)
        y21=0
        x21=+(triangle11[2][0]-triangle11[0][0])/np.sqrt(2)
        x22=+(triangle11[2][0]-triangle11[0][0])/np.sqrt(2)

        #print('ll'+str([x21,x22,y21,y22]))

        pupil_illuminated_only0_in_only1[((u0-p21[0])*np.cos(-angleRad21)+(v0-p21[1])*np.sin(-angleRad21)<x22) ] = True  
        
        
        pupil_illuminated_only1[i_y_min:i_y_max,i_x_min:i_x_max]=pupil_illuminated_only0_in_only1
        
        pupil.illuminated=pupil.illuminated*pupil_illuminated_only1
        time_end_single_square=time.time()
        
        if self.verbosity==1:
            print('Time for cutting out the square is '+str(time_end_single_square-time_start_single_square))    

    
    def _cutRay(self, pupil, p0, angle, thickness,angleunit=None,wide=0):
        """Cut out a ray from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating ray starting point
        @param[in] angle      Ray angle measured CCW from +x.
        @param[in] thickness  Thickness of cutout
        @param[in] angleunit  If None, changes internal units to radians       
        @param[in] wide       Controls the widening of the strut as
                              a function of the distance from the origin
        
        
        """
        if angleunit is None:
            angleRad = angle.asRadians()
        else:   
            angleRad = angle       
        # the 1 is arbitrary, just need something to define another point on
        # the line
        
        
        p1 = (p0[0] + 1, p0[1] + np.tan(angleRad))
        d = PupilFactory._pointLineDistance((self.u, self.v), p0, p1)
        
        radial_distance=14.34*np.sqrt((self.u-p0[0])**2+(self.v-p0[1])**2)
        
        pupil.illuminated[(d < 0.5*thickness*(1+wide*radial_distance)) &
                          ((self.u - p0[0])*np.cos(angleRad) +
                           (self.v - p0[1])*np.sin(angleRad) >= 0)] = False  
        
        

        """
        # print and save commands for debugging
        print('p0: '+str(p0))
        print('angleRad: '+str(angleRad))
        print('thickness: '+str(thickness))
        print('wide: '+str(wide))
        print('**********')
        np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/d'+str(p0[0])+'_'+str(p0[1])+'_'+str(wide),d)
        np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/p0'+str(p0[0])+'_'+str(p0[1])+'_'+str(wide),p0)  
        np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/pupil_illuminated'+str(p0[0])+'_'+str(p0[1])+'_'+str(wide),pupil.illuminated)   
        np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/u',self.u)
        np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/v',self.v)
        """

    def _addRay(self, pupil, p0, angle, thickness,angleunit=None):
        """Add a ray from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating ray starting point
        @param[in] angle      Ray angle measured CCW from +x.
        @param[in] thickness  Thickness of cutout
        """
        if angleunit is None:
            angleRad = angle.asRadians()
        else:   
            angleRad = angle       
        # the 1 is arbitrary, just need something to define another point on
        # the line
        p1 = (p0[0] + 1, p0[1] + np.tan(angleRad))
        d = PupilFactory._pointLineDistance((self.u, self.v), p0, p1)
        pupil.illuminated[(d < 0.5*thickness) &
                          ((self.u - p0[0])*np.cos(angleRad) +
                           (self.v - p0[1])*np.sin(angleRad) >= 0)] = True                   
                     
class PFSPupilFactory(PupilFactory):
    """!Pupil obscuration function factory for PFS 
    """
    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,\
                 x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,slitHolder_frac_dx,verbosity=None,\
                     wide_0=0,wide_23=0,wide_43=0,misalign=0):
        """!Construct a PupilFactory.


        @param[in] pupilSize               Size in meters of constructed Pupils.
        @param[in] npix                    Constructed Pupils will be npix x npix.
        @param[in] input_angle
        @param[in] hscFrac        
        @param[in] strutFrac
        @param[in] slitFrac
        @param[in] slitFrac_dy
        @param[in] x_fiber
        @param[in] y_fiber        
        @param[in] effective_ilum_radius
        @param[in] frd_sigma
        @param[in] frd_lorentz_factor        
        @param[in] det_vert
        @param[in] slitHolder_frac_dx
        @param[in] verbosity    
        @param[in] wide_0
        @param[in] wide_23
        @param[in] wide_43
        """
        
        self.verbosity=verbosity
        if self.verbosity==1:
            print('Entering PFSPupilFactory class')
        # wide here    
        #PupilFactory.__init__(self, pupilSize,npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,
        #                      x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,verbosity=self.verbosity)

        PupilFactory.__init__(self, pupilSize,npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,
                              x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,verbosity=self.verbosity,
                              wide_0=wide_0,wide_23=wide_23,wide_43=wide_43,misalign=misalign)

        self.x_fiber=x_fiber
        self.y_fiber=y_fiber      
        self.slitHolder_frac_dx=slitHolder_frac_dx
        self._spiderStartPos=[np.array([ 0.,  0.]), np.array([ 0.,  0.]), np.array([ 0.,  0.])]
        self._spiderAngles=[0,np.pi*2/3,np.pi*4/3]
        self.effective_ilum_radius=effective_ilum_radius
        
        self.wide_0=wide_0
        self.wide_23=wide_23
        self.wide_43=wide_43
        self.misalign=misalign

    def _horizonRotAngle(self,resultunit=None):
        """!Compute rotation angle of camera with respect to horizontal
        coordinates from self.visitInfo.

        @returns horizon rotation angle.
        
        """
        
        if resultunit is None:
            print('error')
            #parAng = Angle(self.input_angle)
            #return parAng.wrap()
        else:   
            return 0   
        

    def getPupil(self, point):
        """!Calculate a Pupil at a given point in the focal plane.

        @param point  Point2D indicating focal plane coordinates.
        @returns      Pupil
        """
        if self.verbosity==1:
            print('Entering getPupil (function inside PFSPupilFactory)')
            
        # called subaruRadius as it was taken from the code fitting pupil for HSC on Subaru
        subaruRadius = (self.pupilSize/2)*1

        hscFrac = self.hscFrac  # linear fraction
        hscRadius = hscFrac * subaruRadius
        slitFrac = self.slitFrac # linear fraction
        subaruSlit = slitFrac*subaruRadius
        strutFrac = self.strutFrac # linear fraction
        subaruStrutThick = strutFrac*subaruRadius
        
        # y-position of the slit
        slitFrac_dy = self.slitFrac_dy 
        
        # relic from the HSC code
            # See DM-8589 for more detailed description of following parameters
            # d(lensCenter)/d(theta) in meters per degree
            # lensRate = 0.0276 * 3600 / 128.9 * subaruRadius
            # d(cameraCenter)/d(theta) in meters per degree
        hscRate = 2.62 / 1000 * subaruRadius
        hscPlateScale = 380  
        thetaX = point[0] * hscPlateScale 
        thetaY = point[1] * hscPlateScale 

        pupil = self._fullPupil()
        
        camX = thetaX * hscRate
        camY = thetaY * hscRate

        # creating FRD effects
        single_element=np.linspace(-1,1,len(pupil.illuminated), endpoint=True,dtype=np.float32)
        u_manual=np.tile(single_element,(len(single_element),1))
        v_manual=np.transpose(u_manual)  
        center_distance=np.sqrt((u_manual-self.x_fiber*hscRate*hscPlateScale*12)**2+(v_manual-self.y_fiber*hscRate*hscPlateScale*12)**2)
        frd_sigma=self.frd_sigma
        sigma=2*frd_sigma
        
        
        
        
        pupil_frd=(1/2*(scipy.special.erf((-center_distance+self.effective_ilum_radius)/sigma)+\
                        scipy.special.erf((center_distance+self.effective_ilum_radius)/sigma)))
            
        #print('misalign '+str(self.misalign) )
        
        # misaligment starts here
        
        ################
        time_misalign_start=time.time()
        
        position_of_center_0=np.where(center_distance==np.min(center_distance))
        position_of_center=[position_of_center_0[1][0],position_of_center_0[0][0]]
        
        position_of_center_0_x=position_of_center_0[0][0]
        position_of_center_0_y=position_of_center_0[1][0]
        
        distances_to_corners=np.array([np.sqrt(position_of_center[0]**2+position_of_center[1]**2),
        np.sqrt((len(pupil_frd)-position_of_center[0])**2+position_of_center[1]**2),
        np.sqrt((position_of_center[0])**2+(len(pupil_frd)-position_of_center[1])**2),
        np.sqrt((len(pupil_frd)-position_of_center[0])**2+(len(pupil_frd)-position_of_center[1])**2)])
        
        max_distance_to_corner=np.max(distances_to_corners)
        ################3
        threshold_value=0.5
        left_from_center=np.where(pupil_frd[position_of_center_0_x][0:position_of_center_0_y]<threshold_value)[0]
        right_from_center=np.where(pupil_frd[position_of_center_0_x][position_of_center_0_y:]<threshold_value)[0]+position_of_center_0_y
        
        up_from_center=np.where(pupil_frd[:,position_of_center_0_y][position_of_center_0_x:]<threshold_value)[0]+position_of_center_0_x
        down_from_center=np.where(pupil_frd[:,position_of_center_0_y][:position_of_center_0_x]<threshold_value)[0]
        
        if len(left_from_center)>0:
            size_of_05_left=position_of_center_0_y-np.max(left_from_center)
        else:
            size_of_05_left=0
            
        if len(right_from_center)>0:
            size_of_05_right=np.min(right_from_center)-position_of_center_0_y
        else:
            size_of_05_right=0
            
        if len(up_from_center)>0:
            size_of_05_up=np.min(up_from_center)-position_of_center_0_x
        else:
            size_of_05_up=0
            
        if len(down_from_center)>0:
            size_of_05_down=position_of_center_0_x-np.max(down_from_center)
        else:
            size_of_05_down=0
            
        sizes_4_directions=np.array([size_of_05_left,size_of_05_right,size_of_05_up,size_of_05_down])
        max_size=np.max(sizes_4_directions)
        imageradius=max_size   


        
        radiusvalues=np.linspace(0,int(np.ceil(max_distance_to_corner)),int(np.ceil(max_distance_to_corner))+1)
        
        
        sigtotp=sigma*550
        
        
        dif_due_to_mis_class=Pupil_misalign(radiusvalues,imageradius,sigtotp,self.misalign)
        dif_due_to_mis=dif_due_to_mis_class()
        
        scaling_factor_pixel_to_physical=max_distance_to_corner/np.max(center_distance)
        distance_int=np.round(center_distance*scaling_factor_pixel_to_physical).astype(int)        
        
        pupil_frd_with_mis=pupil_frd+dif_due_to_mis[distance_int]
        pupil_frd_with_mis[pupil_frd_with_mis>1]=1
        

        
        
        time_misalign_end=time.time()
        
        if self.verbosity==1:
            print('Time to execute illumination considerations due to misalignment '+str(time_misalign_end-time_misalign_start))
        
        ####
        # misaligment ends here
        
        # misaligment ends here
            
        pupil_lorentz=(np.arctan(2*(self.effective_ilum_radius-center_distance)/(4*sigma))+\
                       np.arctan(2*(self.effective_ilum_radius+center_distance)/(4*sigma)))/(2*np.arctan((2*self.effective_ilum_radius)/(4*sigma)))



        pupil.illuminated= (pupil_frd+1*self.frd_lorentz_factor*pupil_lorentz)/(1+self.frd_lorentz_factor)


        #print('sigma '+str(sigma))
        #print('self.effective_ilum_radius '+str(self.effective_ilum_radius))
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/center_distance',center_distance)


        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/pupil_frd',pupil_frd)
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/pupil_lorentz',pupil_lorentz)       
        #np.save('/Users/nevencaplar/Documents/PFS/TigerAnalysis/Test/pupil_illuminated',pupil.illuminated)
        
        
        # Cout out the acceptance angle of the camera
        #self._cutCircleExterior(pupil, (0.0, 0.0), subaruRadius)        

        #print(self.det_vert)          
        # Cut out detector shadow
        #self._cutSquare(pupil, (camX, camY), hscRadius,self.input_angle,self.det_vert)       
         
        #No vignetting of this kind for the spectroscopic camera
        #self._cutCircleExterior(pupil, (lensX, lensY), lensRadius)
     
        # Cut out spider shadow     
        #print('self.wide_0,self.wide_23,self.wide_43 '+str([self.wide_0,self.wide_23,self.wide_43]))
        
        #for pos, angle in zip(self._spiderStartPos, self._spiderAngles):
        #    x = pos[0] + camX
        #    y = pos[1] + camY
        #    
            #print('[x,y,angle)'+str([x,y,angle]))
        #    if angle==0:
        #        self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_0)
        #    if angle==np.pi*2/3:
        #        self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_23)
        #    if angle==np.pi*4/3:
        #        self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_43)
        
            
        
        # cut out slit shadow
        #self._cutRay(pupil, (2,slitFrac_dy/18),-np.pi,subaruSlit,'rad') 
        
        # cut out slit holder shadow
        #also subaruSlit/3 not fitted, just put roughly correct number
        #self._cutRay(pupil, (self.slitHolder_frac_dx/18,1),-np.pi/2,subaruSlit/3,'rad')   
        
        #if self.verbosity==1:
        #    print('Finished with getPupil')
        
        pupil_lorentz=(np.arctan(2*(self.effective_ilum_radius-center_distance)/(4*sigma))+np.arctan(2*(self.effective_ilum_radius+center_distance)/(4*sigma)))/(2*np.arctan((2*self.effective_ilum_radius)/(4*sigma)))

       
        pupil_frd=np.copy(pupil_frd_with_mis)
        pupil.illuminated= (pupil_frd+1*self.frd_lorentz_factor*pupil_lorentz)/(1+self.frd_lorentz_factor)
        
        # Cout out the acceptance angle of the camera
        self._cutCircleExterior(pupil, (0.0, 0.0), subaruRadius)        
         
        # Cut out detector shadow
        #print( '(camX, camY): '+str( (camX, camY)))
        
        self._cutSquare(pupil, (camX, camY), hscRadius,self.input_angle,self.det_vert)       
        
        #No vignetting of this kind for the spectroscopic camera
        #self._cutCircleExterior(pupil, (lensX, lensY), lensRadius)
        
        # Cut out spider shadow
        for pos, angle in zip(self._spiderStartPos, self._spiderAngles):
            x = pos[0] + camX
            y = pos[1] + camY
            

            if angle==0:
                #print('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_0)
            if angle==np.pi*2/3:
                #print('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_23)
            if angle==np.pi*4/3:
                #print('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad',self.wide_43)
        
        
            
        
        # cut out slit shadow
        self._cutRay(pupil, (2,slitFrac_dy/18),-np.pi,subaruSlit*1.05,'rad') 
        
        # cut out slit holder shadow
        #also subaruSlit/3 not fitted, just put roughly correct number
        self._cutRay(pupil, (self.slitHolder_frac_dx/18,1),-np.pi/2,subaruSlit*0.3,'rad')   
        
        if self.verbosity==1:
            print('Finished with getPupil')
        
        return pupil

class Pupil_misalign(object):
    
    """
    
    """
    
    
    def __init__(self,radiusvalues,imageradius,sigtotp,misalign):
        
        
        self.radiusvalues=radiusvalues
        self.imageradius=imageradius
        self.sigtotp=sigtotp
        self.misalign=misalign
        
    def wapp(self,A):
        #Approximation function by Jim Gunn to approximate and correct for the
        #widening of width due to the angular misalignment convolution. This
        #is used to basically scale the contribution of angular misalignment and FRD
        #A = angmis/sigFRD
        wappA = np.sqrt( 1 + A*A*(1+A*A)/(2 + 1.5*A*A))
        return wappA 
    def fcorr(self,x,A):
        #The function scaled so that it keeps the same (approximate) width value
        #after angular convolution
        correctedfam = self.fcon(x*self.wapp(A),A);
        return correctedfam
    def fcon(self,x,A):
        #For more detail about this method, see "Analyzing Radial Profiles for FRD
        #and Angular Misalignment", by Jim Gunn, 16/06/13.
        wt = [0.1864, 0.1469, 0.1134, 0.1066, 0.1134, 0.1469, 0.1864]   # from Jim Gunn's white paper,
        #wt contains the normalized integrals under the angular misalignment
        #convolution kernel, i.e., C(1-(x/angmisp)^2)^{-1/2} for |x|<angmisp and 0
        #elsewhere. Note that the edges' centers are at +/- a, so they are
        #integrated over an effective half of the length of the others.
        temp = np.zeros(np.size(x))
        for index in range(7):
            temp = temp + wt[index]*self.ndfc(x+(index-3)/3*A);
        angconvolved = temp;
        return angconvolved
    def ndfc(self,x):
        #Standard model dropoff from a Gaussian convolution, normalized to brightness 1, radius (rh) 0, and sigTOT 1
        #print(len(x))
        ndfcfun = 1 - (0.5*erf(x / np.sqrt(2)) + 0.5); 
        return ndfcfun
    def FA(self,r, rh, sigTOT, A):
        #Function that takes all significant variables of the dropoff and
        #normalizes the curve to be comparable to ndfc
        #r = vector of radius values, in steps of pixels
        #rh = radius of half-intensity. Effectively the size of the radius of the dropoff
        #sigTOT = total width of the convolution kernel that recreates the width of the dropoff between 85% and 15% illumination. Effectively just think of this as sigma
        #A = angmis/sigFRD, that is, the ratio between the angular misalignment and the sigma due to only FRD. Usually this is on the order of 1-3.
        FitwithAngle = self.fcorr((r-rh)/sigTOT, A);
        return FitwithAngle


    def __call__(self):
        
        no_mis=self.FA(self.radiusvalues,self.imageradius,self.sigtotp,0)
        with_mis=self.FA(self.radiusvalues,self.imageradius,self.sigtotp,self.misalign)
        dif_due_to_mis=with_mis - no_mis
        
        return dif_due_to_mis






class ZernikeFitter_PFS(object):
    
    """!
    Class to create  donut images in PFS
    
    Despite its name, it does not actually ``fits'' the paramters describing the donuts, it ``just'' creates the images
    
    The final image is made by the convolution of
    1. an OpticalPSF (constructed using FFT)
    2. an input fiber image 
    3. and other convolutions such as CCD charge diffusion
    
    The OpticalPSF part includes
    1.1. description of pupil
    1.2.specification of an arbitrary number of zernike wavefront aberrations 
    
    This code uses lmfit to initalize the parameters.
    
    Calls Psf_position
    Calls Pupil classes (which ones?)
    
    Called by LN_PFS_Single (function constructModelImage_PFS_naturalResolution)
    
    
    """

    def __init__(self,image=None,image_var=None,image_mask=None,pixelScale=None,wavelength=None,
                 diam_sic=None,npix=None,pupilExplicit=None,
                 wf_full_Image=None,radiometricEffectArray_Image=None,
                 ilum_Image=None,dithering=None,save=None,
                 pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,use_wf_grid=None,
                 zmaxInit=None,extraZernike=None,simulation_00=None,verbosity=None,
                 double_sources=None,double_sources_positions_ratios=None,test_run=None,
                 explicit_psf_position=None,use_only_chi=False,use_center_of_flux=False,
                 *args):
        
        """
        @param image        image to analyze
        @param image_var    variance image
        @param image_mask
        @param pixelScale   pixel scale in arcseconds 
        @param wavelength
        @param diam_sic
        @param npix
        @param pupilExplicit
        @param wf_full_Image
        @param radiometricEffectArray_Image
        @param ilum_Image
        @param dithering
        @param save
        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF
        @param use_wf_grid
        @param zmaxInit
        @param extraZernike
        @param simulation_00
        @param verbosity
        @param double_sources
        @param double_sources_positions_ratios
        @param test_run
        @param explicit_psf_position
        """
        
        # if you do not pass the image that you wish to compare, the model will default to creating 41x41 image
        if image is None:
            image=np.ones((41,41))
            self.image = image
        else:
            self.image = image
        
        # if you do not pass the  variance image that you wish to compare, the default variance image has value of '1' everywhere
        if image_var is None:
            image_var=np.ones((41,41))
            self.image_var=image_var
        else:
            self.image_var = image_var
            
        if image_mask is None:
            image_mask=np.zeros(image.shape)
            self.image_mask = image_mask
        else:
            self.image_mask = image_mask

        #flux = number of counts in the image
        flux = float(np.sum(image))
        self.flux=flux    
            
        # if you do not pass the value for wavelength it will default to 794 nm, which is roughly in the middle of the red detector
        if wavelength is None:
            wavelength=794
            self.wavelength=wavelength
        else:
            self.wavelength=wavelength       
        
        # This is size of the pixel in arcsec for PFS red arm in focus
        # calculated with http://www.wilmslowastro.com/software/formulae.htm
        # pixel size in microns/focal length in mm x 206.3
        # pixel size = 15 microns, focal length = 149.2 mm (138 aperature x 1.1 f number)
        if pixelScale is None:
            pixelScale=20.76
            self.pixelScale=pixelScale
        else:
            self.pixelScale=pixelScale
        #print('self.pixelScale: '+str(self.pixelScale))
        
        # Exit pupil size in focus, default is 139.5237e-3 meters (taken from Zemax)
        if diam_sic is None:
            diam_sic=139.5327e-3
            self.diam_sic=diam_sic
        else:
            self.diam_sic=diam_sic
        
        # when creating pupils it will have size of npix pixels
        if npix is None:
            npix=1536
            self.npix=npix
        else:
            self.npix=npix   
        
        # puilExplicit can be used to pass explicitly the image of the pupil instead of creating it from the supplied parameters 
        if pupilExplicit is None:
            pupilExplicit==False
            self.pupilExplicit=pupilExplicit
        else:
            self.pupilExplicit=pupilExplicit      
            
        # radiometricEffectArray_Image can be used to pass explicitly the illumination of the exit pupil instead of creating it from the supplied parameters    
        if radiometricEffectArray_Image is None:
            radiometricEffectArray_Image==False
            self.radiometricEffectArray_Image=radiometricEffectArray_Image
        else:
            self.radiometricEffectArray_Image=radiometricEffectArray_Image  
                
        # effective size of pixels, which can be differ from physical size of pixels due to dithering
        if dithering is None:
            dithering=1
            self.dithering=dithering
            # effective pixel scale is the same as physical pixel scale
            self.pixelScale_effective=self.pixelScale/dithering
        else:
            self.dithering=dithering         
            self.pixelScale_effective=self.pixelScale/dithering         
            
        
        if save in (None,0):
            save=None
            self.save=save
        else:
            save=1
            self.save=save

        if pupil_parameters is None:
            self.pupil_parameters=pupil_parameters
        else:
            self.pupil_parameters=pupil_parameters

        if use_pupil_parameters is None:
            self.use_pupil_parameters=use_pupil_parameters
        else:
            self.use_pupil_parameters=use_pupil_parameters
            self.args = args
            
        if use_optPSF is None:
            self.use_optPSF=use_optPSF
        else:
            self.use_optPSF=use_optPSF
            
        self.use_wf_grid=use_wf_grid    
            
        self.zmax=zmaxInit
        
        self.simulation_00=simulation_00
        if self.simulation_00==True:
            self.simulation_00=1

        self.extraZernike=extraZernike
        
        self.verbosity=verbosity
        
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios
        
        self.test_run=test_run
        
        self.explicit_psf_position=explicit_psf_position
        self.use_only_chi=use_only_chi
        self.use_center_of_flux=use_center_of_flux
        
        
        if self.verbosity==1:
            print('np.__version__' +str(np.__version__))
            #print('skimage.__version__' +str(skimage.__version__))
            print('scipy.__version__' +str(scipy.__version__))
            print('Zernike_Module.__version__' +str(__version__))

    
    def initParams(self,z4Init=None, dxInit=None,dyInit=None,hscFracInit=None,strutFracInit=None,
                   focalPlanePositionInit=None,fiber_rInit=None,
                  slitFracInit=None,slitFrac_dy_Init=None,apodizationInit=None,radiometricEffectInit=None,
                   trace_valueInit=None,serial_trace_valueInit=None,pixel_effectInit=None,backgroundInit=None,
                   x_ilumInit=None,y_ilumInit=None,radiometricExponentInit=None,
                   x_fiberInit=None,y_fiberInit=None,effective_ilum_radiusInit=None,
                   grating_linesInit=None,scattering_radiusInit=None,scattering_slopeInit=None,scattering_amplitudeInit=None,
                   fluxInit=None,frd_sigmaInit=None,frd_lorentz_factorInit=None,
                   det_vertInit=None,slitHolder_frac_dxInit=None,wide_0Init=None,wide_23Init=None,wide_43Init=None,misalignInit=None):
        """Initialize lmfit Parameters object.
        
        @param zmax                      Total number of Zernike aberrations used
        @param z4Init                    Initial Z4 aberration value in waves (that is 2*np.pi*wavelengths)
        
        # pupil parameters
        @param hscFracInit               Value determining how much of the exit pupil obscured by the central obscuration(detector) 
        @param strutFracInit             Value determining how much of the exit pupil is obscured by a single strut
        @param focalPlanePositionInit    2-tuple for position of the central obscuration(detector) in the focal plane
        @param slitFracInit              Value determining how much of the exit pupil is obscured by slit
        @param slitFrac_dy_Init          Value determining what is the vertical position of the slit in the exit pupil
        
        #
        @param wide_0
        @param wide_23
        @param wide_34
        @param misalign 
        
        #non-uniform illumination
        @param radiometricEffectInit     parameter describing non-uniform illumination of the pupil (1-params['radiometricEffect']**2*r**2)**(params['radiometricExponent'])
        @param radiometricExponentInit   parameter describing non-uniform illumination of the pupil (1-params['radiometricEffect']**2*r**2)**(params['radiometricExponent'])
        @param x_ilumInit                x-position of the center of illumination of the exit pupil
        @param y_ilumInit                y-position of the center of illumination of the exit pupil
        
        # illumination due to fiber, parameters
        @param x_fiberInit               position of the fiber misaligment in the x direction
        @param y_fiberInit               position of the fiber misaligment in the y direction
        @param effective_ilum_radiusInit fraction of the maximal radius of the illumination of the exit pupil   
        @param frd_sigma                 sigma of Gaussian convolving only outer edge, mimicking FRD
        @param frd_lorentz_factor        strength of the lorentzian factor describing wings of the pupil illumination
        
        # further pupil parameters      
        @param det_vert                  multiplicative factor determining vertical size of the detector obscuration
        @param slitHolder_frac_dx        dx position of slit holder

        # convolving (postprocessing) parameters
        @param grating_lines             number of effective lines in the grating
        @param scattering_slopeInit      slope of scattering
        @param scattering_amplitudeInit  amplitude of scattering compared to optical PSF
        @param pixel_effectInit          sigma describing charge diffusion effect [in units of 15 microns]
        @param fiber_rInit               radius of perfect tophat fiber, as seen on the detector [in units of 15 microns]         
        @param fluxInit                  total flux in generated image compared to input image (needs to be 1 or very close to 1)

        #not used anymore
        @param dxInit                    (not used in this version of the code - parameter determing position of PSF on detector)
        @param dyInit                    (not used in this version of the code - parameter determing position of PSF on detector )
        @param apodizationInit           (not used in this iteration of the code) by how much pixels to convolve the pupil image to apodize the strucutre - !
        @param trace_valueInit           (not used in this iteration of the code) inital value for adding vertical component to the data
        @param serial_trace_valueInit    (not used in this iteration of the code)inital value for adding horizontal component to the data      
        
        
        """


        if self.verbosity==1:
            print(' ')
            print('Initializing ZernikeFitter_PFS')
            print('Verbosity parameter is: '+str(self.verbosity))
            print('Highest Zernike polynomial is (zmax): '+str(self.zmax))

        params = lmfit.Parameters()
        z_array=[]

        if z4Init is None:
            params.add('z4', 0.0)

        else:
            params.add('z4', z4Init)
            
        for i in range(5, self.zmax+1):
            params.add('z{}'.format(i), 0.0)

        if dxInit is None:
            params.add('dx', 0.0)
        else:
            params.add('dx', dxInit)

        if dyInit is None:
            params.add('dy', 0.0)
        else:
            params.add('dy', dyInit)   
      
        if hscFracInit is None:
            params.add('hscFrac', 0)
        else:
            params.add('hscFrac',hscFracInit)        

        if strutFracInit is None:
            params.add('strutFrac', 0)
        else:
            params.add('strutFrac', strutFracInit)             

        if focalPlanePositionInit is None:
            params.add('dxFocal', 0.0) 
            params.add('dyFocal', 0.0) 
        else:
            params.add('dxFocal', focalPlanePositionInit[0]) 
            params.add('dyFocal', focalPlanePositionInit[1])  

        if slitFracInit is None:
            params.add('slitFrac', 0)
        else:
            params.add('slitFrac', slitFracInit)     
            
        if slitFrac_dy_Init is None:
            params.add('slitFrac_dy', 0)
        else:
            params.add('slitFrac_dy', slitFrac_dy_Init)  

        if fiber_rInit is None:
            params.add('fiber_r', 1.8)
        else:
            params.add('fiber_r', fiber_rInit)  
                
        if radiometricEffectInit is None:
            params.add('radiometricEffect', 0)
        else:
            params.add('radiometricEffect', radiometricEffectInit)    
            
        if trace_valueInit is None:
            params.add('trace_value', 0)
        else:
            params.add('trace_value', trace_valueInit)  
            
        if serial_trace_valueInit is None:
            params.add('serial_trace_value', 0)
        else:
            params.add('serial_trace_value', serial_trace_valueInit)  
            
        if pixel_effectInit is None:
            params.add('pixel_effect', 1)
        else:
            params.add('pixel_effect', pixel_effectInit)       
            
        if backgroundInit is None:
            params.add('background', 0)
        else:
            params.add('background', backgroundInit)    
            
        if fluxInit is None:
            params.add('flux', 1)
        else:
            params.add('flux', fluxInit)                

        if x_ilumInit is None:
            params.add('x_ilum', 1)
        else:
            params.add('x_ilum', x_ilumInit)   
            
        if y_ilumInit is None:
            params.add('y_ilum', 1)
        else:
            params.add('y_ilum', y_ilumInit)   

        if radiometricExponentInit is None:
            params.add('radiometricExponent', 0.25)
        else:
            params.add('radiometricExponent', radiometricExponentInit)  
            
        if x_ilumInit is None:
            params.add('x_fiber', 1)
        else:
            params.add('x_fiber', x_fiberInit)   

        if effective_ilum_radiusInit is None:
            params.add('effective_ilum_radius', 1)
        else:
            params.add('effective_ilum_radius', effective_ilum_radiusInit)   

        if y_fiberInit is None:
            params.add('y_fiber', 0)
        else:
            params.add('y_fiber', y_fiberInit)          
            
        if grating_linesInit is None:
            params.add('grating_lines', 100000)
        else:
            params.add('grating_lines', grating_linesInit)   
            
        if scattering_slopeInit is None:
            params.add('scattering_slope', 2)
        else:
            params.add('scattering_slope', scattering_slopeInit)   

        if scattering_amplitudeInit is None:
            params.add('scattering_amplitude', 10**-2)
        else:
            params.add('scattering_amplitude', scattering_amplitudeInit)   
 
        if frd_sigmaInit is None:
            params.add('frd_sigma', 0.02)
        else:
            params.add('frd_sigma', frd_sigmaInit)  
            
        if frd_lorentz_factorInit is None:
            params.add('frd_lorentz_factor', 0.5)
        else:
            params.add('frd_lorentz_factor', frd_lorentz_factorInit) 
            
            
        if det_vertInit is None:
            params.add('det_vert', 1)
        else:
            params.add('det_vert', det_vertInit)   
            
        if wide_0Init is None:
            params.add('wide_0', 0)
        else:
            params.add('wide_0', wide_0Init)   

        if wide_23Init is None:
            params.add('wide_23', 0)
        else:
            params.add('wide_23', wide_23Init)   

        if wide_43Init is None:
            params.add('wide_43', 0)
        else:
            params.add('wide_43', wide_43Init)               
            
        if misalignInit is None:
            params.add('misalign', 0)
        else:
            params.add('misalign', misalignInit)  

        if slitHolder_frac_dxInit is None:
            params.add('slitHolder_frac_dx', 0)
        else:
            params.add('slitHolder_frac_dx', slitHolder_frac_dxInit)             
            

        self.params = params
        self.optPsf=None
        self.z_array=z_array


    def constructModelImage_PFS_naturalResolution(self,params=None,shape=None,pixelScale=None,
                                                  use_optPSF=None,extraZernike=None,return_intermediate_images=False):
        """Construct model image given the set of parameters
        
        calls _getOptPsf_naturalResolution and optPsf_postprocessing
        gets called by LN_PFS_single.lnlike_Neven
        
        @param params                                         lmfit.Parameters object or python dictionary with
                                                              param values to use, or None to use self.params
        @param shape                                          (nx, ny) shape for model image, or None to use
                                                              the shape of self.maskedImage
        @param pixelScale                                     pixel scale in arcseconds to use for model image,
                                                              or None to use self.pixelScale.
        @param use_optPSF                                     use previously generated optical PSF and conduct only postprocessing
        @param extraZernike                                   Zernike beyond z22
        @param return_intermediate_images                     return intermediate images created during the run

        @returns                                              0.numpy array image with the same flux as the input image, 1. psf_position
        """
        
        if self.verbosity==1:
            print(' ')
            print('Entering constructModelImage_PFS_naturalResolution')
        
        if params is None:
            params = self.params
            
        if shape is None:
            shape = self.image.shape 

        if pixelScale is None:
            pixelScale = self.pixelScale
                  
            
        try:
            parameter_values = params.valuesdict()
        except AttributeError:
            parameter_values = params

        use_optPSF=self.use_optPSF

        if extraZernike is None:
            extraZernike=None
            self.extraZernike=extraZernike
        else:
            extraZernike=list(extraZernike)
            self.extraZernike=extraZernike

        # This give image in nyquist resolution
        # if not explicitly stated to the full procedure
        if use_optPSF is None:
            if return_intermediate_images==False:
                optPsf=self._getOptPsf_naturalResolution(parameter_values,return_intermediate_images=return_intermediate_images)
            else:
                optPsf,ilum,wf_grid_rot=self._getOptPsf_naturalResolution(parameter_values,return_intermediate_images=return_intermediate_images)    
        else:
            #if first iteration still generated image
            if self.optPsf is None:
                if return_intermediate_images==False:
                    optPsf=self._getOptPsf_naturalResolution(parameter_values,return_intermediate_images=return_intermediate_images)
                else:
                    optPsf,ilum,wf_grid_rot=self._getOptPsf_naturalResolution(parameter_values,return_intermediate_images=return_intermediate_images)   
                self.optPsf=optPsf
            else:
                optPsf=self.optPsf



                
        # at the moment, no difference in optPsf_postprocessing depending on return_intermediate_images
        optPsf_cut_fiber_convolved_downsampled,psf_position=self._optPsf_postprocessing(optPsf,return_intermediate_images=return_intermediate_images)

        if self.save==1:
            if socket.gethostname()=='IapetusUSA' or socket.gethostname()=='tiger2-sumire.princeton.edu':
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf',optPsf)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled',optPsf_cut_fiber_convolved_downsampled) 
            else:                 
                pass    
        
        if return_intermediate_images==False:
            return optPsf_cut_fiber_convolved_downsampled,psf_position
        if return_intermediate_images==True:
            return optPsf_cut_fiber_convolved_downsampled,ilum,wf_grid_rot,psf_position    
        
        if self.verbosity==1:
            print('Finished with constructModelImage_PFS_naturalResolution')
            print(' ')   
            
            
    def _optPsf_postprocessing(self,optPsf,return_intermediate_images=False):
        
        """
        @input              optPsf
        @param              return_intermediate_images
        
        Takes optical psf and postprocesses it to generate final image
        
        """
        
        
        time_start_single=time.time()
        if self.verbosity==1:
            print(' ')
            print('Entering optPsf_postprocessing')
        
        
        params = self.params
        shape = self.image.shape 
        double_sources=self.double_sources
        
        # all of the parameters for the creation of the image
        # very stupidly called ``v'' without any reason whatsoever
        v = params.valuesdict()
        
       # how much is my generated image oversampled compared to final image
        oversampling_original=(self.pixelScale_effective)/self.scale_ModelImage_PFS_naturalResolution
        
        if self.verbosity==1:
            print('optPsf.shape: '+str(optPsf.shape))
            print('oversampling_original: ' +str(oversampling_original))
            #print('type(optPsf) '+str(type(optPsf[0][0])))

        
        # determine the size, so that from the huge generated image we can cut out only the central portion (1.4 times larger than the size of actual image)
        size_of_central_cut=int(oversampling_original*self.image.shape[0]*1.4)
        
        if size_of_central_cut > optPsf.shape[0]:
            # if larger than size of image, cut the image
            # fail if not enough space
            size_of_central_cut=optPsf.shape[0]
            #print('size:'+str(int(oversampling_original*self.image.shape[0]*1.0)))
            if self.verbosity==1:
                print('size_of_central_cut modified to '+str(size_of_central_cut))

            
            assert int(oversampling_original*self.image.shape[0]*1.0)<optPsf.shape[0]
  
        
        assert size_of_central_cut<=optPsf.shape[0]
        if self.verbosity==1:
            print('size_of_central_cut: '+str(size_of_central_cut))
            
        # cut part which you need to form the final image 
        # set oversampling to 1 so you are not resizing the image, and dx=0 and dy=0 so that you are not moving around, i.e., you are cutting the central region
        optPsf_cut=Psf_position.cut_Centroid_of_natural_resolution_image(image=optPsf,size_natural_resolution=size_of_central_cut+1,oversampling=1,dx=0,dy=0)
        if self.verbosity==1:
            print('optPsf_cut.shape'+str(optPsf_cut.shape))

        # we want to reduce oversampling to be roughly around 10 to make things computationaly easier
        # if oversamplign_original is smaller than 20 (in case of dithered images), make res coarser by factor of 2
        # otherwise do it by 4
        if oversampling_original< 20:
            oversampling=np.round(oversampling_original/2)
        else:
            #oversampling=np.round(oversampling_original/4)
            oversampling=10
        if self.verbosity==1:
            print('oversampling:' +str(oversampling))
        
        # what will be the size of the image after you resize it to the from ``oversampling_original'' to ``oversampling'' ratio
        size_of_optPsf_cut_downsampled=np.int(np.round(size_of_central_cut/(oversampling_original/oversampling)))
        if self.verbosity==1:
            print('optPsf_cut.shape[0]'+str(optPsf_cut.shape[0]))
            print('size_of_optPsf_cut_downsampled: '+str(size_of_optPsf_cut_downsampled))
            #print('type(optPsf_cut) '+str(type(optPsf_cut[0][0])))
                    
        # make sure that optPsf_cut_downsampled is an array which has an odd size - increase size by 1 if needed

        #if (size_of_optPsf_cut_downsampled % 2) == 0:
        #    optPsf_cut_downsampled=skimage.transform.resize(optPsf_cut,(size_of_optPsf_cut_downsampled+1,size_of_optPsf_cut_downsampled+1),mode='constant',order=3)
        #else:
        #    optPsf_cut_downsampled=skimage.transform.resize(optPsf_cut,(size_of_optPsf_cut_downsampled,size_of_optPsf_cut_downsampled),mode='constant',order=3)
       
        if (size_of_optPsf_cut_downsampled % 2) == 0:
            optPsf_cut_downsampled=resize(optPsf_cut,(size_of_optPsf_cut_downsampled+1,size_of_optPsf_cut_downsampled+1))
        else:
            optPsf_cut_downsampled=resize(optPsf_cut,(size_of_optPsf_cut_downsampled,size_of_optPsf_cut_downsampled))

        if self.verbosity==1:        
            print('optPsf_cut_downsampled.shape: '+str(optPsf_cut_downsampled.shape))
            #print('type(optPsf_cut_downsampled) '+str(type(optPsf_cut_downsampled[0][0])))
        
        # gives middle point of the image to used for calculations of scattered light 
        mid_point_of_optPsf_cut_downsampled=int(optPsf_cut_downsampled.shape[0]/2)
        
        # gives the size of one pixel in optPsf_downsampled in microns
        # one physical pixel is 15 microns
        # effective size is 15 / dithering
        size_of_pixels_in_optPsf_cut_downsampled=(15/self.dithering)/oversampling
        
        # size of the created optical PSF images in microns
        size_of_optPsf_cut_in_Microns=size_of_pixels_in_optPsf_cut_downsampled*(optPsf_cut_downsampled.shape[0])
        if self.verbosity==1:   
            print('size_of_optPsf_cut_in_Microns: '+str(size_of_optPsf_cut_in_Microns))
    
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
    
        # we now apply various effects that are separate from pure optical PSF 
        # Those include
        # 1. scattered light
        # 2. convolution with fiber
        # 3. CCD difusion
        # 4. grating effects
        
        # We then finish with the centering algorithm to move our created image to fit the input science image
        # 5. centering
        
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
        # 1. scattered light
        
        # create grid to apply scattered light
        pointsx = np.linspace(-(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,\
                              (size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,num=optPsf_cut_downsampled.shape[0], dtype=np.float32)
        pointsy =np.linspace(-(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,\
                             (size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,num=optPsf_cut_downsampled.shape[0]).astype(np.float32)
        xs, ys = np.meshgrid(pointsx, pointsy)
        r0 = np.sqrt((xs-0)** 2 + (ys-0)** 2)+.01


        if self.verbosity==1:
            print('postprocessing parameters:')
            print(str(['grating_lines','scattering_slope','scattering_amplitude','pixel_effect','fiber_r']))
            print(str([v['grating_lines'],v['scattering_slope'],v['scattering_amplitude'],v['pixel_effect'],v['fiber_r']]))
            print('type(pointsx): '+str(type(pointsx[0])))

        # creating scattered light code
        scattered_light_kernel=(r0**(-v['scattering_slope']))
        
        # the line below from previous code where I terminated scattering radius dependece below certain radius which could have change (changed on Oct 04, 2018)
        # keep for some more time for historic reasons
        # scattered_light_kernel[r0<v['scattering_radius']]=v['scattering_radius']**(-v['scattering_slope'])
        
        scattered_light_kernel[r0<7.5]=7.5**(-v['scattering_slope'])
        scattered_light_kernel[scattered_light_kernel == np.inf] = 0
        scattered_light_kernel=scattered_light_kernel*(v['scattering_amplitude'])/(10*np.max(scattered_light_kernel))
        
        # convolve the psf with the scattered light kernel to create scattered light component
        #scattered_light=custom_fftconvolve(optPsf_cut_downsampled,scattered_light_kernel)
        #scattered_light=scipy.signal.fftconvolve(optPsf_cut_downsampled, scattered_light_kernel, mode='same')
        scattered_light=signal.fftconvolve(optPsf_cut_downsampled, scattered_light_kernel, mode='same')        
        
        #print('type(scattered_light[0][0])'+str(type(scattered_light[0][0])))
        # add back the scattering to the image
        optPsf_cut_downsampled_scattered=optPsf_cut_downsampled+scattered_light        
        
        if self.verbosity==1:
            print('optPsf_cut_downsampled_scattered.shape:' +str(optPsf_cut_downsampled_scattered.shape))
            #print('type(optPsf_cut_downsampled_scattered[0][0])'+str(type(optPsf_cut_downsampled_scattered[0][0])))
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
        # 2. convolution with fiber

        # create tophat2d
        # physical quantities do not change with dithering, so multiply with self.dithering (applies also to steps 3 and 4)
        fiber = astropy.convolution.Tophat2DKernel(oversampling*v['fiber_r']*self.dithering,mode='oversample').array
        # create array with zeros with size of the current image, which we will fill with fiber array in the middle
        fiber_padded=np.zeros_like(optPsf_cut_downsampled_scattered,dtype=np.float32)
        mid_point_of_optPsf_cut_downsampled=int(optPsf_cut_downsampled.shape[0]/2)
        fiber_array_size=fiber.shape[0]
        # fill the zeroes image with fiber here
        fiber_padded[int(mid_point_of_optPsf_cut_downsampled-fiber_array_size/2)+1:int(mid_point_of_optPsf_cut_downsampled+fiber_array_size/2)+1,\
                     int(mid_point_of_optPsf_cut_downsampled-fiber_array_size/2)+1:int(mid_point_of_optPsf_cut_downsampled+fiber_array_size/2)+1]=fiber

                     
        # legacy code for is the line below, followed by the currently used code
        #optPsf_fiber_convolved=scipy.signal.fftconvolve(optPsf_downsampled_scattered, fiber, mode = 'same') 
        
        # convolve with fiber 
        #optPsf_cut_fiber_convolved=custom_fftconvolve(optPsf_cut_downsampled_scattered,fiber_padded)
        #optPsf_cut_fiber_convolved=scipy.signal.fftconvolve(optPsf_cut_downsampled_scattered, fiber_padded, mode='same')
        optPsf_cut_fiber_convolved=signal.fftconvolve(optPsf_cut_downsampled_scattered, fiber_padded, mode='same')
         
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
        # 3. CCD difusion
        
        #pixels are not perfect detectors
        # charge diffusion in our optical CCDs, can be well described with a Gaussian 
        # sigma is around 7 microns (Jim Gunn - private communication). This is controled in our code by @param 'pixel_effect'
        pixel_gauss=Gaussian2DKernel(oversampling*v['pixel_effect']*self.dithering).array.astype(np.float32)
        pixel_gauss_padded=np.pad(pixel_gauss,int((len(optPsf_cut_fiber_convolved)-len(pixel_gauss))/2),'constant',constant_values=0)
        
        # assert that gauss_padded array did not produce empty array
        assert np.sum(pixel_gauss_padded)>0
 
        #optPsf_cut_pixel_response_convolved=scipy.signal.fftconvolve(optPsf_cut_fiber_convolved, pixel_gauss_padded, mode='same')
        optPsf_cut_pixel_response_convolved=signal.fftconvolve(optPsf_cut_fiber_convolved, pixel_gauss_padded, mode='same')



      
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########        
        # 4. grating effects
        
        # following grating calculation is done
        # assuming that 15 microns covers wavelength range of 0.07907 nm
        #(assuming that 4300 pixels in real detector uniformly covers 340 nm)
        grating_kernel=np.ones((optPsf_cut_pixel_response_convolved.shape[0],1),dtype=np.float32)
        for i in range(len(grating_kernel)):
            grating_kernel[i]=Ifun16Ne((i-int(optPsf_cut_pixel_response_convolved.shape[0]/2))*0.07907*10**-9/(self.dithering*oversampling)+self.wavelength*10**-9,self.wavelength*10**-9,v['grating_lines'])
        grating_kernel=grating_kernel/np.sum(grating_kernel)
        
        # I should implement custom_fft function (custom_fftconvolve), as above
        # This is 1D convolution so it would need a bit of work, and I see that behavior is fine
        #optPsf_cut_grating_convolved=scipy.signal.fftconvolve(optPsf_cut_pixel_response_convolved, grating_kernel, mode='same')
        optPsf_cut_grating_convolved=signal.fftconvolve(optPsf_cut_pixel_response_convolved, grating_kernel, mode='same') 

       
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########   
        # 5. centering
        # This is the part which creates the final image
        
        # if you have requsted a simulated image without movement, `simulation_00' will not be `None' and the code goes into the small ``if statment'' below
        # otherwise, if `simulation_00' is `None' the statment is skipped and the code does not create image with optical center at (0,0)
        if self.verbosity==1:
            print('simulation_00 parameter:' +str(self.simulation_00))
        if self.simulation_00 is not None:
            # needs to be improved and made sure that you take the oversampling into account
            optPsf_cut_grating_convolved_simulation_cut=Psf_position.cut_Centroid_of_natural_resolution_image(optPsf_cut_grating_convolved,20*oversampling,1,+1,+1)
            optPsf_cut_grating_convolved_simulation_cut=optPsf_cut_grating_convolved_simulation_cut/np.sum(optPsf_cut_grating_convolved_simulation_cut)
            np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_grating_convolved_simulation_cut',optPsf_cut_grating_convolved_simulation_cut)
            
            optPsf_cut_grating_convolved_simulation_cut_odd=Psf_position.cut_Centroid_of_natural_resolution_image(optPsf_cut_grating_convolved,21*oversampling,1,+1,+1)
            optPsf_cut_grating_convolved_simulation_cut_odd=optPsf_cut_grating_convolved_simulation_cut_odd/np.sum(optPsf_cut_grating_convolved_simulation_cut_odd)
            np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_grating_convolved_simulation_cut_odd',optPsf_cut_grating_convolved_simulation_cut_odd)
            
            # still create some sort of optPsf_cut_fiber_convolved_downsampled in order to be consistent
            
        else:
            
            pass
            
            
        # the algorithm  finds (or at least should find) the best downsampling combination automatically 
        if self.verbosity==1:
            print('are we invoking double sources (1 or True if yes): '+str(self.double_sources)) 
            print('double source position/ratio is:' +str(self.double_sources_positions_ratios))
            
        # initialize the class which does the centering - the separation between the class and the main function in the class, 
        # ``find_single_realization_min_cut'', is a bit blurry and unsatisfactory
        single_Psf_position=Psf_position(optPsf_cut_grating_convolved, int(round(oversampling)),shape[0] ,
                                         double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,
                                                                               verbosity=self.verbosity,save=self.save)
        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for postprocessing up to single_Psf_position protocol is '+str(time_end_single-time_start_single))        
        #  run the code for centering
        time_start_single=time.time()
        # set simulation_00='None', the simulated at 00 image has been created above
        
        # changes to Psf_position introduced in 0.46a
        optPsf_cut_fiber_convolved_downsampled,psf_position=single_Psf_position.find_single_realization_min_cut(optPsf_cut_grating_convolved,
                                                                               int(round(oversampling)),shape[0],self.image,self.image_var,self.image_mask,
                                                                               v_flux=v['flux'],simulation_00='None',
                                                                               double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,
                                                                               verbosity=self.verbosity,explicit_psf_position=self.explicit_psf_position,
                                                                               use_only_chi=self.use_only_chi,use_center_of_flux=self.use_center_of_flux)

        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for single_Psf_position protocol is '+str(time_end_single-time_start_single))
            #print('type(optPsf_cut_fiber_convolved_downsampled[0][0])'+str(type(optPsf_cut_fiber_convolved_downsampled[0][0])))
   
        if self.verbosity==1:
            print('Sucesfully created optPsf_cut_fiber_convolved_downsampled') 
        
        if self.save==1:
            if socket.gethostname()=='IapetusUSA' or socket.gethostname()=='tiger2-sumire.princeton.edu':
                np.save(TESTING_FINAL_IMAGES_FOLDER+'pixel_gauss_padded',pixel_gauss_padded)                
                
                
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut',optPsf_cut)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_downsampled',optPsf_cut_downsampled)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_downsampled_scattered',optPsf_cut_downsampled_scattered)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'r0',r0)               
                np.save(TESTING_FINAL_IMAGES_FOLDER+'scattered_light',scattered_light)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'scattered_light_kernel',scattered_light_kernel)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'fiber',fiber)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'fiber_padded',fiber_padded)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_downsampled_scattered',optPsf_cut_downsampled_scattered)        
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved',optPsf_cut_fiber_convolved) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_pixel_response_convolved',optPsf_cut_pixel_response_convolved) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_grating_convolved',optPsf_cut_grating_convolved) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'grating_kernel',grating_kernel)
        
        if self.verbosity==1:      
            print('Finished with optPsf_postprocessing')
            print(' ')
        
        # at the moment, the output is the same but there is a possibility to add intermediate outputs
        if return_intermediate_images==False:
            return optPsf_cut_fiber_convolved_downsampled,psf_position
        

        if return_intermediate_images==True:
            return optPsf_cut_fiber_convolved_downsampled,psf_position
            
    
    
    @lru_cache(maxsize=3)
    def _get_Pupil(self,params):
        
        if self.verbosity==1:
            print(' ')
            print('Entering _get_Pupil (function inside ZernikeFitter_PFS)')        
        
        diam_sic=self.diam_sic
        npix=self.npix
         
        if self.verbosity==1:
            print('Size of the pupil (npix): '+str(npix))        
       
        
       
        """
        init of pupil is:
            
        def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,\
             x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,slitHolder_frac_dx,verbosity=None,\
                 wide_0=0,wide_23=0,wide_43=0,misalign=0):
                
        """
        
        
        Pupil_Image=PFSPupilFactory(diam_sic,npix,
                                np.pi/2,
                              self.pupil_parameters[0],self.pupil_parameters[1],
                              self.pupil_parameters[4],self.pupil_parameters[5],
                              self.pupil_parameters[6],self.pupil_parameters[7],self.pupil_parameters[8],
                              self.pupil_parameters[9],self.pupil_parameters[10],self.pupil_parameters[11],self.pupil_parameters[12],
                              verbosity=self.verbosity,
                              wide_0=self.pupil_parameters[13],wide_23=self.pupil_parameters[14],wide_43=self.pupil_parameters[15],
                              misalign=self.pupil_parameters[16])
        
        point=[self.pupil_parameters[2],self.pupil_parameters[3]]
        pupil=Pupil_Image.getPupil(point)

        if self.save==1:
            if socket.gethostname()=='IapetusUSA' or socket.gethostname()=='tiger2-sumire.princeton.edu':
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'pupil.illuminated',pupil.illuminated.astype(np.float32))
        
        if self.verbosity==1:
            print('Finished with _get_Pupil')    
        
        return pupil
        
    
    def _getOptPsf_naturalResolution(self,params,return_intermediate_images=False):
        
        """ !returns optical PSF, given the initialized parameters 
        called by constructModelImage_PFS_naturalResolution
        
         @param params                                       parameters
         @param return_intermediate_images
        """

        if self.verbosity==1:
            print(' ')
            print('Entering _getOptPsf_naturalResolution')       
       
        
        ################################################################################
        # pupil and illumination of the pupil
        ################################################################################
        time_start_single_1=time.time()
        if self.verbosity==1:
            print('use_pupil_parameters: '+str(self.use_pupil_parameters))
            print('pupil_parameters if you are explicity passing use_pupil_parameters: '+str(self.pupil_parameters))

        # parmeters ``i'' just to precision in the construction of ``pupil_parameters'' array
        i=4    
        if self.use_pupil_parameters is None:
            pupil_parameters=np.array([params['hscFrac'.format(i)],params['strutFrac'.format(i)],
                                       params['dxFocal'.format(i)],params['dyFocal'.format(i)],
                                       params['slitFrac'.format(i)],params['slitFrac_dy'.format(i)],
                                       params['x_fiber'.format(i)],params['y_fiber'.format(i)],params['effective_ilum_radius'.format(i)],
                                       params['frd_sigma'.format(i)],params['frd_lorentz_factor'.format(i)],params['det_vert'.format(i)],params['slitHolder_frac_dx'.format(i)],
                                       params['wide_0'.format(i)],params['wide_23'.format(i)],params['wide_43'.format(i)],params['misalign'.format(i)]])
            self.pupil_parameters=pupil_parameters
        else:
            pupil_parameters=np.array(self.pupil_parameters)
            
        diam_sic=self.diam_sic
        
        if self.verbosity==1:
            print(['hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy'])
            print(['x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx'])
            print(['wide_0','wide_23','wide_43','misalign'])
            print('set of pupil_parameters I. : '+str(self.pupil_parameters[:6]))
            print('set of pupil_parameters II. : '+str(self.pupil_parameters[6:6+7]))    
            print('set of pupil_parameters III. : '+str(self.pupil_parameters[13:])) 
        time_start_single_2=time.time()


      


        # initialize galsim.Aperature class
        # the output will be the size of pupil.illuminated
        # if you are passing explicit pupil model ... 
        if self.pupilExplicit is None:
            
            pupil=self._get_Pupil(tuple(pupil_parameters))
            
            aper = galsim.Aperture(
                diam = pupil.size,
                pupil_plane_im = pupil.illuminated.astype(np.float32),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None) 
        else:

            if self.verbosity==1: 
                print('Using provided pupil and skipping _get_Pupil function')
            aper = galsim.Aperture(
                diam = self.diam_sic,
                pupil_plane_im = self.pupilExplicit.astype(np.float32),
                pupil_plane_scale = self.diam_sic/self.npix,
                pupil_plane_size = None)   
            


        if self.verbosity==1:    
            if self.pupilExplicit is None:
                print('Requested pupil size is (pupil.size) [m]: '+str(pupil.size))
                print('One pixel has size of (pupil.scale) [m]: '+str(pupil.scale))
                print('Requested pupil has so many pixels (pupil_plane_im): '+str(pupil.illuminated.astype(np.int16).shape))
            else:
                print('Supplied pupil size is (diam_sic) [m]: '+str(self.diam_sic))
                print('One pixel has size of (diam_sic/npix) [m]: '+str(self.diam_sic/self.npix))
                print('Requested pupil has so many pixels (pupilExplicit): '+str(self.pupilExplicit.shape))
            

        time_end_single_2=time.time()            
        if self.verbosity==1:
            print('Time for _get_Pupil function is '+str(time_end_single_2-time_start_single_2))   
   
        time_start_single_3=time.time()          
        # create array with pixels=1 if the area is illuminated and 0 if it is obscured
        ilum=np.array(aper.illuminated, dtype=np.float32)
        assert np.sum(ilum)>0, str(self.pupil_parameters)
        
        # padding to get exact multiple when we are in focus
        # focus recognized by the fact that ilum is 1024 pixels large
        # deprecated as I always use same size of the pupil, for all amounts of defocus
        if len(ilum)==1024:
            ilum_padded=np.zeros((1158,1158))
            ilum_padded[67:67+1024,67:67+1024]=ilum
            ilum=ilum_padded
                   
        # gives size of the illuminated image
        lower_limit_of_ilum=int(ilum.shape[0]/2-self.npix/2)
        higher_limit_of_ilum=int(ilum.shape[0]/2+self.npix/2)
        if self.verbosity==1: 
            print('lower_limit_of_ilum: ' +str(lower_limit_of_ilum))
            print('higher_limit_of_ilum: ' +str(higher_limit_of_ilum))
        
        #what am I doing here?
        if self.pupilExplicit is None:
            ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]=ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]*pupil.illuminated
        else:
            ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]=ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]*self.pupilExplicit.astype(np.float32)


        if self.verbosity==1: 
            print('Size after padding zeros to 2x size and extra padding to get size suitable for FFT: '+str(ilum.shape))
               
        # maximum extent of pupil image in units of radius of the pupil, needed for next step
        size_of_ilum_in_units_of_radius=ilum.shape[0]/self.npix
        
        if self.verbosity==1:  
            print('size_of_ilum_in_units_of_radius: '+str(size_of_ilum_in_units_of_radius))
            


        # do not caculate the ``radiometric effect (difference between entrance and exit pupil) if paramters are too small to make any difference
        # if that is the case just declare the ``ilum_radiometric'' to be the same as ilum
        # i.e., the illumination of the exit pupil is the same as the illumination of the entrance pupil
        if params['radiometricExponent']<0.01 or params['radiometricEffect']<0.01:
            if self.verbosity==1:  
                print('skiping ``radiometric effect\'\' ')
            ilum_radiometric=ilum
            
            
        else:
            if self.verbosity==1: 
                print('radiometric parameters are: ')     
                print('x_ilum,y_ilum,radiometricEffect,radiometricExponent'+str([params['x_ilum'],params['y_ilum'],params['radiometricEffect'],params['radiometricExponent']]))    

            # add the change of flux between the entrance and exit pupil
            # end product is radiometricEffectArray
            points = np.linspace(-size_of_ilum_in_units_of_radius, size_of_ilum_in_units_of_radius,num=ilum.shape[0])
            xs, ys = np.meshgrid(points, points)
            _radius_coordinate = np.sqrt((xs-params['x_ilum']*params['dxFocal'])** 2 + (ys-params['y_ilum']*params['dyFocal'])** 2)
            
        
            # change in v_0.14
            # ilumination to which radiometric effet has been applied, describing difference betwen entrance and exit pupil
            radiometricEffectArray=(1+params['radiometricEffect']*_radius_coordinate**2)**(-params['radiometricExponent'])
            ilum_radiometric=np.nan_to_num(radiometricEffectArray*ilum,0) 
     
        # this is where you can introduce some apodization in the pupil image by using the line below
        # the apodization sigma is set to that in focus it is at 0.75
        # for larger images, scale according to the size of the input image which is to be FFT-ed
        # 0.75 is an arbitrary number
        apodization_sigma=((len(ilum_radiometric))/1158)**0.875*0.75
        #apodization_sigma=0.75
        time_start_single_4=time.time()
        
        # old code where I applied Gaussian to the whole ilum image
        #ilum_radiometric_apodized = gaussian_filter(ilum_radiometric, sigma=apodization_sigma)
        
        # cut out central region, apply Gaussian on the center region and return to the full size image
        # done to spped up the calculation
        ilum_radiometric_center_region=ilum_radiometric[(lower_limit_of_ilum-int(np.ceil(3*apodization_sigma))):(higher_limit_of_ilum+int(np.ceil(3*apodization_sigma))),\
                                        (lower_limit_of_ilum-int(np.ceil(3*apodization_sigma))):(higher_limit_of_ilum+int(np.ceil(3*apodization_sigma)))]
        
        ilum_radiometric_center_region_apodized=gaussian_filter(ilum_radiometric_center_region, sigma=apodization_sigma)
        
        ilum_radiometric_apodized=np.copy(ilum_radiometric)
        ilum_radiometric_apodized[(lower_limit_of_ilum-int(np.ceil(3*apodization_sigma))):(higher_limit_of_ilum+int(np.ceil(3*apodization_sigma))),\
                                        (lower_limit_of_ilum-int(np.ceil(3*apodization_sigma))):(higher_limit_of_ilum+int(np.ceil(3*apodization_sigma)))]=ilum_radiometric_center_region_apodized        
        
        time_end_single_4=time.time()
        if self.verbosity==1:
            print('Time to apodize the pupil: '+str(time_end_single_4-time_start_single_4))  
            print('type(ilum_radiometric_apodized)'+str(type(ilum_radiometric_apodized[0][0])))     
        # put pixels for which amplitude is less than 0.01 to 0
        r_ilum_pre=np.copy(ilum_radiometric_apodized)
        r_ilum_pre[ilum_radiometric_apodized>0.01]=1
        r_ilum_pre[ilum_radiometric_apodized<0.01]=0
        ilum_radiometric_apodized_bool=r_ilum_pre.astype(bool)
        
        # manual creation of aper.u and aper.v (mimicking steps which were automatically done in galsim)
        # this gives position information about each point in the exit pupil so we can apply wavefront to it

    
        #aperu_manual=[]
        #for i in range(len(ilum_radiometric_apodized_bool)):
        #    aperu_manual.append(np.linspace(-diam_sic*(size_of_ilum_in_units_of_radius/2),diam_sic*(size_of_ilum_in_units_of_radius/2),len(ilum_radiometric_apodized_bool), endpoint=True))
        single_line_aperu_manual=np.linspace(-diam_sic*(size_of_ilum_in_units_of_radius/2),diam_sic*(size_of_ilum_in_units_of_radius/2),len(ilum_radiometric_apodized_bool), endpoint=True)
        aperu_manual=np.tile(single_line_aperu_manual, len(single_line_aperu_manual)).reshape(len(single_line_aperu_manual),len(single_line_aperu_manual))

        
        # full grid
        #u_manual=np.array(aperu_manual)
        u_manual=aperu_manual
        v_manual=np.transpose(aperu_manual)     
        
        # select only parts of the grid that are actually illuminated        
        u=u_manual[ilum_radiometric_apodized_bool]
        v=v_manual[ilum_radiometric_apodized_bool]
        
        time_end_single_3=time.time()
        if self.verbosity==1:
            print('Time for postprocessing pupil after _get_Pupil '+str(time_end_single_3-time_start_single_3))      
            
        time_end_single_1=time.time()
        if self.verbosity==1:
            print('Time for pupil and illumination calculation is '+str(time_end_single_1-time_start_single_1)) 

        ################################################################################
        # wavefront
        ################################################################################
        # create wavefront across the exit pupil   
        
        time_start_single=time.time()
        if self.verbosity==1:  
            print('')    
            print('Starting creation of wavefront')    
   
        
        aberrations_init=[0.0,0,0.0,0.0]
        aberrations = aberrations_init
        # list of aberrations where we set z4, z11, z22 etc. to 0 do study behaviour of non-focus terms
        aberrations_0=list(np.copy(aberrations_init))  
        for i in range(4, self.zmax + 1):
            aberrations.append(params['z{}'.format(i)]) 
            if i in [4,11,22]:
                aberrations_0.append(0)
            else:
                aberrations_0.append(params['z{}'.format(i)]) 
        
        
        if self.extraZernike==None:
            pass
        else:
            aberrations_extended=np.concatenate((aberrations,self.extraZernike),axis=0)

        
        if self.verbosity==1:   
            print('diam_sic [m]: '+str(diam_sic))
            print('aberrations: '+str(aberrations))
            print('aberrations moved to z4=0: '+str(aberrations_0))
            print('aberrations extra: '+str(self.extraZernike))
            print('wavelength [nm]: '+str(self.wavelength))
        
        if self.extraZernike==None:
            optics_screen = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations,lam_0=self.wavelength)
            if self.save==1:
                # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations_0,lam_0=self.wavelength)
        else:
            optics_screen = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations_extended,lam_0=self.wavelength)      
            if self.save==1:
                # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations_0,lam_0=self.wavelength)            
       
        screens = galsim.PhaseScreenList(optics_screen)   
        if self.save==1:
            # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
            screens_fake_0 = galsim.PhaseScreenList(optics_screen_fake_0)  
        
        time_end_single=time.time()

        ################################################################################
        # combining pupil illumination and wavefront
        ################################################################################        
        

        # apply wavefront to the array describing illumination
        #print(self.use_wf_grid)

        if self.use_wf_grid is None:
            
            wf = screens.wavefront(u, v, None, 0)
            if self.save==1:
                wf_full = screens.wavefront(u_manual, v_manual, None, 0)
            wf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.float32)
            wf_grid[ilum_radiometric_apodized_bool] = (wf/self.wavelength)
            wf_grid_rot=wf_grid
        else:
            # if you want to pass an explit wavefront, it goes here

            wf_grid=self.use_wf_grid
            wf_grid_rot=wf_grid
        

        
        #wf_grid_rot[756+1200:756+1215,756+1200:756+1215]=1.1*(wf_grid_rot[756+1200:756+1215,756+1200:756+1215])
        
        
        #if self.save==1 and self.extraZernike==None:
        if self.save==1:
            # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
            if self.verbosity==1:
                print('creating wf_full_fake_0')
            wf_full_fake_0 = screens_fake_0.wavefront(u_manual, v_manual, None, 0)
        
        
        # exponential of the wavefront
        expwf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.complex64)
        expwf_grid[ilum_radiometric_apodized_bool] =ilum_radiometric_apodized[ilum_radiometric_apodized_bool]*np.exp(2j*np.pi * wf_grid_rot[ilum_radiometric_apodized_bool])
        
        if self.verbosity==1:
            print('Time for wavefront and wavefront/pupil combining is '+str(time_end_single-time_start_single)) 
            #print('type(expwf_grid)'+str(type(expwf_grid[0][0])))
        ################################################################################
        # FFT
        ################################################################################    

        ######################################################################
        # Different implementations of the Fourier code

        # legacy code
        # do Fourier via galsim and square it to create image
        # ftexpwf = galsim.fft.fft2(expwf_grid,shift_in=True,shift_out=True)
        
        # uncoment to get timming 
        #time_start_single=time.time()
        #ftexpwf =np.fft.fftshift(np.fft.fft2(np.fft.fftshift(expwf_grid)))
        #img_apod = np.abs(ftexpwf)**2
        #time_end_single=time.time()
        #if self.verbosity==1:
        #    print('Time for FFT is '+str(time_end_single-time_start_single))
        #    print('type(np.fft.fftshift(expwf_grid)'+str(type(np.fft.fftshift(expwf_grid)[0][0])))
        #    print('type(np.fft.fft2(np.fft.fftshift(expwf_grid)))'+str(type(np.fft.fft2(np.fft.fftshift(expwf_grid))[0][0])))
        #    print('type(ftexpwf)'+str(type(ftexpwf[0][0])))
        #    print('type(img_apod)'+str(type(img_apod[0][0])))
            
        time_start_single=time.time()
        ftexpwf =np.fft.fftshift(scipy.fftpack.fft2(np.fft.fftshift(expwf_grid)))
        img_apod = np.abs(ftexpwf)**2
        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for FFT is '+str(time_end_single-time_start_single))
            #print('type(np.fft.fftshift(expwf_grid)'+str(type(np.fft.fftshift(expwf_grid)[0][0])))
            #print('type(np.fft.fft2(np.fft.fftshift(expwf_grid)))'+str(type(scipy.fftpack.fft2(np.fft.fftshift(expwf_grid))[0][0])))
            #print('type(ftexpwf)'+str(type(ftexpwf[0][0])))
            #print('type(img_apod)'+str(type(img_apod[0][0])))            
 

        # code if we decide to use pyfftw - does not work with fftshift
        # time_start_single=time.time()
        # ftexpwf =np.fft.fftshift(pyfftw.builders.fft2(np.fft.fftshift(expwf_grid)))
        # img_apod = np.abs(ftexpwf)**2
        # time_end_single=time.time()
        # print('Time for FFT is '+str(time_end_single-time_start_single))
        ######################################################################


        # size in arcseconds of the image generated by the code
        scale_ModelImage_PFS_naturalResolution=sky_scale(size_of_ilum_in_units_of_radius*self.diam_sic,self.wavelength)
        self.scale_ModelImage_PFS_naturalResolution=scale_ModelImage_PFS_naturalResolution
 
        if self.save==1:
            if socket.gethostname()=='IapetusUSA' or socket.gethostname()=='tiger2-sumire.princeton.edu':
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'aperilluminated',aper.illuminated)  
                # dont save as we do not generate this array in vast majority of cases (status on July 1, 2020)
                #np.save(TESTING_PUPIL_IMAGES_FOLDER+'radiometricEffectArray',radiometricEffectArray)     
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum',ilum)   
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum_radiometric',ilum_radiometric) 
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum_radiometric_apodized',ilum_radiometric_apodized) 
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum_radiometric_apodized_bool',ilum_radiometric_apodized_bool)

                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'u_manual',u_manual) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'v_manual',v_manual) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'u',u) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'v',v) 
                
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_grid',wf_grid)  
                if self.use_wf_grid is None:
                    np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full',wf_full) 
                #if self.extraZernike==None:
                #    np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full_fake_0',wf_full_fake_0)       
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full_fake_0',wf_full_fake_0)  
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'expwf_grid',expwf_grid)   

        if self.verbosity==1:
            print('Finished with _getOptPsf_naturalResolution')  
            print(' ')
        
        if return_intermediate_images==False:
            return img_apod
        if return_intermediate_images==True:
            # return the image, pupil, illumination applied to the pupil
            return img_apod,ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum],wf_grid_rot




    def _chi_PFS(self, params):
        """Compute 'chi' image: (data - model)/sigma
        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        modelImg = self.constructModelImage_PFS(params)
        sigma = np.sqrt(self.image_var)
        chi = (self.image - modelImg)/sigma
        chi_without_nan=[]
        chi_without_nan = chi.ravel()[~np.isnan(chi.ravel())]
        print("chi**2/d.o.f. is:"+str(np.mean((chi_without_nan)**2)))
        return chi_without_nan
       
    def best_image_Neven(self, params):
        """
        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        modelImg = self.constructModelImage_Neven(params)
        return modelImg
    
    def residual_image_Neven(self, params):
        """
        @param params  lmfit.Parameters object.
        @returns       residual 
        """
        modelImg = self.constructModelImage_Neven(params)
        return (self.image - modelImg)
    
    def fit_emcee(self):
        """Do the fit using emcee
        @returns  result as an lmfit.MinimizerResult.
        """
        print("Doing fit using emcee")
        mini = lmfit.Minimizer(self._chi_PFS, self.params)
        self.result = mini.emcee(nwalkers=64,burn=100, steps=200, thin=1, 
                                 is_weighted=True,ntemps=1,workers=1,**self.kwargs)
        return self.result 
    
    def fit_LM(self):
        """Do the fit using Levenberg-Marquardt 
        @returns  result as an lmfit.MinimizerResult.
        """
        print("Doing fit using Levenberg-Marquardt")
        self.result = lmfit.minimize(self._chi_PFS, self.params,**self.kwargs)
        return self.result
    
    def report(self, *args, **kwargs):
        """Return a string with fit results."""
        return lmfit.fit_report(self.result, *args, **kwargs) 

    def Exit_pupil_size(x):
        return (656.0581848529348+ 0.7485514705882259*x)*10**-3

    def F_number_size(x):
        return -1.178009972058799 - 0.00328027941176467* x

    def Pixel_size(x):
        return 206265 *np.arctan(0.015/((-1.178009972058799 - 0.00328027941176467* x)*(656.0581848529348 + 0.7485514705882259* x)))


class LN_PFS_multi_same_spot(object):
 
    """!
    
    Class to compute likelihood of the multiple donut images, of the same spot taken at different defocuses
    
    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax,save=1)    
    def model_return(allparameters_proposal):
        return model(allparameters_proposal,return_Image=True)
    
    calls LN_PFS_single
    
    Called by Tokovinin_multi
    
    
    """
    
    
    def __init__(self,list_of_sci_images,list_of_var_images,list_of_mask_images=None,wavelength=None,dithering=None,save=None,verbosity=None,
             pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,list_of_wf_grid=None,
             zmax=None,extraZernike=None,pupilExplicit=None,simulation_00=None,
             double_sources=None,double_sources_positions_ratios=None,npix=None,
             list_of_defocuses=None,fit_for_flux=True,test_run=False,list_of_psf_positions=None,
             use_center_of_flux=False): 

     
        """
        @param list_of_sci_images                      list of science images, list of 2d array
        @param list_of_var_images                      list of variance images, 2d arrays, which are the same size as sci_image
        @param list_of_mask_images                     list of mask images, 2d arrays, which are the same size as sci_image
        @param dithering                               dithering, 1=normal, 2=two times higher resolution, 3=not supported
        @param save                                    save intermediate result in the process (set value at 1 for saving)
        @param verbosity                               verbosity of the process (set value at 1 for full output)
        
        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF
        
        @param zmax                                    largest Zernike order used (11 or 22, or larger than 22)
        @param extraZernike                            array consisting of higher order zernike (if using higher order than 22)
        @param pupilExplicit
        
        @param simulation_00                           resulting image will be centered with optical center in the center of the image 
                                                       and not fitted acorrding to the sci_image
        @param double_sources                          1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /       arrray with parameters describing relative position\
                                                       and relative flux of the secondary source(s)
        @param npxix                                   size of the pupil (1536 reccomended)
        @param list_of_defocuses                       list of defocuses at which images are taken (float or string?)
        
        @param fit_for_flux                            automatically fit for the best flux level that minimizes the chi**2
        @param test_run                                if True, skips the creation of model and return science image - useful for testing
                                                       interaction of outputs of the module in broader setting quickly 
        @param explicit_psf_position                   gives position of the opt_psf
        """      



                
        if verbosity is None:
            verbosity=0
                
        if use_pupil_parameters is not None:
            assert pupil_parameters is not None

        if double_sources is not None and double_sources is not False:      
            assert np.sum(np.abs(double_sources_positions_ratios))>0    

        if zmax is None:
            zmax=11
       
        """             
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax==22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']    
        """  
        
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'wide_0','wide_23','wide_43','misalign',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax>=22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'wide_0','wide_23','wide_43','misalign',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  

        
        self.list_of_sci_images=list_of_sci_images
        self.list_of_var_images=list_of_var_images
        
        if list_of_mask_images is None:
            list_of_mask_images=[]
            for i in range(len(list_of_sci_images)):
                mask_image=np.zeros(list_of_sci_images[i].shape)
                list_of_mask_images.append(mask_image)
                
        
        self.list_of_mask_images=list_of_mask_images
        
        
        #self.mask_image=mask_image
        #self.sci_image=sci_image
        #self.var_image=var_image
        self.wavelength=wavelength
        self.dithering=dithering
        self.save=save
        self.pupil_parameters=pupil_parameters
        self.use_pupil_parameters=use_pupil_parameters
        self.use_optPSF=use_optPSF
        self.pupilExplicit=pupilExplicit
        self.simulation_00=simulation_00      
        self.zmax=zmax
        self.extraZernike=extraZernike
        self.verbosity=verbosity
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios
        self.npix=npix
        self.fit_for_flux=fit_for_flux
        self.list_of_defocuses=list_of_defocuses
        self.test_run=test_run
        if list_of_psf_positions is None:
            list_of_psf_positions=[None]*len(list_of_sci_images)
        self.list_of_psf_positions=list_of_psf_positions
        if list_of_wf_grid is None:
            list_of_wf_grid=[None]*len(list_of_sci_images)
        self.list_of_wf_grid=list_of_wf_grid
        
        #self.use_only_chi=use_only_chi
        self.use_center_of_flux=use_center_of_flux
      
        
    def move_parametrizations_from_1d_to_2d(self,allparameters_parametrizations_1d,zmax=None):
        
        # 22 parameters has len of 61
        if zmax is None:
            zmax=int((len(allparameters_parametrizations_1d)-61)/2+22)
        
        assert len(allparameters_parametrizations_1d.shape)==1
        

        z_parametrizations=allparameters_parametrizations_1d[:19*2].reshape(19,2)
        g_parametrizations=np.transpose(np.vstack((np.zeros(len(allparameters_parametrizations_1d[19*2:19*2+23])),\
                                                   allparameters_parametrizations_1d[19*2:19*2+23])))
            
            
        if zmax>22:
            extra_Zernike_parameters_number=zmax-22
            z_extra_parametrizations=allparameters_parametrizations_1d[19*2+23:].reshape(extra_Zernike_parameters_number,2)    
        
        if zmax<=22:
            allparameters_parametrizations_2d=np.vstack((z_parametrizations,g_parametrizations))   
        if zmax>22:
            allparameters_parametrizations_2d=np.vstack((z_parametrizations,g_parametrizations,z_extra_parametrizations))   
        
        
        #print('allparameters_parametrizations_2d[41]: '+ str(allparameters_parametrizations_2d[41]))
        #assert allparameters_parametrizations_2d[41][1] >= 0.98
        #assert allparameters_parametrizations_2d[41][1] <= 1.02        
        
        return allparameters_parametrizations_2d
        
        
    def create_list_of_allparameters(self,allparameters_parametrizations,list_of_defocuses=None,zmax=None):
        """
        given the parametrizations (in either 1d or 2d ), create list_of_allparameters to be used in analysis of single images
        
        """
        
        #print('allparameters_parametrizations '+str(allparameters_parametrizations))
        
        if zmax is None:
            zmax=self.zmax
        
        # if you have passed parameterization in 1d, move to 2d 
        #print("allparameters_parametrizations.type: "+str(type(allparameters_parametrizations)))
        #print("allparameters_parametrizations.len: "+str(+len(allparameters_parametrizations)))
        #print("allparameters_parametrizations.shape: "+str(allparameters_parametrizations.shape))
        if len(allparameters_parametrizations.shape)==1:
            allparameters_parametrizations=self.move_parametrizations_from_1d_to_2d(allparameters_parametrizations)
            
        list_of_allparameters=[]
        
        # if this is only a single image, just return the input
        if list_of_defocuses==None:
            return allparameters_parametrizations
        else:
            list_of_defocuses_int=self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses)
            #print(list_of_defocuses_int)
            # go through the list of defocuses, and create the allparameters array for each defocus
            for i in range(len(list_of_defocuses)):
                list_of_allparameters.append(self.create_allparameters_single(list_of_defocuses_int[i],allparameters_parametrizations,zmax))
            
            #print(list_of_allparameters)
            
            return list_of_allparameters
            
          
            
    def value_at_defocus(self,mm,a,b=None):
        """
        transform parameters of linear fit to a value at a given defocus (in mm) 
        
        @param mm                     slit defocus in mm
        @param a                      linear parameter
        @param b                      contstant offset
        
        """
        
        if b==None:
            return a
        else:
            return a*mm+b 
        
    def create_allparameters_single(self,mm,array_of_polyfit_1_parameterizations,zmax=None):
        """
        transforms 1d array linear fits as a function of defocus of parametrizations into form acceptable for creating single images 
        workhorse function used by create_list_of_allparameters
        
        @param mm [float]                               defocus of the slit
        @param array_of_polyfit_1_parameterizations     parameters describing linear fit for the parameters as a function of focus
        @param zmax                                     largest Zernike used
        
        """
        
        if zmax==None:
            # if len is 42, the zmax is 22
            zmax=array_of_polyfit_1_parameterizations.shape[0]-42+22
            if zmax>22:
                extra_Zernike_parameters_number=zmax-22
        else:
            extra_Zernike_parameters_number=zmax-22
            

        
        #for single case, up to z11
        if zmax==11:
            z_parametrizations=array_of_polyfit_1_parameterizations[:8]
            g_parametrizations=array_of_polyfit_1_parameterizations[8:]
            
            
            allparameters_proposal_single=np.zeros((8+len(g_parametrizations)))
            
            for i in range(0,8,1):
                allparameters_proposal_single[i]=self.value_at_defocus(mm,z_parametrizations[i][0],z_parametrizations[i][1])      
        
            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[i+8]=g_parametrizations[i][1] 
                
        if zmax>=22:
            z_parametrizations=array_of_polyfit_1_parameterizations[:19]
            g_parametrizations=array_of_polyfit_1_parameterizations[19:19+23]
            
            if extra_Zernike_parameters_number>0:
                z_extra_parametrizations=array_of_polyfit_1_parameterizations[42:]
            
            
            allparameters_proposal_single=np.zeros((19+len(g_parametrizations)+extra_Zernike_parameters_number))
            
            
            for i in range(0,19,1):
                #print(str([i,mm,z_parametrizations[i]]))
                allparameters_proposal_single[i]=self.value_at_defocus(mm,z_parametrizations[i][0],z_parametrizations[i][1])      
        
            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[19+i]=g_parametrizations[i][1] 
                
            for i in range(0,extra_Zernike_parameters_number,1):
                #print(str([i,mm,z_parametrizations[i]]))
                allparameters_proposal_single[19+len(g_parametrizations)+i]=self.value_at_defocus(mm,z_extra_parametrizations[i][0],z_extra_parametrizations[i][1])      
            
        return allparameters_proposal_single           
    
    def transform_list_of_defocuses_from_str_to_float(self,list_of_defocuses):
        """
        transfroms list of defocuses from strings to float values

        
        @param list_of_defocuses                        list of defocuses in string form (e.g., [m4,m25,0,p15,p4])
        
        """        
        
        list_of_defocuses_float=[]
        for i in range(len(list_of_defocuses)):
            if list_of_defocuses[i][0]=='0':
                list_of_defocuses_float.append(0)
            else:
                if list_of_defocuses[i][0]=='m':
                    sign=-1
                if list_of_defocuses[i][0]=='p':
                    sign=+1
                if len(list_of_defocuses[i])==2:
                    list_of_defocuses_float.append(sign*float(list_of_defocuses[i][1:]))
                else:
                    list_of_defocuses_float.append(sign*float(list_of_defocuses[i][1:])/10)
                    
        return list_of_defocuses_float
            
    
    def create_resonable_allparameters_parametrizations(self,array_of_allparameters,list_of_defocuses_input,zmax,remove_last_n=None):
        """
        given parameters for single defocus images and their defocuses, create parameterizations (1d functions) for multi-image fit across various defocuses
        inverse of function `create_list_of_allparameters`
        
        @param array_of_allparameters                        array with parameters of defocus, 2d array with shape [n(list_of_defocuses),number of parameters]
        @param list_of_defocuses_input                       list of strings at which defocuses are the data from array_of_allparameters
        @param zmax                                          largest Zernike order considered (has to be same for input and output) 
        @param remove_last_n                                 do not do the fit for the last 'n' parameters, default=2
        """           
        
        if remove_last_n is None:
            remove_last_n=2
        
        list_of_defocuses_int=self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses_input)
        if remove_last_n>0:
            array_of_allparameters=array_of_allparameters[:,:-remove_last_n]
        
        if zmax<=22:
            len_of_iterations=array_of_allparameters.shape[1]
        else:
            len_of_iterations=42+zmax-22
            
        list_of_polyfit_1_parameter=[]
        for i in range(len_of_iterations):
            #print([i,array_of_allparameters.shape[1]])
            if i<array_of_allparameters.shape[1]:
                #print('i'+str(i)+' '+str(array_of_allparameters[:,i]))
                polyfit_1_parameter=np.polyfit(x=list_of_defocuses_int,y=array_of_allparameters[:,i],deg=1)
            else:
                #print('i'+str(i)+' '+'None')
                # if you have no input for such high level of Zernike, set it at zero
                polyfit_1_parameter=np.array([0,0])
                
            #print('i_polyfit'+str(i)+' '+str(polyfit_1_parameter))
            list_of_polyfit_1_parameter.append(polyfit_1_parameter)
            
        array_of_polyfit_1_parameterizations=np.array(list_of_polyfit_1_parameter)        
        
        #list_of_defocuses_output_int=self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses_input)
        #list_of_allparameters=[]
        #for i in list_of_defocuses_output_int:
        #    allparameters_proposal_single=self.create_allparameters_single(i,array_of_polyfit_1_parameterizations,zmax=self.zmax)
        #    list_of_allparameters.append(allparameters_proposal_single)
            
        return array_of_polyfit_1_parameterizations
        

    def lnlike_Neven_multi_same_spot(self,list_of_allparameters_input,return_Images=False,\
                                     use_only_chi=False,multi_background_factor=3):
        
        self.use_only_chi=use_only_chi
        
        list_of_single_res=[]
        if return_Images==True:
            list_of_single_model_image=[]
            list_of_single_allparameters=[]
            list_of_single_chi_results=[]
      
        if len(self.list_of_sci_images)==len(list_of_allparameters_input):
            list_of_allparameters=np.copy(list_of_allparameters_input)

        else:
            allparametrization=list_of_allparameters_input
            
            print('self.list_of_defocuses: '+str(self.list_of_defocuses))
            print('allparametrization.type: '+str(allparametrization.type))
            list_of_allparameters=self.create_list_of_allparameters(allparametrization,list_of_defocuses=self.list_of_defocuses)

            if self.verbosity==1:
                print('Starting LN_PFS_multi_same_spot for parameters-hash '+str(hash(str(allparametrization.data)))+' at '+str(time.time())+' in thread '+str(threading.get_ident())) 
        
        assert len(self.list_of_sci_images)==len(list_of_allparameters)
        

        #print(len(self.list_of_sci_images))
        #print(len(list_of_allparameters)) 

        
   
        # use same weights, experiment
        #if use_only_chi==True:
        #    renormalization_of_var_sum=np.ones((len(self.list_of_sci_images)))*len(self.list_of_sci_images)
        #    central_index=int(len(self.list_of_sci_images)/2)
        #    renormalization_of_var_sum[central_index]=1
            
        #else:
            
  
        # find image with lowest variance - pressumably the one in focus
        array_of_var_sum=np.array(list(map(np.sum,self.list_of_var_images)))
        index_of_max_var_sum=np.where(array_of_var_sum==np.min(array_of_var_sum))[0][0]
        # find what variance selectes top 20% of pixels
        # this is done to weight more the images in focus and less the image out of focus in the 
        # final likelihood result
        #quantile_08_focus=np.quantile(self.list_of_sci_images[index_of_max_var_sum],0.8)
        
        
        list_of_var_sums=[]
        for i in range(len(list_of_allparameters)):
            # taking from create_chi_2_almost function in LN_PFS_single
            
            
            mask_image=self.list_of_mask_images[i]
            var_image=self.list_of_var_images[i]
            sci_image=self.list_of_sci_images[i]
            # array that has True for values which are good and False for bad values
            inverted_mask=~mask_image.astype(bool)
            
            try:
                if sci_image.shape[0]==20:
                    multi_background_factor=3
                    
        
                mean_value_of_background_via_var=np.mean([np.median(var_image[0]),np.median(var_image[-1]),\
                                                      np.median(var_image[:,0]),np.median(var_image[:,-1])])*multi_background_factor
             
                mean_value_of_background_via_sci=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
                                                      np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*multi_background_factor
                    
                mean_value_of_background=np.max([mean_value_of_background_via_var,mean_value_of_background_via_sci])
            except:
                pass

            # select only images with above 80% percentile of the image with max variance?         
            var_image_masked=var_image*inverted_mask
            var_image_masked_without_nan = var_image_masked.ravel()[var_image_masked.ravel()>mean_value_of_background]
            
            
            
            if use_only_chi==True:
                # if you level is too high
                if len(var_image_masked_without_nan)==0:
                    var_sum=-1
                else:
                    #var_sum=-(1)*(np.sum(np.sqrt(np.abs(var_image_masked_without_nan))))
                    var_sum=-1

            else:


                # if you level is too high
                if len(var_image_masked_without_nan)==0:
                    var_sum=-(1)
                else:
                    var_sum=-(1)*(np.mean(np.abs(var_image_masked_without_nan)))
            list_of_var_sums.append(var_sum)
    
    
            # renormalization needs to be reconsidered?
            array_of_var_sum=np.array(list_of_var_sums)
            max_of_array_of_var_sum=np.max(array_of_var_sum)
            
    
            renormalization_of_var_sum=array_of_var_sum/max_of_array_of_var_sum
            #print('renormalization_of_var_sum'+str(renormalization_of_var_sum))
        list_of_psf_positions_output=[]



        for i in range(len(list_of_allparameters)):
            
            # if image is in focus which at this point is the size of image with 20
            
            if (self.list_of_sci_images[i].shape)[0]==20:
                if self.use_center_of_flux==True:
                    use_center_of_flux=True
                else:
                    use_center_of_flux=False
            else:
                use_center_of_flux=False
                
            
            

            if self.verbosity==1:
                print('################################')
                print('analyzing image '+str(i+1)+' out of '+str(len(list_of_allparameters)))
                print(' ')           

            # if this is the first image, do the full analysis, generate new pupil and illumination
            if i==0:
                model_single=LN_PFS_single(self.list_of_sci_images[i],self.list_of_var_images[i],self.list_of_mask_images[i],
                                           wavelength=self.wavelength,dithering=self.dithering,save=self.save,verbosity=self.verbosity,
                pupil_parameters=self.pupil_parameters,use_pupil_parameters=self.use_pupil_parameters,use_optPSF=self.use_optPSF,
                use_wf_grid=self.list_of_wf_grid[i],
                zmax=self.zmax,extraZernike=self.extraZernike,pupilExplicit=self.pupilExplicit,simulation_00=self.simulation_00,
                double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,npix=self.npix,
                fit_for_flux=self.fit_for_flux,test_run=self.test_run,explicit_psf_position=self.list_of_psf_positions[i],
                use_only_chi=self.use_only_chi,use_center_of_flux=use_center_of_flux)

                res_single_with_intermediate_images=model_single(list_of_allparameters[i],\
                                                                 return_Image=True,return_intermediate_images=True,
                                                                 use_only_chi=use_only_chi,\
                                                                 multi_background_factor=multi_background_factor)
                #print(res_single_with_intermediate_images)
                if res_single_with_intermediate_images==-np.inf:
                    return -np.inf
                if type(res_single_with_intermediate_images)==tuple:
                    if res_single_with_intermediate_images[0]==-np.inf:
                        return -np.inf                    
                likelihood_result=res_single_with_intermediate_images[0]
                model_image=res_single_with_intermediate_images[1]
                allparameters=res_single_with_intermediate_images[2]                        
                pupil_explicit_0=res_single_with_intermediate_images[3]
                wf_grid_rot=res_single_with_intermediate_images[4]
                chi_results=res_single_with_intermediate_images[5]  
                psf_position=res_single_with_intermediate_images[6]
                
                
                list_of_single_res.append(likelihood_result)
                list_of_psf_positions_output.append(psf_position)
                if return_Images==True:
                    list_of_single_model_image.append(model_image)
                    list_of_single_allparameters.append(allparameters)
                    list_of_single_chi_results.append(chi_results)
                
                
            # and if this is not the first image, use the pupil and illumination used in the first image
            else:               
                
                model_single=LN_PFS_single(self.list_of_sci_images[i],self.list_of_var_images[i],self.list_of_mask_images[i],\
                                           wavelength=self.wavelength,dithering=self.dithering,save=self.save,verbosity=self.verbosity,
                pupil_parameters=self.pupil_parameters,use_pupil_parameters=self.use_pupil_parameters,use_optPSF=self.use_optPSF,
                use_wf_grid=self.list_of_wf_grid[i],
                zmax=self.zmax,extraZernike=self.extraZernike,pupilExplicit=pupil_explicit_0,simulation_00=self.simulation_00,
                double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,npix=self.npix,
                fit_for_flux=self.fit_for_flux,test_run=self.test_run,explicit_psf_position=self.list_of_psf_positions[i],
                use_only_chi=self.use_only_chi,use_center_of_flux=use_center_of_flux)
                if return_Images==False:

                    res_single_without_intermediate_images=model_single(list_of_allparameters[i],\
                                                                        return_Image=return_Images,\
                                                                        use_only_chi=use_only_chi,\
                                                                        multi_background_factor=multi_background_factor)
                    #print(res_single_without_intermediate_images)
            
                    likelihood_result=res_single_without_intermediate_images[0]
                    psf_position=res_single_with_intermediate_images[-1]
                    #print(likelihood_result)
                    list_of_single_res.append(likelihood_result)
                    list_of_psf_positions_output.append(psf_position)

                if return_Images==True:
                    res_single_with_an_image=model_single(list_of_allparameters[i],return_Image=return_Images,use_only_chi=use_only_chi)
                    if res_single_with_an_image==-np.inf:
                        return -np.inf
                    likelihood_result=res_single_with_an_image[0]
                    model_image=res_single_with_an_image[1]
                    allparameters=res_single_with_an_image[2]
                    chi_results=res_single_with_an_image[3]
                    psf_position=res_single_with_an_image[-1]
                    
                    list_of_single_res.append(likelihood_result)
                    list_of_single_model_image.append(model_image)
                    list_of_single_allparameters.append(allparameters)
                    list_of_single_chi_results.append(chi_results)                    
                    list_of_psf_positions_output.append(psf_position)                   
                    # possibly implement intermediate images here



        
        array_of_single_res=np.array(list_of_single_res)
        array_of_psf_positions_output=np.array(list_of_psf_positions_output)

        
        # renormalization
        
        if self.verbosity==1:
            print('################################')
            print('Likelihoods returned per individual images are: '+str(array_of_single_res))
            print('Mean likelihood is '+str(np.mean(array_of_single_res)))
            

  
        #mean_res_of_multi_same_spot=np.mean(array_of_single_res)
        mean_res_of_multi_same_spot=np.mean(array_of_single_res/renormalization_of_var_sum)

        
        if self.verbosity==1:
            print('################################')
            print('Renormalized likelihoods returned per individual images are: '+str(array_of_single_res/renormalization_of_var_sum))
            print('Renormalization factors are: ' + str(renormalization_of_var_sum))
            print('Mean renormalized likelihood is '+str(mean_res_of_multi_same_spot))
            print('array_of_psf_positions_output: '+str(array_of_psf_positions_output))
        
        if self.verbosity==1:
            #print('Ending LN_PFS_multi_same_spot for parameters-hash '+str(hash(str(allparametrization.data)))+' at '+str(time.time())+' in thread '+str(threading.get_ident()))   
            print('Ending LN_PFS_multi_same_spot at time '+str(time.time())+' in thread '+str(threading.get_ident()))   
            print(' ')        
        
        if return_Images==False:
            return mean_res_of_multi_same_spot
        if return_Images==True:
            # 0. mean_res_of_multi_same_spot - mean likelihood per images, renormalized
            # 1. list_of_single_res - likelihood per image, not renormalized
            # 2. list_of_single_model_image - list of created model images
            # 3. list_of_single_allparameters - list of parameters per image?
            # 4. list_of_single_chi_results - list of arrays describing quality of fitting
            #           1. chi2_max value, 2. Qvalue, 3. chi2/d.o.f., 4. chi2_max/d.o.f.  
            # 5. array_of_psf_positions_output - list showing the centering of images
            
            return mean_res_of_multi_same_spot,list_of_single_res,list_of_single_model_image,\
                list_of_single_allparameters,list_of_single_chi_results,array_of_psf_positions_output
        


    def __call__(self, list_of_allparameters,return_Images=False,use_only_chi=False,multi_background_factor=3):
            return self.lnlike_Neven_multi_same_spot(list_of_allparameters,return_Images=return_Images,\
                                                     use_only_chi=use_only_chi,\
                                                     multi_background_factor=multi_background_factor)
            
class Tokovinin_multi(object):
    
    """
    
    # improvments possible - modify by how much to move parameters based on the previous step
    # in simplied H, take new model into account where doing changes
    
    ########################################
    # old 
    final_model_result,list_of_final_model_result,list_of_image_final,\
    allparameters_parametrization_proposal_after_iteration,\
    list_of_finalinput_parameters,list_of_after_chi2,list_of_final_psf_positions
    
    returns:
        
    0. likelihood averaged over all images
    1. likelihood per image (output from model_multi)
    2. out_images
    3. list of final model images
    4. parametrization after the function
    5. list of parameters per image 
    6. list of chi2 per image 
    7. list of psf position of image
    
    ########################################    
    # new version
    initial_model_result,final_model_result,\
    list_of_initial_model_result,list_of_final_model_result,\
    out_images, pre_images, list_of_image_final,\
    allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
    list_of_initial_input_parameters, list_of_finalinput_parameters,\
    list_of_pre_chi2,list_of_after_chi2,\
    list_of_psf_positions,list_of_final_psf_positions,\
    [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions] 
    
    0. likelihood averaged over all images (before the function)    
    1. likelihood averaged over all images (after the function)
    2. likelihood per image (output from model_multi) (before the function)
    3. likelihood per image (output from model_multi) (after the function)
    4. out_images
    5. list of initial model images
    6. list of final model images 
    7. parametrization before the function
    8. parametrization after the function
    9. list of parameters per image (before the function)
    10. list of parameters per image (after the function)
    11. list of chi2 per image (before the function)
    12. list of chi2 per image (after the function)
    13. list of psf position of image (function the function)
    14. list of psf position of image (after the function)
    
    15. [uber_images_normalized,uber_M0,H,array_of_delta_z_parametrizations_None,list_of_final_psf_positions]
    15.0. uber_images_normalized
    15.1. uber_M0
    15.2. H
    15.3. array_of_delta_z_parametrizations_None
    15.4. list_of_final_psf_positions
    
    
    
    """
    
    
    
    def __init__(self,list_of_sci_images,list_of_var_images,list_of_mask_images=None,
                 wavelength=None,dithering=None,save=None,verbosity=None,
             pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,list_of_wf_grid=None,
             zmax=None,extraZernike=None,pupilExplicit=None,simulation_00=None,
             double_sources=None,double_sources_positions_ratios=None,npix=None,
             list_of_defocuses=None,fit_for_flux=True,test_run=False,list_of_psf_positions=None,
             num_iter=None,move_allparameters=None,pool=None): 

     
        """
        @param list_of_sci_images                      list of science images, list of 2d array
        @param list_of_var_images                      list of variance images, 2d arrays, which are the same size as sci_image
        @param list_of_mask_images                     list of mask images, 2d arrays, which are the same size as sci_image
        @param wavelength
        @param dithering                               dithering, 1=normal, 2=two times higher resolution, 3=not supported
        @param save                                    save intermediate result in the process (set value at 1 for saving)
        @param verbosity                               verbosity of the process (set value at 2 for full output, 1 only in Tokovinin, 0==nothing)
        
        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF
        
        @param zmax                                    largest Zernike order used (11 or 22, or larger than 22)
        @param extraZernike                            array consisting of higher order zernike (if using higher order than 22)
        @param pupilExplicit
        
        @param simulation_00                           resulting image will be centered with optical center in the center of the image 
                                                       and not fitted acorrding to the sci_image
        @param double_sources                          1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /       arrray with parameters describing relative position\
                                                       and relative flux of the secondary source(s)
        @param npxix                                   size of the pupil (1536 reccomended)
        @param list_of_defocuses                       list of defocuses at which images are taken (float or string?)
        
        @param fit_for_flux                            automatically fit for the best flux level that minimizes the chi**2
        @param test_run                                if True, skips the creation of model and return science image - useful for testing
                                                       interaction of outputs of the module in broader setting quickly 
                                                       
        @param list_of_psf_positions                   gives position of the opt_psf
        @param num_iter                                number of iteration
        @param move_allparameters                      if True change all parameters i.e., also ``global'' parameters, i.e., not just wavefront parameters
        @param pool                                    pass pool of workers to calculate
        
        array of changes due to movement due to wavefront changes
        """      



                
        if verbosity is None:
            verbosity=0
                
        if use_pupil_parameters is not None:
            assert pupil_parameters is not None

        if double_sources is not None and double_sources is not False:      
            assert np.sum(np.abs(double_sources_positions_ratios))>0    

        if zmax is None:
            zmax=22
                       
            
        """             
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax==22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']    
        """  
        
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'wide_0','wide_23','wide_43','misalign',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax>=22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'wide_0','wide_23','wide_43','misalign',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
        
        self.list_of_sci_images=list_of_sci_images
        self.list_of_var_images=list_of_var_images
        
        if list_of_mask_images is None:
            list_of_mask_images=[]
            for i in range(len(list_of_sci_images)):
                mask_image=np.zeros(list_of_sci_images[i].shape)
                list_of_mask_images.append(mask_image)
                
        
        self.list_of_mask_images=list_of_mask_images
        
        
        # implement custom variance image here
        
        
        #self.mask_image=mask_image
        #self.sci_image=sci_image
        #self.var_image=var_image
        self.wavelength=wavelength
        self.dithering=dithering
        self.save=save
        self.pupil_parameters=pupil_parameters
        self.use_pupil_parameters=use_pupil_parameters
        self.use_optPSF=use_optPSF
        self.pupilExplicit=pupilExplicit
        self.simulation_00=simulation_00      
        self.zmax=zmax
        self.extraZernike=extraZernike
        self.verbosity=verbosity
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios
        self.npix=npix
        self.fit_for_flux=fit_for_flux
        self.list_of_defocuses=list_of_defocuses
        self.test_run=test_run
        if list_of_psf_positions is None:
            list_of_psf_positions=[None]*len(list_of_sci_images)
        self.list_of_psf_positions=list_of_psf_positions
        if list_of_wf_grid is None:
            list_of_wf_grid=[None]*len(list_of_sci_images)
        self.list_of_wf_grid=list_of_wf_grid    
        self.list_of_defocuses=list_of_defocuses
        self.move_allparameters=move_allparameters
        self.num_iter=num_iter
        self.pool=pool
        
        if self.verbosity>=1:
            self.verbosity_model=self.verbosity-1
        else:
            self.verbosity_model=self.verbosity
            
        # parameter that control if the intermediate outputs are saved to the hard disk
        save=False
        self.save=save
        
    def Tokovinin_algorithm_chi_multi(self,allparameters_parametrization_proposal,\
                                      return_Images=False,num_iter=None,previous_best_result=None,
                                      use_only_chi=False,multi_background_factor=3,up_to_which_z=None):
        
        
        """
        @param allparameters_parametrization_proposal
        @param return_Images
        @param num_iter
        @param previous_best_result
        @param use_only_chi
        @param multi_background_factor
        """
        
        
        if self.verbosity>=1:
            print('##########################################################################################')
            print('##########################################################################################')
            print('Starting Tokovinin_algorithm_chi_multi with num_iter: '+str(num_iter))
            print('Tokovinin, return_Images: '+str(return_Images))
            print('Tokovinin, num_iter: '+str(num_iter))
            print('Tokovinin, use_only_chi: '+str(use_only_chi))
            print('Tokovinin, multi_background_factor: '+str(multi_background_factor))
            
            print('allparameters_parametrization_proposal'+str(allparameters_parametrization_proposal))
            print('allparameters_parametrization_proposal.shape'+str(allparameters_parametrization_proposal.shape))
        

        
        list_of_sci_images=self.list_of_sci_images
        list_of_var_images=self.list_of_var_images
        list_of_mask_images=self.list_of_mask_images
        
        
        

        
        double_sources_positions_ratios=self.double_sources_positions_ratios
        list_of_defocuses_input_long=self.list_of_defocuses
        
        if num_iter is None:
            if self.num_iter is not None:
                num_iter=self.num_iter
                
        move_allparameters=self.move_allparameters
        
        # if you passed previous best result, set the list_of_explicit_psf_positions
        # by default it is put as the last element in the last cell in the previous_best_result output
        if previous_best_result is not None:
            # to be compatible with versions before 0.45
            if len(previous_best_result)==5:
                self.list_of_psf_positions=previous_best_result[-1]   
            else:
                self.list_of_psf_positions=previous_best_result[-1][-1]   

            


        
        #########################################################################################################
        # Create initial modeling as basis for future effort
        # the outputs of this section are 0. pre_model_result, 1. model_results, 2. pre_images,
        # 3. pre_input_parameters, 4. chi_2_before_iteration_array, 5. list_of_psf_positions
        if self.verbosity>=1:
            print('list_of_defocuses analyzed: '+str(list_of_defocuses_input_long))
            

        #print('self.list_of_psf_positions in Tokovinin: '+str(self.list_of_psf_positions))
            
        model_multi=LN_PFS_multi_same_spot(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                           wavelength=self.wavelength,dithering=self.dithering,save=self.save,zmax=self.zmax,verbosity=self.verbosity_model,\
                                           double_sources=self.double_sources, double_sources_positions_ratios=self.double_sources_positions_ratios,\
                                           npix=self.npix,\
                                           list_of_defocuses=list_of_defocuses_input_long,\
                                           fit_for_flux=self.fit_for_flux,test_run=self.test_run,\
                                           list_of_psf_positions=self.list_of_psf_positions)   
        
        if self.verbosity>=1:    
            print('****************************')        
            print('Starting Tokovinin procedure with num_iter: '+str(num_iter))
            print('Initial testing proposal is: '+str(allparameters_parametrization_proposal))
        time_start_single=time.time()
        
        
        # create list of minchains, one per each image        
        list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,\
                                                                  list_of_defocuses=list_of_defocuses_input_long,zmax=self.zmax)
        
        # if the parametrization is 2d array, move it into 1d shape    
        if len(allparameters_parametrization_proposal.shape)==2:
            allparameters_parametrization_proposal=move_parametrizations_from_2d_shape_to_1d_shape(allparameters_parametrization_proposal)
            
 

        if self.verbosity>=1:
            print('Starting premodel analysis with num_iter: '+str(num_iter)) 
            
        # results from initial run, before running fitting algorithm
        # pre_model_result - mean likelihood across all images, renormalized
        # model_results - likelihood per image, not renormalized
        # pre_images - list of created model images
        # pre_input_parameters - list of parameters per image?
        # chi_2_before_iteration_array - list of lists describing quality of fitting      
        # list_of_psf_positions -?        
        try:
            #print('checkpoint 0')
            #print('len(list_of_minchain): '+str(len(list_of_minchain)))
            #print('list_of_minchain[0] '+str(list_of_minchain[0]))
            pre_model_result,model_results,pre_images,pre_input_parameters,chi_2_before_iteration_array,list_of_psf_positions=\
                model_multi(list_of_minchain,return_Images=True,use_only_chi=use_only_chi,\
                            multi_background_factor=multi_background_factor)
            #print('checkpoint 1')
            # modify variance image according to the models that have just been created
            ######## first time modifying variance image
            list_of_single_model_image=pre_images
            list_of_var_images_via_model=[]
            for index_of_single_image in range(len(list_of_sci_images)):
                single_var_image_via_model=create_custom_var(modelImg=list_of_single_model_image[index_of_single_image],sci_image=list_of_sci_images[index_of_single_image],
                                                var_image=list_of_var_images[index_of_single_image],mask_image=list_of_mask_images[index_of_single_image])
                
                list_of_var_images_via_model.append(single_var_image_via_model)
                   
            #print('checkpoint 2')   
            # replace the variance images provided with these custom variance images
            list_of_var_images=list_of_var_images_via_model
            self.list_of_var_images=list_of_var_images
                
        except Exception as e: 
            print('Exception is: '+str(e))
            print('Exception type is: '+str(repr(e)))
            print(traceback.print_exc())
            if self.verbosity>=1:
                print('Premodel analysis failed')
            # if the modelling failed
            # returning 7 nan values to be consistent with what would be the return if the algorithm passed    
            # at position 0 return extremly likelihood to indicate failure
            # at position 3 return the input parametrization
            #return -9999999,np.nan,np.nan,allparameters_parametrization_proposal,np.nan,np.nan,np.nan
            return -9999999,-9999999, np.nan,np.nan,np.nan, np.nan, np.nan,
            allparameters_parametrization_proposal,allparameters_parametrization_proposal,
            np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    
            
        if self.verbosity>=1:
            print('list_of_psf_positions at the input stage: '+str(np.array(list_of_psf_positions)))
            

        if self.save==True:
            np.save('/tigress/ncaplar/Results/allparameters_parametrization_proposal_'+str(num_iter),\
                    allparameters_parametrization_proposal)   
            np.save('/tigress/ncaplar/Results/pre_images_'+str(num_iter),\
                    pre_images)   
            np.save('/tigress/ncaplar/Results/pre_input_parameters_'+str(num_iter),\
                    pre_input_parameters)   
            np.save('/tigress/ncaplar/Results/list_of_sci_images_'+str(num_iter),\
                    list_of_sci_images)  
            np.save('/tigress/ncaplar/Results/list_of_var_images_'+str(num_iter),\
                    list_of_var_images)  
            np.save('/tigress/ncaplar/Results/list_of_mask_images_'+str(num_iter),\
                    list_of_mask_images) 

                
    
        
    
        # this needs to change - do I ever use this?!?
        #chi_2_before_iteration=chi_2_before_iteration_array[2]

        # extract the parameters which will not change in this function, i.e., not-wavefront parameters
        nonwavefront_par=list_of_minchain[0][19:42]
        time_end_single=time.time()
        if self.verbosity>=1:
            print('Total time taken for premodel analysis with num_iter '+str(num_iter)+' was  '+str(time_end_single-time_start_single)+' seconds')
            print('chi_2_before_iteration is: '+str(chi_2_before_iteration_array))  

            print('Ended premodel analysis ')    
            print('***********************') 

        
        # import science images and determine the flux mask      
        list_of_mean_value_of_background=[]
        list_of_flux_mask=[]
        list_of_sci_image_std=[]
        for i in range(len(list_of_sci_images)):
            sci_image=list_of_sci_images[i]
            var_image=list_of_var_images[i]
            
            # do not use this for images in focus or near focus
            # probably needs to be done better than via shape measurment
            #
            if sci_image.shape[0]==20:
                multi_background_factor=3
                
    
            mean_value_of_background_via_var=np.mean([np.median(var_image[0]),np.median(var_image[-1]),\
                                                  np.median(var_image[:,0]),np.median(var_image[:,-1])])*multi_background_factor
         
            mean_value_of_background_via_sci=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
                                                  np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*multi_background_factor
                
            mean_value_of_background=np.max([mean_value_of_background_via_var,mean_value_of_background_via_sci])
            if self.verbosity>1:
                print(str(multi_background_factor)+'x mean_value_of_background in image with index'+str(i)+' is estimated to be: '+str(mean_value_of_background))
                

            list_of_mean_value_of_background.append(mean_value_of_background)
 
       
        list_of_flux_mask=[]
        for i in range(len(list_of_sci_images)):
            sci_image=list_of_sci_images[i]
            var_image=list_of_var_images[i]
            flux_mask=sci_image>(list_of_mean_value_of_background[i])
            # normalized science image

            sci_image_std=sci_image/np.sqrt(var_image)
            list_of_sci_image_std.append(sci_image_std)
            list_of_flux_mask.append(flux_mask)
            
        # find postions for focus image in the raveled images
        if len(list_of_flux_mask)>1:    
            len_of_flux_masks=np.array(list(map(np.sum,list_of_flux_mask)))
            position_of_most_focus_image=np.where(len_of_flux_masks==np.min(len_of_flux_masks))[0][0]
            position_focus_1=np.sum(len_of_flux_masks[:position_of_most_focus_image])
            position_focus_2=np.sum(len_of_flux_masks[:position_of_most_focus_image+1])

        self.list_of_flux_mask=list_of_flux_mask
        self.list_of_sci_image_std=list_of_sci_image_std
        ######################################################################################################### 
        # masked science image
        list_of_I=[]
        list_of_I_std=[]    
        list_of_std_image=[]
        for i in range(len(list_of_sci_images)):
            
            sci_image=list_of_sci_images[i]
            sci_image_std=list_of_sci_image_std[i]
            flux_mask=list_of_flux_mask[i]
            std_image=np.sqrt(list_of_var_images[i][flux_mask]).ravel()
            
            I=sci_image[flux_mask].ravel()        
            #I=((sci_image[flux_mask])/np.sum(sci_image[flux_mask])).ravel()
            I_std=((sci_image_std[flux_mask])/1).ravel()
            #I_std=((sci_image_std[flux_mask])/np.sum(sci_image_std[flux_mask])).ravel()
            
            list_of_I.append(I)
            list_of_std_image.append(std_image)
            list_of_I_std.append(I_std)
            
        ### addition May22
        array_of_sci_image_std=np.array(list_of_sci_image_std)
        list_of_std_sum=[]
        for i in range(len(list_of_sci_image_std)):
            list_of_std_sum.append(np.sum(list_of_std_image[i]))
            
        array_of_std_sum=np.array(list_of_std_sum)
        array_of_std_sum=array_of_std_sum/np.min(array_of_std_sum)
        
        list_of_std_image_renormalized=[]
        for i in range(len(list_of_std_image)):
            list_of_std_image_renormalized.append(list_of_std_image[i]*array_of_std_sum[i]) 
        # 
        uber_std=[item for sublist in list_of_std_image_renormalized for item in sublist]
    
        # join all I,I_std from all individual images into one uber I,I_std  
        uber_I=[item for sublist in list_of_I for item in sublist]
        #uber_std=[item for sublist in list_of_std_image for item in sublist]
        #uber_I_std=[item for sublist in list_of_I_std for item in sublist]    

        
        uber_I=np.array(uber_I)
        uber_std=np.array(uber_std)
        
        uber_I_std=uber_I/uber_std  

        if self.save==True:
            np.save('/tigress/ncaplar/Results/list_of_sci_images_'+str(num_iter),\
                    list_of_sci_images)   
            np.save('/tigress/ncaplar/Results/list_of_mean_value_of_background_'+str(num_iter),\
                    list_of_mean_value_of_background)  
            np.save('/tigress/ncaplar/Results/list_of_flux_mask_'+str(num_iter),\
                    list_of_flux_mask)   
            np.save('/tigress/ncaplar/Results/uber_std_'+str(num_iter),\
                    uber_std)               
            np.save('/tigress/ncaplar/Results/uber_I_'+str(num_iter),\
                    uber_I)   
                
                
            
        #np.save('/tigress/ncaplar/Results/uber_I_std',\
        #        uber_I_std)              
                
        # set number of extra Zernike
        #number_of_extra_zernike=0
        #twentytwo_or_extra=22
        # numbers that make sense are 11,22,37,56,79,106,137,172,211,254
        
        #if number_of_extra_zernike is None:
        #    number_of_extra_zernike=0
        #else:
        number_of_extra_zernike=self.zmax-22
        
        
        
        #########################################################################################################
        # Start of the iterative process
        
        number_of_non_decreses=[0]
        
        for iteration_number in range(1): 
    
    
            if iteration_number==0:
                
                # initial SVD treshold
                thresh0 = 0.02
            else:
                pass
                
        
    
        
            ######################################################################################################### 
            # starting real iterative process here
            # create changes in parametrizations    
                
            # list of how much to move Zernike coefficents
            #list_of_delta_z=[]
            #for z_par in range(3,22+number_of_extra_zernike):
            #    list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))
        
            # list of how much to move Zernike coefficents
            # possibly needs to me modified to be smarther and take into account that every second parameter gets ``amplified'' in defocus
            #list_of_delta_z_parametrizations=[]
            #for z_par in range(0,19*2+2*number_of_extra_zernike):
            #    list_of_delta_z_parametrizations.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))        
    
            # this should produce reasonable changes in multi analysis
            list_of_delta_z_parametrizations=[]
            for z_par in range(0,19*2+2*number_of_extra_zernike):
                z_par_i=z_par+4
                # if this is the parameter that change
                if np.mod(z_par_i,2)==0:
                    list_of_delta_z_parametrizations.append(0.1*0.05/np.sqrt(z_par_i))
                if np.mod(z_par_i,2)==1:
                    list_of_delta_z_parametrizations.append(0.05/np.sqrt(z_par_i))
                        
            array_of_delta_z_parametrizations=np.array(list_of_delta_z_parametrizations)*(1)
               
            
            if iteration_number==0: 
                pass
            else:
                #array_of_delta_z_parametrizations=first_proposal_Tokovnin/4
                array_of_delta_z_parametrizations=np.maximum(array_of_delta_z_parametrizations,first_proposal_Tokovnin/4)
                
                
            # this code might work with global parameters?
            array_of_delta_global_parametrizations=np.array([0.1,0.02,0.1,0.1,0.1,0.1,
                                            0.3,1,0.1,0.1,
                                            0.15,0.15,0.1,
                                            0.07,0.05,0.05,0.4,
                                            30000,0.5,0.001,
                                            0.05,0.05,0.01])
            #array_of_delta_global_parametrizations=array_of_delta_global_parametrizations/1
            array_of_delta_global_parametrizations=array_of_delta_global_parametrizations/10
    
            # array, randomized delta extra zernike
            #array_of_delta_randomize=np.random.standard_normal(len(list_of_delta_z))*1.2+1
            #array_of_delta_parametrizations_randomize=np.random.standard_normal(len(list_of_delta_z_parametrizations))*1
            #array_of_delta_z_parametrizations=+np.array(list_of_delta_z_parametrizations)*array_of_delta_parametrizations_randomize
            
            

            
            if move_allparameters==True:
                array_of_delta_all_parametrizations=np.concatenate((array_of_delta_z_parametrizations[0:19*2],\
                                                                    array_of_delta_global_parametrizations, array_of_delta_z_parametrizations[19*2:]))
    
            if self.save==True:
                np.save('/tigress/ncaplar/Results/array_of_delta_z_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                        array_of_delta_z_parametrizations)        
                np.save('/tigress/ncaplar/Results/array_of_delta_global_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                        array_of_delta_global_parametrizations)      
                if move_allparameters==True:
                    np.save('/tigress/ncaplar/Results/array_of_delta_all_parametrizations_'+str(num_iter)+'_'+str(iteration_number),\
                            array_of_delta_all_parametrizations)                      
                    
                    
          
        
            # initialize 
            # if this is the first iteration of the iterative algorithm
            if iteration_number==0:
                
                
                
                thresh=thresh0
                all_global_parametrization_old=allparameters_parametrization_proposal[19*2:19*2+23]
                if number_of_extra_zernike==0:
                    all_wavefront_z_parametrization_old=allparameters_parametrization_proposal[0:19*2]
                else:
                    # if you want more Zernike
                    if len(allparameters_parametrization_proposal)==19*2+23:
                        # if you did not pass explicit extra Zernike, start with zeroes
                        all_wavefront_z_parametrization_old=np.concatenate((allparameters_parametrization_proposal[0:19*2],np.zeros(2*number_of_extra_zernike)))
                    else:
                        all_wavefront_z_parametrization_old=np.concatenate((allparameters_parametrization_proposal[0:19*2],allparameters_parametrization_proposal[19*2+23:]))
        
                pass
            # if this is not a first iteration
            else:
                # errors in the typechecker for 10 lines below are fine
                if self.verbosity==1:
                    print('array_of_delta_z in '+str(iteration_number)+' '+str(array_of_delta_z_parametrizations))
                # code analysis programs might suggest that there is an error here, but everything is ok
                #chi_2_before_iteration=np.copy(chi_2_after_iteration)
                # copy wavefront from the end of the previous iteration
                
                
                all_wavefront_z_parametrization_old=np.copy(all_wavefront_z_parametrization_new)
                if move_allparameters==True:
                    all_global_parametrization_old=np.copy(all_global_parametrization_new)
                if self.verbosity>=1:
                    if did_chi_2_improve==1:
                        print('did_chi_2_improve: yes')
                    else:
                        print('did_chi_2_improve: no')                
                if did_chi_2_improve==0:
                    thresh=thresh0
                else:
                    thresh=thresh*0.5
        
            ######################################################################################################### 
            # create a model with input parameters from previous iteration
            
            list_of_all_wavefront_z_parameterization=[]
        
            up_to_z22_parametrization_start=all_wavefront_z_parametrization_old[0:19*2]     
            from_z22_parametrization_start=all_wavefront_z_parametrization_old[19*2:]  
            global_parametrization_start=all_global_parametrization_old
            
            if self.verbosity>=1:            
                print('up_to_z22_parametrization_start: '+str(up_to_z22_parametrization_start))
                print('nonwavefront_par: '+str(nonwavefront_par))     
                print('from_z22_parametrization_start'+str(from_z22_parametrization_start))
            
            #print('iteration '+str(iteration_number)+' shape of up_to_z22_parametrization_start is: '+str(up_to_z22_parametrization_start.shape))
            if move_allparameters==True:
                initial_input_parameterization=np.concatenate((up_to_z22_parametrization_start,global_parametrization_start,from_z22_parametrization_start))
            else:
                initial_input_parameterization=np.concatenate((up_to_z22_parametrization_start,nonwavefront_par,from_z22_parametrization_start))
            
            if self.verbosity>=1:
                print('initial input parameters in iteration '+str(iteration_number)+' are: '+str(initial_input_parameterization))
                print('moving input wavefront parameters in iteration '+str(iteration_number)+' by: '+str(array_of_delta_z_parametrizations))
            if move_allparameters==True:
                print('moving global input parameters in iteration '+str(iteration_number)+' by: '+str(array_of_delta_global_parametrizations))
                
            
            if self.save==True:
                np.save('/tigress/ncaplar/Results/initial_input_parameterization_'+str(num_iter)+'_'+str(iteration_number),\
                        initial_input_parameterization)                        
    
            #print('len initial_input_parameterization '+str(len(initial_input_parameterization)))
            
            list_of_minchain=model_multi.create_list_of_allparameters(initial_input_parameterization,list_of_defocuses=list_of_defocuses_input_long,zmax=self.zmax)
            #list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,list_of_defocuses=list_of_defocuses_input_long,zmax=56)
    
            # moved in under `else` statment
            #res_multi=model_multi(list_of_minchain,return_Images=True,use_only_chi=use_only_chi,\
            #                      multi_background_factor=multi_background_factor)            
            
            # if this is the first iteration take over the results from premodel run
            if iteration_number==0:
                initial_model_result,list_of_initial_model_result,list_of_image_0,\
                        list_of_initial_input_parameters,list_of_pre_chi2,list_of_psf_positions=\
                        pre_model_result,model_results,pre_images,pre_input_parameters,chi_2_before_iteration_array,list_of_psf_positions
            else:
                res_multi=model_multi(list_of_minchain,return_Images=True,use_only_chi=use_only_chi,\
                                  multi_background_factor=multi_background_factor)    
                #mean_res_of_multi_same_spot_proposal,list_of_single_res_proposal,list_of_single_model_image_proposal,\
                #            list_of_single_allparameters_proposal,list_of_single_chi_results_proposal=res_multi  
                initial_model_result,list_of_initial_model_result,list_of_image_0,\
                            list_of_initial_input_parameters,list_of_pre_chi2,list_of_psf_positions=res_multi   
                # modify variance image according to the models that have just been created
                ######## second time modifying variance image
                list_of_single_model_image=list_of_image_0
                list_of_var_images_via_model=[]
                for index_of_single_image in range(len(list_of_sci_images)):
                    single_var_image_via_model=create_custom_var(modelImg=list_of_single_model_image[index_of_single_image],sci_image=list_of_sci_images[index_of_single_image],
                                                    var_image=list_of_var_images[index_of_single_image],mask_image=list_of_mask_images[index_of_single_image])
                    
                    list_of_var_images_via_model.append(single_var_image_via_model)
                # replace the variance images provided with these custom variance images
                list_of_var_images=list_of_var_images_via_model
                self.list_of_var_images=list_of_var_images
            
            #initial_model_result,image_0,initial_input_parameters,pre_chi2=model(initial_input_parameters,return_Image=True,return_intermediate_images=False)
            if self.save==True:
                np.save('/tigress/ncaplar/Results/list_of_initial_model_result_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_initial_model_result)                        
                np.save('/tigress/ncaplar/Results/list_of_image_0_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_image_0)    
                np.save('/tigress/ncaplar/Results/list_of_initial_input_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_initial_input_parameters)   
                np.save('/tigress/ncaplar/Results/list_of_pre_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_pre_chi2)      
                np.save('/tigress/ncaplar/Results/list_of_psf_positions_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_psf_positions)     
    
            ######################################################################################################### 
            # divided model images by their standard deviations
            
            list_of_image_0_std=[]
            for i in range(len(list_of_image_0)):
                # normalizing by standard deviation image
                # May 22 modification
                STD=np.sqrt(list_of_var_images[i]) *array_of_std_sum[i]   
                image_0=list_of_image_0[i]
                list_of_image_0_std.append(image_0/STD)

            ######################################################################################################### 
            # updated science images divided by std (given that we created new custom variance images, via model)
            
        
            ######################################################################################################### 
            # mask model images at the start of this iteration, before modifying parameters
            # create uber_M0
            
            list_of_M0=[]
            list_of_M0_std=[]
            for i in range(len(list_of_image_0_std)):
                
                image_0=list_of_image_0[i]
                image_0_std=list_of_image_0_std[i]
                flux_mask=list_of_flux_mask[i]
                # what is list_of_mask_images?
    
                M0=image_0[flux_mask].ravel()            
                #M0=((image_0[flux_mask])/np.sum(image_0[flux_mask])).ravel()
                M0_std=((image_0_std[flux_mask])/1).ravel()
                #M0_std=((image_0_std[flux_mask])/np.sum(image_0_std[flux_mask])).ravel()
                
                list_of_M0.append(M0)
                list_of_M0_std.append(M0_std)
        
            # join all M0,M0_std from invidiual images into one uber M0,M0_std
            uber_M0=[item for sublist in list_of_M0 for item in sublist]
            uber_M0_std=[item for sublist in list_of_M0_std for item in sublist]    
            
            uber_M0=np.array(uber_M0)
            uber_M0_std=np.array(uber_M0_std)
            
            #uber_M0=uber_M0/np.sum(uber_M0)
            #uber_M0_std=uber_M0_std/np.sum(uber_M0_std)
        
            self.uber_M0=uber_M0
            self.uber_M0_std=uber_M0_std
        
            if self.save==True:
                np.save('/tigress/ncaplar/Results/uber_M0_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M0)             
                np.save('/tigress/ncaplar/Results/uber_M0_std_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M0_std)     
    
    
        
            ######################################################################################################### 
            # difference between model (uber_M0) and science (uber_I) at start of this iteration
            
            # non-std version
            # not used, that is ok, we are at the moment using std version
            IM_start=np.sum(np.abs(np.array(uber_I)-np.array(uber_M0)))        
            # std version 
            IM_start_std=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M0_std)))    
            
            if len(list_of_flux_mask)>1:
                IM_start_focus=np.sum(np.abs(np.array(uber_I)-np.array(uber_M0))[position_focus_1:position_focus_2]) 
                IM_start_std_focus=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M0_std))[position_focus_1:position_focus_2]) 
            
            # mean of differences of our images - should we use mean?; probably not... needs to be normalized?
            unitary_IM_start=np.mean(IM_start)  
            unitary_IM_start_std=np.mean(IM_start_std)  
                
            #print list_of_IM_start_std
            if self.verbosity==1:
                print('np.sum(np.abs(I-M0)) before iteration '+str(num_iter)+'_'+str(iteration_number)+': '+str(unitary_IM_start))  
                print('np.sum(np.abs(I_std-M0_std)) before iteration '+str(num_iter)+'_'+str(iteration_number)+': '+str(unitary_IM_start_std))  
            #print('np.sum(np.abs(I_std-M0_std)) before iteration '+str(iteration_number)+': '+str(unitary_IM_start_std))  
        
        
                    
            ######################################################################################################### 
            # create list of new parametrizations to be tested
            # combine the old wavefront parametrization with the delta_z_parametrization 
            
            # create two lists:
            # 1. one contains only wavefront parametrizations
            # 2. second contains the whole parametrizations
            #print('checkpoint 0')
            if move_allparameters==True:
                list_of_all_wavefront_z_parameterization=[]
                list_of_input_parameterizations=[]
                for z_par in range(19*2):
                    all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
            
                    up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                    from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
        
                    parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)  
                    #print('checkpoint 1')
                for g_par in range(23):
                    all_global_parametrization_list=np.copy(all_global_parametrization_old)
                    all_global_parametrization_list[g_par]=all_global_parametrization_list[g_par]+array_of_delta_global_parametrizations[g_par]
                    #list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
            
                    up_to_z22_start=all_wavefront_z_parametrization_old[0:19*2]     
                    from_z22_start=all_wavefront_z_parametrization_old[19*2:]  
        
                    parametrization_proposal=np.concatenate((up_to_z22_start,all_global_parametrization_list,from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)  
                    #print('checkpoint 2')
                for z_par in range(19*2,len(all_wavefront_z_parametrization_old)):
                    all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
            
                    up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                    from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
        
                    parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)  
                    #print('checkpoint 3')
                
            else:
                list_of_all_wavefront_z_parameterization=[]
                list_of_input_parameterizations=[]
                for z_par in range(len(all_wavefront_z_parametrization_old)):
                    all_wavefront_z_parametrization_list=np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par]=all_wavefront_z_parametrization_list[z_par]+array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)
            
                    up_to_z22_start=all_wavefront_z_parametrization_list[0:19*2]     
                    from_z22_start=all_wavefront_z_parametrization_list[19*2:]  
        
                    parametrization_proposal=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)    
                    #print('checkpoint 4')
            
            ######################################################################################################### 
            # Starting testing new set of parameters
            # Creating new images
            
            out_ln=[]
            out_ln_ind=[]
            out_images=[]
            out_parameters=[]
            out_chi2=[]
            out_pfs_positions=[]
        
            if self.verbosity>=1:
                print('We are now inside of the pool loop number '+str(iteration_number)+' with num_iter: '+str(num_iter))

            # actually it is parametrization
            # list of (56-3)*2 sublists, each one with (56-3)*2 + 23 values
            time_start=time.time()
        
            # This assume that Zernike parameters go up to 56
            # I need to pass each of 106 parametrization to model_multi BUT
            # model_multi actually takes list of parameters, not parametrizations
            # I need list that has 106 sublists, each one of those being 9x(53+23)
            # 9 == number of images
            # 53 == number of Zernike parameters (56-3)
            # 23 == number of global parameters
            uber_list_of_input_parameters=[]
            for i in range(len(list_of_input_parameterizations)):
    
               list_of_input_parameters=model_multi.create_list_of_allparameters(list_of_input_parameterizations[i],\
                                                                                 list_of_defocuses=list_of_defocuses_input_long,zmax=self.zmax)
               uber_list_of_input_parameters.append(list_of_input_parameters)
               
            #save the uber_list_of_input_parameters
            if self.save==True:
                np.save('/tigress/ncaplar/Results/uber_list_of_input_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_list_of_input_parameters)    
                    

            # pass new model_multi that has fixed pos (October 6, 2020)   
            # should have same paramter as staring model_multi, apart from list_of_psf_positions (maybe variance?, but prob not)
            model_multi_out=LN_PFS_multi_same_spot(list_of_sci_images,list_of_var_images,list_of_mask_images=list_of_mask_images,\
                                 wavelength=self.wavelength,dithering=self.dithering,save=self.save,zmax=self.zmax,verbosity=self.verbosity_model,double_sources=self.double_sources,\
                                 double_sources_positions_ratios=double_sources_positions_ratios,npix=self.npix,
                                 fit_for_flux=self.fit_for_flux,test_run=self.test_run,list_of_psf_positions=list_of_psf_positions)   

                
                
                
            if move_allparameters==True:    
                self.array_of_delta_all_parametrizations=array_of_delta_all_parametrizations  
            else:
                self.array_of_delta_z_parametrizations=array_of_delta_z_parametrizations      
                
                
            ######## start of creating H
            
            # H is normalized difference between pixel values of model image that result from changing the j-th term
            # This is expensive because we have to generate new image for each change of j-th term
            if previous_best_result==None:
                    
                if self.verbosity>=1:
                    print('self.pool parameter is: '+str(self.pool))
                
                # generate images
                if self.pool is None:   
                    out1=map(partial(model_multi_out, return_Images=True,\
                                     use_only_chi=use_only_chi,multi_background_factor=multi_background_factor), uber_list_of_input_parameters)
                else:
                    out1=self.pool.map(partial(model_multi_out, return_Images=True,\
                                               use_only_chi=use_only_chi,multi_background_factor=multi_background_factor), uber_list_of_input_parameters)
                out1=list(out1)
                time_end=time.time()
                if self.verbosity>=1:
                    print('time_end-time_start for creating model_multi_out '+str(time_end-time_start))    
                    
                #out1=map(partial(model, return_Image=True), input_parameters)
                #out1=list(out1)
                
                #print('len(out1)'+str(len(out1)))
                #print('out1[0].shape'+str(len(out1[0])))
            
                # normalization of the preinput run? (what did I mean that)
                pre_input_parameters=np.array(pre_input_parameters)
                if self.verbosity>=1:
                    print('pre_input_parameters.shape '+str(pre_input_parameters.shape))
                    print('pre_input_parameters[0][0:5] '+str(pre_input_parameters[0][0:5]))
        
                
                # select the column specifying the flux normalization from the input images
                array_of_normalizations_pre_input=pre_input_parameters[:,41]
            

                #out1=a_pool.map(model,input_parameters,repeat(True))
                for i in range(len(uber_list_of_input_parameters)):
                    #print(i)
                    
                    #    initial_model_result,list_of_initial_model_result,list_of_image_0,\
                    #                list_of_initial_input_parameters,list_of_pre_chi2
                    
                    # outputs are 
                    # 0. mean likelihood
                    # 1. list of individual res (likelihood)
                    # 2. list of science images
                    # 3. list of parameters used
                    # 4. list of quality measurments
                    
                    
                    out_images_pre_renormalization=np.array(out1[i][2])
                    out_parameters_single_move=np.array(out1[i][3])
                    # replace the normalizations in the output imags with the normalizations from the input images
                    array_of_normalizations_out=out_parameters_single_move[:,41]
                    out_renormalization_parameters=array_of_normalizations_pre_input/array_of_normalizations_out
                    
                    
                    out_ln.append(out1[i][0])
                    out_ln_ind.append(out1[i][1])
                    #print('out_images_pre_renormalization.shape: '+str(out_images_pre_renormalization.shape))
                    #print('out_renormalization_parameters.shape: '+str(out_renormalization_parameters.shape))
                    #np.save('/tigress/ncaplar/Results/out_images_pre_renormalization',out_images_pre_renormalization)
                    
                    out_images_step=[]
                    for l in range(len(out_renormalization_parameters)):
                        out_images_step.append(out_images_pre_renormalization[l]*out_renormalization_parameters[l])  
                    out_images.append(out_images_step)                        
                    
                    #out_images.append(out_images_pre_renormalization*out_renormalization_parameters)
                    out_parameters.append(out1[i][3])
                    out_chi2.append(out1[i][4])
                    out_pfs_positions.append(out1[i][5])
                    
                    # we use these out_images to study the differences due to changing parameters,
                    # we do not want the normalization to affect things (and position of optical center)
                    # so we renormalize to that multiplication constants are the same as in the input 
                    
                    
                    
                    
                time_end=time.time()
                if self.verbosity>=1:
                    print('time_end-time_start for whole model_multi_out '+str(time_end-time_start))
            
                if self.save==True:
                    np.save('/tigress/ncaplar/Results/out_images_'+str(num_iter)+'_'+str(iteration_number),\
                            out_images)    
                    np.save('/tigress/ncaplar/Results/out_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                            out_parameters)  
                    np.save('/tigress/ncaplar/Results/out_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                            out_chi2) 
                
                ######################################################################################################### 
                # Normalize created images
                
                # We created (zmax*2) x N images, where N is the number of defocused images
                
                ###########
                # !old comments as of April 2021
                # !loop over all of (zmax*2) combinations and double-normalize and ravel N images
                # !double-normalize (I used to do that before, not anymore) = set sum of each image to 1 and then set the sum of all raveled images to 1
                # !hm.....
                ###########
                
                # join all images together
                list_of_images_normalized_uber=[]
                list_of_images_normalized_std_uber=[]
                # go over (zmax-3)*2 images
                for j in range(len(out_images)):
                    # two steps for what could have been achived in one, but to ease up transition from previous code 
                    out_images_single_parameter_change=out_images[j]
                    optpsf_list=out_images_single_parameter_change
                    
                    # flux image has to correct per image
                    #  mask images that have been created in the fitting procedure with the appropriate flux mask
                    images_normalized=[]
                    for i in range(len(optpsf_list)):
                        
                        flux_mask=list_of_flux_mask[i]
                        images_normalized.append((optpsf_list[i][flux_mask]).ravel())                
                        # !old double-normalizing code
                        # !images_normalized.append((optpsf_list[i][flux_mask]/np.sum(optpsf_list[i][flux_mask])).ravel())
                    
                        
                    
                    images_normalized_flat=[item for sublist in images_normalized for item in sublist]  
                    images_normalized_flat=np.array(images_normalized_flat)
                    #images_normalized_flat=np.array(images_normalized_flat)/len(optpsf_list)        
                    
                    # list of (zmax-3)*2 raveled images
                    list_of_images_normalized_uber.append(images_normalized_flat)
                    
                    # same but divided by STD
                    #images_normalized_std=[]
                    #for i in range(len(optpsf_list)):   
                        # seems that I am a bit more verbose here with my definitions
                        #optpsf_list_i=optpsf_list[i]
                        
                        
                        # do I want to generate new STD images, from each image?
                        # May 22 modification
                        #STD=list_of_sci_image_std[i]*array_of_std_sum[i]
                        #optpsf_list_i_STD=optpsf_list_i/STD    
                        #flux_mask=list_of_flux_mask[i]
                        #images_normalized_std.append((optpsf_list_i_STD[flux_mask]/np.sum(optpsf_list_i_STD[flux_mask])).ravel())
                    
                    # join all images together
                    #images_normalized_std_flat=[item for sublist in images_normalized_std for item in sublist]  
                    # normalize so that the sum is still one
                    #images_normalized_std_flat=np.array(images_normalized_std_flat)/len(optpsf_list)
                    
                    #list_of_images_normalized_std_uber.append(images_normalized_std_flat)
                    
                # create uber images_normalized,images_normalized_std    
                # images that have zmax*2 rows and very large number of columns (number of non-masked pixels from all N images)
                uber_images_normalized=np.array(list_of_images_normalized_uber)    
                #uber_images_normalized_std=np.array(list_of_images_normalized_std_uber)          
        
                if self.save==True:
                    np.save('/tigress/ncaplar/Results/uber_images_normalized_'+str(num_iter)+'_'+str(iteration_number),\
                            uber_images_normalized)  
                
                #np.save('/tigress/ncaplar/Results/uber_images_normalized_std_'+str(num_iter)+'_'+str(iteration_number),\
                #        uber_images_normalized_std)  
        
        
                
                
                #single_wavefront_parameter_list=[]
                #for i in range(len(out_parameters)):
                #    single_wavefront_parameter_list.append(np.concatenate((out_parameters[i][:19],out_parameters[i][42:])) )
            
            
                
                ######################################################################################################### 
                # Core Tokovinin algorithm
            
                
                if self.verbosity>=1:
                    print('images_normalized (uber).shape: '+str(uber_images_normalized.shape))
                    print('array_of_delta_z_parametrizations[:,None].shape'+str(array_of_delta_z_parametrizations[:,None].shape))
                # equation A1 from Tokovinin 2006
                # new model minus old model
                if move_allparameters==True:
                    H=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_all_parametrizations[:,None])    
                    #H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/array_of_delta_z_parametrizations[:,None]) 
                    H_std=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_all_parametrizations[:,None])/uber_std.ravel()[:,None]               
                else:                
                    H=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_z_parametrizations[:,None])    
                    #H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/array_of_delta_z_parametrizations[:,None]) 
                    H_std=np.transpose(np.array((uber_images_normalized-uber_M0))/array_of_delta_z_parametrizations[:,None])/uber_std.ravel()[:,None]     
                
                array_of_delta_z_parametrizations_None=np.copy(array_of_delta_z_parametrizations[:,None])
                
            else:
                H=self.create_simplified_H(previous_best_result)
                
                H_std=H/uber_std.ravel()[:,None] 
            
            ######## end of creating H
            
            
            
            
            
            
            
            
            if self.save==True and previous_best_result==None:
                np.save('/tigress/ncaplar/Results/array_of_delta_z_parametrizations_None_'+str(num_iter)+'_'+str(iteration_number),\
                        array_of_delta_z_parametrizations_None)              
            
            
            
            if self.save==True:
                np.save('/tigress/ncaplar/Results/H_'+str(num_iter)+'_'+str(iteration_number),\
                        H)  
            if self.save==True:
                np.save('/tigress/ncaplar/Results/H_std_'+str(num_iter)+'_'+str(iteration_number),\
                        H_std)                  
            
            
            
            first_proposal_Tokovnin,first_proposal_Tokovnin_std=self.create_first_proposal_Tokovnin(H,H_std,\
                                                                uber_I,uber_M0,uber_std,up_to_which_z=up_to_which_z)
            
            """
            #print('np.mean(H,axis=0).shape)'+str(np.mean(H,axis=0).shape))
            singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))<0.01]
            non_singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))>0.01]
            #print('non_singlular_parameters.shape)'+str(non_singlular_parameters.shape))
            H=H[:,non_singlular_parameters]
            H_std=H_std[:,non_singlular_parameters]
        
            HHt=np.matmul(np.transpose(H),H)
            HHt_std=np.matmul(np.transpose(H_std),H_std) 
            #print('svd thresh is '+str(thresh))
            #invHHt=svd_invert(HHt,thresh)
            #invHHt_std=svd_invert(HHt_std,thresh)
            invHHt=np.linalg.inv(HHt)        
            invHHt_std=np.linalg.inv(HHt_std)
        
            invHHtHt=np.matmul(invHHt,np.transpose(H))
            invHHtHt_std=np.matmul(invHHt_std,np.transpose(H_std))
    
        
            # I is uber_I now (science images)
            # M0 is uber_M0 now (set of models before the iteration)
            first_proposal_Tokovnin=np.matmul(invHHtHt,uber_I-uber_M0)
            #first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,uber_I_std-uber_M0_std)
            first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,(uber_I-uber_M0)/uber_std.ravel())  
   

   
            # if you have removed certain parameters because of the singularity, return them here, with no change
            if len(singlular_parameters)>0:
                for i in range(len(singlular_parameters)):
                    first_proposal_Tokovnin=np.insert(first_proposal_Tokovnin,singlular_parameters[i],0)
                    first_proposal_Tokovnin_std=np.insert(first_proposal_Tokovnin_std,singlular_parameters[i],0)            
            #print('first_proposal_Tokovnin_std'+str(first_proposal_Tokovnin_std.shape))
            #print('invHHtHt_std.shape'+str(invHHtHt_std.shape))
            
            """
            
            if self.verbosity>=1:
                print('first_proposal_Tokovnin[:5] is: '+str(first_proposal_Tokovnin[:8*2]))
                print('first_proposal_Tokovnin_std[:5] is: '+str(first_proposal_Tokovnin_std[:8*2]))               
                try:
                    print('ratio is of proposed to initial parameters (std) is: '+str(first_proposal_Tokovnin_std/array_of_delta_z_parametrizations))
                except:
                    pass
        
            #Tokovnin_proposal=0.7*first_proposal_Tokovnin
            if move_allparameters==True:
                Tokovnin_proposal=np.zeros((129,))
                #Tokovnin_proposal[non_singlular_parameters]=0.7*first_proposal_Tokovnin_std            
                Tokovnin_proposal[non_singlular_parameters]=1*first_proposal_Tokovnin_std     
                
                all_parametrization_new=np.copy(initial_input_parameterization)
                allparameters_parametrization_proposal_after_iteration_before_global_check=all_parametrization_new+Tokovnin_proposal
                # tests if the global parameters would be out of bounds - if yes, reset them to the limit values
                global_parametrization_proposal_after_iteration_before_global_check=\
                    allparameters_parametrization_proposal_after_iteration_before_global_check[19*2:19*2+23]
                checked_global_parameters=check_global_parameters(global_parametrization_proposal_after_iteration_before_global_check,test_print=1)
                
                allparameters_parametrization_proposal_after_iteration=np.copy(allparameters_parametrization_proposal_after_iteration_before_global_check)
                allparameters_parametrization_proposal_after_iteration[19*2:19*2+23]=checked_global_parameters
    
                
                
            else:
                #Tokovnin_proposal=0.7*first_proposal_Tokovnin_std
                Tokovnin_proposal=1*first_proposal_Tokovnin_std
    
            if self.verbosity>=1:
                print('Tokovnin_proposal[:5] is: '+str(Tokovnin_proposal[:5]))
                if self.zmax>35:
                    print('Tokovnin_proposal[38:43] is: '+str(Tokovnin_proposal[38:43]))
            #print('all_wavefront_z_parametrization_old in '+str(iteration_number)+' '+str(all_wavefront_z_parametrization_old[:5]))
            #print('Tokovnin_proposal[:5] is: '+str(Tokovnin_proposal[:5]))
            #print('Tokovnin_proposal.shape '+str(Tokovnin_proposal.shape))

            # if the Tokovinin proposal is not made, return the initial result 
            if len(Tokovnin_proposal)<10:
                #return initial_model_result,list_of_initial_model_result,list_of_image_0,\
                #    allparameters_parametrization_proposal,list_of_initial_input_parameters,list_of_pre_chi2,list_of_psf_positions
                return initial_model_result,initial_model_result,\
                       list_of_initial_model_result,list_of_initial_model_result,\
                       out_images,list_of_image_0,list_of_image_0,\
                       allparameters_parametrization_proposal,allparameters_parametrization_proposal,\
                       list_of_initial_input_parameters,list_of_initial_input_parameters,\
                       list_of_pre_chi2,list_of_pre_chi2,\
                       list_of_psf_positions, list_of_psf_positions   

    
                break                

            #print('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))
            if move_allparameters==True:
                #all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)    
                #all_global_parametrization_new=np.copy(all_global_parametrization_old)
                #all_parametrization_new=np.copy(initial_input_parameterization)
                
                #allparameters_parametrization_proposal_after_iteration=all_parametrization_new+Tokovnin_proposal
                
                up_to_z22_end=allparameters_parametrization_proposal_after_iteration[:19*2]
                from_z22_end=allparameters_parametrization_proposal_after_iteration[19*2+23:]
                all_wavefront_z_parametrization_new=np.concatenate((up_to_z22_end,from_z22_end))
                
                all_global_parametrization_new=allparameters_parametrization_proposal_after_iteration[19*2:19*2+23]
                
                
            else:
                all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                all_wavefront_z_parametrization_new=all_wavefront_z_parametrization_new+Tokovnin_proposal
                up_to_z22_end=all_wavefront_z_parametrization_new[:19*2]
                from_z22_end=all_wavefront_z_parametrization_new[19*2:]
                allparameters_parametrization_proposal_after_iteration=np.concatenate((up_to_z22_end,nonwavefront_par,from_z22_end))
        
            if self.save==True:
                np.save('/tigress/ncaplar/Results/first_proposal_Tokovnin'+str(num_iter)+'_'+str(iteration_number),\
                        first_proposal_Tokovnin) 
                np.save('/tigress/ncaplar/Results/first_proposal_Tokovnin_std'+str(num_iter)+'_'+str(iteration_number),\
                        first_proposal_Tokovnin_std)   
                np.save('/tigress/ncaplar/Results/allparameters_parametrization_proposal_after_iteration_'+str(num_iter)+'_'+str(iteration_number),\
                        allparameters_parametrization_proposal_after_iteration)    
        
            #########################
            # Creating single exposure with new proposed parameters and seeing if there is improvment    
            time_start_final=time.time()
            
            
            
            

            list_of_parameters_after_iteration=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal_after_iteration,\
                                                                                        list_of_defocuses=list_of_defocuses_input_long,zmax=self.zmax)
            res_multi=model_multi(list_of_parameters_after_iteration,return_Images=True,\
                                  use_only_chi=use_only_chi,multi_background_factor=multi_background_factor)
    
            if self.verbosity>=1:
                print('allparameters_parametrization_proposal_after_iteration '+str(allparameters_parametrization_proposal_after_iteration[0:5]))
                print('list_of_parameters_after_iteration[0][0:5] '+str(list_of_parameters_after_iteration[0][0:5]))
    
            final_model_result,list_of_final_model_result,list_of_image_final,\
                        list_of_finalinput_parameters,list_of_after_chi2,list_of_final_psf_positions=res_multi
            ### third (last?) time modifying variance image
            list_of_single_model_image=list_of_image_final
            list_of_var_images_via_model=[]
            for index_of_single_image in range(len(list_of_sci_images)):
                single_var_image_via_model=create_custom_var(modelImg=list_of_single_model_image[index_of_single_image],sci_image=list_of_sci_images[index_of_single_image],
                                                var_image=list_of_var_images[index_of_single_image],mask_image=list_of_mask_images[index_of_single_image])
                
                list_of_var_images_via_model.append(single_var_image_via_model)
            # replace the variance images provided with these custom variance images
            list_of_var_images=list_of_var_images_via_model
            self.list_of_var_images=list_of_var_images
            
                        
                        
    
            time_end_final=time.time()
            if self.verbosity>=1:
                print('Total time taken for final iteration was '+str(time_end_final-time_start_final)+' seconds with num_iter: '+str(num_iter))
    
            if self.save==True:
                np.save('/tigress/ncaplar/Results/list_of_final_model_result_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_final_model_result)                        
                np.save('/tigress/ncaplar/Results/list_of_image_final_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_image_final)    
                np.save('/tigress/ncaplar/Results/list_of_finalinput_parameters_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_finalinput_parameters)   
                np.save('/tigress/ncaplar/Results/list_of_after_chi2_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_after_chi2)        
                np.save('/tigress/ncaplar/Results/list_of_final_psf_positions_'+str(num_iter)+'_'+str(iteration_number),\
                        list_of_final_psf_positions)                       
    
            if self.verbosity>=1:
                print('list_of_final_psf_positions : '+str(list_of_psf_positions))
                  
           
            ######################################################################################################### 
            # divided model images by their standard deviations
            
            list_of_image_final_std=[]
            for i in range(len(list_of_image_0)):
                # normalizing by standard deviation image
                # May 22 modification
                STD=np.sqrt(list_of_var_images[i]) *array_of_std_sum[i]   
                image_final=list_of_image_final[i]
                list_of_image_final_std.append(image_final/STD)
            
        
            ######################################################################################################### 
            #  masked model images after this iteration (mask by flux criteria)
            
            
            list_of_M_final=[]
            list_of_M_final_std=[]
            for i in range(len(list_of_image_final_std)):
                
                image_final=list_of_image_final[i]
                image_final_std=list_of_image_final_std[i]
                flux_mask=list_of_flux_mask[i]
                # what is list_of_mask_images?
                
                #M_final=((image_final[flux_mask])/np.sum(image_final[flux_mask])).ravel()
                M_final=(image_final[flux_mask]).ravel()
                #M_final_std=((image_final_std[flux_mask])/np.sum(image_final_std[flux_mask])).ravel()
                M_final_std=((image_final_std[flux_mask])/1).ravel()
                
                list_of_M_final.append(M_final)
                list_of_M_final_std.append(M_final_std)
        
            # join all M0,M0_std from invidiual images into one uber M0,M0_std
            uber_M_final=[item for sublist in list_of_M_final for item in sublist]
            uber_M_final_std=[item for sublist in list_of_M_final_std for item in sublist]   
           
            uber_M_final=np.array(uber_M_final)
            uber_M_final_std=np.array(uber_M_final_std)
            
            uber_M_final_linear_prediction=uber_M0+ self.create_linear_aproximation_prediction(H,first_proposal_Tokovnin)
            uber_M_final_std_linear_prediction=uber_M0_std+ self.create_linear_aproximation_prediction(H_std,first_proposal_Tokovnin_std)            
                
            if self.save==True:
                np.save('/tigress/ncaplar/Results/uber_M_final_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M_final)                        
                np.save('/tigress/ncaplar/Results/uber_M_final_std_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M_final_std)    
            if self.save==True:
                np.save('/tigress/ncaplar/Results/uber_M_final_linear_prediction_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M_final_linear_prediction)                        
                np.save('/tigress/ncaplar/Results/uber_M_final_std_linear_prediction_'+str(num_iter)+'_'+str(iteration_number),\
                        uber_M_final_std_linear_prediction)  
    
            
            ####
            # Seeing if there is an improvment
            # Quality measure is the sum of absolute differences of uber_I_std (all images/std) and uber_M_final_std (all models / std)
            # how closely is that correlated with improvments in final_model_result?
            
            # non-std version
            # not used, that is ok, we are at the moment using std version
            IM_final=np.sum(np.abs(np.array(uber_I)-np.array(uber_M_final)))        
            # std version 
            IM_final_std=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M_final_std))) 
            
            # linear prediction versions
            IM_final_linear_prediction=np.sum(np.abs(np.array(uber_I)-np.array(uber_M_final_linear_prediction)))        
            # std version 
            IM_final_std_linear_prediction=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M_final_std_linear_prediction))) 
            
            # do a separate check on the improvment measure for the image in focus, when applicable
            if len(list_of_flux_mask)>1:
                IM_final_focus=np.sum(np.abs(np.array(uber_I)-np.array(uber_M_final))[position_focus_1:position_focus_2]) 
                IM_final_std_focus=np.sum(np.abs(np.array(uber_I_std)-np.array(uber_M_final_std))[position_focus_1:position_focus_2]) 
                
            
            

            
            if self.verbosity>=1:
                print('I-M_start before iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_start))    
                print('I-M_final after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_final))
                print('IM_final_linear_prediction after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '\
                      +str(IM_final_linear_prediction))
                if len(list_of_flux_mask)>1:
                    print('I-M_start_focus before iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_start_focus))    
                    print('I-M_final_focus after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_final_focus))
                
                
                print('I_std-M_start_std after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_start_std))        
                print('I_std-M_final_std after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_final_std))
                print('IM_final_std_linear_prediction after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '\
                      +str(IM_final_std_linear_prediction))
                if len(list_of_flux_mask)>1:
                    print('I-M_start_focus_std before iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_start_std_focus))    
                    print('I-M_final_focus_std after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(IM_final_std_focus))
                

                
                print('Likelihood before iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(initial_model_result))
                print('Likelihood after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(final_model_result))

                print('Likelihood before iteration  '+str(iteration_number)+' with num_iter '+str(num_iter)+', per image: '+str(list_of_initial_model_result))
                print('Likelihood after iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+', per image: '+str(list_of_final_model_result))                

                #print('chi_2_after_iteration/chi_2_before_iteration '+str(chi_2_after_iteration/chi_2_before_iteration ))
                print('IM_final/IM_start with num_iter '+str(num_iter)+': '+str(IM_final/IM_start))
                print('IM_final_std/IM_start_std with num_iter '+str(num_iter)+': '+str(IM_final_std/IM_start_std))
                if len(list_of_flux_mask)>1:
                    print('IM_final_focus/IM_start_focus with num_iter '+str(num_iter)+': '+str(IM_final_focus/IM_start_focus))
                    print('IM_final_std_focus/IM_start_std_focus with num_iter '+str(num_iter)+': '+str(IM_final_std_focus/IM_start_std_focus))
                    
                print('#########################################################')

        
            ##################
            # If improved take new parameters, if not dont
            
            # TEST, May18 2021
            # if more images, test that everything AND focus image has improved
            if len(list_of_flux_mask)>1:
                if IM_final_std/IM_start_std <1.0 and IM_final_std_focus/IM_start_std_focus <1.25 :  
                    condition_for_improvment=True
                else:
                    condition_for_improvment=False
            else:
                # if you are having only one image
                if IM_final_std/IM_start_std <1.0:
                    condition_for_improvment=True
            
            if self.verbosity>=1:
                print('condition_for_improvment in iteration '+str(iteration_number)+' with num_iter '+str(num_iter)+': '+str(condition_for_improvment))
            if condition_for_improvment==True :        
                #when the quality measure did improve
                did_chi_2_improve=1
                number_of_non_decreses.append(0)
                if self.verbosity>=1:
                    print('number_of_non_decreses:' + str(number_of_non_decreses))
                    print('current value of number_of_non_decreses is: '+str(np.sum(number_of_non_decreses)))
                    print('##########################################################################################')
                    print('##########################################################################################')
            else:
                #when the quality measure did not improve
                did_chi_2_improve=0
                # resetting all parameters
                if move_allparameters==True:
                    all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                    all_global_parametrization_new=np.copy(all_global_parametrization_old)
                    allparameters_parametrization_proposal_after_iteration=initial_input_parameterization
                else:
                    all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                    #chi_2_after_iteration=chi_2_before_iteration
                    up_to_z22_end=all_wavefront_z_parametrization_new[:19*2]
                    from_z22_start=all_wavefront_z_parametrization_new[19*2:]
                    allparameters_parametrization_proposal_after_iteration=np.concatenate((up_to_z22_start,nonwavefront_par,from_z22_start))
                thresh=thresh0
                number_of_non_decreses.append(1)
                if self.verbosity>=1:
                    print('number_of_non_decreses:' + str(number_of_non_decreses))
                    print('current value of number_of_non_decreses is: '+str(np.sum(number_of_non_decreses)))
                    print('##########################################################################################')
                    print('##########################################################################################')
                    
                final_model_result=initial_model_result
                list_of_final_model_result=list_of_initial_model_result
                list_of_image_final=pre_images
                allparameters_parametrization_proposal_after_iteration=allparameters_parametrization_proposal
                list_of_finalinput_parameters=list_of_initial_input_parameters
                list_of_after_chi2=list_of_pre_chi2
                list_of_final_psf_positions=list_of_psf_positions
        
            if np.sum(number_of_non_decreses)==1:
                if return_Images==False:
                    return final_model_result
                else:
                    if previous_best_result==None:
                        return initial_model_result,final_model_result,\
                list_of_initial_model_result,list_of_final_model_result,\
                out_images, pre_images, list_of_image_final,\
                allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
                list_of_initial_input_parameters, list_of_finalinput_parameters,\
                list_of_pre_chi2,list_of_after_chi2,\
                list_of_psf_positions,list_of_final_psf_positions,\
                [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions]                 
                    else:
                        return  initial_model_result,final_model_result,\
                list_of_initial_model_result,list_of_final_model_result,\
                out_images, pre_images, list_of_image_final,\
                allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
                list_of_initial_input_parameters, list_of_finalinput_parameters,\
                list_of_pre_chi2,list_of_after_chi2,\
                list_of_psf_positions,list_of_final_psf_positions
                
                break
        
        # if return_Images==False just return the mean likelihood
        if return_Images==False:
            return final_model_result
        else:
            # if you return images, return full 
            if previous_best_result==None:
                # 0. likelihood averaged over all images (before the function) 1. ikelihood averaged over all images (after the function) 
                # 2. likelihood per image (output from model_multi) (before the function) 3. likelihood per image (output from model_multi) (after the function)
                # 4. out_images 5. list of initial model images 6. list of final model images 
                # 7. parametrization before the function 8. parametrization after the function
                # 9. list of parameters per image (before the function) 10. list of parameters per image (after the function)
                # 11. list of chi2 per image (before the function) 12. list of chi2 per image (after the function)
                # 13. list of psf position of image (function the function) 14. list of psf position of image (after the function)
                return initial_model_result,final_model_result,\
                list_of_initial_model_result,list_of_final_model_result,\
                out_images, pre_images, list_of_image_final,\
                allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
                list_of_initial_input_parameters, list_of_finalinput_parameters,\
                list_of_pre_chi2,list_of_after_chi2,\
                list_of_psf_positions,list_of_final_psf_positions,\
                [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions] 
                #return final_model_result,list_of_final_model_result,out_images,list_of_image_final,\
                #    allparameters_parametrization_proposal_after_iteration,list_of_finalinput_parameters,\
                #        list_of_after_chi2,list_of_final_psf_positions,\
                #        [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions] 
                        
            else:
                return initial_model_result,final_model_result,\
                list_of_initial_model_result,list_of_final_model_result,\
                out_images, pre_images, list_of_image_final,\
                allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
                list_of_initial_input_parameters, list_of_finalinput_parameters,\
                list_of_pre_chi2,list_of_after_chi2,\
                list_of_psf_positions,list_of_final_psf_positions

    
    def create_simplified_H(self,previous_best_result,multi_background_factor=3):
        
        """
        create matrix H using the provided images and changes
        
        H - normalized difference between pixel values
        
        it assumes that the changes are still the same in this iteration as in the previous iteration
        
        """
        
        # to be compatable with version before 0.45 where previous_best_result was actually only the last part of len=5
        # 
        if len(previous_best_result)==5:
            previous_best_result=previous_best_result
        else:
            # if you are passing the whole best result, separte the parts of the result
            main_body_of_best_result=previous_best_result[:-1]
            previous_best_result=previous_best_result[-1]


        # we need actual final model images from the previous best result
        #list_of_image_0_from_previous_best_result=main_body_of_best_result[6]
        #list_of_image_0=list_of_image_0_from_previous_best_result
        
        # we need actual initial model images from the previous best result
        # this will be used to evalute change of model due to changes in singel wavefront parameters
        # i.e., to estimate matrix H
        list_of_image_0_from_previous_best_result=main_body_of_best_result[5]
        list_of_image_0=list_of_image_0_from_previous_best_result
        
        ########################################################
        # import science images and determine the flux mask     
        #
        #list_of_mean_value_of_background=[]
        #list_of_flux_mask=[]
        #list_of_sci_image_std=[]
        #for i in range(len(self.list_of_sci_images)):
        #    sci_image=self.list_of_sci_images[i]
        #    var_image=self.list_of_var_images[i]
        #
        #    mean_value_of_background_via_var=np.mean([np.median(var_image[0]),np.median(var_image[-1]),\
        #                                          np.median(var_image[:,0]),np.median(var_image[:,-1])])*multi_background_factor
        # 
        #    mean_value_of_background_via_sci=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
        #                                          np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*multi_background_factor
        #        
        #    mean_value_of_background=np.max([mean_value_of_background_via_var,mean_value_of_background_via_sci])
        #    if self.verbosity>1:
        #        print(str(multi_background_factor)+'x mean_value_of_background in image with index'+str(i)+' is estimated to be: '+str(mean_value_of_background))
        #        
        #
        #    list_of_mean_value_of_background.append(mean_value_of_background)
        #    flux_mask=sci_image>(mean_value_of_background)
        #
        #    
            # normalized science image
        #   var_image=self.list_of_var_images[i]
        #   sci_image_std=sci_image/np.sqrt(var_image)
        #   list_of_sci_image_std.append(sci_image_std)
        #    list_of_flux_mask.append(flux_mask)
            
        list_of_flux_mask=self.list_of_flux_mask 
        list_of_sci_image_std=self.list_of_sci_image_std
            
        ######################################################################################################### 
        # divided model images by their standard deviations
        
        list_of_image_0_std=[]
        for i in range(len(list_of_image_0)):
            # normalizing by standard deviation image
            STD=np.sqrt(self.list_of_var_images[i])    
            image_0=list_of_image_0[i]
            list_of_image_0_std.append(image_0/STD)            
            
        ########################################################        
             
        # mask model images at the start of this iteration, before modifying parameters
        # create uber_M0_previous_best - uber_M0 derived from previous best, but with current flux mask
        
        list_of_M0=[]
        list_of_M0_std=[]
        for i in range(len(list_of_image_0_std)):
            
            image_0=list_of_image_0[i]
            image_0_std=list_of_image_0_std[i]
            flux_mask=list_of_flux_mask[i]


            M0=image_0[flux_mask].ravel()            
            #M0=((image_0[flux_mask])/np.sum(image_0[flux_mask])).ravel()
            M0_std=((image_0_std[flux_mask])/1).ravel()
            #M0_std=((image_0_std[flux_mask])/np.sum(image_0_std[flux_mask])).ravel()
            
            list_of_M0.append(M0)
            list_of_M0_std.append(M0_std)
    
        # join all M0,M0_std from invidiual images into one uber M0,M0_std
        uber_M0_previous_best=[item for sublist in list_of_M0 for item in sublist]
        uber_M0_previous_best_std=[item for sublist in list_of_M0_std for item in sublist]    
        
        uber_M0_previous_best=np.array(uber_M0_previous_best)
        uber_M0_previous_best_std=np.array(uber_M0_previous_best_std)
        
        #uber_M0=uber_M0/np.sum(uber_M0)
        #uber_M0_std=uber_M0_std/np.sum(uber_M0_std)
    
        self.uber_M0_previous_best=uber_M0_previous_best
        self.uber_M0_previous_best_std=uber_M0_previous_best_std
        
        ########################################################      
        # uber_images_normalized_previous_best, but with current flux mask
        
        # previous uber images - not used 
        #uber_images_normalized_previous_best_old_flux_mask=previous_best_result[0]        
        # previous uber model - not used 
        #uber_M0_previous_best_old_flux_mask=previous_best_result[1]
        
        # we need out images showing difference from original image due to 
        # changing a single Zernike parameter
        out_images=main_body_of_best_result[4]   
        
        # join all images together
        list_of_images_normalized_uber=[]
        list_of_images_normalized_std_uber=[]
        # go over (zmax-3)*2 images
        for j in range(len(out_images)):
            # two steps for what could have been achived in one, but to ease up transition from previous code 
            out_images_single_parameter_change=out_images[j]
            optpsf_list=out_images_single_parameter_change
            
            # flux image has to correct per image
            #  mask images that have been created in the fitting procedure with the appropriate flux mask
            images_normalized=[]
            for i in range(len(optpsf_list)):
                
                flux_mask=list_of_flux_mask[i]
                if j==0:

                    #print('sum_flux_in images'+str([i,np.sum(flux_mask)]))
                    pass
                images_normalized.append((optpsf_list[i][flux_mask]).ravel())                
                # !old double-normalizing code
                # !images_normalized.append((optpsf_list[i][flux_mask]/np.sum(optpsf_list[i][flux_mask])).ravel())
            
                
            
            images_normalized_flat=[item for sublist in images_normalized for item in sublist]  
            images_normalized_flat=np.array(images_normalized_flat)
            #images_normalized_flat=np.array(images_normalized_flat)/len(optpsf_list)        
            
            # list of (zmax-3)*2 raveled images
            list_of_images_normalized_uber.append(images_normalized_flat)
            
            # same but divided by STD
            images_normalized_std=[]
            for i in range(len(optpsf_list)):   
                # seems that I am a bit more verbose here with my definitions
                optpsf_list_i=optpsf_list[i]
                
                
                # do I want to generate new STD images, from each image?
                STD=list_of_sci_image_std[i]
                optpsf_list_i_STD=optpsf_list_i/STD    
                flux_mask=list_of_flux_mask[i]
                #images_normalized_std.append((optpsf_list_i_STD[flux_mask]/np.sum(optpsf_list_i_STD[flux_mask])).ravel())
            
            # join all images together
            #images_normalized_std_flat=[item for sublist in images_normalized_std for item in sublist]  
            # normalize so that the sum is still one
            #images_normalized_std_flat=np.array(images_normalized_std_flat)/len(optpsf_list)
            
            #list_of_images_normalized_std_uber.append(images_normalized_std_flat)
            
        # create uber images_normalized,images_normalized_std    
        # images that have zmax*2 rows and very large number of columns (number of non-masked pixels from all N images)
        uber_images_normalized_previous_best=np.array(list_of_images_normalized_uber)            
        
        
        
        ########################################################              
        # current model image
        uber_M0=self.uber_M0
        
        # current change of the parameters
        if self.move_allparameters==True:    
            array_of_delta_all_parametrizations=self.array_of_delta_all_parametrizations  
        else:
            array_of_delta_parametrizations=self.array_of_delta_z_parametrizations  
        

        # previous uber model
        #uber_M0_previous_best=previous_best_result[1]
        # previous H (not used)
        #H_previous_best=previous_best_result[2]
        # how much has delta parametrizations changed in the previous result
        array_of_delta_parametrizations_None_previous_best=previous_best_result[3]
        
        # ratio between current parametrization and the previous (provided) changed parametrization
        ratio_of_parametrizations=(array_of_delta_parametrizations[:,None]/array_of_delta_parametrizations_None_previous_best)
        
        # create the array of how wavefront changes the uber_model by multiply the changes with new ratios
        array_of_wavefront_changes=np.transpose(ratio_of_parametrizations*\
                                                np.array(uber_images_normalized_previous_best-uber_M0_previous_best)/(array_of_delta_parametrizations_None_previous_best))  
        
        # difference between current model image and previous model image
        
        #print('uber_images_normalized_previous_best.shape'+str(uber_images_normalized_previous_best.shape))
        #print('uber_M0_previous_best.shape'+str(uber_M0_previous_best.shape))
        #print('uber_M0.shape'+str(uber_M0.shape))
        
        # change between the initial model in this step and the imported model
        global_change=uber_M0-uber_M0_previous_best
        # global_change_proposed
        global_change_proposed=(uber_M0-uber_M0_previous_best)/array_of_delta_parametrizations[:,None]

        
        # H is a change of wavefront 
        H=array_of_wavefront_changes 

        # H is a change of wavefront and change of model (is it?)        
        # H=array_of_wavefront_changes + global_change[:,None]
        
        #np.save('/tigress/ncaplar/Results/global_change_'+str(2)+'_'+str(0),\
        #            global_change)     

        #np.save('/tigress/ncaplar/Results/global_change_proposed_'+str(2)+'_'+str(0),\
        #            global_change_proposed)                            
        #np.save('/tigress/ncaplar/Results/array_of_wavefront_changes_'+str(2)+'_'+str(0),\
        #            array_of_wavefront_changes)    
        #np.save('/tigress/ncaplar/Results/H_'+str(2)+'_'+str(0),\
        #            array_of_wavefront_changes)    

        return H
    
    def create_first_proposal_Tokovnin(self,H,H_std,uber_I,uber_M0,uber_std,up_to_which_z=None):
        
        
            H_shape=H.shape
            
            #print('H_shape'+str(H_shape))
            #print('up_to_which_z:'+str(up_to_which_z))
            
            if up_to_which_z!=None:
                #H=H[:,1:(up_to_which_z-3)*2:2]
                #H_std=H_std[:,1:(up_to_which_z-3)*2:2]    
                
                H=H[:,0:(up_to_which_z-3)*2]
                H_std=H_std[:,0:(up_to_which_z-3)*2] 
                
            else:
                pass
            
            H=np.nan_to_num(H,0)
            
            #print('np.mean(H,axis=0).shape)'+str(np.mean(H,axis=0).shape))
            singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))<0.001]
            non_singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))>0.001]
            
            #print('np.abs((np.mean(H,axis=0)))'+str(np.abs((np.mean(H,axis=0)))))
            #print('non_singlular_parameters.shape)'+str(non_singlular_parameters.shape))
            #print('singlular_parameters)'+str(singlular_parameters))
            
            H=H[:,non_singlular_parameters]
            H_std=H_std[:,non_singlular_parameters]
    
            HHt=np.matmul(np.transpose(H),H)
            HHt_std=np.matmul(np.transpose(H_std),H_std) 
            #print('svd thresh is '+str(thresh))
            #invHHt=svd_invert(HHt,thresh)
            #invHHt_std=svd_invert(HHt_std,thresh)
            invHHt=np.linalg.inv(HHt)        
            invHHt_std=np.linalg.inv(HHt_std)
    
            invHHtHt=np.matmul(invHHt,np.transpose(H))
            invHHtHt_std=np.matmul(invHHt_std,np.transpose(H_std))
    
    
            # I is uber_I now (science images)
            # M0 is uber_M0 now (set of models before the iteration)
            first_proposal_Tokovnin=np.matmul(invHHtHt,uber_I-uber_M0)
            #first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,uber_I_std-uber_M0_std)
            first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,(uber_I-uber_M0)/uber_std.ravel())  
    
    
            #print('first_proposal_Tokovnin.shape before sing'+str(first_proposal_Tokovnin.shape))
    
            # if you have removed certain parameters because of the singularity, return them here, with no change
            if len(singlular_parameters)>0:
                for i in range(len(singlular_parameters)):
                    first_proposal_Tokovnin=np.insert(first_proposal_Tokovnin,singlular_parameters[i],0)
                    first_proposal_Tokovnin_std=np.insert(first_proposal_Tokovnin_std,singlular_parameters[i],0)    
                    
            #print('first_proposal_Tokovnin.shape after sing'+str(first_proposal_Tokovnin.shape))
                        
            if up_to_which_z!=None:
                #H=H[:,1:(up_to_which_z-3)*2:2]
                #H_std=H_std[:,1:(up_to_which_z-3)*2:2]    
                
                first_proposal_Tokovnin_0=np.zeros((H_shape[1]))
                first_proposal_Tokovnin_0_std=np.zeros((H_shape[1]))
                #print('first_proposal_Tokovnin_0.shape'+str(first_proposal_Tokovnin_0.shape))
                
                #print('up_to_which_z: '+str(up_to_which_z))
                #print('first_proposal_Tokovnin:' +str(first_proposal_Tokovnin))
                
                #print(first_proposal_Tokovnin_0[0:(up_to_which_z-3)*2].shape)
                #print(first_proposal_Tokovnin.shape)
                first_proposal_Tokovnin_0[0:(up_to_which_z-3)*2]=first_proposal_Tokovnin
                first_proposal_Tokovnin_0_std[0:(up_to_which_z-3)*2]=first_proposal_Tokovnin_std
                
                first_proposal_Tokovnin=first_proposal_Tokovnin_0
                first_proposal_Tokovnin_std=first_proposal_Tokovnin_0_std
            else:
                pass 
    
            return first_proposal_Tokovnin,first_proposal_Tokovnin_std
        
    def create_linear_aproximation_prediction(self,H,first_proposal_Tokovnin): 
        return np.dot(H,first_proposal_Tokovnin)
        


    def __call__(self, allparameters_parametrization_proposal,return_Images=True,num_iter=None,\
                 previous_best_result=None,use_only_chi=False,multi_background_factor=3):
            return self.Tokovinin_algorithm_chi_multi(allparameters_parametrization_proposal,return_Images=return_Images,num_iter=num_iter,\
                                                  previous_best_result=previous_best_result,use_only_chi=use_only_chi,\
                                                  multi_background_factor=multi_background_factor)



class LN_PFS_single(object):
    
    """!
    
    Class to compute likelihood of the donut image, given the sci and var image
    Also the prinicpal way to get the images via ``return_Image'' option 
    
    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,use_pupil_parameters=None,zmax=zmax,save=1)    
    def model_return(allparameters_proposal):
        return model(allparameters_proposal,return_Image=True)
    
    Calls ZernikeFitter_PFS class (constructModelImage_PFS_naturalResolution function )in order to create images
    
    Called by LN_PFS_multi_same_spot
    
    """
        
    def __init__(self,sci_image,var_image,
                 mask_image=None,
                 wavelength=None,dithering=None,save=None,verbosity=None,
                 pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,use_wf_grid=None,
                 zmax=None,extraZernike=None,pupilExplicit=None,simulation_00=None,
                 double_sources=None,double_sources_positions_ratios=None,npix=None,
                 fit_for_flux=None,test_run=None,explicit_psf_position=None,
                 use_only_chi=False,use_center_of_flux=False):    
        """
        @param sci_image                               science image, 2d array
        @param var_image                               variance image, 2d array,same size as sci_image
        @param mask_image                              mask image, 2d array,same size as sci_image
        @param dithering                               dithering, 1=normal, 2=two times higher resolution, 3=not supported
        @param save                                    save intermediate result in the process (set value at 1 for saving)
        @param verbosity                               verbosity of the process (set value at 1 for full output)
        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF
        @param zmax                                    largest Zernike order used (11 or 22)
        @param extraZernike                            array consistingin of higher order zernike (if using higher order than 22)
        @param pupilExplicit
        @param simulation_00                           resulting image will be centered with optical center in the center of the image 
                                                       and not fitted acorrding to the sci_image
        @param double_sources                          1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /       arrray with parameters describing relative position\
                                                       and relative flux of the secondary source(s)
        @param npxix                                   size of the pupil
        
        @param fit_for_flux                            automatically fit for the best flux level that minimizes the chi**2
        @param test_run                                if True, skips the creation of model and return science image - useful for testing
                                                       interaction of outputs of the module in broader setting quickly 
        @param explicit_psf_position                   gives position of the opt_psf
        """        
                
        if verbosity is None:
            verbosity=0
                
        if use_pupil_parameters is not None:
            assert pupil_parameters is not None
            
        if double_sources is not None and double_sources is not False:      
            assert np.sum(np.abs(double_sources_positions_ratios))>0    

        if zmax is None:
            zmax=11
            
        """              
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax>=22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'radiometricEffect','radiometricExponent','x_ilum','y_ilum',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
        """  
        
        if zmax==11:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                          'wide_0','wide_23','wide_43','misalign',
                          'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                          'grating_lines','scattering_slope','scattering_amplitude',
                          'pixel_effect','fiber_r','flux']         
        if zmax>=22:
            self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
              'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
              'wide_0','wide_23','wide_43','misalign',
              'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
              'grating_lines','scattering_slope','scattering_amplitude',
              'pixel_effect','fiber_r','flux']  
        
        
        
        if mask_image is None:
            mask_image=np.zeros(sci_image.shape)
        self.mask_image=mask_image
        self.sci_image=sci_image
        self.var_image=var_image
        self.dithering=dithering
        self.pupil_parameters=pupil_parameters
        self.use_pupil_parameters=use_pupil_parameters
        self.use_optPSF=use_optPSF
        self.pupilExplicit=pupilExplicit
        self.simulation_00=simulation_00 
        if self.simulation_00==True:
            self.simulation_00=1

        self.zmax=zmax
        self.extraZernike=extraZernike
        self.verbosity=verbosity
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios
        self.fit_for_flux=fit_for_flux
        if test_run==None:
            self.test_run=False
        else:
            self.test_run=test_run
        self.explicit_psf_position=explicit_psf_position

        # if npix is not specified automatically scale the image
        # this will create images which will have different pupil size for different sizes of science image
        # and this will influence the results
        if npix is None:
            if dithering is None or dithering==1:
                npix=int(math.ceil(int(1024*sci_image.shape[0]/(20*4)))*2)
            else:
                npix=int(math.ceil(int(1024*sci_image.shape[0]/(20*4*self.dithering)))*2)
        else:
            self.npix=npix
            
   
        if verbosity==1:
            print('Science image shape is: '+str(sci_image.shape))
            print('Top left pixel value of the science image is: '+str(sci_image[0][0]))
            print('Variance image shape is: '+str(sci_image.shape))
            print('Top left pixel value of the variance image is: '+str(var_image[0][0]))     
            print('Mask image shape is: '+str(sci_image.shape))
            print('Sum of mask image is: '+str(np.sum(mask_image)))    
            print('Dithering value is: '+str(dithering)) 
            print('')
            
            print('supplied extra Zernike parameters (beyond zmax): '+str(extraZernike))
        
        """
        parameters that go into ZernikeFitter_PFS
        def __init__(self,image=None,image_var=None,image_mask=None,pixelScale=None,wavelength=None,
             diam_sic=None,npix=None,pupilExplicit=None,
             wf_full_Image=None,radiometricEffectArray_Image=None,
             ilum_Image=None,dithering=None,save=None,
             pupil_parameters=None,use_pupil_parameters=None,
             use_optPSF=None,use_wf_grid=None,
             zmaxInit=None,extraZernike=None,simulation_00=None,verbosity=None,
             double_sources=None,double_sources_positions_ratios=None,
             test_run=None,explicit_psf_position=None,*args):
        """
        
        
        
        # how are these two approaches different?
        if pupil_parameters is None:
            single_image_analysis=ZernikeFitter_PFS(sci_image,var_image,image_mask=mask_image,pixelScale=None,wavelength=wavelength,
                                                    diam_sic=None,npix=npix,pupilExplicit=pupilExplicit,\
                                                    wf_full_Image=None,radiometricEffectArray_Image=None,
                                                    ilum_Image=None,dithering=dithering,save=save,\
                                                    pupil_parameters=pupil_parameters,use_pupil_parameters=use_pupil_parameters,
                                                    use_optPSF=use_optPSF,use_wf_grid=use_wf_grid,
                                                    zmaxInit=zmax,extraZernike=extraZernike,simulation_00=simulation_00,verbosity=verbosity,\
                                                    double_sources=double_sources,double_sources_positions_ratios=double_sources_positions_ratios,\
                                                    test_run=test_run,explicit_psf_position=explicit_psf_position,
                                                    use_only_chi=use_only_chi,use_center_of_flux=use_center_of_flux)  
            single_image_analysis.initParams(zmax)
            self.single_image_analysis=single_image_analysis
        else:

            single_image_analysis=ZernikeFitter_PFS(sci_image,var_image,image_mask=mask_image,npix=npix,dithering=dithering,save=save,\
                                                    pupil_parameters=pupil_parameters,use_pupil_parameters=use_pupil_parameters,
                                                    extraZernike=extraZernike,simulation_00=simulation_00,verbosity=verbosity,\
                                                    double_sources=double_sources,double_sources_positions_ratios=double_sources_positions_ratios,\
                                                    test_run=test_run,explicit_psf_position=explicit_psf_position,
                                                    use_only_chi=use_only_chi,use_center_of_flux=use_center_of_flux)  
           
            single_image_analysis.initParams(zmax,hscFracInit=pupil_parameters[0],strutFracInit=pupil_parameters[1],
                   focalPlanePositionInit=(pupil_parameters[2],pupil_parameters[3]),slitFracInit=pupil_parameters[4],
                  slitFrac_dy_Init=pupil_parameters[5],x_fiberInit=pupil_parameters[6],y_fiberInit=pupil_parameters[7],
                  effective_ilum_radiusInit=pupil_parameters[8],frd_sigmaInit=pupil_parameters[9],
                  det_vertInit=pupil_parameters[10],slitHolder_frac_dxInit=pupil_parameters[11],
                  wide_0Init=pupil_parameters[12],wide_23Init=pupil_parameters[13],wide_43Init=pupil_parameters[14],
                  misalignInit=pupil_parameters[15])
            
            self.single_image_analysis=single_image_analysis

    def create_custom_var(self, modelImg,sci_image,var_image,mask_image=None):
        """
        
        
        The algorithm creates variance map from the model image provided
        The connection between variance and flux is determined from the provided science image and variance image
        
        
        @param modelImg     model image
        @param sci_image    scientific image 
        @param var_image    variance image        
        @param mask_image   mask image     
        
        All of inputs have to be 2d np.arrays with same size
        
        Returns the np.array with same size as inputs
        
        introduced in v0.41
        
        """
        if mask_image is None:
            sci_pixels=sci_image.ravel()
            var_pixels=var_image.ravel()   
        else:
            sci_pixels=sci_image[mask_image==0].ravel()
            var_pixels=var_image[mask_image==0].ravel()
        z=np.polyfit(sci_pixels,var_pixels,deg=2)
        p1=np.poly1d(z)
        custom_var_image=p1(sci_image)
        

        return custom_var_image


    def create_chi_2_almost(self,modelImg,sci_image,var_image,mask_image,use_only_chi=False,multi_background_factor=3):
        """
        @param modelImg    model image
        @param sci_image    scientific image 
        @param var_image    variance image        
        @param mask_image   mask image  
        @param use_only_chi if True, the program is reporting np.abs(chi), not chi^2
        
        
        returns array with 5 values (descriptions below with use_only_chi=False)
        1. normal chi**2
        2. what is 'instrinsic' chi**2, i.e., just sum((scientific image)**2/variance)
        3. 'Q' value = sum(abs(model - scientific image))/sum(scientific image)
        4. chi**2 reduced
        5. chi**2 reduced 'intrinsic'
        
        """ 

        try:
            
            if sci_image.shape[0]==20:
                multi_background_factor=3
                
            
            mean_value_of_background_via_var=np.mean([np.median(var_image[0]),np.median(var_image[-1]),\
                                                  np.median(var_image[:,0]),np.median(var_image[:,-1])])*multi_background_factor
         
            mean_value_of_background_via_sci=np.mean([np.median(sci_image[0]),np.median(sci_image[-1]),\
                                                  np.median(sci_image[:,0]),np.median(sci_image[:,-1])])*multi_background_factor
                
            mean_value_of_background=np.max([mean_value_of_background_via_var,mean_value_of_background_via_sci])
            
            flux_mask=sci_image>(mean_value_of_background)
            inverted_flux_mask=flux_mask.astype(bool)
        except:
            inverted_flux_mask=np.ones(sci_image.shape)
        
        
        
        
        # array that has True for values which are good and False for bad values
        inverted_mask=~mask_image.astype(bool)
        
  
        # strengthen the mask by taking in the account only bright pixels, which have passed the flux cut
        inverted_mask=inverted_mask*inverted_flux_mask
      
        # at the moment plug it in here
        use_custom_var=True
        if use_custom_var==True:
            custom_var_image=self.create_custom_var(modelImg,sci_image,var_image,mask_image)
            # overload var_image with newly created image
            var_image=custom_var_image
        
        # apply the mask on all of the images (sci, var and model)        
        var_image_masked=var_image*inverted_mask
        sci_image_masked=sci_image*inverted_mask
        modelImg_masked=modelImg*inverted_mask
        
        # sigma values
        sigma_masked = np.sqrt(var_image_masked)
        
        
        
        # chi array
        chi = (sci_image_masked - modelImg_masked)/sigma_masked
        # chi intrinsic, i.e., without subtracting model
        chi_intrinsic=(sci_image_masked/sigma_masked)
        
        
        #chi2_intrinsic=np.sum(sci_image_masked**2/var_image_masked)
        
        # ravel and remove bad values
        chi_without_nan = chi.ravel()[~np.isnan(chi.ravel())]
        chi_intrinsic_without_nan=chi_intrinsic.ravel()[~np.isnan(chi_intrinsic.ravel())]
        
        if use_only_chi==False:
            
            # square it
            chi2_res=(chi_without_nan)**2
            chi2_intrinsic_res=(chi_intrinsic_without_nan)**2
        else:
            # do not square it
            # keep the names, but careful as they are not squared quantities
            chi2_res=np.abs(chi_without_nan)**1
            chi2_intrinsic_res=np.abs(chi_intrinsic_without_nan)**1        

        #print('use_only_chi variable in create_chi_2_almost is: '+str(use_only_chi))
        #print('chi2_res '+str(np.sum(chi2_res)))
        #print('chi2_intrinsic_res '+str(np.sum(chi2_intrinsic_res)))
        
        # calculates 'Q' values
        Qlist=np.abs((sci_image_masked - modelImg_masked))
        Qlist_without_nan=Qlist.ravel()[~np.isnan(Qlist.ravel())]
        sci_image_without_nan=sci_image_masked.ravel()[~np.isnan(sci_image_masked.ravel())]
        Qvalue = np.sum(Qlist_without_nan)/np.sum(sci_image_without_nan)
        
        # return the result
        return [np.sum(chi2_res),np.sum(chi2_intrinsic_res),Qvalue,np.mean(chi2_res),np.mean(chi2_intrinsic_res)]
    
    def lnlike_Neven(self,allparameters,return_Image=False,return_intermediate_images=False,use_only_chi=False,multi_background_factor=3):
        """
        report likelihood given the parameters of the model
        give -np.inf if outside of the parameters range specified below 
        
        return_Image
        return_intermediate_images
        use_only_chi
        
        
        """ 
        time_lnlike_start=time.time()
        
        if self.verbosity==1:
            print('')
            print('Entering lnlike_Neven')
            print('allparameters '+str(allparameters))
        
        if self.pupil_parameters is not None:    
            if len(allparameters)<25:
                allparameters=add_pupil_parameters_to_all_parameters(allparameters,self.pupil_parameters)
            else:
                allparameters=add_pupil_parameters_to_all_parameters(remove_pupil_parameters_from_all_parameters(allparameters),self.pupil_parameters)
        

        if self.zmax<=22:
            zmax_number=self.zmax-3              
        else:
            zmax_number=19
        zparameters=allparameters[0:zmax_number]
            
        globalparameters=allparameters[len(zparameters):len(zparameters)+23]
        
        #if self.fit_for_flux==True:
        #    globalparameters=np.concatenate((globalparameters,np.array([1])))

        


        # internal parameter for debugging change value to 1 to see which parameters are failling
        test_print=0
        if self.verbosity==1:
            test_print=1    
            
        
            
        #When running big fits these are limits which ensure that the code does not wander off in totally non physical region
        #hsc frac
        if globalparameters[0]<0.6 or globalparameters[0]>0.8:
            print('globalparameters[0] outside limits; value: '+str(globalparameters[0])) if test_print == 1 else False 
            return -np.inf
        
         #strut frac
        if globalparameters[1]<0.07 or globalparameters[1]>0.13:
            print('globalparameters[1] outside limits') if test_print == 1 else False 
            return -np.inf
        
        #slit_frac < strut frac 
        #if globalparameters[4]<globalparameters[1]:
            #print('globalparameters[1] not smaller than 4 outside limits')
            #return -np.inf
  
         #dx Focal
        if globalparameters[2]>0.4:
            print('globalparameters[2] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[2]<-0.4:
            print('globalparameters[2] outside limits') if test_print == 1 else False 
            return -np.inf
        
        # dy Focal
        if globalparameters[3]>0.4:
            print('globalparameters[3] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[3]<-0.4:
            print('globalparameters[3] outside limits') if test_print == 1 else False 
            return -np.inf

        # slitFrac
        if globalparameters[4]<0.05:
            print('globalparameters[4] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[4]>0.09:
            print('globalparameters[4] outside limits') if test_print == 1 else False 
            return -np.inf

        # slitFrac_dy
        if globalparameters[5]<-0.5:
            print('globalparameters[5] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[5]>0.5:
            print('globalparameters[5] outside limits') if test_print == 1 else False 
            return -np.inf
        
        # wide_0
        if globalparameters[6]<0:
            print('globalparameters[6] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[6]>1:
            print('globalparameters[6] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        # wide_23
        if globalparameters[7]<0:
            print('globalparameters[7] outside limits') if test_print == 1 else False 
            return -np.inf
        # changed in w_23
        if globalparameters[7]>1:
            print('globalparameters[7] outside limits') if test_print == 1 else False 
            return -np.inf 
        
        # wide_43
        if globalparameters[8]<0:
            print('globalparameters[8] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[8]>1:
            print('globalparameters[8] outside limits') if test_print == 1 else False 
            return -np.inf
        
        # misalign
        if globalparameters[9]<0:
            print('globalparameters[9] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[9]>12:
            print('globalparameters[9] outside limits') if test_print == 1 else False 
            return -np.inf   
        
        # x_fiber
        if globalparameters[10]<-0.4:
            print('globalparameters[10] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[10]>0.4:
            print('globalparameters[10] outside limits') if test_print == 1 else False 
            return -np.inf      
      
        # y_fiber
        if globalparameters[11]<-0.4:
            print('globalparameters[11] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[11]>0.4:
            print('globalparameters[11] outside limits') if test_print == 1 else False 
            return -np.inf        
  
        # effective_radius_illumination
        if globalparameters[12]<0.7:
            print('globalparameters[12] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[12]>1.0:
            print('globalparameters[12] outside limits with value '+str(globalparameters[12])) if test_print == 1 else False 
            return -np.inf  
 
        # frd_sigma
        if globalparameters[13]<0.01:
            print('globalparameters[13] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[13]>.4:
            print('globalparameters[13] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        #frd_lorentz_factor
        if globalparameters[14]<0.01:
            print('globalparameters[14] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[14]>1:
            print('globalparameters[14] outside limits') if test_print == 1 else False 
            return -np.inf  

        # det_vert
        if globalparameters[15]<0.85:
            print('globalparameters[15] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[15]>1.15:
            print('globalparameters[15] outside limits') if test_print == 1 else False 
            return -np.inf  

        # slitHolder_frac_dx
        if globalparameters[16]<-0.8:
            print('globalparameters[16] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[16]>0.8:
            print('globalparameters[16] outside limits') if test_print == 1 else False 
            return -np.inf  
     
        # grating_lines
        if globalparameters[17]<1200:
            print('globalparameters[17] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[17]>120000:
            print('globalparameters[17] outside limits') if test_print == 1 else False 
            return -np.inf  
            
        # scattering_slope
        if globalparameters[18]<1.5:
            print('globalparameters[18] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[18]>+3.0:
            print('globalparameters[18] outside limits') if test_print == 1 else False 
            return -np.inf 

        # scattering_amplitude
        if globalparameters[19]<0:
            print('globalparameters[19] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[19]>+0.4:
            print('globalparameters[19] outside limits') if test_print == 1 else False 
            return -np.inf             
        
        # pixel_effect
        if globalparameters[20]<0.15:
            print('globalparameters[20] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[20]>+0.8:
            print('globalparameters[20] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        # fiber_r
        if globalparameters[21]<1.74:
            print('globalparameters[21] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[21]>+1.98:
            print('globalparameters[21] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        # flux
        if self.fit_for_flux==True:
            globalparameters[22]=1
        else:          
            if globalparameters[22]<0.98:
                print('globalparameters[22] outside limits') if test_print == 1 else False 
                return -np.inf
            if globalparameters[22]>1.02:
                print('globalparameters[22] outside limits') if test_print == 1 else False 
                return -np.inf      

        

        x=self.create_x(zparameters,globalparameters)       
        for i in range(len(self.columns)):
            self.single_image_analysis.params[self.columns[i]].set(x[i]) 

        
        if len(allparameters)>len(self.columns):
            if self.verbosity==1:
                print('We are going higher than Zernike 22!')
            extra_Zernike_parameters=allparameters[len(self.columns):]
            if self.verbosity==1:
                print('extra_Zernike_parameters '+str(extra_Zernike_parameters))
        else:
            extra_Zernike_parameters=None
            if self.verbosity==1:
                print('No extra Zernike (beyond zmax)')
                
        # if it is not a test run, run the actual code        
        if self.test_run==False:
            # this try statment avoids code crashing when code tries to analyze weird combination of parameters which fail to produce an image    
            try:
                if return_intermediate_images==False:
                    modelImg,psf_position = self.single_image_analysis.constructModelImage_PFS_naturalResolution(self.single_image_analysis.params,\
                                                                                                extraZernike=extra_Zernike_parameters,return_intermediate_images=return_intermediate_images)  
                if return_intermediate_images==True:
                    modelImg,ilum,wf_grid_rot,psf_position = self.single_image_analysis.constructModelImage_PFS_naturalResolution(self.single_image_analysis.params,\
                                                                                                extraZernike=extra_Zernike_parameters,return_intermediate_images=return_intermediate_images)                  
            except IndexError:
                return -np.inf,-np.inf
        else:
            randomizer_array=np.random.randn(self.sci_image.shape[0],self.sci_image.shape[1]) /100+1
            if return_intermediate_images==False:
                
                modelImg=self.sci_image*randomizer_array
                psf_position=[0,0]
                
                if self.verbosity==1:
                    print('Careful - the model image is created in a test_run')
            else:

                #ilum_test=np.ones((3072,3072))
                ilum_test=np.ones((30,30))    
            
                #wf_grid_rot=np.load(TESTING_WAVEFRONT_IMAGES_FOLDER+'ilum.npy')

                #wf_grid_rot_test=np.ones((3072,3072))
                wf_grid_rot_test=np.ones((30,30))

                psf_position_test=[0,0]
                
                modelImg,ilum,wf_grid_rot,psf_position =self.sci_image*randomizer_array,ilum_test,wf_grid_rot_test,psf_position_test
                if self.verbosity==1:
                    print('Careful - the model image is created in a test_run')
                    print('test run with return_intermediate_images==True!')

        # if image is in focus, which at the moment is size of post stamp image of 20 by 20
        #print('self.sci_image.shape[0]'+str(self.sci_image.shape[0]))
        if self.sci_image.shape[0]==20:
            # apply the procedure from https://github.com/Subaru-PFS/drp_stella/blob/master/python/pfs/drp/stella/subtractSky2d.py
            # `image` from the pipeline is `sci_image` here
            # `psfImage` from the pipelin is `modelImg` here
            # `image.mask` from the pipeline is `mask_image` here
            # `image.variance` from the pipeline is `var_image` here    
            
            inverted_mask=~self.mask_image.astype(bool)
            
            modelDotModel = np.sum(modelImg[inverted_mask]**2)
            modelDotData = np.sum(modelImg[inverted_mask]*self.sci_image[inverted_mask])
            modelDotModelVariance = np.sum(modelImg[inverted_mask]**2*self.var_image[inverted_mask])
            flux = modelDotData/modelDotModel
            fluxErr = np.sqrt(modelDotModelVariance)/modelDotModel
            
            modelImg=modelImg*flux
            if self.verbosity==1:
                print('Image in focus, using pipeline normalization; multiplying all values in the model by '+str(flux)) 
            
        else:
            
        
            if self.fit_for_flux==True:
                if self.verbosity==1:
                    print('Internally fitting for flux; disregarding passed value for flux')
                    
                def find_flux_fit(flux_fit):
                    return self.create_chi_2_almost(flux_fit*modelImg,self.sci_image,self.var_image,self.mask_image,use_only_chi=use_only_chi)[0]     
                
                flux_fitting_result = scipy.optimize.shgo(find_flux_fit,bounds=[(0.98,1.02)],iters=6)
                flux=flux_fitting_result.x[0]
                if len(allparameters)==42:
                    allparameters[-1]=flux
                if len(allparameters)==41:
                    allparameters=np.concatenate((allparameters,np.array([flux])))
                else:
                    #print('here')
                    #print(allparameters[41])
                    if (allparameters[41]<1.1) and (allparameters[41]>0.9):
                        allparameters[41]=flux
                    else:
                        pass
                #print('flux: '+str(flux))
                #print(len(allparameters))
                #print(allparameters)
                
                modelImg=modelImg*flux
                if self.verbosity==1:
                    print('Internally fitting for flux; multiplying all values in the model by '+str(flux))
            else:
                pass


           

        
        
        # returns 0. chi2 value, 1. chi2_max value, 2. Qvalue, 3. chi2/d.o.f., 4. chi2_max/d.o.f.  
        chi_2_almost_multi_values=self.create_chi_2_almost(modelImg,self.sci_image,self.var_image,self.mask_image,\
                                                           use_only_chi=use_only_chi,\
                                                           multi_background_factor=multi_background_factor)
        chi_2_almost=chi_2_almost_multi_values[0]
        chi_2_almost_max=chi_2_almost_multi_values[1]
        chi_2_almost_dof=chi_2_almost_multi_values[3]
        chi_2_almost_max_dof=chi_2_almost_multi_values[4]

        
        
        # res stands for ``result'' 
        if use_only_chi==False:
            # reporting likelihood in chi^2 case
            res=-(1/2)*(chi_2_almost+np.sum(np.log(2*np.pi*self.var_image)))
        else:
            # reporting np.abs(chi) per d.o.f.
            res=-(1/1)*(chi_2_almost_dof)

        time_lnlike_end=time.time()  
        if self.verbosity==True:
            print('Finished with lnlike_Neven')
            if use_only_chi==False:
                print('chi_2_almost/d.o.f is '+str(chi_2_almost_dof)+'; chi_2_almost_max_dof is '+str(chi_2_almost_max_dof)+' log(improvment) is '+str(np.log10(chi_2_almost_dof/chi_2_almost_max_dof)))
            else:
                print('chi_almost/d.o.f is '+str(chi_2_almost_dof)+'; chi_almost_max_dof is '+str(chi_2_almost_max_dof)+' log(improvment) is '+str(np.log10(chi_2_almost_dof/chi_2_almost_max_dof)))
     
            print('The likelihood reported is: '+str(res))
            print('multiprocessing.current_process() '+str(current_process())+' thread '+str(threading.get_ident()))
            print(str(platform.uname()))
            print('Time for lnlike_Neven function in thread '+str(threading.get_ident())+' is: '+str(time_lnlike_end-time_lnlike_start) +str(' seconds'))    
            print(' ')
  
                    
        if return_Image==False:
            return res,psf_position
        else:
            # if return_Image==True return: 0. likelihood, 1. model image, 2. parameters, 3. [0. chi**2, 1. chi**2_max, 2. chi**2/dof, 3. chi**2_max/dof]
            if return_intermediate_images==False:
                return res,modelImg,allparameters,\
                    [chi_2_almost,chi_2_almost_max,chi_2_almost_dof,chi_2_almost_max_dof],psf_position
            if return_intermediate_images==True:
                return res,modelImg,allparameters,ilum,wf_grid_rot,\
                    [chi_2_almost,chi_2_almost_max,chi_2_almost_dof,chi_2_almost_max_dof],psf_position
            
   
            
    def create_x(self,zparameters,globalparameters):
        """
        Given the zparameters and globalparameters separtly, this code moves them in a single array     
        
        @param zparameters        Zernike coefficents
        @param globalparameters   other parameters describing the system
        """ 
        x=np.zeros((len(zparameters)+len(globalparameters)))
        for i in range(len(zparameters)):
            x[i]=zparameters[i]     
    
    
        for i in range(len(globalparameters)):
            x[int(len(zparameters)/1)+i]=globalparameters[i]      
        
    
        return x

    def __call__(self, allparameters,return_Image=False,return_intermediate_images=False,\
                 use_only_chi=False,multi_background_factor=3):
        return self.lnlike_Neven(allparameters,return_Image=return_Image,\
                                 return_intermediate_images=return_intermediate_images,\
                                 use_only_chi=use_only_chi,\
                                 multi_background_factor=multi_background_factor)

class LNP_PFS(object):
    def __init__(self,  image=None,image_var=None):
        self.image=image
        self.image_var=image_var
    def __call__(self, image=None,image_var=None):
        return 0.0

class PFSLikelihoodModule(object):
    """
    PFSLikelihoodModule class for calculating a likelihood for cosmoHammer.ParticleSwarmOptimizer
    """

    def __init__(self,model,explicit_wavefront=None):
        """
        
        Constructor of the PFSLikelihoodModule
        """
        self.model=model
        self.explicit_wavefront=explicit_wavefront

    def computeLikelihood(self, ctx):
        """
        Computes the likelihood using information from the context
        """
        # Get information from the context. This can be results from a core
        # module or the parameters coming from the sampler
        params= ctx.getParams()[0]
        return_Images_value=ctx.getParams()[1]
   
        print('params'+str(params))

        # Calculate a likelihood up to normalization
        lnprob = self.model(params,return_Images=return_Images_value)
        

        
        #print('current_process is: '+str(current_process())+str(lnprob))
        #print(params)
        #print('within computeLikelihood: parameters-hash '+str(hash(str(params.data)))+'/threading: '+str(threading.get_ident()))
        
        #sys.stdout.flush()
        # Return the likelihood
        return lnprob

    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        #e.g. load data from files

        print("PFSLikelihoodModule setup done")


    
class Zernike_Analysis(object):
    """!
    
    !!!!! DEPRECATED !!!!
    Zernike_Analysis is its own class now
    
    
    Class for analysing results of the cluster run
    """

    def __init__(self, date,obs,single_number,eps,arc=None,dataset=None):
        """!

        @param[in]
        """
        if arc is None:
            arc=''

        if dataset==0:
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Nov_14/Stamps_cleaned"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=8603
                elif arc=="Ne":
                    single_number_focus=8693  

        if dataset==1:   
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Feb_5/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=11748
                elif arc=="Ne":
                    single_number_focus=11748+607  
 
        if dataset==2:
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_May_28/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=17017+54
                if arc=="Ne":
                    single_number_focus=16292  
                if arc=="Kr":
                    single_number_focus=17310+54  
                
        if dataset==3:  
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Jun_25/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=19238+54
                elif arc=="Ne":
                    single_number_focus=19472  
                    
        if dataset==4:  
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/ReducedData/Data_Aug_14/Stamps_cleaned/"
            if arc is not None:         
                if arc=="HgAr":
                    single_number_focus=21346+54
                elif arc=="Ne":
                    single_number_focus==21550+54  
                if str(arc)=="Kr":
                    single_number_focus==21754+54                     
                    
            ##########################
            # import data
        if obs==8600:
            print("Not implemented for December 2018 data")
        else:
            sci_image =np.load(STAMPS_FOLDER+'sci'+str(obs)+str(single_number)+str(arc)+'_2Stacked.npy')
            mask_image =np.load(STAMPS_FOLDER+'mask'+str(obs)+str(single_number)+str(arc)+'_2Stacked.npy')
            var_image =np.load(STAMPS_FOLDER+'var'+str(obs)+str(single_number)+str(arc)+'_2Stacked.npy')
            sci_image_focus_large =np.load(STAMPS_FOLDER+'sci'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')
            var_image_focus_large =np.load(STAMPS_FOLDER+'var'+str(single_number_focus)+str(single_number)+str(arc)+'_Stacked_large.npy')   
        
        
        self.sci_image=sci_image
        self.var_image=var_image
        self.mask_image=mask_image
        """
        columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent',
                      'x_ilum','y_ilum',
                      'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
                      'pixel_effect','fiber_r','flux']    
        """
        columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                                  'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                                  'wide_0','wide_23','wide_43','misalign',
                                  'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                                  'grating_lines','scattering_slope','scattering_amplitude',
                                  'pixel_effect','fiber_r','flux']     

        
        self.columns=columns
        
        

        
        RESULT_FOLDER='/Users/nevencaplar/Documents/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
        if os.path.exists(RESULT_FOLDER):
            pass
        else:
            RESULT_FOLDER='/Volumes/My Passport for Mac/Old_Files/PFS/TigerAnalysis/ResultsFromTiger/'+date+'/'
        
        self.RESULT_FOLDER=RESULT_FOLDER
        
        IMAGES_FOLDER='/Users/nevencaplar/Documents/PFS/Images/'+date+'/'
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        self.IMAGES_FOLDER=IMAGES_FOLDER

        self.date=date
        self.obs=obs
        self.single_number=single_number
        self.eps=eps
        self.arc=arc
        
        method='P'
        self.method=method
        
    def create_list_of_var_or_ln_sums(self,sigma_offset=0):
        list_of_var_sums=[]
        for i in range(len(self.list_of_var_images)):
            # taking from create_chi_2_almost function in LN_PFS_single
    
    
            mask_image=self.list_of_mask_images[i]
            var_image=self.list_of_var_images[i]
            # array that has True for values which are good and False for bad values
            inverted_mask=~mask_image.astype(bool)
    
            #         
            var_image_masked=var_image*inverted_mask
            var_image_masked_without_nan = var_image_masked.ravel()[var_image_masked.ravel()>0]
    
            var_sum=-(1/2)*(len(var_image_masked_without_nan)*sigma_offset+np.sum(np.log(2*np.pi*var_image_masked_without_nan)))
    
            list_of_var_sums.append(var_sum)  
            
        array_of_var_sums=np.array(list_of_var_sums)    
        return array_of_var_sums        
        
    
    def create_likelihood(self):
        #chain_Emcee1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')
        #likechain_Emcee1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')

        # get chain number 0, which is has lowest temperature
        #likechain0_Emcee1=likechain_Emcee1[0]
        #chain0_Emcee1=chain_Emcee1[0]

        #chain_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
       
        like_min_Emcee2=[]
        for i in range(likechain_Emcee2.shape[1]):
            like_min_Emcee2.append(np.min(np.abs(likechain_Emcee2[:,i]))  )     
            
        #chain_Swarm1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Swarm1.npy')
        likechain_Swarm1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')
        
        like_min_swarm1=[]
        for i in range(likechain_Swarm1.shape[0]):
            like_min_swarm1.append(np.min(np.abs(likechain_Swarm1[i]))  )  
        #likechain0_Emcee2=likechain_Emcee2[0]
        #chain0_Emcee2=chain_Emcee2[0]        
        
        #chain_Swarm2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Swarm2.npy')
        likechain_Swarm2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')        

        like_min_swarm2=[]
        for i in range(likechain_Swarm2.shape[0]):
            like_min_swarm2.append(np.min(np.abs(likechain_Swarm2[i]))  )  
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        
        # get chain number 0, which is has lowest temperature
        if len(likechain_Emcee3)<=4:
            likechain0_Emcee3=likechain_Emcee3[0]
            chain0_Emcee3=chain_Emcee3[0]     
        else:
            likechain0_Emcee3=likechain_Emcee3
            chain0_Emcee3=chain_Emcee3
        # check the shape of the chain (number of walkers, number of steps, number of parameters)
        print('(number of walkers, number of steps, number of parameters): '+str(chain0_Emcee3.shape))
        
        # see the best chain, in numpy and pandas form
        minchain=chain0_Emcee3[np.abs(likechain0_Emcee3)==np.min(np.abs(likechain0_Emcee3))][0]
        #print(minchain)
        self.minchain=minchain
        like_min_Emcee3=[]
        
        #for i in range(likechain0_Emcee1.shape[1]):
        #    like_min.append(np.min(np.abs(likechain0_Emcee1[:,i])))
        
        #for i in range(likechain0_Emcee2.shape[1]):
        #    like_min.append(np.min(np.abs(likechain0_Emcee2[:,i])))    
        
        for i in range(likechain0_Emcee3.shape[1]):
            like_min_Emcee3.append(np.min(np.abs(likechain0_Emcee3[:,i]))  )  
        
        #print(len(like_min_swarm1))
        #print(len(like_min_Emcee2))
        #print(len(like_min_swarm2))
        #print(len(like_min_Emcee3))        
        like_min=like_min_swarm1+like_min_Emcee2+like_min_swarm2+like_min_Emcee3
        #print(len(like_min))                
        print('minimal likelihood is: '+str(np.min(like_min)))   
        chi2=(np.array(like_min)*(2)-np.log(2*np.pi*np.sum(self.var_image)))/(self.sci_image.shape[0])**2
        print('minimal chi2 reduced is: '+str(np.min(chi2)))
        
        return minchain,like_min

    def create_likelihood_multi(self):
        self.obs=8627
        #chain_Emcee1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')
        likechain_Emcee1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Multi_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')

        # get chain number 0, which is has lowest temperature
        likechain0_Emcee1=likechain_Emcee1[0]
        #chain0_Emcee1=chain_Emcee1[0]

        #chain_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Multi_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        
        likechain0_Emcee2=likechain_Emcee2[0]
        #chain0_Emcee2=chain_Emcee2[0]        
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Multi_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee3.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Multi_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee3.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3[0]
        chain0_Emcee3=chain_Emcee3[0]     
        
        # check the shape of the chain (number of walkers, number of steps, number of parameters)
        print('(number of walkers, number of steps, number of parameters): '+str(chain0_Emcee3.shape))
        
        # see the best chain, in numpy and pandas form
        minchain=chain0_Emcee3[np.abs(likechain0_Emcee3)==np.min(np.abs(likechain0_Emcee3))][0]
        #print(minchain)
        self.minchain=minchain
        like_min=[]
        for i in range(likechain0_Emcee1.shape[1]):
            like_min.append(np.min(np.abs(likechain0_Emcee1[:,i])))
        
        for i in range(likechain0_Emcee2.shape[1]):
            like_min.append(np.min(np.abs(likechain0_Emcee2[:,i])))    
        
        for i in range(likechain0_Emcee3.shape[1]):
            like_min.append(np.min(np.abs(likechain0_Emcee3[:,i]))  )  
            
            
        print('minimal likelihood is: '+str(np.min(like_min)))   
        chi2=(np.array(like_min)*(2)-np.log(2*np.pi*np.sum(self.var_image)))/(self.sci_image.shape[0])**2
        print('minimal chi2 reduced is: '+str(np.min(chi2)))
        
        return minchain,like_min
    
    def create_chains(self):
         
        #chain_Emcee1=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')
        #likechain_Emcee1=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee1.npy')

        # get chain number 0, which is has lowest temperature
        #likechain0_Emcee1=likechain_Emcee1[0]
        #chain0_Emcee1=chain_Emcee1[0]

        #chain_Emcee2=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        #likechain_Emcee2=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+'Emcee2.npy')
        
        #likechain0_Emcee2=likechain_Emcee2[0]
        #chain0_Emcee2=chain_Emcee2[0]        
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3   
    
    
    def create_chains_Emcee_2(self):
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee3.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3   
    
    
    def create_chains_Emcee_1(self):
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Emcee2.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3       
    
    def create_chains_swarm_2(self):
              
        
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm2.npy')
        
        # get chain number 0, which is has lowest temperature
        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3      

    def create_chains_swarm_1(self):
              
        chain_Emcee3=np.load(self.RESULT_FOLDER+'chain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')
        likechain_Emcee3=np.load(self.RESULT_FOLDER+'likechain'+str(self.date)+'_Single_'+str(self.method)+'_'+str(self.obs)+str(self.single_number)+str(self.eps)+str(self.arc)+'Swarm1.npy')

        likechain0_Emcee3=likechain_Emcee3
        chain0_Emcee3=chain_Emcee3     
        
        self.chain0_Emcee3=chain0_Emcee3
        self.likechain0_Emcee3=likechain0_Emcee3
        
        return chain0_Emcee3,likechain0_Emcee3        
    
    def create_basic_comparison_plot(self): 

        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        sci_image=self.sci_image
        var_image=self.var_image
        mask_image=self.mask_image
        
        mask_image_inverse=np.logical_not(mask_image).astype(int)
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
        
        plt.figure(figsize=(14,14))


        plt.subplot(221)
        plt.imshow(res_iapetus,origin='lower',vmax=np.max(np.abs(sci_image)))
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Model')
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image*mask_image_inverse,origin='lower',vmax=np.max(np.abs(sci_image)))
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Data')
        plt.grid(False)
        plt.subplot(223)
        plt.imshow((sci_image-res_iapetus)*mask_image_inverse,origin='lower',cmap='bwr',vmin=-np.max(np.abs(sci_image))/20,vmax=np.max(np.abs(sci_image))/20)
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='black')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='black')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Residual (data - model)')
        plt.grid(False)
        plt.subplot(224)
        #plt.imshow((res_iapetus-sci_image)/np.sqrt(var_image),origin='lower',cmap='bwr',vmax=np.max(np.abs((res_iapetus-sci_image)/np.sqrt(var_image))),vmin=-np.max(np.abs((res_iapetus-sci_image)/np.sqrt(var_image))))
        plt.imshow(((sci_image-res_iapetus)/np.sqrt(var_image))*mask_image_inverse,origin='lower',cmap='bwr',vmax=5,vmin=-5)

        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='black')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='black')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('chi map')
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-10.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))
  
    def create_basic_comparison_plot_log(self):      
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        sci_image=self.sci_image
        var_image=self.var_image
        mask_image=self.mask_image
        
        mask_image_inverse=np.logical_not(mask_image).astype(int)
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
        
        
        plt.figure(figsize=(14,14))
        plt.subplot(221)
        plt.imshow(res_iapetus,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Model')
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image*mask_image_inverse,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Data')
        plt.grid(False)
        plt.subplot(223)
        plt.imshow((np.abs(sci_image-res_iapetus))*mask_image_inverse,origin='lower',vmax=np.max(np.abs(sci_image))/20,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('abs(Residual (  data-model))')
        plt.grid(False)
        plt.subplot(224)
        plt.imshow(((sci_image-res_iapetus)**2/((1)*var_image))*mask_image_inverse,origin='lower',vmin=1,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('chi**2 map')
        np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-7.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        print('mask_image_inverse'+str(np.sum(mask_image_inverse)))
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))     
 

    def create_basic_comparison_plot_log_artifical(self):      
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        noise=cs=self.create_artificial_noise()
        sci_image=self.sci_image
        var_image=self.var_image
        size=sci_image.shape[0]
        if size==40:
            dithering=2
        else:
            dithering=1
        
        
        plt.figure(figsize=(14,14))
        plt.subplot(221)
        plt.imshow(res_iapetus+noise,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Model with artifical noise')
        plt.grid(False)
        plt.subplot(222)
        plt.imshow(sci_image,origin='lower',vmin=1,vmax=np.max(np.abs(sci_image)),norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('Data')
        plt.grid(False)
        plt.subplot(223)
        plt.imshow(np.abs(res_iapetus-sci_image),origin='lower',vmax=np.max(np.abs(sci_image))/20,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('abs(Residual (model - data))')
        plt.grid(False)
        plt.subplot(224)
        plt.imshow((res_iapetus-sci_image)**2/((1)*var_image),origin='lower',vmin=1,norm=LogNorm())
        plt.plot(np.ones(len(sci_image))*(size/2-3.5),np.array(range(len(sci_image))),'--',color='white')
        plt.plot(np.ones(len(sci_image))*((size/2-dithering*3.5)+7*dithering),np.array(range(len(sci_image))),'--',color='white')
        cbar=plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_ticks([10,10**2,10**3,10**4,10**5])
        plt.title('chi**2 map')
        print(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image)))
        np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))
        plt.tight_layout(pad=0.0, w_pad=1.8, h_pad=-7.0)
        print('chi**2 reduced is: '+str(np.sum((res_iapetus-sci_image)**2/((var_image.shape[0]*var_image.shape[1])*var_image))))
        print('Abs of residual divided by total flux is: '+str(np.sum(np.abs((res_iapetus-sci_image)))/np.sum((res_iapetus))))
        print('Abs of residual divided by largest value of a flux in the image is: '+str(np.max(np.abs((res_iapetus-sci_image)/np.max(res_iapetus)))))          
        
    def create_artificial_noise(self):
        var_image=self.var_image
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        
        artifical_noise=np.zeros_like(res_iapetus)
        artifical_noise=np.array(artifical_noise)
        for i in range(len(artifical_noise)):
            for j in range(len(artifical_noise)):
                artifical_noise[i,j]=np.random.randn()*np.sqrt(var_image[i,j])       
                
        return artifical_noise
    
    def create_cut_plots(self):
        
        var_image=self.var_image
        artifical_noise=self.create_artificial_noise()
        sci_image=self.sci_image
        optPsf_cut_fiber_convolved_downsampled=np.load(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled.npy')
        res_iapetus=optPsf_cut_fiber_convolved_downsampled
        
        mid_point_of_sci_image=int(sci_image.shape[0]/2)
        
        plt.figure(figsize=(25,10))
        
        plt.subplot(121)
        plt.title('horizontal direction')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image])*(1/2)),'--',color='black')
        
        
        plt.legend(fontsize=25)
        
        plt.subplot(122)
        plt.title('wavelength direction')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[:,mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[:,mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image])*(1/2)),'--',color='black')
        plt.legend(fontsize=20)      

        plt.figure(figsize=(30,10))
        
        plt.subplot(121)
        plt.title('horizontal direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[mid_point_of_sci_image]+artifical_noise[mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[mid_point_of_sci_image]*(1/2))),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(res_iapetus[mid_point_of_sci_image]*(1/2))),'-',color='blue',label='FWHM of model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(res_iapetus[mid_point_of_sci_image]-sci_image[mid_point_of_sci_image])),'red',linestyle='--',label='abs(residual)')
        plt.legend(fontsize=15)
        
        plt.subplot(122)
        plt.title('wavelength direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(res_iapetus[:,mid_point_of_sci_image]+artifical_noise[:,mid_point_of_sci_image]),'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(sci_image[:,mid_point_of_sci_image])),'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(sci_image[:,mid_point_of_sci_image]*(1/2))),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.log10(np.max(res_iapetus[:,mid_point_of_sci_image]*(1/2))),'-',color='blue',label='FWHM of model')
        plt.plot(np.array(range(len(res_iapetus))),np.log10(np.abs(res_iapetus[:,mid_point_of_sci_image]-sci_image[:,mid_point_of_sci_image])),'red',linestyle='--',label='abs(residual)')
        plt.legend(fontsize=15)

        plt.figure(figsize=(30,10))
        
        plt.subplot(121)
        plt.title('horizontal direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[mid_point_of_sci_image]+artifical_noise[mid_point_of_sci_image],'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),sci_image[mid_point_of_sci_image],'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[mid_point_of_sci_image]-sci_image[mid_point_of_sci_image],'red',linestyle='--',label='residual')
        plt.errorbar(np.array(range(len(res_iapetus))),sci_image[mid_point_of_sci_image],yerr=1*np.sqrt(var_image[mid_point_of_sci_image]),color='orange',fmt='o')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(sci_image[mid_point_of_sci_image]*(1/2)),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(res_iapetus[mid_point_of_sci_image]*(1/2)),'-',color='blue',label='FWHM of model')
        plt.legend(fontsize=15)
        
        plt.subplot(122)
        plt.title('wavelength direction, with noise')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[:,mid_point_of_sci_image]+artifical_noise[:,mid_point_of_sci_image],'blue',linestyle='--',label='model')
        plt.plot(np.array(range(len(res_iapetus))),sci_image[:,mid_point_of_sci_image],'orange',linestyle='--',label='data')
        plt.plot(np.array(range(len(res_iapetus))),res_iapetus[:,mid_point_of_sci_image]-sci_image[:,mid_point_of_sci_image],'red',linestyle='--',label='residual')
        plt.errorbar(np.array(range(len(res_iapetus))),sci_image[:,mid_point_of_sci_image],yerr=1*np.sqrt(var_image[:,mid_point_of_sci_image]),color='orange',fmt='o')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(sci_image[mid_point_of_sci_image]*(1/2)),'-',color='orange',label='FWHM of data')
        plt.plot(np.array(range(len(res_iapetus))),np.ones(len(res_iapetus))*np.max(res_iapetus[mid_point_of_sci_image]*(1/2)),'-',color='blue',label='FWHM of model')
        plt.legend(fontsize=15)    

    def create_corner_plots(self):
        IMAGES_FOLDER='/Users/nevencaplar/Documents/PFS/Images/'+self.date+'/'
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        print('Images are in folder: '+str(IMAGES_FOLDER))
    
        import corner
        minchain=self.minchain
        columns=self.columns
        chain0_Emcee3=self.chain0_Emcee3
        #likechain_Emcee3=self.likechain_Emcee3

        matplotlib.rcParams.update({'font.size': 16})
        flatchain0=np.reshape(chain0_Emcee3,(chain0_Emcee3.shape[0]*chain0_Emcee3.shape[1],chain0_Emcee3.shape[2]))
        
        figure=corner.corner(flatchain0[:,0:8], labels=columns[0:8],
                          truths=list(minchain[0:8]))
        figure.savefig(IMAGES_FOLDER+'zparameters.png')
        
        flatchain0=np.reshape(chain0_Emcee3,(chain0_Emcee3.shape[0]*chain0_Emcee3.shape[1],chain0_Emcee3.shape[2]))

        figure=corner.corner(flatchain0[:,8:], labels=columns[8:],
                          truths=list(minchain[8:]))
        figure.savefig(IMAGES_FOLDER+'/globalparameters.png')
        


class Psf_position(object):
    """
    Class that deals with positioning the PSF model in respect to the data
    
    inputs are:
        
        image                                       oversampled model image
        oversampling                                by how much is the the oversampled image oversampled
        size_natural_resolution                     size of the final image
        simulation_00                               True if simulate at optical center at 0,0
        double_sources
        double_sources_positions_ratios
        verbosity
        save
    
    """

    def __init__(self, image,oversampling,size_natural_resolution, simulation_00=False,\
                 double_sources=False,double_sources_positions_ratios=[0,0],verbosity=0,save=None):
        
        self.image=image
        self.oversampling=oversampling
        self.size_natural_resolution=size_natural_resolution
        self.simulation_00=simulation_00
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios
        self.verbosity=verbosity
        if save is None:
            save=0
            self.save=save
        else:
            save=1
            self.save=save
        

    def cut_Centroid_of_natural_resolution_image(image,size_natural_resolution,oversampling,dx,dy):
        """
        function which takes central part of a larger oversampled image
        
        @param image                          array contaning suggested starting values for a model 
        @param size_natural_resolution        size of new image in natural units 
        @param oversampling                   oversampling
        @param dx                             how much to move in dx direction (fix)
        @param dy                             how much to move in dy direction (fix)
        """
    
        positions_from_where_to_start_cut=[int(len(image)/2-size_natural_resolution/2-dx*oversampling+1),
                                           int(len(image)/2-size_natural_resolution/2-dy*oversampling+1)]
    
        res=image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1]+int(size_natural_resolution),
                     positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0]+int(size_natural_resolution)]
        
        return res

    def find_single_realization_min_cut(self, input_image,oversampling,size_natural_resolution,sci_image,var_image,mask_image,v_flux, simulation_00=False,
                                        double_sources=None,double_sources_positions_ratios=[0,0],verbosity=0,explicit_psf_position=None,
                                        use_only_chi=False,use_center_of_flux=False):
    
        """
        function called by create_optPSF_natural in ZernikeFitter_PFS
        find what is the best starting point to downsample the oversampled image
        
        @param image                                                      image to be analyzed (in our case this will be image of the optical psf convolved with fiber)
        @param oversampling                                               oversampling
        @param size_natural_resolution                                    size of final image (in the ``natural'', i.e., physical resolution)
        @param sci_image_0                                                scientific image
        @param var_image_0                                                variance image
        @param v_flux                                                     flux normalization
        @param simulation_00                                              do not move the center, for making fair comparisons between models - optical center in the center of the image
        @param double_sources                                             are there double sources in the image
        @param double_sources_positions_ratios                            tuple describing init guess for the relation between secondary and primary souces (offset, ratio)
        @param verbosity                                                  verbosity of the algorithm
        @param explicit_psf_position                                      x and y offset
        @param use_only_chi          
        @param use_center_of_flux
        
        calls function create_complete_realization (many times in order to fit the best solution)
        
        
        returns model image in the size of the science image and centered to the science image (unless simulation_00=True or explicit_psf_position has been passed)
        """
        
        self.sci_image=sci_image
        self.var_image=var_image
        self.mask_image=mask_image
        self.v_flux=v_flux


        # if you are just asking for simulated image at (0,0) there is no possibility to create double sources
        if simulation_00==1:
            double_sources=None
            
        if double_sources is None or double_sources is False:
            double_sources_positions_ratios=[0,0]
        
        shape_of_input_img=input_image.shape[0]
        shape_of_sci_image=sci_image.shape[0]
        max_possible_value_to_analyze=int(shape_of_input_img-oversampling)
        min_possible_value_to_analyze=int(oversampling)
        center_point=int(shape_of_input_img/2)
        
        self.shape_of_input_img=shape_of_input_img
        self.shape_of_sci_image=shape_of_sci_image

        if verbosity==1:
            print('parameter use_only_chi in Psf_postion is set to: '+str(use_only_chi))
            print('parameter use_center_of_flux in Psf_postion is set to: '+str(use_center_of_flux))

        # depending on if there is a second source in the image split here
        # double_sources is always None when using simulated images
        if double_sources==None or double_sources is False:
            # if simulation_00 is on just run the realization set at 0
            if simulation_00==1:
                if verbosity==1:
                    print('simulation_00 is set to 1 - I am just returing the image at (0,0) coordinates ')                
      

                # return the solution with x and y is zero
                mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
                =self.create_complete_realization([0,0], return_full_result=True,use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)          
            
            # if you are fitting an actual image go through the full process
            else:

                # if you did not pass explict position search for the best position
                if explicit_psf_position is None:
                    

                    # create one complete realization with default parameters - estimate centorids and use that knowledge to put fitting limits in the next step
                    centroid_of_sci_image=find_centroid_of_flux(sci_image)
  
                    
  
                    
                    initial_complete_realization=self.create_complete_realization([0,0,\
                                                                                   -double_sources_positions_ratios[0]*self.oversampling,\
                                                                                   double_sources_positions_ratios[1]],return_full_result=True,\
                                                                                   use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)[-1]


                    centroid_of_initial_complete_realization=find_centroid_of_flux(initial_complete_realization)



                    #determine offset between the initial guess and the data
                    offset_initial_and_sci=np.array(find_centroid_of_flux(initial_complete_realization))-np.array(find_centroid_of_flux(sci_image))

                    if verbosity==1:
                        print('offset_initial_and_sci: '+str(offset_initial_and_sci))
                    
                    if self.save==1:
                        np.save(TESTING_FINAL_IMAGES_FOLDER+'initial_complete_realization',initial_complete_realization) 
            
                    # search for the best center using scipy ``shgo'' algorithm
                    # set the limits for the fitting procedure
                    y_2sources_limits=[(offset_initial_and_sci[1]-2)*self.oversampling,(offset_initial_and_sci[1]+2)*self.oversampling]
                    x_2sources_limits=[(offset_initial_and_sci[0]-1)*self.oversampling,(offset_initial_and_sci[0]+1)*self.oversampling]
                    # search for best positioning
                    # implement try for secondary too
                    try:
                        #print('(False,use_only_chi,use_center_of_flux)'+str((False,use_only_chi,use_center_of_flux)))
                        primary_position_and_ratio_shgo=scipy.optimize.shgo(self.create_complete_realization,args=(False,use_only_chi,use_center_of_flux),bounds=\
                                                                                 [(x_2sources_limits[0],x_2sources_limits[1]),(y_2sources_limits[0],y_2sources_limits[1])],n=10,sampling_method='sobol',\
                                                                                 options={'ftol':1e-3,'maxev':10})
                            
                        
                        #primary_position_and_ratio=primary_position_and_ratio_shgo
                        primary_position_and_ratio=scipy.optimize.minimize(self.create_complete_realization,args=(False,use_only_chi,use_center_of_flux),x0=primary_position_and_ratio_shgo.x,\
                                                                           method='Nelder-Mead',options={'xatol': 0.00001, 'fatol': 0.00001})    
                        
                        primary_position_and_ratio_x=primary_position_and_ratio.x
                    except:
                        print('search for primary position failed')
                        primary_position_and_ratio_x=[0,0]
                        

    
                    # return the best result, based on the result of the conducted search
                    mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
                    =self.create_complete_realization(primary_position_and_ratio_x, return_full_result=True,use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)
                    
                    if self.save==1:
                        np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_primary_renormalized',single_realization_primary_renormalized) 
                        np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_secondary_renormalized',single_realization_secondary_renormalized)     
                        np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized',complete_realization_renormalized)     
            
                    if self.verbosity==1:
                        if simulation_00!=1:
                            print('We are fitting for only one source')
                            print('One source fitting result is '+str(primary_position_and_ratio_x))   
                            print('type(complete_realization_renormalized)'+str(type(complete_realization_renormalized[0][0])))
                                
                    return complete_realization_renormalized,primary_position_and_ratio_x
                # if you did pass explicit_psf_position for the solution
                else:

                    mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
                    =self.create_complete_realization(explicit_psf_position, return_full_result=True,use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)
  
                        
                    if self.save==1:
                            np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_primary_renormalized',single_realization_primary_renormalized) 
                            np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_secondary_renormalized',single_realization_secondary_renormalized)     
                            np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized',complete_realization_renormalized)     
                
                    if self.verbosity==1:
                            if simulation_00!=1:
                                print('We are passing value for only one source')
                                print('One source fitting result is '+str(explicit_psf_position))   
                                print('type(complete_realization_renormalized)'+str(type(complete_realization_renormalized[0][0])))
                                
                    return complete_realization_renormalized,explicit_psf_position
                        
                
                
        else: 
            # need to make possible that you can pass your own values for double source!!!!
            # !!!!!
            # !!!!!
            #!!!!!
            
            # create one complete realization with default parameters - estimate centroids and use that knowledge to put fitting limits in the next step
            centroid_of_sci_image=find_centroid_of_flux(sci_image)
            #print('initial double_sources_positions_ratios is: '+str(double_sources_positions_ratios))
            initial_complete_realization=self.create_complete_realization([0,0,\
                                                                           -double_sources_positions_ratios[0]*self.oversampling,double_sources_positions_ratios[1]],
                                                                           return_full_result=True,use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)[-1]
            centroid_of_initial_complete_realization=find_centroid_of_flux(initial_complete_realization)
            
            #determine offset between the initial guess and the data
            offset_initial_and_sci=np.array(centroid_of_initial_complete_realization)-np.array(centroid_of_sci_image)
            
            if verbosity==1:
                print('offset_initial_and_sci: '+str(offset_initial_and_sci))
            
            if self.save==1:
                np.save(TESTING_FINAL_IMAGES_FOLDER+'sci_image',sci_image) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'initial_complete_realization',initial_complete_realization) 

            # set the limits for the fitting procedure
            y_2sources_limits=[(offset_initial_and_sci[1]-2)*self.oversampling,(offset_initial_and_sci[1]+2)*self.oversampling]
            x_2sources_limits=[(offset_initial_and_sci[0]-1)*self.oversampling,(offset_initial_and_sci[0]+1)*self.oversampling]
            y_2sources_limits_second_source=[(-self.double_sources_positions_ratios[0]-2)*oversampling,(-self.double_sources_positions_ratios[0]+2)*oversampling]
            
            # search for best result
            # x position, y_position_1st, y_position_2nd, ratio
            primary_secondary_position_and_ratio=scipy.optimize.shgo(self.create_complete_realization,(False,use_only_chi,use_center_of_flux),bounds=\
                                                                     [(x_2sources_limits[0],x_2sources_limits[1]),(y_2sources_limits[0],y_2sources_limits[1]),\
                                                                      (y_2sources_limits_second_source[0],y_2sources_limits_second_source[1]),\
                                                                      (self.double_sources_positions_ratios[1]/2,2*self.double_sources_positions_ratios[1])],n=10,sampling_method='sobol',\
                                                                      options={'maxev':10,'ftol':1e-3})
            
            #return best result
            mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
            =self.create_complete_realization(primary_secondary_position_and_ratio.x,
                                              return_full_result=True,use_only_chi=use_only_chi,use_center_of_light=use_center_of_flux)
    
            if self.save==1:
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_primary_renormalized',single_realization_primary_renormalized) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_secondary_renormalized',single_realization_secondary_renormalized)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized',complete_realization_renormalized)     
    
            if self.verbosity==1:
                print('We are fitting for two sources')
                print('Two source fitting result is '+str(primary_secondary_position_and_ratio.x))   
                print('type(complete_realization_renormalized)'+str(type(complete_realization_renormalized[0][0])))
        
        
        return complete_realization_renormalized,primary_secondary_position_and_ratio.x
    
    
    def create_complete_realization(self, x, return_full_result=False,use_only_chi=False,use_center_of_light=False):
        # need to include masking
        # fits for best chi2, even if algorithm is asking for chi
        """
        create one complete realization of the image from the full oversampled image
        
        @param     x                                                          array contaning x_primary, y_primary, offset in y to secondary source, ratio in flux from secondary to primary
        @bol       return_full_result                                         if True, returns the images iteself (not just chi**2)
        @bol       use_only_chi
        @bol       use_center_of_light
        
        called by find_single_realization_min_cut
        
        calls create_chi_2_almost_Psf_position
        """
        
        #print('len(x): '+str(len(x)))
        #print('x passed to create_complete_realization is: '+str(x))
        #print('return_full_result in create_complete_realization: '+str(return_full_result))     
        #print('use_only_chi in create_complete_realization:  '+str(use_only_chi))
        #print('use_center_of_light in create_complete_realization:  '+str(use_center_of_light))
        #print('image.shape: '+str(self.image.shape))
        
        image=self.image
        # I think I use sci_image only for its shape
        sci_image=self.sci_image
        var_image=self.var_image
        mask_image=self.mask_image
        shape_of_sci_image=self.size_natural_resolution
        oversampling=self.oversampling
        v_flux=self.v_flux
               
        # central position of the create oversampled image
        center_position=int(np.floor(image.shape[0]/2))
        primary_offset_axis_1=x[0]
        primary_offset_axis_0=x[1]    
        
        # if you are only fitting for primary image
        # add zero values for secondary image
        if len(x)==2:
            ratio_secondary=0
        else:        
            ratio_secondary=x[3]

        if len(x)==2:
            secondary_offset_axis_1=0
            secondary_offset_axis_0=0
        else:    
            secondary_offset_axis_1=primary_offset_axis_1
            secondary_offset_axis_0=x[2]+primary_offset_axis_0
            
           
        # offset from x positions
        primary_offset_axis_1_floor=int(np.floor(primary_offset_axis_1)+center_position-shape_of_sci_image/2*oversampling)
        primary_offset_axis_1_ceiling=int(np.ceil(primary_offset_axis_1)+center_position-shape_of_sci_image/2*oversampling)
        primary_offset_axis_1_mod_from_floor=primary_offset_axis_1-int(np.floor(primary_offset_axis_1))
        
        # offset from y positions
        primary_offset_axis_0_floor=int(np.floor(primary_offset_axis_0)+center_position-shape_of_sci_image/2*oversampling)
        primary_offset_axis_0_ceiling=int(np.ceil(primary_offset_axis_0)+center_position-shape_of_sci_image/2*oversampling)
        primary_offset_axis_0_mod_from_floor=primary_offset_axis_0-int(np.floor(primary_offset_axis_0))
    
        # secondary offset from x position (probably superflous, as it should be the same as primary )
        secondary_offset_axis_1_floor=int(np.floor(secondary_offset_axis_1)+center_position-shape_of_sci_image/2*oversampling)
        secondary_offset_axis_1_ceiling=int(np.ceil(secondary_offset_axis_1)+center_position-shape_of_sci_image/2*oversampling)
        secondary_offset_axis_1_mod_from_floor=secondary_offset_axis_1-int(np.floor(secondary_offset_axis_1))
    
        # secondary offset from y position (probably superflous, as it should be the same as primary )
        secondary_offset_axis_0_floor=int(np.floor(secondary_offset_axis_0)+center_position-shape_of_sci_image/2*oversampling)
        secondary_offset_axis_0_ceiling=int(np.ceil(secondary_offset_axis_0)+center_position-shape_of_sci_image/2*oversampling)
        secondary_offset_axis_0_mod_from_floor=secondary_offset_axis_0-int(np.floor(secondary_offset_axis_0))
        
        # if you have to pad the image with zeros on either side go into this part of if statment
        if primary_offset_axis_0_floor<0 or (primary_offset_axis_0_ceiling+oversampling*shape_of_sci_image)>len(image)\
        or primary_offset_axis_1_floor<0 or (primary_offset_axis_1_ceiling+oversampling*shape_of_sci_image)>len(image):
            #print('going into crop loop')
    
            pos_floor_floor = np.array([primary_offset_axis_0_floor, primary_offset_axis_1_floor])
            pos_floor_ceiling = np.array([primary_offset_axis_0_floor, primary_offset_axis_1_ceiling])
            pos_ceiling_floor = np.array([primary_offset_axis_0_ceiling, primary_offset_axis_1_floor])
            pos_ceiling_ceiling = np.array([primary_offset_axis_0_ceiling, primary_offset_axis_1_ceiling])    

            # image in the top right corner, x=1, y=1
            input_img_single_realization_before_downsampling_primary_floor_floor = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image),dtype=np.float32)
            # image in the top left corner, x=0, y=1
            input_img_single_realization_before_downsampling_primary_floor_ceiling = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image),dtype=np.float32)
            # image in the bottom right corner, x=1, y=0
            input_img_single_realization_before_downsampling_primary_ceiling_floor = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image),dtype=np.float32)
            # image in the bottom left corner, x=0, y=0
            input_img_single_realization_before_downsampling_primary_ceiling_ceiling = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image),dtype=np.float32)

            self.fill_crop(image, pos_floor_floor, input_img_single_realization_before_downsampling_primary_floor_floor)
            self.fill_crop(image, pos_floor_ceiling, input_img_single_realization_before_downsampling_primary_floor_ceiling)
            self.fill_crop(image, pos_ceiling_floor, input_img_single_realization_before_downsampling_primary_ceiling_floor)
            self.fill_crop(image, pos_ceiling_ceiling, input_img_single_realization_before_downsampling_primary_ceiling_ceiling)
    
        else:
            # if you do not have to pad, just simply take the part of the original oversampled image
            input_img_single_realization_before_downsampling_primary_floor_floor=image[primary_offset_axis_0_floor:primary_offset_axis_0_floor+oversampling*shape_of_sci_image,\
                                                                       primary_offset_axis_1_floor:primary_offset_axis_1_floor+oversampling*shape_of_sci_image]
            input_img_single_realization_before_downsampling_primary_floor_ceiling=image[primary_offset_axis_0_floor:primary_offset_axis_0_floor+oversampling*shape_of_sci_image,\
                                                                       primary_offset_axis_1_ceiling:primary_offset_axis_1_ceiling+oversampling*shape_of_sci_image]
            input_img_single_realization_before_downsampling_primary_ceiling_floor=image[primary_offset_axis_0_ceiling:primary_offset_axis_0_ceiling+oversampling*shape_of_sci_image,\
                                                                       primary_offset_axis_1_floor:primary_offset_axis_1_floor+oversampling*shape_of_sci_image]
            input_img_single_realization_before_downsampling_primary_ceiling_ceiling=image[primary_offset_axis_0_ceiling:primary_offset_axis_0_ceiling+oversampling*shape_of_sci_image,\
                                                                       primary_offset_axis_1_ceiling:primary_offset_axis_1_ceiling+oversampling*shape_of_sci_image]
        
        
        # construct bilinear interpolation from these 4 images

        input_img_single_realization_before_downsampling_primary=self.bilinear_interpolation(primary_offset_axis_0_mod_from_floor,primary_offset_axis_1_mod_from_floor,\
                                                                                        input_img_single_realization_before_downsampling_primary_floor_floor,input_img_single_realization_before_downsampling_primary_floor_ceiling,\
                                                                                        input_img_single_realization_before_downsampling_primary_ceiling_floor,input_img_single_realization_before_downsampling_primary_ceiling_ceiling)
        
        #print('input_img_single_realization_before_downsampling_primary:'+str(input_img_single_realization_before_downsampling_primary))
        # downsample the primary image    
        single_primary_realization=resize(input_img_single_realization_before_downsampling_primary,(shape_of_sci_image,shape_of_sci_image))
         
        ###################
        # implement - if secondary too far outside the image, do not go through secondary
        # go through secondary loop if the flux ratio is not zero
        if ratio_secondary !=0:
            # if the secondary would be outside
            if secondary_offset_axis_0_floor<0 or (secondary_offset_axis_0_ceiling+oversampling*shape_of_sci_image)>len(image)\
            or secondary_offset_axis_1_floor<0 or (secondary_offset_axis_1_ceiling+oversampling*shape_of_sci_image)>len(image):
                #print('checkpoint here')
                pos_floor_floor = np.array([secondary_offset_axis_0_floor, secondary_offset_axis_1_floor])
                pos_floor_ceiling = np.array([secondary_offset_axis_0_floor, secondary_offset_axis_1_ceiling])
                pos_ceiling_floor = np.array([secondary_offset_axis_0_ceiling, secondary_offset_axis_1_floor])
                pos_ceiling_ceiling = np.array([secondary_offset_axis_0_ceiling, secondary_offset_axis_1_ceiling])    
                #print('pos_floor_floor: '+str(pos_floor_floor))     
                input_img_single_realization_before_downsampling_secondary_floor_floor = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0,dtype=np.float32)
                input_img_single_realization_before_downsampling_secondary_floor_ceiling = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0,dtype=np.float32)
                input_img_single_realization_before_downsampling_secondary_ceiling_floor = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0,dtype=np.float32)
                input_img_single_realization_before_downsampling_secondary_ceiling_ceiling= np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0,dtype=np.float32)
        
                self.fill_crop(image, pos_floor_floor, input_img_single_realization_before_downsampling_secondary_floor_floor)
                self.fill_crop(image, pos_floor_ceiling, input_img_single_realization_before_downsampling_secondary_floor_ceiling)
                self.fill_crop(image, pos_ceiling_floor, input_img_single_realization_before_downsampling_secondary_ceiling_floor)
                self.fill_crop(image, pos_ceiling_ceiling, input_img_single_realization_before_downsampling_secondary_ceiling_ceiling)
                
            else:
                
                
                input_img_single_realization_before_downsampling_secondary_floor_floor=image[secondary_offset_axis_0_floor:secondary_offset_axis_0_floor+oversampling*shape_of_sci_image,\
                                                                           secondary_offset_axis_1_floor:secondary_offset_axis_1_floor+oversampling*shape_of_sci_image]
                input_img_single_realization_before_downsampling_secondary_floor_ceiling=image[secondary_offset_axis_0_floor:secondary_offset_axis_0_floor+oversampling*shape_of_sci_image,\
                                                                           secondary_offset_axis_1_ceiling:secondary_offset_axis_1_ceiling+oversampling*shape_of_sci_image]
                input_img_single_realization_before_downsampling_secondary_ceiling_floor=image[secondary_offset_axis_0_ceiling:secondary_offset_axis_0_ceiling+oversampling*shape_of_sci_image,\
                                                                           secondary_offset_axis_1_floor:secondary_offset_axis_1_floor+oversampling*shape_of_sci_image]  
                input_img_single_realization_before_downsampling_secondary_ceiling_ceiling=image[secondary_offset_axis_0_ceiling:secondary_offset_axis_0_ceiling+oversampling*shape_of_sci_image,\
                                                                           secondary_offset_axis_1_ceiling:secondary_offset_axis_1_ceiling+oversampling*shape_of_sci_image]
        
            # first two argument input is y,x offset
            # so first axis_1 then axis 0
            # change here from 0.39b to 0.4        
            input_img_single_realization_before_downsampling_secondary=self.bilinear_interpolation(secondary_offset_axis_0_mod_from_floor,secondary_offset_axis_1_mod_from_floor,\
                                                                                        input_img_single_realization_before_downsampling_secondary_floor_floor,input_img_single_realization_before_downsampling_secondary_floor_ceiling,\
                                                                                        input_img_single_realization_before_downsampling_secondary_ceiling_floor,input_img_single_realization_before_downsampling_secondary_ceiling_ceiling)
                    
            single_secondary_realization=resize(input_img_single_realization_before_downsampling_secondary,(shape_of_sci_image,shape_of_sci_image))    
    
        inverted_mask=~mask_image.astype(bool)
    
        #print('ratio_secondary: '+str(ratio_secondary))
        if ratio_secondary !=0:
            complete_realization=single_primary_realization+ratio_secondary*single_secondary_realization
            complete_realization_renormalized=complete_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
        else:
            complete_realization=single_primary_realization
            complete_realization_renormalized=complete_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
            
        #print('checkpoint in create_complete_realization')
        if return_full_result==False:
            chi_2_almost_multi_values=self.create_chi_2_almost_Psf_position(complete_realization_renormalized,sci_image,var_image,mask_image,use_only_chi=use_only_chi,use_center_of_light=use_center_of_light)
            if self.verbosity==1:
                print('chi2 within shgo with use_only_chi '+str(use_only_chi)+' and use_center_of_light '+str(use_center_of_light)+' '+str(x)+' / '+str(chi_2_almost_multi_values))
                #print('chi2 within shgo optimization routine (not chi_2_almost_multi_values): '+str(np.mean((sci_image-complete_realization_renormalized)**2/var_image)))
            return chi_2_almost_multi_values
        else:
            if ratio_secondary !=0:
                #print('ratio_secondary 2nd loop: '+str(ratio_secondary))
                single_primary_realization_renormalized=single_primary_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized=ratio_secondary*single_secondary_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))    
            else:
                #print('ratio_secondary 2nd loop 0: '+str(ratio_secondary))
                single_primary_realization_renormalized=single_primary_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized=np.zeros(single_primary_realization_renormalized.shape)                  
            
            
            if self.save==1:
                np.save(TESTING_FINAL_IMAGES_FOLDER+'image',image)            
                
                np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_primary_floor_floor',input_img_single_realization_before_downsampling_primary_floor_floor)            
                np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_primary_floor_ceiling',input_img_single_realization_before_downsampling_primary_floor_ceiling)             
                np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_primary_ceiling_floor',input_img_single_realization_before_downsampling_primary_ceiling_floor) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_primary_ceiling_ceiling',input_img_single_realization_before_downsampling_primary_ceiling_ceiling) 
    
                np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_primary',input_img_single_realization_before_downsampling_primary) 
                if ratio_secondary !=0:
                    np.save(TESTING_FINAL_IMAGES_FOLDER+'image_full_for_secondary',image) 
         
                    np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_secondary_floor_floor',input_img_single_realization_before_downsampling_secondary_floor_floor) 
                    

                    np.save(TESTING_FINAL_IMAGES_FOLDER+'input_img_single_realization_before_downsampling_secondary',input_img_single_realization_before_downsampling_secondary) 
                    np.save(TESTING_FINAL_IMAGES_FOLDER+'single_secondary_realization',single_secondary_realization) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_primary_realization',single_primary_realization) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_primary_realization_renormalized_within_create_complete_realization',single_primary_realization_renormalized) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_secondary_realization_renormalized_within_create_complete_realization',single_secondary_realization_renormalized)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized_within_create_complete_realization',complete_realization_renormalized)     
            
            # should I modify this function to remove distance from physcial center of mass when using that option
            chi_2_almost_multi_values=self.create_chi_2_almost_Psf_position(complete_realization_renormalized,sci_image,var_image,mask_image,\
                                                                            use_only_chi=use_only_chi,use_center_of_light=use_center_of_light)

            return chi_2_almost_multi_values,\
            single_primary_realization_renormalized,single_secondary_realization_renormalized,complete_realization_renormalized  
            
            #old code that did not include mask...
            #return np.mean((sci_image-complete_realization_renormalized)**2/var_image),\
            #single_primary_realization_renormalized,single_secondary_realization_renormalized,complete_realization_renormalized   

    def create_chi_2_almost_Psf_position(self,modelImg,sci_image,var_image,mask_image,use_only_chi=False,use_center_of_light=False):
        """
        called by create_complete_realization
        
        takes the model image and the data
        
        @param modelImg     model 
        @param sci_image    scientific image 
        @param var_image    variance image
        @param mask_image   mask image
        
        return the measure of quality (chi**2, chi or distance of center of light)
        
        """ 
        
        #print('use_only_chi in create_chi_2_almost_Psf_position '+str(use_only_chi) )
        
        
        inverted_mask=~mask_image.astype(bool)
        
        var_image_masked=var_image*inverted_mask
        sci_image_masked=sci_image*inverted_mask
        modelImg_masked=modelImg*inverted_mask
        
        if use_center_of_light==False:
            if use_only_chi==False:
                chi2=(sci_image_masked - modelImg_masked)**2/var_image_masked
                chi2nontnan=chi2[~np.isnan(chi2)]
            if use_only_chi==True:
                chi2=np.abs((sci_image_masked - modelImg_masked))**1/np.sqrt(var_image_masked)
                chi2nontnan=chi2[~np.isnan(chi2)]
             
            #print('np.mean(chi2nontnan): '+str(np.mean(chi2nontnan)))    
            return np.mean(chi2nontnan)
        else:
            distance_of_flux_center=np.sqrt(np.sum((np.array(find_centroid_of_flux(modelImg_masked))-np.array(find_centroid_of_flux(sci_image_masked)))**2))
            #print('distance_of_flux_center: '+str(distance_of_flux_center))
            return distance_of_flux_center  



    def fill_crop(self, img, pos, crop):
      '''
      Fills `crop` with values from `img` at `pos`, 
      while accounting for the crop being off the edge of `img`.
      *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
      '''
      img_shape, pos, crop_shape = np.array(img.shape,dtype=int), np.array(pos,dtype=int), np.array(crop.shape,dtype=int)
      end = pos+crop_shape
      # Calculate crop slice positions
      crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
      crop_high = crop_shape - np.clip(end-img_shape, a_min=0, a_max=crop_shape)
      crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
      # Calculate img slice positions
      pos = np.clip(pos, a_min=0, a_max=img_shape)
      end = np.clip(end, a_min=0, a_max=img_shape)
      img_slices = (slice(low, high) for low, high in zip(pos, end))
      try:
          crop[tuple(crop_slices)] = img[tuple(img_slices)]    
      except TypeError:
          print('TypeError in fill_crop function')
          #np.save('/home/ncaplar/img',img)
          #np.save('/home/ncaplar/pos',pos)          
          #np.save('/home/ncaplar/crop',crop)
          pass
          
    def bilinear_interpolation(self, y,x,img_floor_floor,img_floor_ceiling,img_ceiling_floor,img_ceiling_ceiling):   
        
        '''
        creates bilinear interpolation given y and x subpixel coordinate and 4 images
        
        input
        
        y - y offset from floor_floor image
        x - x offset from floor_floor image
        
        img_floor_floor - 
        img_floor_ceiling - image offset from img_floor_floor by 1 pixel in x direction
        img_ceiling_floor - image offset from img_floor_floor by 1 pixel in y direction
        img_ceiling_ceiling - image offset from img_floor_floor by 1 pixel in both x and y direction
        
        '''
        
        
        
        # have to check if floor and ceiling definition are ok 
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        # x2=1
        # x1=0
        # y2=1
        # y1=0
        
        # img_floor_floor in top right corner
        # img_ceiling_ceiling in bottom left corner
        # img_floor_ceiling in top left corner
        # img_ceiling_floor in the bottom right corner 
        
        
        return img_floor_floor*(1-x)*(1-y)+img_floor_ceiling*(x)*(1-y)+img_ceiling_floor*(1-x)*(y)+img_ceiling_ceiling*(x)*(y)
        
      
     
    
    def create_trace(self, best_img,norm_of_trace,norm_of_serial_trace):
        if norm_of_trace==0:
            return best_img
        else:
            data_shifted_left_right=np.zeros(np.shape(best_img))
            data_shifted_left_right[:, :] =np.sum(best_img,axis=0)*norm_of_trace
        
            data_shifted_up_down=np.transpose(np.zeros(np.shape(best_img)))         
            data_shifted_up_down[:, :] =np.sum(best_img,axis=1)*norm_of_serial_trace
            data_shifted_up_down=np.transpose(data_shifted_up_down)
        
            return best_img+data_shifted_up_down+data_shifted_left_right     
        
    def estimate_trace_and_serial(self, sci_image,model_image):
    
        model_image=np.sum(sci_image)/np.sum(model_image)*model_image
        
        flux_sci_all_columns_sum_rows=np.sum(sci_image,axis=1)
        flux_sci_all_rows_sum_columns=np.sum(sci_image,axis=0)
        flux_model_all_columns_sum_rows=np.sum(model_image,axis=1)
        flux_model_all_rows_sum_columns=np.sum(model_image,axis=0)
    
        selection_of_faint_rows=flux_sci_all_columns_sum_rows<(np.sort(flux_sci_all_columns_sum_rows)[4]+1)
        selection_of_faint_columns=flux_sci_all_rows_sum_columns<(np.sort(flux_sci_all_rows_sum_columns)[4]+1)
    
        #to determine median value
        #median_rows=int(len(flux_sci_all_columns_sum_rows)/2)
    
        flux_sci_selected_faint_rows_sum_columns=np.sum(sci_image[selection_of_faint_rows],axis=0)
        flux_model_selected_faint_rows_sum_columns=np.sum(model_image[selection_of_faint_rows],axis=0)
        flux_sci_selected_faint_columns_sum_rows=np.sum(sci_image[:,selection_of_faint_columns],axis=1)
        flux_model_selected_faint_columns_sum_rows=np.sum(model_image[:,selection_of_faint_columns],axis=1)
        
        
        proposed_trace=((flux_sci_selected_faint_rows_sum_columns-flux_model_selected_faint_rows_sum_columns)/flux_model_all_rows_sum_columns)[flux_model_all_rows_sum_columns>np.max(flux_model_all_rows_sum_columns)*0.10]
        proposed_trace=np.sort(proposed_trace)[int(len(proposed_trace)/2)]
        
        proposed_serial=((flux_sci_selected_faint_columns_sum_rows-flux_model_selected_faint_columns_sum_rows)/flux_model_all_columns_sum_rows)[flux_model_all_columns_sum_rows>np.max(flux_model_all_columns_sum_rows)*0.10]
        proposed_serial=np.sort(proposed_serial)[int(len(proposed_serial)/2)]
        if proposed_trace<0:
            proposed_trace=0
        else:
            #divided by 5 because this is derived from 5 rows/columns
            proposed_trace=proposed_trace/5
    
        if proposed_serial<0:
            proposed_serial=0
        else:
            proposed_serial=proposed_serial/5
            
        return [proposed_trace,proposed_serial]
    

    
        
# ***********************    
# 'free' (not inside a class) definitions below
# ***********************   


def create_custom_var(modelImg,sci_image,var_image,mask_image=None):
    """
    
    
    The algorithm creates variance map from the model image provided
    The connection between variance and flux is determined from the provided science image and variance image
    
    
    @param modelImg     model image
    @param sci_image    scientific image 
    @param var_image    variance image        
    @param mask_image   mask image     
    
    All of inputs have to be 2d np.arrays with same size
    
    Returns the np.array with same size as inputs
    
    introduced in v0.41
    
    """
    if mask_image is None:
        sci_pixels=sci_image.ravel()
        var_pixels=var_image.ravel()   
    else:
        sci_pixels=sci_image[mask_image==0].ravel()
        var_pixels=var_image[mask_image==0].ravel()
    z=np.polyfit(sci_pixels,var_pixels,deg=2)
    p1=np.poly1d(z)
    custom_var_image=p1(sci_image)
    

    return custom_var_image


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
    # fixed error on September 18, 2020 - before it was missing one transpose, see below
    #inv_matrix = np.dot(u , np.dot( invw, v))
    inv_matrix = np.dot(u , np.dot( np.transpose(invw), v))

    return inv_matrix



def find_centroid_of_flux(image,mask=None):
    """
    function giving the tuple of the position of weighted average of the flux in a square image
    indentical result as calculateCentroid from drp_stella.images
    
    @input image    poststamp image for which to find center
    @input mask     mask, same size as the image
    
    returns tuple with x and y center, in units of pixels
    """
    if mask is None:
        mask=np.ones(image.shape)
    
    x_center=[]
    y_center=[]
    
    # if there are nan values (most likely cosmics), replace them with max value in the rest of the image
    # careful, this can seriously skew the results if not used for this purpose
    max_value_image=np.max(image[~np.isnan(image)])
    image[np.isnan(image)]=max_value_image

    I_x=[]
    for i in range(len(image)):
        I_x.append([i,np.mean(image[:,i]*mask[:,i])])

    I_x=np.array(I_x)

    I_y=[]
    for i in range(len(image)):
        I_y.append([i,np.mean(image[i]*mask[i])])

    I_y=np.array(I_y)


    x_center=(np.sum(I_x[:,0]*I_x[:,1])/np.sum(I_x[:,1]))
    y_center=(np.sum(I_y[:,0]*I_y[:,1])/np.sum(I_y[:,1]))

    return(x_center,y_center)

def create_parInit(allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,\
                   stronger=None,use_optPSF=None,deduced_scattering_slope=None,zmax=None):
    
    """!given the suggested parametrs create array with randomized starting values to supply to fitting code
    
    @param allparameters_proposal            array contaning suggested starting values for a model 
    @param multi                             set to True when you want to analyze more images at once
    @param pupil_parameters                  fix parameters describing the pupil
    @param allparameters_proposal_err        uncertantity on proposed parameters
    @param stronger                          factors which increases all uncertanties by a constant value
    @param use_optPFS                        fix all parameters that give pure optical PSF, except z4 (allowing change in ['z4', 'scattering_slope', 'scattering_amplitude', 'pixel_effect', 'fiber_r', 'flux'])
    @param deduced_scattering_slope     
    @param zmax     
    
    """ 
    
    if multi==True:
        # 
        if len(allparameters_proposal.shape)==2:
            #if you have passed 2d parametrization you have to move it to one 1d
            array_of_polyfit_1_parameterizations=np.copy(allparameters_proposal)
            if zmax==11:
                # not implemented
                pass
            if zmax==22:
                #print('zmax is 22, right: ' +str(zmax))
                #print('len(array_of_polyfit_1_parameterizations[19:]) ' +str(len(array_of_polyfit_1_parameterizations[19:]) ))
                
                
                # if you have passed the parametrization that goes to the zmax=22, depending if you passed value for flux
                if len(array_of_polyfit_1_parameterizations[19:])==23:
                    allparameters_proposal=np.concatenate((array_of_polyfit_1_parameterizations[:19].ravel(),array_of_polyfit_1_parameterizations[19:-1][:,1]))
                if len(array_of_polyfit_1_parameterizations[19:])==22:
                    allparameters_proposal=np.concatenate((array_of_polyfit_1_parameterizations[:19].ravel(),array_of_polyfit_1_parameterizations[19:][:,1]))   
                
                # if you have passed too many 
                if len(array_of_polyfit_1_parameterizations[19:])>23:
                    allparameters_proposal=np.concatenate((array_of_polyfit_1_parameterizations[:19].ravel(),array_of_polyfit_1_parameterizations[19:][:,1]))                   
                    
            if zmax>22:
                # will fail if you ask for z larger than 22 and you have not provided it 
                allparameters_proposal=np.concatenate((array_of_polyfit_1_parameterizations[:19].ravel(),array_of_polyfit_1_parameterizations[19:19+23][:,1],array_of_polyfit_1_parameterizations[42:].ravel()))   
                
                
        # if you have passed 1d parametrizations just copy
        else:
            
            allparameters_proposal=np.copy(allparameters_proposal)
        

    
    # if you are passing explicit estimate for uncertantity of parameters, make sure length is the same as of the parameters
    if allparameters_proposal_err is not None:
        assert len(allparameters_proposal)==len(allparameters_proposal_err)
        
    # default value for multiplying all uncertantity values (making them stronger) is 1     
    if stronger is None:
        stronger=1
        
    # default value for zmax, if None is passed it is set at 22
    if zmax is None:
        zmax=22
        
    # if you are passing fixed scattering slope at number deduced from larger defocused image
    # does not work with multi!!!! (really? - I think it should)
    if zmax==11:   
        if deduced_scattering_slope is not None:
            allparameters_proposal[26]=np.abs(deduced_scattering_slope)
    if zmax==22:
        if deduced_scattering_slope is not None:
            allparameters_proposal[26+11]=np.abs(deduced_scattering_slope)
    
    
    if zmax==11:
        if allparameters_proposal_err is None:
            if multi is None:
                # 8 values describing z4-z11
                allparameters_proposal_err=stronger*np.array([2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                        0.1,0.02,0.1,0.1,0.1,0.1,
                                        0.3,1,0.1,0.1,
                                        0.15,0.15,0.1,
                                        0.07,0.2,0.05,0.4,
                                        30000,0.5,0.01,
                                        0.1,0.05,0.01])
                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26]=0
            else:
                # 16 values describing z4-z11
                allparameters_proposal_err=stronger*np.array([2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                        0.1,0.1,0.1,0.1,0.05,0.1,
                                        0.2, 0.4,0.1,0.1,
                                        0.1,0.1,0.02,0.02,0.5,0.2,0.1,
                                        30000,0.5,0.01,
                                        0.1,0.05,0.01]   )     
    if zmax>=22:
        
        extra_Zernike_parameters_number=zmax-22
        #print('extra_Zernike_parameters_number in parInit:' +str(extra_Zernike_parameters_number))
        if allparameters_proposal_err is None:
            if multi is None or multi is False:
                # 19 values describing z4-z22
                # smaller values for z12-z22
                #['z4','z5','z6','z7','z8','z9','z10','z11',
                #          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22', 
                #'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                #'wide_0','wide_23','wide_43','misalign',
                #'x_fiber','y_fiber','effective_ilum_radius',
                #'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                #'grating_lines','scattering_slope','scattering_amplitude',
                #'pixel_effect','fiber_r','flux']    

                allparameters_proposal_err=stronger*np.array([2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                                     0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,
                                                     0.08,0.03,0.1,0.1,0.016,0.05,
                                                     0.3,0.3,0.3,10,
                                                     0.15,0.15,0.1,
                                                     0.1,0.64,0.05,0.2,
                                                     60000,0.95,0.014,
                                                     0.2,0.14,0.015])
                if extra_Zernike_parameters_number > 0:
                    extra_Zernike_proposal=0.0*np.ones((extra_Zernike_parameters_number,))
                    allparameters_proposal_err=np.concatenate((allparameters_proposal_err,extra_Zernike_proposal))
                
                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26+11]=0
            else:
                # determined from results_of_fit_input_HgAr
                
                allparameters_proposal_err=stronger*np.array([0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                                     0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,
                                                     0.035,0.02,0.1,0.1,0.008,0.05,
                                                     0.3,0.3,0.3,10,
                                                     0.1,0.1,0.1,
                                                     0.08,0.2,0.05,0.1,
                                                     60000,0.4,0.006,
                                                     0.2,0.04,0.015])    
                
                # at the moment zero because I do not want these to actually move around, but perhaps needs to be reconsidered in the future
                extra_Zernike_proposal=0.0*np.ones((extra_Zernike_parameters_number*2,))
                allparameters_proposal_err=np.concatenate((allparameters_proposal_err,extra_Zernike_proposal))
                

    
    
    if pupil_parameters is None:
        number_of_par=len(allparameters_proposal_err)
    else:
        number_of_par=len(allparameters_proposal_err)-len(pupil_parameters)
        
    walkers_mult=6
    nwalkers=number_of_par*walkers_mult

    
    
    if zmax==11:
        if multi is None or multi is False:
            zparameters_flatten=allparameters_proposal[0:8]
            zparameters_flatten_err=allparameters_proposal_err[0:8]
            globalparameters_flatten=allparameters_proposal[8:]
            globalparameters_flatten_err=allparameters_proposal_err[8:]

        else:
            zparameters_flatten=allparameters_proposal[0:8*2]
            zparameters_flatten_err=allparameters_proposal_err[0:8*2]
            globalparameters_flatten=allparameters_proposal[8*2:]
            globalparameters_flatten_err=allparameters_proposal_err[8*2:]
    # if we have 22 or more        
    if zmax>=22:
        if multi is None or multi is False:
            zparameters_flatten=allparameters_proposal[0:8+11]
            zparameters_flatten_err=allparameters_proposal_err[0:8+11]
            globalparameters_flatten=allparameters_proposal[8+11:8+11+23]
            globalparameters_flatten_err=allparameters_proposal_err[8+11:8+11+23]
            zparameters_extra_flatten=allparameters_proposal[(8+11)*1+23:]
            zparameters_extra_flatten_err=allparameters_proposal_err[(8+11)*1+23:]
        else:
            zparameters_flatten=allparameters_proposal[0:(8+11)*2]
            zparameters_flatten_err=allparameters_proposal_err[0:(8+11)*2]
            globalparameters_flatten=allparameters_proposal[(8+11)*2:(8+11)*2+23]
            globalparameters_flatten_err=allparameters_proposal_err[(8+11)*2:(8+11)*2+23]        
            zparameters_extra_flatten=allparameters_proposal[(8+11)*2+23:]
            zparameters_extra_flatten_err=allparameters_proposal_err[(8+11)*2+23:]
            #print('zparameters_flatten '+str(zparameters_flatten))
        
        
    if zmax==11:
        if multi is None:
            try: 
                for i in range(8):
                    if i==0:
                        zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    else:
                        zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    if i==0:
                        zparameters_flat=zparameters_flat_single_par
                    else:
                        zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
            except NameError:
                print('NameError!')
        else:
            try: 
                for i in range(8*2):
                    zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    if i==0:
                        zparameters_flat=zparameters_flat_single_par
                    else:
                        zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
            except NameError:
                print('NameError!')        
                
    # if we have 22 or more            
    if zmax>=22:
        if multi is None or multi is False:
            try: 

                for i in range(8+11):
                    if i==0:
                        zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    else:
                        zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    if i==0:
                        zparameters_flat=zparameters_flat_single_par
                    else:
                        zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
                 
                # if you are going for extra Zernike parameters
                # copied the same code from multi
                for i in range(extra_Zernike_parameters_number):

                        zparameters_extra_flat_single_par=np.concatenate(([zparameters_extra_flatten[i]],\
                                                                         np.random.normal(zparameters_extra_flatten[i],\
                                                                         zparameters_extra_flatten_err[i],nwalkers-1)))

                        #zparameters_extra_flat_single_par=np.random.normal(0,0.05,nwalkers)
                        #print(zparameters_extra_flat_single_par.shape)
                        if i==0:
                            zparameters_extra_flat=zparameters_extra_flat_single_par
                        else:
                            zparameters_extra_flat=np.column_stack((zparameters_extra_flat,zparameters_extra_flat_single_par))
                    
                        
            except NameError:
                print('NameError!')
                
        # in case that multi variable is turned on:        
        else:
            try: 
                for i in range((8+11)*2):
                    #print('i'+str(i))
                    #print('zparameters_flatten[i]: '+str(zparameters_flatten[i]))
                    #print('zparameters_flatten_err[i]: '+str(zparameters_flatten_err[i]))
                    #print('nwalkers-1: '+str(nwalkers-1))
                    #print(np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1))
                    zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],
                                                                np.random.normal(zparameters_flatten[i],\
                                                                zparameters_flatten_err[i],nwalkers-1)))
        
                    if i==0:
                        zparameters_flat=zparameters_flat_single_par
                    else:
                        zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
                        
                # if you are going for extra Zernike parameters
                if zmax>22 :
                    for i in range(extra_Zernike_parameters_number*2):
                        zparameters_extra_flat_single_par=np.concatenate(([zparameters_extra_flatten[i]],\
                                                                         np.random.normal(zparameters_extra_flatten[i],zparameters_extra_flatten_err[i],nwalkers-1)))

                        #zparameters_extra_flat_single_par=np.random.normal(0,0.05,nwalkers)
                        #print(zparameters_extra_flat_single_par.shape)
                        if i==0:
                            zparameters_extra_flat=zparameters_extra_flat_single_par
                        else:
                            zparameters_extra_flat=np.column_stack((zparameters_extra_flat,zparameters_extra_flat_single_par))
                        #print(zparameters_extra_flat.shape)
                        
            except NameError:
                print('NameError!')        
    
    try:

        # hscFrac always positive
        globalparameters_flat_0=np.abs(np.random.normal(globalparameters_flatten[0],globalparameters_flatten_err[0],nwalkers*20))
        globalparameters_flat_0=np.concatenate(([globalparameters_flatten[0]],
                                                globalparameters_flat_0[np.all((globalparameters_flat_0>=0.6,globalparameters_flat_0<=0.8),axis=0)][0:nwalkers-1]))
        # strutFrac always positive
        globalparameters_flat_1_long=np.abs(np.random.normal(globalparameters_flatten[1],globalparameters_flatten_err[1],nwalkers*200))
        globalparameters_flat_1=globalparameters_flat_1_long
        globalparameters_flat_1=np.concatenate(([globalparameters_flatten[1]],
                                                globalparameters_flat_1[np.all((globalparameters_flat_1>=0.07,globalparameters_flat_1<=0.13),axis=0)][0:nwalkers-1]))
        # dxFocal
        globalparameters_flat_2=np.random.normal(globalparameters_flatten[2],globalparameters_flatten_err[2],nwalkers*20)
        globalparameters_flat_2=np.concatenate(([globalparameters_flatten[2]],
                                                globalparameters_flat_2[np.all((globalparameters_flat_2>=-0.4,globalparameters_flat_2<=0.4),axis=0)][0:nwalkers-1]))
        # dyFocal
        globalparameters_flat_3=np.random.normal(globalparameters_flatten[3],globalparameters_flatten_err[3],nwalkers*20)
        globalparameters_flat_3=np.concatenate(([globalparameters_flatten[3]],
                                                globalparameters_flat_3[np.all((globalparameters_flat_3>=-0.4,globalparameters_flat_3<=0.4),axis=0)][0:nwalkers-1]))
        # slitFrac
        globalparameters_flat_4=np.abs(np.random.normal(globalparameters_flatten[4],globalparameters_flatten_err[4],nwalkers*20))
        #print(globalparameters_flatten_err[4])
        globalparameters_flat_4=np.concatenate(([globalparameters_flatten[4]],
                                                globalparameters_flat_4[np.all((globalparameters_flat_4>=0.05,globalparameters_flat_4<=0.09),axis=0)][0:nwalkers-1]))
        # slitFrac_dy
        globalparameters_flat_5=np.abs(np.random.normal(globalparameters_flatten[5],globalparameters_flatten_err[5],nwalkers*20))      
        globalparameters_flat_5=np.concatenate(([globalparameters_flatten[5]],
                                                globalparameters_flat_5[np.all((globalparameters_flat_5>=-0.5,globalparameters_flat_5<=0.5),axis=0)][0:nwalkers-1]))
        # wide_0
        globalparameters_flat_6=np.abs(np.random.normal(globalparameters_flatten[6],globalparameters_flatten_err[6],nwalkers*20))
        globalparameters_flat_6=np.concatenate(([globalparameters_flatten[6]],
                                                globalparameters_flat_6[np.all((globalparameters_flat_6>=0,globalparameters_flat_6<=1),axis=0)][0:nwalkers-1]))
        # wide_23
        globalparameters_flat_7=np.random.normal(globalparameters_flatten[7],globalparameters_flatten_err[7],nwalkers*20)
        globalparameters_flat_7=np.concatenate(([globalparameters_flatten[7]],
                                                globalparameters_flat_7[np.all((globalparameters_flat_7>=0.0,globalparameters_flat_7<=1),axis=0)][0:nwalkers-1]))
        # wide_43 
        globalparameters_flat_8=np.abs(np.random.normal(globalparameters_flatten[8],globalparameters_flatten_err[8],nwalkers*20))
        globalparameters_flat_8=np.concatenate(([globalparameters_flatten[8]],
                                                globalparameters_flat_8[np.all((globalparameters_flat_8>=0,globalparameters_flat_8<=1),axis=0)][0:nwalkers-1]))
        # misalign
        globalparameters_flat_9=np.abs(np.random.normal(globalparameters_flatten[9],globalparameters_flatten_err[9],nwalkers*20))
        globalparameters_flat_9=np.concatenate(([globalparameters_flatten[9]],
                                                globalparameters_flat_9[np.all((globalparameters_flat_9>=0,globalparameters_flat_9<=12),axis=0)][0:nwalkers-1]))
        # x_fiber
        globalparameters_flat_10=np.random.normal(globalparameters_flatten[10],globalparameters_flatten_err[10],nwalkers*20)
        globalparameters_flat_10=np.concatenate(([globalparameters_flatten[10]],
                                                 globalparameters_flat_10[np.all((globalparameters_flat_10>=-0.4,globalparameters_flat_10<=0.4),axis=0)][0:nwalkers-1]))
        # y_fiber
        globalparameters_flat_11=np.random.normal(globalparameters_flatten[11],globalparameters_flatten_err[11],nwalkers*20)
        globalparameters_flat_11=np.concatenate(([globalparameters_flatten[11]],
                                                 globalparameters_flat_11[np.all((globalparameters_flat_11>=-0.4,globalparameters_flat_11<=0.4),axis=0)][0:nwalkers-1]))
        
        #effective_radius_illumination
        globalparameters_flat_12=np.random.normal(globalparameters_flatten[12],globalparameters_flatten_err[12],nwalkers*20)
        globalparameters_flat_12=np.concatenate(([globalparameters_flatten[12]],
                                                 globalparameters_flat_12[np.all((globalparameters_flat_12>=0.7,globalparameters_flat_12<=1.0),axis=0)][0:nwalkers-1]))
        
        if globalparameters_flatten[13]<0.01:
            globalparameters_flatten[13]=0.01
        # frd_sigma
        globalparameters_flat_13=np.random.normal(globalparameters_flatten[13],globalparameters_flatten_err[13],nwalkers*20)
        globalparameters_flat_13=np.concatenate(([globalparameters_flatten[13]],
                                                 globalparameters_flat_13[np.all((globalparameters_flat_13>=0.01,globalparameters_flat_13<=0.4),axis=0)][0:nwalkers-1]))
        
        # frd_lorentz_factor
        globalparameters_flat_14=np.random.normal(globalparameters_flatten[14],globalparameters_flatten_err[14],nwalkers*20)
        globalparameters_flat_14=np.concatenate(([globalparameters_flatten[14]],
                                                 globalparameters_flat_14[np.all((globalparameters_flat_14>=0.01,globalparameters_flat_14<=1),axis=0)][0:nwalkers-1]))       
        
        # det_vert
        globalparameters_flat_15=np.random.normal(globalparameters_flatten[15],globalparameters_flatten_err[15],nwalkers*20)
        globalparameters_flat_15=np.concatenate(([globalparameters_flatten[15]],
                                                 globalparameters_flat_15[np.all((globalparameters_flat_15>=0.85,globalparameters_flat_15<=1.15),axis=0)][0:nwalkers-1]))
        
        #slitHolder_frac_dx
        globalparameters_flat_16=np.random.normal(globalparameters_flatten[16],globalparameters_flatten_err[16],nwalkers*20)
        globalparameters_flat_16=np.concatenate(([globalparameters_flatten[16]],
                                                 globalparameters_flat_16[np.all((globalparameters_flat_16>=-0.8,globalparameters_flat_16<=0.8),axis=0)][0:nwalkers-1]))

        # grating lines
        globalparameters_flat_17=np.random.normal(globalparameters_flatten[17],globalparameters_flatten_err[17],nwalkers*20)
        globalparameters_flat_17=np.concatenate(([globalparameters_flatten[17]],
                                                 globalparameters_flat_17[np.all((globalparameters_flat_17>=1200,globalparameters_flat_17<=120000),axis=0)][0:nwalkers-1]))

        # scattering_slope
        globalparameters_flat_18=np.random.normal(globalparameters_flatten[18],globalparameters_flatten_err[18],nwalkers*20)
        globalparameters_flat_18=np.concatenate(([globalparameters_flatten[18]],
                                                 globalparameters_flat_18[np.all((globalparameters_flat_18>=1.5,globalparameters_flat_18<=3.0),axis=0)][0:nwalkers-1]))
        # scattering_amplitude
        globalparameters_flat_19=np.random.normal(globalparameters_flatten[19],globalparameters_flatten_err[19],nwalkers*20)
        globalparameters_flat_19=np.concatenate(([globalparameters_flatten[19]],
                                                 globalparameters_flat_19[np.all((globalparameters_flat_19>=0.0,globalparameters_flat_19<=0.4),axis=0)][0:nwalkers-1]))
        # pixel_effect
        globalparameters_flat_20=np.random.normal(globalparameters_flatten[20],globalparameters_flatten_err[20],nwalkers*20)
        globalparameters_flat_20=np.concatenate(([globalparameters_flatten[20]],
                                                 globalparameters_flat_20[np.all((globalparameters_flat_20>=0.15,globalparameters_flat_20<=0.8),axis=0)][0:nwalkers-1]))
        
        # fiber_r
        if globalparameters_flatten[21]<1.74:
            globalparameters_flatten[21]=1.8
        
        globalparameters_flat_21=np.random.normal(globalparameters_flatten[21],globalparameters_flatten_err[21],nwalkers*20)
        globalparameters_flat_21=np.concatenate(([globalparameters_flatten[21]],
                                                 globalparameters_flat_21[np.all((globalparameters_flat_21>=1.74,globalparameters_flat_21<=1.98),axis=0)][0:nwalkers-1]))
        
        if len(globalparameters_flatten)==23:
            # flux
            globalparameters_flat_22=np.random.normal(globalparameters_flatten[22],globalparameters_flatten_err[22],nwalkers*20)
            globalparameters_flat_22=np.concatenate(([globalparameters_flatten[22]],
                                                     globalparameters_flat_22[np.all((globalparameters_flat_22>=0.98,globalparameters_flat_22<=1.02),axis=0)][0:nwalkers-1]))
        else:
            pass



        # uncomment in order to troubleshoot and show many parameters generated for each parameter
        """
        for i in [globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3,
                                                   globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_10,
                                                   globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13,
                                                   globalparameters_flat_14,globalparameters_flat_15,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,globalparameters_flat_22]:
            print(str(i[0])+': '+str(len(i)))
        """        
        if pupil_parameters is None:
            if len(globalparameters_flatten)==23:
                #print('considering globalparameters_flatten 23 ')
                #print(globalparameters_flat_0.shape)
                #print(globalparameters_flat_3.shape)
                #print(globalparameters_flat_6.shape)
                #print(globalparameters_flat_9.shape)                
                #print(globalparameters_flat_12.shape)
                #print(globalparameters_flat_15.shape)
                #print(globalparameters_flat_18.shape)
                #print(globalparameters_flat_21.shape)     
                #print(globalparameters_flat_22.shape)
                globalparameters_flat=np.column_stack((globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3,
                                                   globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_10,
                                                   globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13,
                                                   globalparameters_flat_14,globalparameters_flat_15,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,globalparameters_flat_22))
            else:
                print('not considering globalparameters_flatten 23 !!! ')
                globalparameters_flat=np.column_stack((globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3,
                                                   globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_10,
                                                   globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13,
                                                   globalparameters_flat_14,globalparameters_flat_15,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21))                
            
        else:
                        globalparameters_flat=np.column_stack((globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,globalparameters_flat_22))
                        
    except NameError:
        print("NameError")

    
    #print('globalparameters_flat.shape'+str(zparameters_flat.shape) )
    #print('globalparameters_flat.shape'+str(globalparameters_flat.shape) )

    if zmax<=22:    
        allparameters=np.column_stack((zparameters_flat,globalparameters_flat))
    if zmax>22:  
        #print('globalparameters_flat.shape'+str(zparameters_extra_flat.shape) )
        allparameters=np.column_stack((zparameters_flat,globalparameters_flat,zparameters_extra_flat))    
      
    parInit=allparameters.reshape(nwalkers,number_of_par)   
    
    # hm..... relic of some older code, needs cleaning
    if use_optPSF is not None:
        if zmax==11:
            for i in range(1,25):
            #for i in np.concatenate((range(1,7),range(8,25))):
            #for i in range(8,25):
                parInit[:,i]=np.full(len(parInit[:,i]),allparameters_proposal[i])
        else:
            for i in range(1,25+11):
            #for i in np.concatenate((range(1,7),range(8,25))):
            #for i in range(8,25):
                parInit[:,i]=np.full(len(parInit[:,i]),allparameters_proposal[i])          
    else:
        pass
    
    return parInit

def Ifun16Ne (lambdaV,lambda0,Ne):
    """Construct Lorentizan scattering
        @param lambdaV      
        @param lambda0  
        @param Ne                                 number of effective lines
        @returns              
    """
    
    
    return (lambda0/(Ne*np.pi*np.sqrt(2)))**2/((lambdaV-lambda0)**2+(lambda0/(Ne*np.pi*np.sqrt(2)))**2)



def custom_fftconvolve(array1, array2):
    assert array1.shape==array2.shape
    
    
    fft_result=np.fft.fftshift(np.real(np.fft.irfft2(np.fft.rfft2(array1)*np.fft.rfft2(array2),s=np.array(array1.shape))))
    # ensure that the resulting shape is an odd nubmer, needed for fft convolutions later
    if array1.shape[0] % 2 ==0:
        # if the size of an array is even number
        fft_result=fft_result[:fft_result.shape[0]-1,:fft_result.shape[1]-1]  
    else:
        # if the size of an array is an odd number
        fft_result=fft_result[:fft_result.shape[0]-2,:fft_result.shape[1]-2]    
        fft_result=np.pad(fft_result,1,'constant',constant_values=0)
    
    return fft_result


# Taken from galsim.Aperature() code

    # Some quick notes for Josh:
    # - Relation between real-space grid with size theta and pitch dtheta (dimensions of angle)
    #   and corresponding (fast) Fourier grid with size 2*maxk and pitch stepk (dimensions of
    #   inverse angle):
    #     stepk = 2*pi/theta
    #     maxk = pi/dtheta
    # - Relation between aperture of size L and pitch dL (dimensions of length, not angle!) and
    #   (fast) Fourier grid:
    #     dL = stepk * lambda / (2 * pi)
    #     L = maxk * lambda / pi
    # - Implies relation between aperture grid and real-space grid:
    #     dL = lambda/theta
    #     L = lambda/dtheta
def stepK(pupil_plane_scale, lam, scale_unit=galsim.arcsec):
    """Return the Fourier grid spacing for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
    @returns           Fourier grid spacing.
    """
    return 2*np.pi*pupil_plane_scale/(lam*1e-9) * scale_unit/galsim.radians

def maxK(pupil_plane_size,lam, scale_unit=galsim.arcsec):
    """Return the Fourier grid half-size for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
    @returns           Fourier grid half-size.
    """
    return np.pi*pupil_plane_size/(lam*1e-9) * scale_unit/galsim.radians

def sky_scale( pupil_plane_size,lam, scale_unit=galsim.arcsec):
    """Return the image scale for this aperture at given wavelength.
    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image scale.
    """
    return (lam*1e-9) / pupil_plane_size * galsim.radians/scale_unit

def sky_size(pupil_plane_scale, lam, scale_unit=galsim.arcsec):
    """Return the image size for this aperture at given wavelength.
    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image size.
    """
    return (lam*1e-9) / pupil_plane_scale * galsim.radians/scale_unit



def remove_pupil_parameters_from_all_parameters(parameters):
    lenpar=len(parameters)
    return np.concatenate((parameters[:lenpar-23],parameters[lenpar-17:lenpar-13],parameters[lenpar-7:]))

def add_pupil_parameters_to_all_parameters(parameters,pupil_parameters):
    lenpar=len(parameters)
    return np.concatenate((parameters[:lenpar-11],pupil_parameters[:6],parameters[lenpar-11:lenpar-7],pupil_parameters[6:],parameters[lenpar-7:]),axis=0)





# taken https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
# also here https://github.com/scikit-image/scikit-image/issues/2827

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def _reflect_breaks(size: int) -> np.ndarray:
  """Calculate cell boundaries with reflecting boundary conditions."""
  result = np.concatenate([[0], 0.5 + np.arange(size - 1), [size - 1]])
  assert len(result) == size + 1
  return result


def _interval_overlap(first_breaks: np.ndarray,
                      second_breaks: np.ndarray) -> np.ndarray:
  """Return the overlap distance between all pairs of intervals.

  Args:
    first_breaks: breaks between entries in the first set of intervals, with
      shape (N+1,). Must be a non-decreasing sequence.
    second_breaks: breaks between entries in the second set of intervals, with
      shape (M+1,). Must be a non-decreasing sequence.

  Returns:
    Array with shape (N, M) giving the size of the overlapping region between
    each pair of intervals.
  """
  first_upper = first_breaks[1:]
  second_upper = second_breaks[1:]
  upper = np.minimum(first_upper[:, np.newaxis], second_upper[np.newaxis, :])

  first_lower = first_breaks[:-1]
  second_lower = second_breaks[:-1]
  lower = np.maximum(first_lower[:, np.newaxis], second_lower[np.newaxis, :])

  return np.maximum(upper - lower, 0)

#@lru_cache()
def _resize_weights(
    old_size: int, new_size: int, reflect: bool = False) -> np.ndarray:
  """Create a weight matrix for resizing with the local mean along an axis.

  Args:
    old_size: old size.
    new_size: new size.
    reflect: whether or not there are reflecting boundary conditions.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if not reflect:
    old_breaks = np.linspace(0, old_size, num=old_size + 1,dtype=np.float32)
    new_breaks = np.linspace(0, old_size, num=new_size + 1,dtype=np.float32)
  else:
    old_breaks = _reflect_breaks(old_size)
    new_breaks = (old_size - 1) / (new_size - 1) * _reflect_breaks(new_size)

  weights = _interval_overlap(new_breaks, old_breaks)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (new_size, old_size)
  return weights


def resize(array: np.ndarray,
           shape: Tuple[int, ...],
           reflect_axes: Iterable[int] = ()) -> np.ndarray:

  """Resize an array with the local mean / bilinear scaling.

  Works for both upsampling and downsampling in a fashion equivalent to
  block_mean and zoom, but allows for resizing by non-integer multiples. Prefer
  block_mean and zoom when possible, as this implementation is probably slower.

  Args:
    array: array to resize.
    shape: shape of the resized array.
    reflect_axes: iterable of axis numbers with reflecting boundary conditions,
      mirrored over the center of the first and last cell.

  Returns:
    Array resized to shape.

  Raises:
    ValueError: if any values in reflect_axes fall outside the interval
      [-array.ndim, array.ndim).
  """
  reflect_axes_set = set()
  for axis in reflect_axes:
    if not -array.ndim <= axis < array.ndim:
      raise ValueError('invalid axis: {}'.format(axis))
    reflect_axes_set.add(axis % array.ndim)

  output = array
  for axis, (old_size, new_size) in enumerate(zip(array.shape, shape)):
    reflect = axis in reflect_axes_set
    weights = _resize_weights(old_size, new_size, reflect=reflect)

    product = np.tensordot(output, weights, [[axis], [-1]])
    output = np.moveaxis(product, -1, axis)
  return output

def check_global_parameters(globalparameters,test_print=None,fit_for_flux=None):
    #When running big fits these are limits which ensure that the code does not wander off in totally non physical region


    globalparameters_output=np.copy(globalparameters)
    
    # hsc frac
    if globalparameters[0]<=0.6 or globalparameters[0]>=0.8:
        print('globalparameters[0] outside limits; value: '+str(globalparameters[0])) if test_print == 1 else False 
    if globalparameters[0]<=0.6:
        globalparameters_output[0]=0.6
    if globalparameters[0]>0.8:
        globalparameters_output[0]=0.8

     #strut frac
    if globalparameters[1]<0.07 or globalparameters[1]>0.13:
        print('globalparameters[1] outside limits') if test_print == 1 else False 
    if globalparameters[1]<=0.07:
        globalparameters_output[1]=0.07
    if globalparameters[1]>0.13:
        globalparameters_output[1]=0.13

    #slit_frac < strut frac 
    #if globalparameters[4]<globalparameters[1]:
        #print('globalparameters[1] not smaller than 4 outside limits')
        #return -np.inf

     #dx Focal
    if globalparameters[2]<-0.4 or globalparameters[2]>0.4:
        print('globalparameters[2] outside limits') if test_print == 1 else False 
    if globalparameters[2]<-0.4:
        globalparameters_output[2]=-0.4
    if globalparameters[2]>0.4:
        globalparameters_output[2]=0.4

    # dy Focal
    if globalparameters[3]>0.4:
        print('globalparameters[3] outside limits') if test_print == 1 else False 
        globalparameters_output[3]=0.4
    if globalparameters[3]<-0.4:
        print('globalparameters[3] outside limits') if test_print == 1 else False 
        globalparameters_output[3]=-0.4

    # slitFrac
    if globalparameters[4]<0.05:
        print('globalparameters[4] outside limits') if test_print == 1 else False 
        globalparameters_output[4]=0.05
    if globalparameters[4]>0.09:
        print('globalparameters[4] outside limits') if test_print == 1 else False 
        globalparameters_output[4]=0.09

    # slitFrac_dy
    if globalparameters[5]<-0.5:
        print('globalparameters[5] outside limits') if test_print == 1 else False 
        globalparameters_output[5]=-0.5
    if globalparameters[5]>0.5:
        print('globalparameters[5] outside limits') if test_print == 1 else False 
        globalparameters_output[5]=+0.5

    # radiometricEffect / wide_0
    if globalparameters[6]<0:
        print('globalparameters[6] outside limits') if test_print == 1 else False 
        globalparameters_output[6]=0
    if globalparameters[6]>1:
        print('globalparameters[6] outside limits') if test_print == 1 else False 
        globalparameters_output[6]=1

    # radiometricExponent / wide_23
    if globalparameters[7]<0:
        print('globalparameters[7] outside limits') if test_print == 1 else False 
        globalparameters_output[7]=0
    # changed in v0.42
    if globalparameters[7]>1:
        print('globalparameters[7] outside limits') if test_print == 1 else False 
        globalparameters_output[7]=1

    # x_ilum /wide_43
    if globalparameters[8]<0:
        print('globalparameters[8] outside limits') if test_print == 1 else False 
        globalparameters_output[8]=0
    # changed in v0.42
    if globalparameters[8]>1:
        print('globalparameters[8] outside limits') if test_print == 1 else False 
        globalparameters_output[8]=1

    # y_ilum / misalign
    if globalparameters[9]<0:
        print('globalparameters[9] outside limits') if test_print == 1 else False 
        globalparameters_output[9]=0
    if globalparameters[9]>12:
        print('globalparameters[9] outside limits') if test_print == 1 else False 
        globalparameters_output[9]=12

    # x_fiber
    if globalparameters[10]<-0.4:
        print('globalparameters[10] outside limits') if test_print == 1 else False 
        globalparameters_output[10]=-0.4
    if globalparameters[10]>0.4:
        print('globalparameters[10] outside limits') if test_print == 1 else False 
        globalparameters_output[10]=0.4

    # y_fiber
    if globalparameters[11]<-0.4:
        print('globalparameters[11] outside limits') if test_print == 1 else False 
        globalparameters_output[11]=-0.4
    if globalparameters[11]>0.4:
        print('globalparameters[11] outside limits') if test_print == 1 else False 
        globalparameters_output[11]=0.4      

    # effective_radius_illumination
    if globalparameters[12]<0.7:
        print('globalparameters[12] outside limits') if test_print == 1 else False 
        globalparameters_output[12]=0.7
    if globalparameters[12]>1.0:
        print('globalparameters[12] outside limits') if test_print == 1 else False 
        globalparameters_output[12]=1

    # frd_sigma
    if globalparameters[13]<0.01:
        print('globalparameters[13] outside limits') if test_print == 1 else False 
        globalparameters_output[13]=0.01
    if globalparameters[13]>.4:
        print('globalparameters[13] outside limits') if test_print == 1 else False 
        globalparameters_output[13]=0.4 

    #frd_lorentz_factor
    if globalparameters[14]<0.01:
        print('globalparameters[14] outside limits') if test_print == 1 else False 
        globalparameters_output[14]=0.01
    if globalparameters[14]>1:
        print('globalparameters[14] outside limits') if test_print == 1 else False 
        globalparameters_output[14]=1 

    # det_vert
    if globalparameters[15]<0.85:
        print('globalparameters[15] outside limits') if test_print == 1 else False 
        globalparameters_output[15]=0.85
    if globalparameters[15]>1.15:
        print('globalparameters[15] outside limits') if test_print == 1 else False 
        globalparameters_output[15]=1.15

    # slitHolder_frac_dx
    if globalparameters[16]<-0.8:
        print('globalparameters[16] outside limits') if test_print == 1 else False 
        globalparameters_output[16]=-0.8
    if globalparameters[16]>0.8:
        print('globalparameters[16] outside limits') if test_print == 1 else False 
        globalparameters_output[16]=0.8 

    # grating_lines
    if globalparameters[17]<1200:
        print('globalparameters[17] outside limits') if test_print == 1 else False 
        globalparameters_output[17]=1200
    if globalparameters[17]>120000:
        print('globalparameters[17] outside limits') if test_print == 1 else False 
        globalparameters_output[17]=120000 

    # scattering_slope
    if globalparameters[18]<1.5:
        print('globalparameters[18] outside limits') if test_print == 1 else False 
        globalparameters_output[18]=1.5
    if globalparameters[18]>+3.0:
        print('globalparameters[18] outside limits') if test_print == 1 else False 
        globalparameters_output[18]=3 

    # scattering_amplitude
    if globalparameters[19]<0:
        print('globalparameters[19] outside limits') if test_print == 1 else False 
        globalparameters_output[19]=0
    if globalparameters[19]>+0.4:
        print('globalparameters[19] outside limits') if test_print == 1 else False 
        globalparameters_output[19]=0.4          

    # pixel_effect
    if globalparameters[20]<0.15:
        print('globalparameters[20] outside limits') if test_print == 1 else False 
        globalparameters_output[20]=0.15
    if globalparameters[20]>+0.8:
        print('globalparameters[20] outside limits') if test_print == 1 else False 
        globalparameters_output[20]=0.8

    # fiber_r
    if globalparameters[21]<1.74:
        print('globalparameters[21] outside limits') if test_print == 1 else False 
        globalparameters_output[21]=1.74
    if globalparameters[21]>+1.98:
        print('globalparameters[21] outside limits') if test_print == 1 else False 
        globalparameters_output[21] =1.98

    # flux
    if fit_for_flux==True:
        globalparameters_output[22]=1
    else:          
        if globalparameters[22]<0.98:
            print('globalparameters[22] outside limits') if test_print == 1 else False 
            globalparameters_output[22] =0.98
        if globalparameters[22]>1.02:
            print('globalparameters[22] outside limits') if test_print == 1 else False 
            globalparameters_output[22] =1.02

                
    return globalparameters_output

def move_parametrizations_from_2d_shape_to_1d_shape(allparameters_best_parametrization_shape_2d):
    """ 
    change the linear parametrization array in 2d shape to parametrization array in 1d
    
    @param allparameters_best_parametrization_shape_2d        linear parametrization, 2d array
    
    """    
    
    
    if allparameters_best_parametrization_shape_2d.shape[0]>42:
        #  if you are using above Zernike above 22
        #print('we are creating new result with Zernike above 22')
        allparameters_best_parametrization_shape_1d=np.concatenate((allparameters_best_parametrization_shape_2d[:19].ravel(),
                                                    allparameters_best_parametrization_shape_2d[19:19+23][:,1],\
                                                        allparameters_best_parametrization_shape_2d[19+23:].ravel()))
        
    else:
        #print('we are creating new result with Zernike at 22')
        allparameters_best_parametrization_shape_1d=np.concatenate((allparameters_best_parametrization_shape_2d[:19].ravel(),
                                                    allparameters_best_parametrization_shape_2d[19:-1][:,1]))    
        
    return allparameters_best_parametrization_shape_1d
