"""
First created on Mon Aug 13 10:01:03 2018

Versions
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

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""
########################################
#standard library imports
from __future__ import absolute_import, division, print_function
import os
import time
import sys
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
#import pyfftw
#import pandas as pd

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
import skimage.transform
import scipy.optimize as optimize
from scipy.ndimage.filters import gaussian_filter

#lmfit
import lmfit

#matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# needed for resizing routines
from typing import Tuple, Iterable
########################################

__all__ = ['PupilFactory', 'Pupil','ZernikeFitter_PFS','LN_PFS_single','LNP_PFS',\
           'find_centroid_of_flux','create_parInit',\
           'Zernike_Analysis','PFSPupilFactory','custom_fftconvolve','stepK','maxK',\
           'sky_scale','sky_size','create_x','remove_pupil_parameters_from_all_parameters',\
           'resize','_interval_overlap']

__version__ = "0.28b"

############################################################
# name your directory where you want to have files!
PSF_DIRECTORY='/Users/nevencaplar/Documents/PFS/'
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

    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,verbosity=None):
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

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half lenght of the length of square side
        @param[in] angle      angle that the camera is rotated
        """
        pupil_illuminated_only1=np.ones_like(pupil.illuminated)
        
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
        """
        pupil_illuminated_only1=np.ones_like(pupil.illuminated)
        
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

        pupil_illuminated_only0_in_only1=np.zeros((i_y_max-i_y_min,i_x_max-i_x_min))
        
        
        u0=self.u[i_y_min:i_y_max,i_x_min:i_x_max]
        v0=self.v[i_y_min:i_y_max,i_x_min:i_x_max]
            
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

        pupil_illuminated_only0_in_only1[((v0-p21[1])*np.cos(-angleRad21)-(u0-p21[0])*np.sin(-angleRad21)<y22)  ] = True
    
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

        pupil_illuminated_only0_in_only1[ ((v0-p21[1])*np.cos(-angleRad21)-(u0-p21[0])*np.sin(-angleRad21)>y21)] = True
        
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

        pupil_illuminated_only0_in_only1[((u0-p21[0])*np.cos(-angleRad21)+(v0-p21[1])*np.sin(-angleRad21)>x21) ] = True  

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

        pupil_illuminated_only0_in_only1[((u0-p21[0])*np.cos(-angleRad21)+(v0-p21[1])*np.sin(-angleRad21)<x22) ] = True  
        
        
        pupil_illuminated_only1[i_y_min:i_y_max,i_x_min:i_x_max]=pupil_illuminated_only0_in_only1
        
        pupil.illuminated=pupil.illuminated*pupil_illuminated_only1
        time_end_single_square=time.time()
        
        if self.verbosity==1:
            print('Time for cutting out the square is '+str(time_end_single_square-time_start_single_square))    

    
    def _cutRay(self, pupil, p0, angle, thickness,angleunit=None):
        """Cut out a ray from a Pupil.

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
                           (self.v - p0[1])*np.sin(angleRad) >= 0)] = False   

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
    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,slitHolder_frac_dx,verbosity=None):
        """!Construct a PupilFactory.

        @param[in] visitInfo  VisitInfo object for a particular exposure.
        @param[in] pupilSize  Size in meters of constructed Pupils.
        @param[in] npix       Constructed Pupils will be npix x npix.
        """
        
        self.verbosity=verbosity
        if self.verbosity==1:
            print('Entering PFSPupilFactory class')
            
        PupilFactory.__init__(self, pupilSize,npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy,x_fiber,y_fiber,effective_ilum_radius,frd_sigma,frd_lorentz_factor,det_vert,verbosity=self.verbosity)

        self.x_fiber=x_fiber
        self.y_fiber=y_fiber      
        self.slitHolder_frac_dx=slitHolder_frac_dx
        self._spiderStartPos=[np.array([ 0.,  0.]), np.array([ 0.,  0.]), np.array([ 0.,  0.])]
        self._spiderAngles=[1.57-1.57,3.66519-1.57,5.75959-1.57]
        self.effective_ilum_radius=effective_ilum_radius

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
            print('Entering getPupil')
            
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
        single_element=np.linspace(-1,1,len(pupil.illuminated), endpoint=True)
        u_manual=np.tile(single_element,(len(single_element),1))
        v_manual=np.transpose(u_manual)  
        center_distance=np.sqrt((u_manual-self.x_fiber*hscRate*hscPlateScale*12)**2+(v_manual-self.y_fiber*hscRate*hscPlateScale*12)**2)
        frd_sigma=self.frd_sigma
        sigma=2*frd_sigma
        pupil_frd=(1/2*(scipy.special.erf((-center_distance+self.effective_ilum_radius)/sigma)+scipy.special.erf((center_distance+self.effective_ilum_radius)/sigma)))
        pupil_lorentz=(np.arctan(2*(self.effective_ilum_radius-center_distance)/(4*sigma))+np.arctan(2*(self.effective_ilum_radius+center_distance)/(4*sigma)))/(2*np.arctan((2*self.effective_ilum_radius)/(4*sigma)))

        pupil.illuminated= (pupil_frd+1*self.frd_lorentz_factor*pupil_lorentz)/(1+self.frd_lorentz_factor)
        
        # Cout out the acceptance angle of the camera
        self._cutCircleExterior(pupil, (0.0, 0.0), subaruRadius)        
         
        # Cut out detector shadow
        self._cutSquare(pupil, (camX, camY), hscRadius,self.input_angle,self.det_vert)       
        
        #No vignetting of this kind for the spectroscopic camera
        #self._cutCircleExterior(pupil, (lensX, lensY), lensRadius)
        
        # Cut out spider shadow
        for pos, angle in zip(self._spiderStartPos, self._spiderAngles):
            x = pos[0] + camX
            y = pos[1] + camY
            self._cutRay(pupil, (x, y), angle, subaruStrutThick,'rad')
            
        # cut out slit shadow
        self._cutRay(pupil, (2,slitFrac_dy/18),-np.pi,subaruSlit,'rad') 
        
        # cut out slit holder shadow
        #also subaruSlit/3 not fitted, just put roughly correct number
        self._cutRay(pupil, (self.slitHolder_frac_dx/18,1),-np.pi/2,subaruSlit/3,'rad')   
        
        if self.verbosity==1:
            print('Finished with getPupil')
        
        return pupil

class ZernikeFitter_PFS(object):
    
    """!Class to create  donut images in PFS
    The model is constructed using FFT, and consists of the convolution of
    an OpticalPSF and an input fiber image.  The OpticalPSF part includes the
    specification of an arbitrary number of zernike wavefront aberrations. 
    
    This code uses lmfit to initalize the parameters.
    """

    def __init__(self,image=None,image_var=None,image_mask=None,pixelScale=None,wavelength=None,
                 jacobian=None,diam_sic=None,npix=None,pupilExplicit=None,
                 wf_full_Image=None,radiometricEffectArray_Image=None,
                 ilum_Image=None,dithering=None,save=None,
                 pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,
                 zmaxInit=None,extraZernike=None,simulation_00=None,verbosity=None,
                 double_sources=None,double_sources_positions_ratios=None,*args):
        
        """
        @param image        image to analyze
        @param image_var    variance image
        @param pixelScale   pixel scale in arcseconds 
        @param wavelength   wavelenght 
        @param jacobian     An optional 2x2 Jacobian distortion matrix to apply
                            to the forward model.  Note that this is relative to
                            the pixelScale above.  Default is the identity matrix.
        @param diam_sic     diameter of the exit pupil 
        @param npix         number of pixels describing the pupil
        @param pupilExplicit if you want to pass explicit image of the exit pupil instead of creating one within the class
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
            
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float32)
        else:
            self.jacobian = jacobian
        
        # if you do not pass the value for wavelength it will default to 794 nm, which is roughly in the middle of the red detector
        if wavelength is None:
            wavelength=794
            self.wavelength=wavelength
        else:
            self.wavelength=wavelength       
        
        # This is size of the pixel in arcsec for PFS red arm in focus
        # calculated with http://www.wilmslowastro.com/software/formulae.htm
        # pixel size in microns/focal length in mm x 206.3
        # pixel size =15 microns, focal length = 149.2 mm (138 aperature x1.1 f number)
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
            npix=1024
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
                
        # when creating pupils it will have size of npix pixels
        if dithering is None:
            dithering=1
            self.dithering=dithering
            self.pixelScale=self.pixelScale/dithering
        else:
            self.dithering=dithering         
            self.pixelScale=self.pixelScale/dithering         
            
        
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
            
        self.zmax=zmaxInit
        
        self.simulation_00=simulation_00

        self.extraZernike=extraZernike
        
        self.verbosity=verbosity
        
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios

    
    def initParams(self,z4Init=None, dxInit=None,dyInit=None,hscFracInit=None,strutFracInit=None,
                   focalPlanePositionInit=None,fiber_rInit=None,
                  slitFracInit=None,slitFrac_dy_Init=None,apodizationInit=None,radiometricEffectInit=None,
                   trace_valueInit=None,serial_trace_valueInit=None,pixel_effectInit=None,backgroundInit=None,
                   x_ilumInit=None,y_ilumInit=None,radiometricExponentInit=None,
                   x_fiberInit=None,y_fiberInit=None,effective_ilum_radiusInit=None,
                   grating_linesInit=None,scattering_radiusInit=None,scattering_slopeInit=None,scattering_amplitudeInit=None,fluxInit=None,frd_sigmaInit=None,frd_lorentz_factorInit=None,
                   det_vertInit=None,slitHolder_frac_dxInit=None):
        """Initialize lmfit Parameters object.
        
        @param zmax                      Total number of Zernike aberrations used
        @param z4Init                    Initial Z4 aberration value in waves (that is 2*np.pi*wavelengths)
        
        # pupil parameters
        @param hscFracInit               Value determining how much of the exit pupil obscured by the central obscuration(detector) 
        @param strutFracInit             Value determining how much of the exit pupil is obscured by a single strut
        @param focalPlanePositionInit    2-tuple for position of the central obscuration(detector) in the focal plane
        @param slitFracInit              Value determining how much of the exit pupil is obscured by slit
        @param slitFrac_dy_Init          Value determining what is the vertical position of the slit in the exit pupil
        
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

        if slitHolder_frac_dxInit is None:
            params.add('slitHolder_frac_dx', 0)
        else:
            params.add('slitHolder_frac_dx', slitHolder_frac_dxInit)             
            

        self.params = params
        self.optPsf=None
        self.z_array=z_array


    def constructModelImage_PFS_naturalResolution(self,params=None,shape=None,pixelScale=None,jacobian=None,use_optPSF=None,extraZernike=None,hashvalue=None):
        """Construct model image from parameters
        @param params      lmfit.Parameters object or python dictionary with
                           param values to use, or None to use self.params
        @param shape       (nx, ny) shape for model image, or None to use
                           the shape of self.maskedImage
        @param pixelScale  pixel scale in arcseconds to use for model image,
                           or None to use self.pixelScale.

        @returns           numpy array image with the same flux as the input image
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
            
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float32)
        else:
            jacobian = self.jacobian          
            
        try:
            v = params.valuesdict()
        except AttributeError:
            v = params
            
        if hashvalue is None:
            hashvalue = 0
            
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
            optPsf=self._getOptPsf_naturalResolution(v)
        else:
            #if first iteration still generate image
            if self.optPsf is None:
                optPsf=self._getOptPsf_naturalResolution(v)
                self.optPsf=optPsf
            else:
                optPsf=self.optPsf
                
        optPsf_cut_fiber_convolved_downsampled=self.optPsf_postprocessing(optPsf)

        if self.save==1:

            if socket.gethostname()=='IapetusUSA':
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf',optPsf)
                np.save(TESTING_FINAL_IMAGES_FOLDER+'optPsf_cut_fiber_convolved_downsampled',optPsf_cut_fiber_convolved_downsampled) 
            else:                 
                pass    

        if self.verbosity==1:
            print('Finished with constructModelImage_PFS_naturalResolution')
            print(' ')

        return optPsf_cut_fiber_convolved_downsampled
    
    def optPsf_postprocessing(self,optPsf):

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
        oversampling_original=(self.pixelScale)/self.scale_ModelImage_PFS_naturalResolution
        
        if self.verbosity==1:
            print('optPsf.shape: '+str(optPsf.shape))
            print('oversampling_original: ' +str(oversampling_original))

        
        # determine the size, so that from the hughe generated large image we cut the central portion (1.4 times larger than the size of actual image)
        size_of_central_cut=int(oversampling_original*self.image.shape[0]*1.4)
        assert size_of_central_cut<optPsf.shape[0]
        if self.verbosity==1:
            print('size_of_central_cut: '+str(size_of_central_cut))
            
        # cut part which you need image 
        # set oversampling to 1 so you are not resizing and dx=0 and dy=0 so that you are not moving around, i.e., you are cutting the central region
        optPsf_cut=Psf_position.cut_Centroid_of_natural_resolution_image(image=optPsf,size_natural_resolution=size_of_central_cut+1,oversampling=1,dx=0,dy=0)
        if self.verbosity==1:
            print('optPsf_cut.shape'+str(optPsf_cut.shape))

        # we want to reduce oversampling by roughly the factor of 4 to make things easier
        oversampling=np.round(oversampling_original/4)
        if self.verbosity==1:
            print('oversampling:' +str(oversampling))
        
        # what is the size of the images after you resize to the new oversampled ratio
        size_of_optPsf_cut_downsampled=np.round(size_of_central_cut/(oversampling_original/oversampling))
        if self.verbosity==1:
            print('optPsf_cut.shape[0]'+str(optPsf_cut.shape[0]))
            print('size_of_optPsf_cut_downsampled: '+str(size_of_optPsf_cut_downsampled))
                    
        # make sure that optPsf_cut_downsampled is an array which has an odd size - increase size by 1 if needed
        if (size_of_optPsf_cut_downsampled % 2) == 0:
            optPsf_cut_downsampled=skimage.transform.resize(optPsf_cut,(size_of_optPsf_cut_downsampled+1,size_of_optPsf_cut_downsampled+1),mode='constant')
        else:
            optPsf_cut_downsampled=skimage.transform.resize(optPsf_cut,(size_of_optPsf_cut_downsampled,size_of_optPsf_cut_downsampled),mode='constant')
        
        if self.verbosity==1:        
            print('optPsf_cut_downsampled.shape: '+str(optPsf_cut_downsampled.shape))
        
        # gives middle point of the image to used for calculations of scattered light 
        mid_point_of_optPsf_cut_downsampled=int(optPsf_cut_downsampled.shape[0]/2)
        
        # gives the size of one pixel in optPsf_downsampled in microns
        # one pixel is 15 microns
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
        pointsx = np.linspace(-(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,num=optPsf_cut_downsampled.shape[0])
        pointsy =np.linspace(-(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,(size_of_optPsf_cut_in_Microns-size_of_pixels_in_optPsf_cut_downsampled)/2,num=optPsf_cut_downsampled.shape[0])
        xs, ys = np.meshgrid(pointsx, pointsy)
        r0 = np.sqrt((xs-0)** 2 + (ys-0)** 2)+.01

        # creating scattere light code
        scattered_light_kernel=(r0**(-v['scattering_slope']))
        
        # the line below from previous code where I terminated scattering radius dependece below certain radius which could have change (changed on Oct 04, 2018)
        # keep for some more time for historic reasons
        # scattered_light_kernel[r0<v['scattering_radius']]=v['scattering_radius']**(-v['scattering_slope'])
        
        scattered_light_kernel[r0<7.5]=7.5**(-v['scattering_slope'])
        scattered_light_kernel[scattered_light_kernel == np.inf] = 0
        scattered_light_kernel=scattered_light_kernel*(v['scattering_amplitude'])/(10*np.max(scattered_light_kernel))
        
        # convolve the psf with the scattered light kernel to create scattered light component
        scattered_light=custom_fftconvolve(optPsf_cut_downsampled,scattered_light_kernel)

        # add back the scattering to the image
        optPsf_cut_downsampled_scattered=optPsf_cut_downsampled+scattered_light        
        
        if self.verbosity==1:
            print('optPsf_cut_downsampled_scattered.shape:' +str(optPsf_cut_downsampled_scattered.shape))

        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
        # 2. convolution with fiber


        # create tophat2d
        fiber = astropy.convolution.Tophat2DKernel(oversampling*1*v['fiber_r']*self.dithering,mode='oversample').array
        # create array with zeros with size of the current image, which we will fill with fiber array in the middle
        fiber_padded=np.zeros_like(optPsf_cut_downsampled_scattered)
        mid_point_of_optPsf_cut_downsampled=int(optPsf_cut_downsampled.shape[0]/2)
        fiber_array_size=fiber.shape[0]
        # fill the zeroes image with fiber here
        fiber_padded[int(mid_point_of_optPsf_cut_downsampled-fiber_array_size/2)+1:int(mid_point_of_optPsf_cut_downsampled+fiber_array_size/2)+1,\
                     int(mid_point_of_optPsf_cut_downsampled-fiber_array_size/2)+1:int(mid_point_of_optPsf_cut_downsampled+fiber_array_size/2)+1]=fiber

        
        # legacy code for is the line below, followed by the currently used code
        #optPsf_fiber_convolved=scipy.signal.fftconvolve(optPsf_downsampled_scattered, fiber, mode = 'same') 
        
        # convolve with fiber 
        optPsf_cut_fiber_convolved=custom_fftconvolve(optPsf_cut_downsampled_scattered,fiber_padded)
         
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########
        # 3. CCD difusion
        
        #pixels are not perfect detectors
        # charge diffusion in our optical CCDs, can be well described with a Gaussian 
        # sigma is around 7 microns (Jim Gunn - private communication). This is controled in our code by @param 'pixel_effect'
        pixel_gauss=Gaussian2DKernel(oversampling*v['pixel_effect']*self.dithering).array
        pixel_gauss_padded=np.pad(pixel_gauss,int((len(optPsf_cut_fiber_convolved)-len(pixel_gauss))/2),'constant',constant_values=0)
        optPsf_cut_pixel_response_convolved=custom_fftconvolve(optPsf_cut_fiber_convolved, pixel_gauss_padded)
      
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########        
        # 4. grating effects
        
        # following grating calculation is done
        # assuming that 15 microns covers wavelength range of 0.07907 nm
        #(assuming that 4300 pixels in real detector uniformly covers 340 nm)
        grating_kernel=np.ones((optPsf_cut_pixel_response_convolved.shape[0],1))
        for i in range(len(grating_kernel)):
            grating_kernel[i]=Ifun16Ne((i-int(optPsf_cut_pixel_response_convolved.shape[0]/2))*0.07907*10**-9/(self.dithering*oversampling)+self.wavelength*10**-9,self.wavelength*10**-9,v['grating_lines'])
        grating_kernel=grating_kernel/np.sum(grating_kernel)
        
        # I should implement custom_fft function (custom_fftconvolve), as above
        # This is 1D convolution so it would need a bit of work, and I see that behavioru is fine
        optPsf_cut_grating_convolved=scipy.signal.fftconvolve(optPsf_cut_pixel_response_convolved, grating_kernel, mode='same')
        
        #########        #########        #########        #########        #########         #########        #########        #########        #########        #########   
        # 5. centering
        # This is the part which creates the final image
        
        # if you have requsted a simulated image without movement, `simulation_00' will not be `None' and go into this small loop below
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
            
        else:
            pass

        
        # the algorithm  finds (or at least should find) the best downsampling combination automatically 
        if self.verbosity==1:
            print('are we invoking double sources (1 if yes): '+str(self.double_sources)) 
            print('double source position/ratio is:' +str(self.double_sources_positions_ratios))
            
        # initialize the class which does the centering - the separation between the class and the main function in the class, ``find_single_realization_min_cut'', is a bit blurry and unsatisfactory
        single_Psf_position=Psf_position(optPsf_cut_grating_convolved, int(round(oversampling)),shape[0] ,double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,
                                                                               verbosity=self.verbosity)
       
        #  run the code for centering
        time_start_single=time.time()
        optPsf_cut_fiber_convolved_downsampled=single_Psf_position.find_single_realization_min_cut(optPsf_cut_grating_convolved,
                                                                               int(round(oversampling)),shape[0],self.image,self.image_var,self.image_mask,
                                                                               v_flux=v['flux'],simulation_00=self.simulation_00,
                                                                               double_sources=self.double_sources,double_sources_positions_ratios=self.double_sources_positions_ratios,
                                                                               verbosity=self.verbosity)
        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for single_Psf_position protocol is '+str(time_end_single-time_start_single))
   
        if self.verbosity==1:
            print('Sucesfully created optPsf_cut_fiber_convolved_downsampled') 
        
        if self.save==1:
            if socket.gethostname()=='IapetusUSA':
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
        
        return optPsf_cut_fiber_convolved_downsampled
    
    
    @lru_cache(maxsize=3)
    def _get_Pupil(self,params):
        
        if self.verbosity==1:
            print(' ')
            print('Entering _get_Pupil')        
        
        diam_sic=self.diam_sic
        npix=self.npix
         
        if self.verbosity==1:
            print('Size of the pupil (npix): '+str(npix))        
        

        Pupil_Image=PFSPupilFactory(diam_sic,npix,
                                np.pi/2,
                              self.pupil_parameters[0],self.pupil_parameters[1],
                              self.pupil_parameters[4],self.pupil_parameters[5],
                              self.pupil_parameters[6],self.pupil_parameters[7],self.pupil_parameters[8],
                                self.pupil_parameters[9],self.pupil_parameters[10],self.pupil_parameters[11],self.pupil_parameters[12],verbosity=self.verbosity)
        point=[self.pupil_parameters[2],self.pupil_parameters[3]]
        pupil=Pupil_Image.getPupil(point)

        if self.save==1:
            if socket.gethostname()=='IapetusUSA':
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'pupil.illuminated',pupil.illuminated.astype(np.float32))
        
        if self.verbosity==1:
            print('Finished with _get_Pupil')    
            print(' ')
        
        return pupil
        
    
    def _getOptPsf_naturalResolution(self,params):
        
        """ !returns optical PSF
        
         @param params       parameters
        """

        if self.verbosity==1:
            print(' ')
            print('Entering _getOptPsf_naturalResolution')       
       
        if self.verbosity==1:
            print('use_pupil_parameters: '+str(self.use_pupil_parameters))
            print('pupil_parameters if you use_pupil_parameters: '+str(self.pupil_parameters))

        # parmeters ``i'' just to precision in the construction of ``pupil_parameters'' array
        i=4    
        if self.use_pupil_parameters is None:
            pupil_parameters=np.array([params['hscFrac'.format(i)],params['strutFrac'.format(i)],
                                    params['dxFocal'.format(i)],params['dyFocal'.format(i)],
                                  params['slitFrac'.format(i)],params['slitFrac_dy'.format(i)],
                                    params['x_fiber'.format(i)],params['y_fiber'.format(i)],params['effective_ilum_radius'.format(i)],
                                    params['frd_sigma'.format(i)],params['frd_lorentz_factor'.format(i)],params['det_vert'.format(i)],params['slitHolder_frac_dx'.format(i)]])
            self.pupil_parameters=pupil_parameters
        else:
            pupil_parameters=np.array(self.pupil_parameters)
            
        diam_sic=self.diam_sic
        
        if self.verbosity==1:
            print(['hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy','slitFrac','slitFrac_dy'])
            print(['x_fiber','y_fiber','effective_ilum_radius','frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx'])
            print('pupil_parameters: '+str(self.pupil_parameters))
            
        time_start_single=time.time()
        pupil=self._get_Pupil(tuple(pupil_parameters))
        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for single pupil calculation is '+str(time_end_single-time_start_single))


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
            
        if self.verbosity==1:    
            print('Supplied pupil size is (pupil.size) [m]: '+str(pupil.size))
            print('One pixel has size of (pupil.scale) [m]: '+str(pupil.scale))
            print('Supplied pupil has so many pixels (pupil_plane_im): '+str(pupil.illuminated.astype(np.int16).shape))
   
        
        # initialize galsim.Aperature class
        # the output will be the size of pupil.illuminated
        
        if self.pupilExplicit is None:
            aper = galsim.Aperture(
                diam = pupil.size,
                pupil_plane_im = pupil.illuminated.astype(np.float32),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None) 
        else:
            if self.verbosity==1: 
                print('Using provided pupil')
            aper = galsim.Aperture(
                diam =  pupil.size,
                pupil_plane_im = self.pupilExplicit.astype(np.float32),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None)         
        
        
        
        
        if self.verbosity==1:   
            print('Starting creation of wavefront')    
        # create wavefront across the exit pupil      
        if self.extraZernike==None:
            pass
        else:
            aberrations_extended=np.concatenate((aberrations,self.extraZernike),axis=0)

        
        if self.verbosity==1:   
            print('diam_sic: '+str(diam_sic))
            print('aberrations: '+str(aberrations))
            print('aberrations 0 '+str(aberrations_0))
            print('aberrations extra: '+str(self.extraZernike))
            print('self.wavelength: '+str(self.wavelength))
        
        if self.extraZernike==None:
            optics_screen = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations,lam_0=self.wavelength)
            if self.save==1:
                # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations_0,lam_0=self.wavelength)
        else:
            optics_screen = galsim.phase_screens.OpticalScreen(diam=diam_sic,aberrations=aberrations_extended,lam_0=self.wavelength)      
            
        screens = galsim.PhaseScreenList(optics_screen)   
        if self.save==1:
            # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
            screens_fake_0 = galsim.PhaseScreenList(optics_screen_fake_0)  
        
        # create array with pixels=1 if the area is illuminated and 0 if it is obscured
        ilum=np.array(aper.illuminated, dtype=np.float32)
        assert np.sum(ilum)>0, str(self.pupil_parameters)
        
        # padding to get exact multiple when we are in focus
        # focus recognized by the fact that ilum is 1024 pixels large
        if len(ilum)==1024:
            ilum_padded=np.zeros((1158,1158))
            ilum_padded[67:67+1024,67:67+1024]=ilum
            ilum=ilum_padded
                   
        # gives size of the illuminated image
        lower_limit_of_ilum=int(ilum.shape[0]/2-pupil.illuminated.shape[0]/2)
        higher_limit_of_ilum=int(ilum.shape[0]/2+pupil.illuminated.shape[0]/2)
        if self.verbosity==1: 
            print('lower_limit_of_ilum: ' +str(lower_limit_of_ilum))
            print('higher_limit_of_ilum: ' +str(higher_limit_of_ilum))
        ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]=ilum[lower_limit_of_ilum:higher_limit_of_ilum,lower_limit_of_ilum:higher_limit_of_ilum]*pupil.illuminated

        if self.verbosity==1: 
            print('Size after padding zeros to 2x size and extra padding to get size suitable for FFT: '+str(ilum.shape))
               
        # maximum extent of pupil image in units of radius of the pupil, needed for next step
        size_of_ilum_in_units_of_radius=ilum.shape[0]/pupil.illuminated.astype(np.int16).shape[0]
        
        if self.verbosity==1:  
            print('size_of_ilum_in_units_of_radius: '+str(size_of_ilum_in_units_of_radius))
        
        # add the change of flux between the entrance and exit pupil
        # end product is radiometricEffectArray
        points = np.linspace(-size_of_ilum_in_units_of_radius, size_of_ilum_in_units_of_radius,num=ilum.shape[0])
        xs, ys = np.meshgrid(points, points)
        r = np.sqrt((xs-params['x_ilum']*params['dxFocal'])** 2 + (ys-params['y_ilum']*params['dyFocal'])** 2)
        
        # change in v_0.14
        radiometricEffectArray=(1+params['radiometricEffect']*r**2)**(-params['radiometricExponent'])
        ilum_radiometric=np.nan_to_num(radiometricEffectArray*ilum,0) 
     
        # this is where you can introduce some apodization in the pupil image by using the line below
        # the apodization sigma is set to that in focus it is at 0.75
        # for larger images, scale according to the size of the input image which is to be FFT-ed
        # 0.75 is an arbitrary number
        apodization_sigma=((len(ilum_radiometric))/1158)**0.875*0.75
        #apodization_sigma=0.75
        r = gaussian_filter(ilum_radiometric, sigma=apodization_sigma)
        
        # put pixels for which amplitude is less than 0.01 to 0
        r_ilum_pre=np.copy(r)
        r_ilum_pre[r>0.01]=1
        r_ilum_pre[r<0.01]=0
        r_ilum=r_ilum_pre.astype(bool)
        
        # manual creation of aper.u and aper.v (mimicking steps which were automatically done in galsim)
        # this gives position informaition about each point in the exit pupil so we can apply wavefront to it
        aperu_manual=[]
        for i in range(len(r_ilum)):
            aperu_manual.append(np.linspace(-diam_sic*(size_of_ilum_in_units_of_radius/2),diam_sic*(size_of_ilum_in_units_of_radius/2),len(r_ilum), endpoint=True))

        # full grid
        u_manual=np.array(aperu_manual)
        v_manual=np.transpose(aperu_manual)     
        
        # select only parts of the grid that are actually illuminated        
        u=u_manual[r_ilum]
        v=v_manual[r_ilum]

        # apply wavefront to the array describing illumination
        wf = screens.wavefront(u, v, None, 0)
        if self.save==1:
            wf_full = screens.wavefront(u_manual, v_manual, None, 0)
        wf_grid = np.zeros_like(r_ilum, dtype=np.float32)
        wf_grid[r_ilum] = (wf/self.wavelength)
        wf_grid_rot=wf_grid
        
        if self.save==1:
            # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
            wf_full_fake_0 = screens_fake_0.wavefront(u_manual, v_manual, None, 0)
        
        
        # exponential of the wavefront
        expwf_grid = np.zeros_like(r_ilum, dtype=np.complex128)
        expwf_grid[r_ilum] = r[r_ilum]*np.exp(2j*np.pi * wf_grid_rot[r_ilum])
        

        ######################################################################
        # Different implementations of the Fourier code

        # legacy code
        # do Fourier via galsim and square it to create image
        # ftexpwf = galsim.fft.fft2(expwf_grid,shift_in=True,shift_out=True)
        
        # uncoment to get timming 
        time_start_single=time.time()
        ftexpwf =np.fft.fftshift(np.fft.fft2(np.fft.fftshift(expwf_grid)))
        img_apod = np.abs(ftexpwf)**2
        time_end_single=time.time()
        if self.verbosity==1:
            print('Time for FFT is '+str(time_end_single-time_start_single))

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
            if socket.gethostname()=='IapetusUSA':
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'aperilluminated',aper.illuminated)  
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'radiometricEffectArray',radiometricEffectArray)     
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum',ilum)   
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'r_ilum',r_ilum)
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'ilum_radiometric',ilum_radiometric) 
                np.save(TESTING_PUPIL_IMAGES_FOLDER+'r_resize',r) 
    
                
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'u_manual',u_manual) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'v_manual',v_manual) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'u',u) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'v',v) 
                
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_grid',wf_grid)  
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full',wf_full) 
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'wf_full_fake_0',wf_full_fake_0)       
                
                np.save(TESTING_WAVEFRONT_IMAGES_FOLDER+'expwf_grid',expwf_grid)   

        if self.verbosity==1:
            print('Finished with _getOptPsf_naturalResolution')  
            print(' ')
        
        return img_apod




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


class LN_PFS_single(object):
        
    def __init__(self,sci_image,var_image,mask_image=None,dithering=None,save=None,verbosity=None,
                 pupil_parameters=None,use_pupil_parameters=None,use_optPSF=None,
                 zmax=None,extraZernike=None,pupilExplicit=None,simulation_00=None,
                 double_sources=None,double_sources_positions_ratios=None,npix=None):    
        """
        @param sci_image           science image 
        @param var_image           variance image
        """        
                
        if verbosity is None:
            verbosity=0
                
        if use_pupil_parameters is not None:
            assert pupil_parameters is not None
            
        if double_sources is not None:      
            assert np.sum(np.abs(double_sources_positions_ratios))>0    

        if zmax is None:
            zmax=11
                      
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
        self.zmax=zmax
        self.extraZernike=extraZernike
        self.verbosity=verbosity
        self.double_sources=double_sources
        self.double_sources_positions_ratios=double_sources_positions_ratios

        # if npix is not specified automatically scale the image
        # this will create images which will have different pupil size for different sizes of science image
        # and this will influence the results
        if npix is None:
            if dithering is None or dithering is 1:
                npix=int(math.ceil(int(1024*sci_image.shape[0]/(20*4)))*2)
            else:
                npix=int(math.ceil(int(1024*sci_image.shape[0]/(20*4*self.dithering)))*2)
        else:
            self.npix=npix
        print('npix_value:'+str(npix))        
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
        
        if pupil_parameters is None:
            single_image_analysis=ZernikeFitter_PFS(sci_image,var_image,image_mask=mask_image,npix=npix,dithering=dithering,save=save,\
                                                    pupil_parameters=pupil_parameters,use_pupil_parameters=use_pupil_parameters,
                                                    use_optPSF=use_optPSF,zmaxInit=zmax,extraZernike=extraZernike,
                                                    pupilExplicit=pupilExplicit,simulation_00=simulation_00,verbosity=verbosity,double_sources=double_sources,double_sources_positions_ratios=double_sources_positions_ratios)  
            single_image_analysis.initParams(zmax)
            self.single_image_analysis=single_image_analysis
        else:

            single_image_analysis=ZernikeFitter_PFS(sci_image,var_image,image_mask=mask_image,npix=npix,dithering=dithering,save=save,\
                                                    pupil_parameters=pupil_parameters,use_pupil_parameters=use_pupil_parameters,
                                                    extraZernike=extraZernike,simulation_00=simulation_00,verbosity=verbosity,double_sources=double_sources,double_sources_positions_ratios=double_sources_positions_ratios)  
            single_image_analysis.initParams(zmax,hscFracInit=pupil_parameters[0],strutFracInit=pupil_parameters[1],
                   focalPlanePositionInit=(pupil_parameters[2],pupil_parameters[3]),slitFracInit=pupil_parameters[4],
                  slitFrac_dy_Init=pupil_parameters[5],x_fiberInit=pupil_parameters[6],y_fiberInit=pupil_parameters[7],
                  effective_ilum_radiusInit=pupil_parameters[8],frd_sigmaInit=pupil_parameters[9],
                  det_vertInit=pupil_parameters[10],slitHolder_frac_dxInit=pupil_parameters[11]) 
            self.single_image_analysis=single_image_analysis

    def create_chi_2_almost(self,modelImg,sci_image,var_image,mask_image):
        """
        return array with 3 values
        1. normal chi**2
        2. what is 'instrinsic' chi**2, i.e., just sum((scientific image)**2/variance)
        3. 'Q' value = sum(abs(model - scientific image))/sum(scientific image)
        
        @param sci_image    model 
        @param sci_image    scientific image 
        @param var_image    variance image
        """ 
        inverted_mask=~mask_image.astype(bool)
        
        var_image_masked=var_image*inverted_mask
        sci_image_masked=sci_image*inverted_mask
        modelImg_masked=modelImg*inverted_mask
        
        sigma = np.sqrt(var_image_masked)
        chi = (sci_image_masked - modelImg_masked)/sigma
        chi2_intrinsic=np.sum(sci_image_masked**2/var_image_masked)
        
        chi_without_nan=[]
        chi_without_nan = chi.ravel()[~np.isnan(chi.ravel())]
        res=(chi_without_nan)**2
        
        # calculates 'Q' values
        Qlist=np.abs((sci_image_masked - modelImg_masked))
        Qlist_without_nan=Qlist.ravel()[~np.isnan(Qlist.ravel())]
        sci_image_without_nan=sci_image_masked.ravel()[~np.isnan(sci_image_masked.ravel())]
        Qvalue = np.sum(Qlist_without_nan)/np.sum(sci_image_without_nan)
        
        return [np.sum(res),chi2_intrinsic,Qvalue]
    
    def lnlike_Neven(self,allparameters,return_Image=False):
        """
        report likelihood given the parameters of the model
        give -np.inf if outside of the parameters range specified below 
        """ 
        
        if self.verbosity==1:
            print('')
            print('Entering lnlike_Neven')
            print('allparameters '+str(allparameters))
        
        if self.pupil_parameters is not None:    
            if len(allparameters)<25:
                allparameters=add_pupil_parameters_to_all_parameters(allparameters,self.pupil_parameters)
            else:
                allparameters=add_pupil_parameters_to_all_parameters(remove_pupil_parameters_from_all_parameters(allparameters),self.pupil_parameters)
                        
        zparameters=allparameters[0:self.zmax-3]
        globalparameters=allparameters[len(zparameters):]
        #print(zparameters)
        #print(globalparameters)
        


        # internal parameter for debugging change value to 1 to see which parameters are failling
        test_print=0
        if self.verbosity==1:
            test_print=1
        #When running big fits these are limits which ensure that the code does not wander off in totally nonphyical region

        # hsc frac
        if globalparameters[0]<=0.6 or globalparameters[0]>0.8:
            print('globalparameters[0] outside limits') if test_print == 1 else False 
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
        
        # radiometricEffect
        if globalparameters[6]<0:
            print('globalparameters[6] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[6]>1:
            print('globalparameters[6] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        # radiometricExponent
        if globalparameters[7]<0:
            print('globalparameters[7] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[7]>2:
            print('globalparameters[7] outside limits') if test_print == 1 else False 
            return -np.inf 
        
        # x_ilum
        if globalparameters[8]<0.5:
            print('globalparameters[8] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[8]>1.5:
            print('globalparameters[8] outside limits') if test_print == 1 else False 
            return -np.inf
        
        # y_ilum
        if globalparameters[9]<0.5:
            print('globalparameters[9] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[9]>1.5:
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
            print('globalparameters[12] outside limits') if test_print == 1 else False 
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
        if globalparameters[17]>200000:
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
        if globalparameters[20]<0.35:
            print('globalparameters[20] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[20]>+0.8:
            print('globalparameters[20] outside limits') if test_print == 1 else False 
            return -np.inf  
        
         # fiber_r
        if globalparameters[21]<1.78:
            print('globalparameters[21] outside limits') if test_print == 1 else False 
            return -np.inf
        if globalparameters[21]>+1.98:
            print('globalparameters[21] outside limits') if test_print == 1 else False 
            return -np.inf  
        
        # flux
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
                
        if extra_Zernike_parameters is None:
            hash_name=np.abs(hash(tuple(allparameters))) 
        else:
            hash_name=np.abs(hash(tuple(allparameters))) 

        # this try statment avoid code crashing when code tries to analyze weird combination of parameters which fail to produce an image    
        try:   

            modelImg = self.single_image_analysis.constructModelImage_PFS_naturalResolution(self.single_image_analysis.params,extraZernike=extra_Zernike_parameters,hashvalue=hash_name)  
        except IndexError:
            return -np.inf
        
        chi_2_almost_multi_values=self.create_chi_2_almost(modelImg,self.sci_image,self.var_image,self.mask_image)
        chi_2_almost=chi_2_almost_multi_values[0]
        
        #res=-(1/2)*(chi_2_almost+np.log(2*np.pi*np.sum(self.var_image)))
        
        res=-(1/2)*(chi_2_almost+np.sum(np.log(2*np.pi*self.var_image)))

        if self.verbosity==True:
            print('Finished with lnlike_Neven')
            print(' ')
            
        if return_Image==False:
            return res
        else:
            return res,modelImg,allparameters

    def create_x(self,zparameters,globalparameters):
        """
        given the parameters move them in into array
        for a routine which fits at only one defocus this is superfluous bit of code 
        but I am keeping it here for compability with the code that fits many images at different defocus value.
        There the code moves parameters related to wavefront (zparameters) according to specified defocus and
        globalparameters stay the same
        
        
        @param zparameters        Zernike coefficents
        @param globalparameters   other parameters describing the system
        """ 
        x=np.zeros((len(zparameters)+len(globalparameters)))
        for i in range(len(zparameters)):
            x[i]=zparameters[i]     


        for i in range(len(globalparameters)):
            x[int(len(zparameters)/1)+i]=globalparameters[i]      
        
    
        return x
     
    def __call__(self, allparameters,return_Image=False):
        return self.lnlike_Neven(allparameters,return_Image=return_Image)

class LNP_PFS(object):
    def __init__(self,  image=None,image_var=None):
        self.image=image
        self.image_var=image_var
    def __call__(self, image=None,image_var=None):
        return 0.0

class PFSLikelihoodModule(object):
    """
    PFSLikelihoodModule object for calculating a likelihood for cosmoHammer.ParticleSwarmOptimizer
    """

    def __init__(self,model):
        """
        
        Constructor of the PFSLikelihoodModule
        """
        self.model=model

    def computeLikelihood(self, ctx):
        """
        Computes the likelihood using information from the context
        """
        # Get information from the context. This can be results from a core
        # module or the parameters coming from the sampler
        params = ctx.getParams()
        
        
        #print(params)
        # Calculate a likelihood up to normalization
        lnprob = self.model(params)
        #print(str(current_process())+str(lnprob))
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
    Class for analysing results of the cluster run
    """

    def __init__(self, date,obs,single_number,eps,arc=None,dataset=None):
        """!

        @param[in]
        """
        if arc is None:
            arc=''

        if dataset==0:
            STAMPS_FOLDER="/Users/nevencaplar/Documents/PFS/Data_Nov_14/Stamps_cleaned/"
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
        
        columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'hscFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                      'radiometricEffect','radiometricExponent',
                      'x_ilum','y_ilum',
                      'x_fiber','y_fiber','effective_ilum_radius','frd_sigma','det_vert','slitHolder_frac_dx',
                      'grating_lines','scattering_radius','scattering_slope','scattering_amplitude',
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
    Class that deals with positining the PSF model in respect to the data
    
    
    """

    def __init__(self, image,oversampling,size_natural_resolution, simulation_00=False, double_sources=False,double_sources_positions_ratios=[0,0],verbosity=0,save=None):
        
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
        @param size_natural_resolution        how many natural units to create new image (fix)
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
                                        double_sources=None,double_sources_positions_ratios=[0,0],verbosity=0):
    
        """
        function called by create_optPSF_natural
        find what is the best starting point to downsample the oversampled image
        
        @param image                                                      image to be analyzed (in our case this will be image of the optical psf convolved with fiber)
        @param oversampling                                               oversampling
        @param size_natural_resolution                                    size of final image (in the ``natural'', i.e., physical resolution)
        @param sci_image_0                                                scientific image
        @param var_image_0                                                variance image
        @param v_flux                                                     flux normalization
        @param simulation_00                                              do not move the center, for making good comparisons between images
        @param double_sources                                             are there double sources avaliable in the image ?
        @param double_sources_positions_ratios                            tuple describing init guess for the relation between secondary and primary souces (offset, ratio)
        @param verbosity                                                  verbosity of the algorithm
        
        returns model image in the size of the science image and centered to the science image
        """
        
        self.sci_image=sci_image
        self.var_image=var_image
        self.mask_image=mask_image
        self.v_flux=v_flux
        
        # if you are just asking for simulated image at (0,0) there is no possibility to create double sources
        if simulation_00 is 1:
            double_souces=None
            
        if double_sources is None:
            double_sources_positions_ratios=[0,0]
        
        shape_of_input_img=input_image.shape[0]
        shape_of_sci_image=sci_image.shape[0]
        max_possible_value_to_analyze=int(shape_of_input_img-oversampling)
        min_possible_value_to_analyze=int(oversampling)
        center_point=int(shape_of_input_img/2)
        
        self.shape_of_input_img=shape_of_input_img
        self.shape_of_sci_image=shape_of_sci_image
        
        # depending on if there is a second source in the image split here
        # double_sources is always None when using simulated images
        if double_sources==None:
            # if simulation_00 is on just run the realization set at 0
            if simulation_00 is 1:
                if verbosity==1:
                    print('simulation_00 is set to 1 - I am just returing the image at (0,0) coordinates ')                
                
                # return the solution with x and y is zero
                mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
                =self.create_complete_realization([0,0], return_full_result=True)          
            
            # if you are fitting an actual image go through the full process
            else:
                # create one complete realization with default parameters - estimate centorids and use that knowledge to put fitting limits in the next step
                centroid_of_sci_image=find_centroid_of_flux(sci_image)
                initial_complete_realization=self.create_complete_realization([0,0,-double_sources_positions_ratios[0]*self.oversampling,double_sources_positions_ratios[1]],return_full_result=True)[-1]
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
                primary_position_and_ratio=scipy.optimize.shgo(self.create_complete_realization,bounds=\
                                                                         [(x_2sources_limits[0],x_2sources_limits[1]),(y_2sources_limits[0],y_2sources_limits[1])],n=10,sampling_method='sobol',\
                                                                         options={'ftol':1e-3,'maxev':10})
    

                # return the best result, based on the result of the conducted search
                mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
                =self.create_complete_realization(primary_position_and_ratio.x, return_full_result=True)
            
            if self.save==1:
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_primary_renormalized',single_realization_primary_renormalized) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_secondary_renormalized',single_realization_secondary_renormalized)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized',complete_realization_renormalized)     
    
            if self.verbosity==1:
                    if simulation_00 is not 1:
                        print('We are fitting for only one source')
                        print('One source fitting result is '+str(primary_position_and_ratio.x))   
            
            return complete_realization_renormalized
        else:          
            # create one complete realization with default parameters - estimate centorids and use that knowledge to put fitting limits in the next step
            centroid_of_sci_image=find_centroid_of_flux(sci_image)
            initial_complete_realization=self.create_complete_realization([0,0,-double_sources_positions_ratios[0]*self.oversampling,double_sources_positions_ratios[1]],return_full_result=True)[-1]
            centroid_of_initial_complete_realization=find_centroid_of_flux(initial_complete_realization)
            
            #determine offset between the initial guess and the data
            offset_initial_and_sci=np.array(find_centroid_of_flux(initial_complete_realization))-np.array(find_centroid_of_flux(sci_image))
            
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
            primary_secondary_position_and_ratio=scipy.optimize.shgo(self.create_complete_realization,bounds=\
                                                                     [(x_2sources_limits[0],x_2sources_limits[1]),(y_2sources_limits[0],y_2sources_limits[1]),\
                                                                      (y_2sources_limits_second_source[0],y_2sources_limits_second_source[1]),\
                                                                      (self.double_sources_positions_ratios[1]/2,2*self.double_sources_positions_ratios[1])],n=10,sampling_method='sobol',\
                                                                      options={'maxev':10,'ftol':1e-3})
            
            #return best result
            mean_res,single_realization_primary_renormalized,single_realization_secondary_renormalized,complete_realization_renormalized \
            =self.create_complete_realization(primary_secondary_position_and_ratio.x, return_full_result=True)
    
            if self.save==1:
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_primary_renormalized',single_realization_primary_renormalized) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_realization_secondary_renormalized',single_realization_secondary_renormalized)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized',complete_realization_renormalized)     
    
            if self.verbosity==1:
                print('We are fitting for two sources')
                print('Two source fitting result is '+str(primary_secondary_position_and_ratio.x))   
            
            
            return complete_realization_renormalized
    
    
    def create_complete_realization(self, x, return_full_result=False):
        # need to include masking
        """
        create one complete realization of the image from the full oversampled image
        
        @param x                                                          array contaning x_primary, y_primary (y_secondary, ratio_secondary)
        @bol return_full_result                                           if yes, return the images iteself (not just chi**2)
        """
        
        
        image=self.image
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
            input_img_single_realization_before_downsampling_primary_floor_floor = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image))
            # image in the top left corner, x=0, y=1
            input_img_single_realization_before_downsampling_primary_floor_ceiling = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image))
            # image in the bottom right corner, x=1, y=0
            input_img_single_realization_before_downsampling_primary_ceiling_floor = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image))
            # image in the bottom left corner, x=0, y=0
            input_img_single_realization_before_downsampling_primary_ceiling_ceiling = np.zeros((oversampling*shape_of_sci_image, oversampling*shape_of_sci_image))

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
        
        # downsample the primary image    
        single_primary_realization=resize(input_img_single_realization_before_downsampling_primary,(shape_of_sci_image,shape_of_sci_image))
    
        ###################
        # implement - if secondary too far outside the image, do not go through secondary
        if ratio_secondary !=0:
            # go through secondary loop if ratio is not zero
            if secondary_offset_axis_0_floor<0 or (secondary_offset_axis_0_ceiling+oversampling*shape_of_sci_image)>len(image)\
            or secondary_offset_axis_1_floor<0 or (secondary_offset_axis_1_ceiling+oversampling*shape_of_sci_image)>len(image):
                pos_floor_floor = np.array([secondary_offset_axis_0_floor, secondary_offset_axis_1_floor])
                pos_floor_ceiling = np.array([secondary_offset_axis_0_floor, secondary_offset_axis_1_ceiling])
                pos_ceiling_floor = np.array([secondary_offset_axis_0_ceiling, secondary_offset_axis_1_floor])
                pos_ceiling_ceiling = np.array([secondary_offset_axis_0_ceiling, secondary_offset_axis_1_ceiling])    
                
                input_img_single_realization_before_downsampling_secondary_floor_floor = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0)
                input_img_single_realization_before_downsampling_secondary_floor_ceiling = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0)
                input_img_single_realization_before_downsampling_secondary_ceiling_floor = np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0)
                input_img_single_realization_before_downsampling_secondary_ceiling_ceiling= np.full([oversampling*shape_of_sci_image, oversampling*shape_of_sci_image], 0)
        
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
        
        
            input_img_single_realization_before_downsampling_secondary=self.bilinear_interpolation(secondary_offset_axis_1_mod_from_floor,secondary_offset_axis_0_mod_from_floor,\
                                                                                        input_img_single_realization_before_downsampling_secondary_floor_floor,input_img_single_realization_before_downsampling_secondary_floor_ceiling,\
                                                                                        input_img_single_realization_before_downsampling_secondary_ceiling_floor,input_img_single_realization_before_downsampling_secondary_ceiling_ceiling)
                    
            single_secondary_realization=resize(input_img_single_realization_before_downsampling_secondary,(shape_of_sci_image,shape_of_sci_image))    
    
        inverted_mask=~mask_image.astype(bool)
    
    
        if ratio_secondary !=0:
            complete_realization=single_primary_realization+ratio_secondary*single_secondary_realization
            complete_realization_renormalized=complete_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
        else:
            complete_realization=single_primary_realization
            complete_realization_renormalized=complete_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
            
            
        if return_full_result==False:
            chi_2_almost_multi_values=self.create_chi_2_almost_Psf_position(complete_realization_renormalized,sci_image,var_image,mask_image)
            if self.verbosity==1:
                print('chi2 within shgo optimization routine (chi_2_almost_multi_values): '+str(chi_2_almost_multi_values))
                #print('chi2 within shgo optimization routine (not chi_2_almost_multi_values): '+str(np.mean((sci_image-complete_realization_renormalized)**2/var_image)))
            return chi_2_almost_multi_values
        else:
            if ratio_secondary !=0:
                single_primary_realization_renormalized=single_primary_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized=ratio_secondary*single_secondary_realization*(np.sum(sci_image[inverted_mask])*v_flux/np.sum(complete_realization[inverted_mask]))    
            else:
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
                    np.save(TESTING_FINAL_IMAGES_FOLDER+'single_secondary_realization',single_secondary_realization) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_primary_realization',single_primary_realization) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_primary_realization_renormalized_within_create_complete_realization',single_primary_realization_renormalized) 
                np.save(TESTING_FINAL_IMAGES_FOLDER+'single_secondary_realization_renormalized_within_create_complete_realization',single_secondary_realization_renormalized)     
                np.save(TESTING_FINAL_IMAGES_FOLDER+'complete_realization_renormalized_within_create_complete_realization',complete_realization_renormalized)     
            
            #print('mask_image:'+str(np.sum(mask_image)))
            chi_2_almost_multi_values=self.create_chi_2_almost_Psf_position(complete_realization_renormalized,sci_image,var_image,mask_image)

            return chi_2_almost_multi_values,\
            single_primary_realization_renormalized,single_secondary_realization_renormalized,complete_realization_renormalized  
            
            #old code that did not include mask...
            #return np.mean((sci_image-complete_realization_renormalized)**2/var_image),\
            #single_primary_realization_renormalized,single_secondary_realization_renormalized,complete_realization_renormalized   

    def create_chi_2_almost_Psf_position(self,modelImg,sci_image,var_image,mask_image):
        """
        return array with 3 values
        1. normal chi**2
        2. what is 'instrinsic' chi**2, i.e., just sum((scientific image)**2/variance)
        3. 'Q' value = sum(abs(model - scientific image))/sum(scientific image)
        
        @param sci_image    model 
        @param sci_image    scientific image 
        @param var_image    variance image
        """ 
        inverted_mask=~mask_image.astype(bool)
        
        var_image_masked=var_image*inverted_mask
        sci_image_masked=sci_image*inverted_mask
        modelImg_masked=modelImg*inverted_mask
        
        chi2=(sci_image_masked - modelImg_masked)**2/var_image_masked
        chi2nontnan=chi2[~np.isnan(chi2)]
        return np.mean(chi2nontnan)


    def fill_crop(self, img, pos, crop):
      '''
      Fills `crop` with values from `img` at `pos`, 
      while accounting for the crop being off the edge of `img`.
      *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
      '''
      img_shape, pos, crop_shape = np.array(img.shape), np.array(pos), np.array(crop.shape),
      end = pos+crop_shape
      # Calculate crop slice positions
      crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
      crop_high = crop_shape - np.clip(end-img_shape, a_min=0, a_max=crop_shape)
      crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
      # Calculate img slice positions
      pos = np.clip(pos, a_min=0, a_max=img_shape)
      end = np.clip(end, a_min=0, a_max=img_shape)
      img_slices = (slice(low, high) for low, high in zip(pos, end))
      crop[tuple(crop_slices)] = img[tuple(img_slices)]    
      
    def bilinear_interpolation(self, y,x,img_floor_floor,img_floor_ceiling,img_ceiling_floor,img_ceiling_ceiling):     
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

    

def find_centroid_of_flux(image):
    """
    function giving the position of weighted average of the flux in a square image
    
    @param iamge    input image 
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

def create_parInit(allparameters_proposal,multi=None,pupil_parameters=None,allparameters_proposal_err=None,stronger=None,use_optPSF=None,deduced_scattering_slope=None,zmax=None):
    
    """!given the suggested parametrs create array with randomized starting values to supply to fitting code
    
    @param allparameters_proposal    array contaning suggested starting values for a model 
    @param multi                     (not working) when you want to analyze more images at once
    @param pupil_parameters          fixed parameters describign the pupil
    @param allparameters_proposal_err error on proposed parameters
    @param stronger                 factors which increases all errors
    @param use_optPFS               fix all parameters that give pure optical PSF, except z4 (allowing change in ['z4', 'scattering_slope', 'scattering_amplitude', 'pixel_effect', 'fiber_r', 'flux'])
    """ 
    if allparameters_proposal_err is not None:
        assert len(allparameters_proposal)==len(allparameters_proposal_err)
    if stronger is None:
        stronger=1

     # fixed scattering slope at number deduced from larger defocused image
    if zmax is None:
        zmax=11

    if zmax==11:   
        if deduced_scattering_slope is not None:
            allparameters_proposal[26]=np.abs(deduced_scattering_slope)
    else:
        if deduced_scattering_slope is not None:
            allparameters_proposal[26+11]=np.abs(deduced_scattering_slope)
    
    if zmax==11:
        if allparameters_proposal_err is None:
            if multi is None:
                allparameters_proposal_err=np.array([2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                        0.1,0.02,0.1,0.1,0.1,0.1,
                                        0.3,1,0.1,0.1,
                                        0.15,0.15,0.1,
                                        0.07,0.2,0.05,0.4,
                                        30000,0.5,0.01,
                                        0.1,0.05,0.01])
                if stronger is not None:
                    allparameters_proposal_err=stronger*allparameters_proposal_err
                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26]=0
            else:
                allparameters_proposal_err=[2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                        0.1,0.1,0.1,0.1,0.05,0.1,
                                        0.2, 0.4,0.1,0.1,
                                        0.1,0.1,0.02,0.02,0.5,0.2,0.1,
                                        30000,0.5,0.01,
                                        0.1,0.05,0.01]        
    else:
        if allparameters_proposal_err is None:
            if multi is None:
                # updated Jul 29 2019
                allparameters_proposal_err=np.array([2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                                     0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,
                                        0.08,0.03,0.1,0.1,0.016,0.05,
                                        0.3,1,0.1,0.1,
                                        0.15,0.15,0.1,
                                        0.1,0.64,0.05,0.2,
                                        60000,0.95,0.014,
                                        0.2,0.14,0.015])
                if stronger is not None:
                    allparameters_proposal_err=stronger*allparameters_proposal_err
                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26+11]=0
            else:
                print('not implemented')
                allparameters_proposal_err=[2,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                                        0.1,0.1,0.1,0.1,0.05,0.1,
                                        0.2, 0.4,0.1,0.1,
                                        0.1,0.1,0.02,0.02,0.5,0.2,0.1,
                                        30000,0.5,0.01,
                                        0.1,0.05,0.01]         
    
    
    if pupil_parameters is None:
        number_of_par=len(allparameters_proposal)
    else:
        number_of_par=len(allparameters_proposal)-len(pupil_parameters)
        
    walkers_mult=6
    nwalkers=number_of_par*walkers_mult

    
    
    if zmax==11:
        if multi is None:
            zparameters_flatten=allparameters_proposal[0:8]
            zparameters_flatten_err=allparameters_proposal_err[0:8]
            globalparameters_flatten=allparameters_proposal[8:]
            globalparameters_flatten_err=allparameters_proposal_err[8:]

        else:
            zparameters_flatten=allparameters_proposal[0:8*2]
            zparameters_flatten_err=allparameters_proposal_err[0:8*2]
            globalparameters_flatten=allparameters_proposal[8*2:]
            globalparameters_flatten_err=allparameters_proposal_err[8*2:]
    else:
        if multi is None:
            zparameters_flatten=allparameters_proposal[0:8+11]
            zparameters_flatten_err=allparameters_proposal_err[0:8+11]
            globalparameters_flatten=allparameters_proposal[8+11:]
            globalparameters_flatten_err=allparameters_proposal_err[8+11:]
            #print(globalparameters_flatten_err)
        else:
            zparameters_flatten=allparameters_proposal[0:(8+11)*2]
            zparameters_flatten_err=allparameters_proposal_err[0:(8+11)*2]
            globalparameters_flatten=allparameters_proposal[(8+11)*2:]
            globalparameters_flatten_err=allparameters_proposal_err[(8+11)*2:]        
        
        
    #print(globalparameters_flatten)
    #print(globalparameters_flatten_err)
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
    else:
        if multi is None:
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
            except NameError:
                print('NameError!')
        else:
            try: 
                for i in range((8+11)*2):
                    zparameters_flat_single_par=np.concatenate(([zparameters_flatten[i]],np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1)))
                    if i==0:
                        zparameters_flat=zparameters_flat_single_par
                    else:
                        zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
            except NameError:
                print('NameError!')        
        
    #print(globalparameters_flatten[0])
    #print(globalparameters_flatten_err[0])       
    
    try:
        #print(globalparameters_flatten)
        # hscFrac always positive
        globalparameters_flat_0=np.abs(np.random.normal(globalparameters_flatten[0],globalparameters_flatten_err[0],nwalkers*20))
        globalparameters_flat_0=np.concatenate(([globalparameters_flatten[0]],
                                                globalparameters_flat_0[np.all((globalparameters_flat_0>0.6,globalparameters_flat_0<0.8),axis=0)][0:nwalkers-1]))
        # strutFrac always positive
        globalparameters_flat_1_long=np.abs(np.random.normal(globalparameters_flatten[1],globalparameters_flatten_err[1],nwalkers*200))
        globalparameters_flat_1=globalparameters_flat_1_long
        globalparameters_flat_1=np.concatenate(([globalparameters_flatten[1]],
                                                globalparameters_flat_1[np.all((globalparameters_flat_1>0.07,globalparameters_flat_1<0.13),axis=0)][0:nwalkers-1]))
        # dxFocal
        globalparameters_flat_2=np.random.normal(globalparameters_flatten[2],globalparameters_flatten_err[2],nwalkers*20)
        globalparameters_flat_2=np.concatenate(([globalparameters_flatten[2]],
                                                globalparameters_flat_2[np.all((globalparameters_flat_2>-0.4,globalparameters_flat_2<0.4),axis=0)][0:nwalkers-1]))
        # dyFocal
        globalparameters_flat_3=np.random.normal(globalparameters_flatten[3],globalparameters_flatten_err[3],nwalkers*20)
        globalparameters_flat_3=np.concatenate(([globalparameters_flatten[3]],
                                                globalparameters_flat_3[np.all((globalparameters_flat_3>-0.4,globalparameters_flat_3<0.4),axis=0)][0:nwalkers-1]))
        # slitFrac
        globalparameters_flat_4=np.abs(np.random.normal(globalparameters_flatten[4],globalparameters_flatten_err[4],nwalkers*20))
        #print(globalparameters_flatten_err[4])
        globalparameters_flat_4=np.concatenate(([globalparameters_flatten[4]],
                                                globalparameters_flat_4[np.all((globalparameters_flat_4>0.05,globalparameters_flat_4<0.09),axis=0)][0:nwalkers-1]))
        # slitFrac_dy
        globalparameters_flat_5=np.abs(np.random.normal(globalparameters_flatten[5],globalparameters_flatten_err[5],nwalkers*20))      
        globalparameters_flat_5=np.concatenate(([globalparameters_flatten[5]],
                                                globalparameters_flat_5[np.all((globalparameters_flat_5>-0.5,globalparameters_flat_5<0.5),axis=0)][0:nwalkers-1]))
        # radiometricEffect
        globalparameters_flat_6=np.abs(np.random.normal(globalparameters_flatten[6],globalparameters_flatten_err[6],nwalkers*20))
        globalparameters_flat_6=np.concatenate(([globalparameters_flatten[6]],
                                                globalparameters_flat_6[np.all((globalparameters_flat_6>0,globalparameters_flat_6<=1),axis=0)][0:nwalkers-1]))
        # radiometricExponent
        globalparameters_flat_7=np.random.normal(globalparameters_flatten[7],globalparameters_flatten_err[7],nwalkers*20)
        globalparameters_flat_7=np.concatenate(([globalparameters_flatten[7]],
                                                globalparameters_flat_7[np.all((globalparameters_flat_7>0.0,globalparameters_flat_7<2),axis=0)][0:nwalkers-1]))
        # x_ilum 
        globalparameters_flat_8=np.abs(np.random.normal(globalparameters_flatten[8],globalparameters_flatten_err[8],nwalkers*20))
        globalparameters_flat_8=np.concatenate(([globalparameters_flatten[8]],
                                                globalparameters_flat_8[np.all((globalparameters_flat_8>0.5,globalparameters_flat_8<1.5),axis=0)][0:nwalkers-1]))
        # y_ilum
        globalparameters_flat_9=np.abs(np.random.normal(globalparameters_flatten[9],globalparameters_flatten_err[9],nwalkers*20))
        globalparameters_flat_9=np.concatenate(([globalparameters_flatten[9]],
                                                globalparameters_flat_9[np.all((globalparameters_flat_9>0.5,globalparameters_flat_9<1.5),axis=0)][0:nwalkers-1]))
        # x_fiber
        globalparameters_flat_10=np.random.normal(globalparameters_flatten[10],globalparameters_flatten_err[10],nwalkers*20)
        globalparameters_flat_10=np.concatenate(([globalparameters_flatten[10]],
                                                 globalparameters_flat_10[np.all((globalparameters_flat_10>-0.4,globalparameters_flat_10<0.4),axis=0)][0:nwalkers-1]))
        # y_fiber
        globalparameters_flat_11=np.random.normal(globalparameters_flatten[11],globalparameters_flatten_err[11],nwalkers*20)
        globalparameters_flat_11=np.concatenate(([globalparameters_flatten[11]],
                                                 globalparameters_flat_11[np.all((globalparameters_flat_11>-0.4,globalparameters_flat_11<0.4),axis=0)][0:nwalkers-1]))
        
        #effective_radius_illumination
        globalparameters_flat_12=np.random.normal(globalparameters_flatten[12],globalparameters_flatten_err[12],nwalkers*20)
        globalparameters_flat_12=np.concatenate(([globalparameters_flatten[12]],
                                                 globalparameters_flat_12[np.all((globalparameters_flat_12>0.7,globalparameters_flat_12<1.0),axis=0)][0:nwalkers-1]))
        
        if globalparameters_flatten[13]<0.01:
            globalparameters_flatten[13]=0.01
        # frd_sigma
        globalparameters_flat_13=np.random.normal(globalparameters_flatten[13],globalparameters_flatten_err[13],nwalkers*20)
        globalparameters_flat_13=np.concatenate(([globalparameters_flatten[13]],
                                                 globalparameters_flat_13[np.all((globalparameters_flat_13>=0.01,globalparameters_flat_13<0.4),axis=0)][0:nwalkers-1]))
        
        # frd_lorentz_factor
        globalparameters_flat_14=np.random.normal(globalparameters_flatten[14],globalparameters_flatten_err[14],nwalkers*20)
        globalparameters_flat_14=np.concatenate(([globalparameters_flatten[14]],
                                                 globalparameters_flat_14[np.all((globalparameters_flat_14>0.01,globalparameters_flat_14<1),axis=0)][0:nwalkers-1]))       
        
        # det_vert
        globalparameters_flat_15=np.random.normal(globalparameters_flatten[15],globalparameters_flatten_err[15],nwalkers*20)
        globalparameters_flat_15=np.concatenate(([globalparameters_flatten[15]],
                                                 globalparameters_flat_15[np.all((globalparameters_flat_15>0.85,globalparameters_flat_15<1.15),axis=0)][0:nwalkers-1]))
        
        #slitHolder_frac_dx
        globalparameters_flat_16=np.random.normal(globalparameters_flatten[16],globalparameters_flatten_err[16],nwalkers*20)
        globalparameters_flat_16=np.concatenate(([globalparameters_flatten[16]],
                                                 globalparameters_flat_16[np.all((globalparameters_flat_16>-0.8,globalparameters_flat_16<0.8),axis=0)][0:nwalkers-1]))

        # grating lines
        globalparameters_flat_17=np.random.normal(globalparameters_flatten[17],globalparameters_flatten_err[17],nwalkers*20)
        globalparameters_flat_17=np.concatenate(([globalparameters_flatten[17]],
                                                 globalparameters_flat_17[np.all((globalparameters_flat_17>1200,globalparameters_flat_17<120000),axis=0)][0:nwalkers-1]))

        # scattering_slope
        globalparameters_flat_18=np.random.normal(globalparameters_flatten[18],globalparameters_flatten_err[18],nwalkers*20)
        globalparameters_flat_18=np.concatenate(([globalparameters_flatten[18]],
                                                 globalparameters_flat_18[np.all((globalparameters_flat_18>1.5,globalparameters_flat_18<3.0),axis=0)][0:nwalkers-1]))
        # scattering_amplitude
        globalparameters_flat_19=np.random.normal(globalparameters_flatten[19],globalparameters_flatten_err[19],nwalkers*20)
        globalparameters_flat_19=np.concatenate(([globalparameters_flatten[19]],
                                                 globalparameters_flat_19[np.all((globalparameters_flat_19>0.0,globalparameters_flat_19<0.4),axis=0)][0:nwalkers-1]))
        # pixel_effect
        globalparameters_flat_20=np.random.normal(globalparameters_flatten[20],globalparameters_flatten_err[20],nwalkers*20)
        globalparameters_flat_20=np.concatenate(([globalparameters_flatten[20]],
                                                 globalparameters_flat_20[np.all((globalparameters_flat_20>0.35,globalparameters_flat_20<0.8),axis=0)][0:nwalkers-1]))
        
        # fiber_r
        if globalparameters_flatten[21]<1.78:
            globalparameters_flatten[21]=1.8
        
        globalparameters_flat_21=np.random.normal(globalparameters_flatten[21],globalparameters_flatten_err[21],nwalkers*20)
        globalparameters_flat_21=np.concatenate(([globalparameters_flatten[21]],
                                                 globalparameters_flat_21[np.all((globalparameters_flat_21>1.78,globalparameters_flat_21<1.98),axis=0)][0:nwalkers-1]))
        
        # flux
        globalparameters_flat_22=np.random.normal(globalparameters_flatten[22],globalparameters_flatten_err[22],nwalkers*20)
        globalparameters_flat_22=np.concatenate(([globalparameters_flatten[22]],
                                                 globalparameters_flat_22[np.all((globalparameters_flat_22>0.98,globalparameters_flat_22<1.02),axis=0)][0:nwalkers-1]))



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
            globalparameters_flat=np.column_stack((globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3,
                                                   globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_10,
                                                   globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13,
                                                   globalparameters_flat_14,globalparameters_flat_15,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,globalparameters_flat_22))
            
        else:
                        globalparameters_flat=np.column_stack((globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,globalparameters_flat_22))
                        
    except NameError:
        print("NameError")

    
    #print(globalparameters_flat.shape) 
    
    allparameters=np.column_stack((zparameters_flat,globalparameters_flat))
    
    parInit=allparameters.reshape(nwalkers,number_of_par) 
    
    
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

def value_at_defocus(mm,a,b=None):
    if b==None:
        return a
    else:
        return a*mm+b 
    
def create_x(mm,parameters):
    """
    transfrom multi analysis variable into single analysis variables
    only for z11
    
    @param mm     defocu at LAM
    @param parameters multi parameters
    
    """
    #for single case, up to z11
    
    zparameters=parameters[:16]
    globalparameters=parameters[16:]
    
    
    x=np.zeros((8+len(globalparameters)))
    for i in range(0,8,1):
        x[i]=value_at_defocus(mm,zparameters[i*2],zparameters[i*2+1])      

    for i in range(len(globalparameters)):
        x[int(len(zparameters)/2)+i]=globalparameters[i] 
    

    return x

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

@lru_cache()
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
    old_breaks = np.linspace(0, old_size, num=old_size + 1)
    new_breaks = np.linspace(0, old_size, num=new_size + 1)
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

