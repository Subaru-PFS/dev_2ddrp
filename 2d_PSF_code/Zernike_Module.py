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
Jun 26, 2019; 0.21d -> 0.21e included variable ``dataset'',
                             which denots which data we are using in the analysis
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
Nov 19, 2020: 0.35d -> 0.36 realized that vertical strut is different than others -
                            first, simplest implementation
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
Apr 27, 2021: 0.44f -> 0.45 Tokovinin now works much quicker with multi_background_factor
                            (create_simplified_H updated)
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
Jun 15, 2021: 0.46b -> 0.46c change limit on the initial cut of the oversampled image,
                             in order to handle bluer data
Jun 19, 2021: 0.46c -> 0.46d changed skimage.transform.resize to resize,
                             to avoid skimage.transform not avaliable in LSST
Jun 20, 2021: 0.46d -> 0.46e changed scipy.signal to signal,
                             and require that optPsf_cut_downsampled_scattered size is int /
                             no change to unit test
Jun 24, 2021: 0.46e -> 0.47 removed resize and introduced galsim resizing in Psf_position,
                             to be consistent with LSST pipeline
Jun 25, 2021: 0.47  -> 0.47a introduced galsim resizing in the first downsampling from natural resolution
                             to default=10 oversampling also
Jul 11, 2021: 0.47a -> 0.47b changed a minus factor in secondary position estimation
Jul 12, 2021: 0.47b -> 0.47c inital offset in positioning had a wrong +- sign in front
Jul 23, 2021: 0.47c -> 0.47d (only) added comments and explanations
Jul 26, 2021: 0.47d -> 0.47e changed default oversampling to 11
Jul 27, 2021: 0.47e -> 0.47f offset done in galsim, but downsampling via resize function
Aug 26, 2021: 0.47f -> 0.47g direct minimization when use_center_of_flux=True
Aug 30, 2021: 0.47g -> 0.48 offset done in LSST code now
Sep 02, 2021: 0.48 -> 0.48a done cleaning offset code (PIPE2D-880)
Sep 15, 2021: 0.48a -> 0.48b removed minor bug where array_of_var_sum was called too early,
                             and could fail if nan value was present
Sep 27, 2021: 0.48b -> 0.48c added explicit bool conversion to double_sources
Oct 05, 2021: 0.48c -> 0.48d further explicit bool(double_sources) covnersion in ln_pfs_single
Oct 08, 2021: 0.48d -> 0.48e Pep8 cleaning
Oct 15, 2021: 0.48e -> 0.48f forced a randomseed number in create_parInit function
Oct 25, 2021: 0.48f -> 0.49 set half of init values in create_parInit to be same as init value
Oct 26, 2021: 0.49 -> 0.49a modified create_custom_var that it does lin fit if 2nd degree fit is convex
Oct 28, 2021: 0.49a -> 0.49b modified create_custom_var so that it does not fall below min(var) value
Nov 01, 2021: 0.49b -> 0.49c create_custom_var does not change var image from step to step anymore
Nov 02, 2021: 0.49c -> 0.49d eliminated std varianble from create_simplified_H
Nov 03, 2021: 0.49d -> 0.49e PIPE2D-930; fixed reusing list_of_variance in Tokovinin
Nov 03, 2021: 0.49e -> 0.50 PIPE2D-931; modified creation of polyfit for variance image higher up
                            so it is done only once per sci/var/mask image combination
Nov 20, 2021: 0.50 -> 0.50a Hilo modifications
Dec 06, 2021: 0.50a -> 0.51 Zernike_estimation_preparation class
Dec 09, 2021: 0.51 -> 0.51a introduced `fixed_single_spot`
Feb 11, 2022: 0.51a -> 0.51b unified index parameter allowed to vary
Mar 18, 2022: 0.51b -> 0.51c introduced div_same par, controlling how many particles are same
Mar 24, 2022: 0.51c -> 0.51d multiple small changes, for running same illum in fiber
Apr 03, 2022: 0.51d -> 0.51e test is now analysis_type_fiber == "fixed_fiber_par"
May 05, 2022: 0.51e -> 0.51f added documentation
May 09, 2022: 0.51f -> 0.51g replaced print with logging
May 24, 2022: 0.51g -> 0.51h small changes to output testing directory
May 26, 2022: 0.51h -> 0.51i linting fixes
Jun 01, 2022: 0.51i -> 0.52 im1.setCenter(0,0), to be compatible with galsim 2.3.4

@author: Neven Caplar
@contact: ncaplar@princeton.edu
@web: www.ncaplar.com
"""
########################################
# standard library imports
# from __future__ import absolute_import, division, logging.info_function
from functools import partial
from typing import Tuple, Iterable

# import matplotlib
# from matplotlib.colors import LogNorm
# import matplotlib.pyplot as plt
import lmfit
from scipy.linalg import svd
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import scipy.fftpack
import scipy.misc
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.special import erf
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import Tophat2DKernel
import lsst.afw.math
import lsst.afw.image
import lsst.afw
import lsst
import galsim
import traceback
# import platform
import threading
# from multiprocessing import current_process
import numpy as np
import pandas as pd
import os
import time
# import sys
import math
import socket
import sys
import pickle
import logging

# PFS imports
from pfs.utils.fiberids import FiberIds
gfm = FiberIds()

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
# logging.info(np.__config__)


########################################
# Related third party imports
# none at the moment

########################################
# Local application/library specific imports
# galsim
galsim.GSParams.maximum_fft_size = 12000

# lsst

# astropy
# import astropy
# import astropy.convolution

# scipy
# import scipy
# import skimage.transform
# import scipy.optimize as optimize
# for svd_invert function

# lmfit

# matplotlib

# needed for resizing routines

# for distributing image creation in Tokovinin algorithm
########################################

__all__ = [
    'PupilFactory',
    'Pupil',
    'ZernikeFitterPFS',
    'LN_PFS_multi_same_spot',
    'LN_PFS_single',
    'LNP_PFS',
    'find_centroid_of_flux',
    'create_parInit',
    'PFSPupilFactory',
    'custom_fftconvolve',
    'stepK',
    'maxK',
    'sky_scale',
    'sky_size',
    'remove_pupil_parameters_from_all_parameters',
    'resize',
    '_interval_overlap',
    'svd_invert',
    'Tokovinin_multi',
    'find_centroid_of_flux',
    'create_popt_for_custom_var',
    'create_custom_var_from_popt',
    'Zernike_estimation_preparation']

__version__ = "0.52"

# classes Pupil, PupilFactory and PFSPupilFactory have different form of documentation,
# compared to other classes as they have been imported from code written by Josh Meyers


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

    Based on the code by Josh Meyers, developed for HSC camera
    Contains functions that can create various obscurations in the camera
    """

    def __init__(
            self,
            pupilSize,
            npix,
            input_angle,
            detFrac,
            strutFrac,
            slitFrac,
            slitFrac_dy,
            x_fiber,
            y_fiber,
            effective_ilum_radius,
            frd_sigma,
            frd_lorentz_factor,
            det_vert,
            wide_0=0,
            wide_23=0,
            wide_43=0,
            misalign=0,
            verbosity=0):
        """Construct a PupilFactory.
        Parameters
        ----------
        pupilSize: `float`
            Size of the exit pupil [m]
        npix: `int`
            Constructed Pupils will be npix x npix
        input_angle: `float`
            Angle of the pupil (for all practical purposes fixed an np.pi/2)
        detFrac: `float`
            Value determining how much of the exit pupil obscured by the
            central obscuration(detector)
        strutFrac: `float`
            Value determining how much of the exit pupil is obscured
            by a single strut
        slitFrac: `float`
             Value determining how much of the exit pupil is obscured by slit
        slitFrac_dy: `float`
            Value determining what is the vertical position of the slit
            in the exit pupil
        x_fiber: `float`
            Position of the fiber misaligment in the x direction
        y_fiber: `float`
            Position of the fiber misaligment in the y direction
        effective_ilum_radius: `float`
            Fraction of the maximal radius of the illumination
            of the exit pupil that is actually illuminated
        frd_sigma: `float`
            Sigma of Gaussian convolving only outer edge, mimicking FRD
        frd_lorentz_factor: `float`
            Strength of the lorentzian factor describing wings
        det_vert: `float`
            Multiplicative factor determining vertical size
            of the detector obscuration
        wide_0: `float`
            Widening of the strut at 0 degrees
        wide_23: `float`
            Widening of the strut at the top-left corner
        wide_43: `float`
            Widening of the strut at the bottom-left corner
        misalign: `float`
            Describing the amount of misaligment
        verbosity: `int`
            How verbose during evaluation (1 = full verbosity)
        """
        self.verbosity = verbosity
        if self.verbosity == 1:
            logging.info('Entering PupilFactory class')
            logging.info('Entering PupilFactory class')

        self.pupilSize = pupilSize
        self.npix = npix
        self.input_angle = input_angle
        self.detFrac = detFrac
        self.strutFrac = strutFrac
        self.pupilScale = pupilSize / npix
        self.slitFrac = slitFrac
        self.slitFrac_dy = slitFrac_dy
        self.effective_ilum_radius = effective_ilum_radius
        self.frd_sigma = frd_sigma
        self.frd_lorentz_factor = frd_lorentz_factor
        self.det_vert = det_vert

        self.wide_0 = wide_0
        self.wide_23 = wide_23
        self.wide_43 = wide_43
        self.misalign = misalign

        u = (np.arange(npix, dtype=np.float32) - (npix - 1) / 2) * self.pupilScale
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
        return np.abs(dy21 * x0 - dx21 * y0 + x2 * y1 - y2 * x1) / np.hypot(dy21, dx21)

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
        theta = np.arctan(self.u / self.v) + thetarot

        pupil.illuminated[r2 > r**2 * b**2 / (b**2 * (np.cos(theta))**2 + r**2 * (np.sin(theta))**2)] = False

    def _cutSquare(self, pupil, p0, r, angle, det_vert):
        """Cut out the interior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half lenght of the length of square side
        @param[in] angle      angle that the camera is rotated
        @param[in] det_vert   multiplicative factor that distorts the square into a rectangle
        """
        pupil_illuminated_only1 = np.ones_like(pupil.illuminated, dtype=np.float32)

        time_start_single_square = time.time()

        ###########################################################
        # Central square
        if det_vert is None:
            det_vert = 1

        x21 = -r / 2 * det_vert * 1
        x22 = +r / 2 * det_vert * 1
        y21 = -r / 2 * 1
        y22 = +r / 2 * 1
        i_max = self.npix / 2 - 0.5
        i_min = -i_max

        i_y_max = int(np.round((x22 + p0[1]) / self.pupilScale - (i_min)))
        i_y_min = int(np.round((x21 + p0[1]) / self.pupilScale - (i_min)))
        i_x_max = int(np.round((y22 + p0[0]) / self.pupilScale - (i_min)))
        i_x_min = int(np.round((y21 + p0[0]) / self.pupilScale - (i_min)))

        assert angle == np.pi / 2
        # angleRad = angle

        camX_value_for_f_multiplier = p0[0]
        camY_value_for_f_multiplier = p0[1]

        # logging.info(camX_value_for_f_multiplier,camY_value_for_f_multiplier)
        camY_Max = 0.02
        f_multiplier_factor = (-camX_value_for_f_multiplier * 100 / 3) * \
            (np.abs(camY_value_for_f_multiplier) / camY_Max) + 1
        # f_multiplier_factor=1
        if self.verbosity == 1:
            logging.info('f_multiplier_factor for size of detector triangle is: ' + str(f_multiplier_factor))

        pupil_illuminated_only0_in_only1 = np.zeros((i_y_max - i_y_min, i_x_max - i_x_min))

        u0 = self.u[i_y_min:i_y_max, i_x_min:i_x_max]
        v0 = self.v[i_y_min:i_y_max, i_x_min:i_x_max]

        # factor that is controling how big is the triangle in the corner of the detector?
        f = 0.2
        f_multiplier = f_multiplier_factor / 1

        ###########################################################
        # Lower right corner
        x21 = -r / 2
        x22 = +r / 2
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert

        f_lr = np.copy(f) * (1 / f_multiplier)

        angleRad21 = -np.pi / 4
        triangle21 = [[p0[0] + x22, p0[1] + y21],
                      [p0[0] + x22, p0[1] + y21 - y21 * f_lr],
                      [p0[0] + x22 - x22 * f_lr, p0[1] + y21]]

        p21 = triangle21[0]
        y22 = (triangle21[1][1] - triangle21[0][1]) / np.sqrt(2)
        y21 = 0
        x21 = (triangle21[2][0] - triangle21[0][0]) / np.sqrt(2)
        x22 = -(triangle21[2][0] - triangle21[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((v0 - p21[1]) * np.cos(-angleRad21)
                                          - (u0 - p21[0]) * np.sin(-angleRad21) < y22)] = True

        ###########################################################
        # Upper left corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert
        # angleRad12 = -np.pi / 4
        f_ul = np.copy(f) * (1 / f_multiplier)

        triangle12 = [[p0[0] + x21, p0[1] + y22],
                      [p0[0] + x21, p0[1] + y22 - y22 * f_ul],
                      [p0[0] + x21 - x21 * f_ul, p0[1] + y22]]

        p21 = triangle12[0]
        y22 = 0
        y21 = (triangle12[1][1] - triangle12[0][1]) / np.sqrt(2)
        x21 = -(triangle12[2][0] - triangle12[0][0]) / np.sqrt(2)
        x22 = +(triangle12[2][0] - triangle12[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((v0 - p21[1]) * np.cos(-angleRad21)
                                          - (u0 - p21[0]) * np.sin(-angleRad21) > y21)] = True

        ###########################################################
        # Upper right corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert
        f_ur = np.copy(f) * f_multiplier

        triangle22 = [[p0[0] + x22, p0[1] + y22],
                      [p0[0] + x22, p0[1] + y22 - y22 * f_ur],
                      [p0[0] + x22 - x22 * f_ur, p0[1] + y22]]

        p21 = triangle22[0]
        y22 = -0
        y21 = +(triangle22[1][1] - triangle22[0][1]) / np.sqrt(2)
        x21 = +(triangle22[2][0] - triangle22[0][0]) / np.sqrt(2)
        x22 = -(triangle22[2][0] - triangle22[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((u0 - p21[0]) * np.cos(-angleRad21)
                                          + (v0 - p21[1]) * np.sin(-angleRad21) > x21)] = True

        ###########################################################
        # Lower left corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert
        f_ll = np.copy(f) * f_multiplier

        triangle11 = [[p0[0] + x21, p0[1] + y21],
                      [p0[0] + x21, p0[1] + y21 - y21 * f_ll],
                      [p0[0] + x21 - x21 * f_ll, p0[1] + y21]]

        p21 = triangle11[0]
        y22 = -(triangle11[1][1] - triangle11[0][1]) / np.sqrt(2)
        y21 = 0
        x21 = +(triangle11[2][0] - triangle11[0][0]) / np.sqrt(2)
        x22 = +(triangle11[2][0] - triangle11[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((u0 - p21[0]) * np.cos(-angleRad21)
                                          + (v0 - p21[1]) * np.sin(-angleRad21) < x22)] = True

        pupil_illuminated_only1[i_y_min:i_y_max, i_x_min:i_x_max] = pupil_illuminated_only0_in_only1

        pupil.illuminated = pupil.illuminated * pupil_illuminated_only1
        time_end_single_square = time.time()

        if self.verbosity == 1:
            logging.info('Time for cutting out the square is '
                         + str(time_end_single_square - time_start_single_square))

    def _cutRay(self, pupil, p0, angle, thickness, angleunit=None, wide=0):
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

        radial_distance = 14.34 * np.sqrt((self.u - p0[0])**2 + (self.v - p0[1])**2)

        pupil.illuminated[(d < 0.5 * thickness * (1 + wide * radial_distance))
                          & ((self.u - p0[0]) * np.cos(angleRad)
                             + (self.v - p0[1]) * np.sin(angleRad) >= 0)] = False

    def _addRay(self, pupil, p0, angle, thickness, angleunit=None):
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
        pupil.illuminated[(d < 0.5 * thickness)
                          & ((self.u - p0[0]) * np.cos(angleRad)
                             + (self.v - p0[1]) * np.sin(angleRad) >= 0)] = True

    def pfiIlum_2d(self, pupil, fiber_id, p0=(0, 0)):
        """Apply PFI correction to the illumination

        Parameters
        ----------
        fiber_id: `int`
            Fiber id number
        p0: 2-tuple
            Center of the pupil

        Returns
        ----------
        None
            Nothing returned. PFI correction applied to pupil.illuminated.
        """
        # array where each element is a distance from the p0 point
        # where p0 is an arbitrary point placed in the pupil
        r_dist = np.sqrt((self.u - p0[0])**2 + (self.v - p0[1])**2)
        # array with a difference between pfi ilumination and dcb illumination
        pfiIlum_2d = self._pfiIlum_1d(fiber_id, r_dist)
        # apply the difference to the pupil.illumination
        pupil.illuminated = pupil.illuminated * pfiIlum_2d
        return

    def _pfiIlum_1d(self, fiber_id, r_dist):
        """Return 1d radial profile for a given fiber

        Parameters
        ----------
        fiber_id: `int`
            Fiber id number

        Returns
        ----------
        array
        """
        rad = gfm.data['rad'][fiber_id - 1]  # fiberid - 1 ??
        ang = self._rad_to_angle(rad)
        _data = self._radial_profile(ang)
        _fun = interp1d(_data[0], _data[1], bounds_error=False, fill_value='extrapolate')
        return _fun(r_dist)

    def _rad_to_angle(self, rad):
        """Transform radius to an angle on the focal plane

        Parameters
        ----------
        rad: `float`
            Radial position of the fiber on the focal plane

        Returns
        ----------
        ang: `float`
            Angular position of the fiber on the focal plane
        """
        angles = [0, 6, 12, 18, 24, 30, 36, 42]
        r = [0, 32.0183, 64.128, 96.4280, 129.0284, 162.0553, 195.6696, 230.0966]
        z = np.polyfit(r, angles, deg=2)
        p_rad_to_angle = np.poly1d(z)
        ang = p_rad_to_angle(rad)
        return ang

    def _radial_profile(self, ang):
        """Return 1d theoretical radial profile for a given angle of the fiber on the focal plane

        Parameters
        ----------
        ang: `float`
            Angular position of the fiber on the focal plane

        Returns
        ----------
        """
        df_subarusb = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                               'data/subarusb.csv'
                                               ),
                                  sep=',', header='infer', skiprows=0)
        df_subarusb = df_subarusb[0:79]  # discarding NaNs
        # scale = 330. / 0.185  # modifiable
        scale = self.pupilSize / 2 / 0.185
        angles = [0, 6, 12, 18, 24, 30, 36, 42]
        radius = df_subarusb[df_subarusb.columns[0]].values.astype('float') * scale
        profile = df_subarusb[df_subarusb.columns[1:]].values.astype('float')
        f = interp2d(angles, radius, profile, kind='linear')
        return np.array([radius, f(ang, radius).T[0]])


class PFSPupilFactory(PupilFactory):
    """Pupil obscuration function factory for PFS

    Based on the code by Josh Meyers, initially developed for HSC camera
    Invokes PupilFactory to create obscurations of the camera
    Adds various illumination effects which are specified to the spectrographs
    """

    def __init__(
            self,
            pupilSize,
            npix,
            input_angle,
            detFrac,
            strutFrac,
            slitFrac,
            slitFrac_dy,
            x_fiber,
            y_fiber,
            effective_ilum_radius,
            frd_sigma,
            frd_lorentz_factor,
            det_vert,
            slitHolder_frac_dx,
            fiber_id=None,
            wide_0=0,
            wide_23=0,
            wide_43=0,
            misalign=0,
            verbosity=0):
        """!Construct a PupilFactory.


        Parameters
        ----------
        pupilSize: `float`
            Size of the exit pupil [m]
        npix: `int`
             Constructed Pupils will be npix x npix
        input_angle: `float`
            Angle of the pupil (for all practical purposes fixed an np.pi/2)
        detFrac: `float`
             Value determining how much of the exit pupil obscured by the
             central obscuration(detector)
        strutFrac: `float`
            Value determining how much of the exit pupil is obscured
            by a single strut
        slitFrac: `float`
             Value determining how much of the exit pupil is obscured by slit
        slitFrac_dy: `float`
            Value determining what is the vertical position of the slit
            in the exit pupil
        x_fiber: `float`
              Position of the fiber misaligment in the x direction
        y_fiber: `float`
              Position of the fiber misaligment in the y direction
        effective_ilum_radius: `float`
            Fraction of the maximal radius of the illumination
            of the exit pupil that is actually illuminated
        frd_sigma: `float`
            Sigma of Gaussian convolving only outer edge, mimicking FRD
        frd_lorentz_factor: `float`
            Strength of the lorentzian factor describing wings
        det_vert: `float`
             Multiplicative factor determining vertical size
             of the detector obscuration
        wide_0: `float`
           Widening of the strut at 0 degrees
        wide_23: `float`
          Widening of the strut at the top-left corner
        wide_43: `float`
              Widening of the strut at the bottom-left corner
        misalign: `float`
         Describing the amount of misaligment
        verbosity: `int`
            How verbose during evaluation (1 = full verbosity)
        """
        self.verbosity = verbosity
        if self.verbosity == 1:
            logging.info('Entering PFSPupilFactory class')

        PupilFactory.__init__(
            self,
            pupilSize,
            npix,
            input_angle,
            detFrac,
            strutFrac,
            slitFrac,
            slitFrac_dy,
            x_fiber,
            y_fiber,
            effective_ilum_radius,
            frd_sigma,
            frd_lorentz_factor,
            det_vert,
            verbosity=self.verbosity,
            wide_0=wide_0,
            wide_23=wide_23,
            wide_43=wide_43,
            misalign=misalign)

        self.x_fiber = x_fiber
        self.y_fiber = y_fiber
        self.slitHolder_frac_dx = slitHolder_frac_dx
        self._spiderStartPos = [np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])]
        self._spiderAngles = [0, np.pi * 2 / 3, np.pi * 4 / 3]
        self.effective_ilum_radius = effective_ilum_radius
        self.fiber_id = fiber_id

        self.wide_0 = wide_0
        self.wide_23 = wide_23
        self.wide_43 = wide_43
        self.misalign = misalign

    def getPupil(self, point):
        """!Calculate a Pupil at a given point in the focal plane.

        @param point  Point2D indicating focal plane coordinates.
        @returns      Pupil
        """
        if self.verbosity == 1:
            logging.info('Entering getPupil (function inside PFSPupilFactory)')

        # called subaruRadius as it was taken from the code fitting pupil for HSC on Subaru
        subaruRadius = (self.pupilSize / 2) * 1

        detFrac = self.detFrac  # linear fraction
        hscRadius = detFrac * subaruRadius
        slitFrac = self.slitFrac  # linear fraction
        subaruSlit = slitFrac * subaruRadius
        strutFrac = self.strutFrac  # linear fraction
        subaruStrutThick = strutFrac * subaruRadius

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

        # ###############################
        # Adding PFI illumination
        # ###############################

        # it modifies pupil.illuminated
        if self.fiber_id is not None:
            self.pfiIlum_2d(pupil, self.fiber_id)

        # ###############################
        # Creating FRD effects
        # ###############################
        single_element = np.linspace(-1, 1, len(pupil.illuminated), endpoint=True, dtype=np.float32)
        u_manual = np.tile(single_element, (len(single_element), 1))
        v_manual = np.transpose(u_manual)
        center_distance = np.sqrt((u_manual - self.x_fiber * hscRate * hscPlateScale * 12)
                                  ** 2 + (v_manual - self.y_fiber * hscRate * hscPlateScale * 12)**2)
        frd_sigma = self.frd_sigma
        sigma = 2 * frd_sigma

        pupil_frd = pupil.illuminated * \
            (1 / 2 * (scipy.special.erf((-center_distance + self.effective_ilum_radius) / sigma)
                      + scipy.special.erf((center_distance + self.effective_ilum_radius) / sigma)))

        # ###############################
        # Adding misaligment in this section
        # ###############################
        time_misalign_start = time.time()

        position_of_center_0 = np.where(center_distance == np.min(center_distance))
        position_of_center = [position_of_center_0[1][0], position_of_center_0[0][0]]

        position_of_center_0_x = position_of_center_0[0][0]
        position_of_center_0_y = position_of_center_0[1][0]

        distances_to_corners = np.array([np.sqrt(position_of_center[0]**2 + position_of_center[1]**2),
                                         np.sqrt((len(pupil_frd) - position_of_center[0])**2
                                                 + position_of_center[1]**2),
                                         np.sqrt((position_of_center[0])**2
                                                 + (len(pupil_frd) - position_of_center[1])**2),
                                         np.sqrt((len(pupil_frd) - position_of_center[0])**2
                                                 + (len(pupil_frd) - position_of_center[1])**2)])

        max_distance_to_corner = np.max(distances_to_corners)
        threshold_value = 0.5
        left_from_center = np.where(pupil_frd[position_of_center_0_x]
                                    [0:position_of_center_0_y] < threshold_value)[0]
        right_from_center = \
            np.where(pupil_frd[position_of_center_0_x][position_of_center_0_y:] < threshold_value)[0] +\
            position_of_center_0_y

        up_from_center = \
            np.where(pupil_frd[:, position_of_center_0_y][position_of_center_0_x:] < threshold_value)[0] +\
            position_of_center_0_x
        down_from_center = np.where(pupil_frd[:, position_of_center_0_y]
                                    [:position_of_center_0_x] < threshold_value)[0]

        if len(left_from_center) > 0:
            size_of_05_left = position_of_center_0_y - np.max(left_from_center)
        else:
            size_of_05_left = 0

        if len(right_from_center) > 0:
            size_of_05_right = np.min(right_from_center) - position_of_center_0_y
        else:
            size_of_05_right = 0

        if len(up_from_center) > 0:
            size_of_05_up = np.min(up_from_center) - position_of_center_0_x
        else:
            size_of_05_up = 0

        if len(down_from_center) > 0:
            size_of_05_down = position_of_center_0_x - np.max(down_from_center)
        else:
            size_of_05_down = 0

        sizes_4_directions = np.array([size_of_05_left, size_of_05_right, size_of_05_up, size_of_05_down])
        max_size = np.max(sizes_4_directions)
        imageradius = max_size

        radiusvalues = np.linspace(
            0, int(
                np.ceil(max_distance_to_corner)), int(
                np.ceil(max_distance_to_corner)) + 1)

        sigtotp = sigma * 550

        dif_due_to_mis_class = Pupil_misalign(radiusvalues, imageradius, sigtotp, self.misalign)
        dif_due_to_mis = dif_due_to_mis_class()

        scaling_factor_pixel_to_physical = max_distance_to_corner / np.max(center_distance)
        distance_int = np.round(center_distance * scaling_factor_pixel_to_physical).astype(int)

        pupil_frd_with_mis = pupil_frd + dif_due_to_mis[distance_int]
        pupil_frd_with_mis[pupil_frd_with_mis > 1] = 1

        time_misalign_end = time.time()

        if self.verbosity == 1:
            logging.info('Time to execute illumination considerations due to misalignment '
                         + str(time_misalign_end - time_misalign_start))

        # ###############################
        # Lorentz scattering
        # ###############################
        pupil_lorentz = (np.arctan(2 * (self.effective_ilum_radius - center_distance) / (4 * sigma))
                         + np.arctan(2 * (self.effective_ilum_radius + center_distance) / (4 * sigma))) /\
                        (2 * np.arctan((2 * self.effective_ilum_radius) / (4 * sigma)))

        pupil_frd = np.copy(pupil_frd_with_mis)
        pupil.illuminated = (pupil_frd + 1 * self.frd_lorentz_factor
                             * pupil_lorentz) / (1 + self.frd_lorentz_factor)

        # Cut out the acceptance angle of the camera
        self._cutCircleExterior(pupil, (0.0, 0.0), subaruRadius)

        # Cut out detector shadow
        self._cutSquare(pupil, (camX, camY), hscRadius, self.input_angle, self.det_vert)

        # No vignetting of this kind for the spectroscopic camera
        # self._cutCircleExterior(pupil, (lensX, lensY), lensRadius)

        # Cut out spider shadow
        for pos, angle in zip(self._spiderStartPos, self._spiderAngles):
            x = pos[0] + camX
            y = pos[1] + camY

            if angle == 0:
                # logging.info('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_0)
            if angle == np.pi * 2 / 3:
                # logging.info('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_23)
            if angle == np.pi * 4 / 3:
                # logging.info('cutRay applied to strut at angle '+str(angle))
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_43)

        # cut out slit shadow
        self._cutRay(pupil, (2, slitFrac_dy / 18), -np.pi, subaruSlit * 1.05, 'rad')

        # cut out slit holder shadow
        # subaruSlit/3 is roughly the width of the holder
        self._cutRay(pupil, (self.slitHolder_frac_dx / 18, 1), -np.pi / 2, subaruSlit * 0.3, 'rad')

        if self.verbosity == 1:
            logging.info('Finished with getPupil')

        return pupil


class Pupil_misalign(object):
    """Apply misaligment correction to the illumination of the pupil

    Developed by Brent Belland (Caltech)
    Copied here without modifications
    """

    def __init__(self, radiusvalues, imageradius, sigtotp, misalign):

        self.radiusvalues = radiusvalues
        self.imageradius = imageradius
        self.sigtotp = sigtotp
        self.misalign = misalign

    def wapp(self, A):
        # Approximation function by Jim Gunn to approximate and correct for the
        # widening of width due to the angular misalignment convolution. This
        # is used to basically scale the contribution of angular misalignment and FRD
        # A = angmis/sigFRD
        wappA = np.sqrt(1 + A * A * (1 + A * A) / (2 + 1.5 * A * A))
        return wappA

    def fcorr(self, x, A):
        # The function scaled so that it keeps the same (approximate) width value
        # after angular convolution
        correctedfam = self.fcon(x * self.wapp(A), A)
        return correctedfam

    def fcon(self, x, A):
        # For more detail about this method, see "Analyzing Radial Profiles for FRD
        # and Angular Misalignment", by Jim Gunn, 16/06/13.
        wt = [0.1864, 0.1469, 0.1134, 0.1066, 0.1134, 0.1469, 0.1864]   # from Jim Gunn's white paper,
        # wt contains the normalized integrals under the angular misalignment
        # convolution kernel, i.e., C(1-(x/angmisp)^2)^{-1/2} for |x|<angmisp and 0
        # elsewhere. Note that the edges' centers are at +/- a, so they are
        # integrated over an effective half of the length of the others.
        temp = np.zeros(np.size(x))
        for index in range(7):
            temp = temp + wt[index] * self.ndfc(x + (index - 3) / 3 * A)
        angconvolved = temp
        return angconvolved

    def ndfc(self, x):
        # Standard model dropoff from a Gaussian convolution, normalized to brightness 1,
        # radius (rh) 0, and sigTOT 1
        # logging.info(len(x))
        ndfcfun = 1 - (0.5 * erf(x / np.sqrt(2)) + 0.5)
        return ndfcfun

    def FA(self, r, rh, sigTOT, A):
        # Function that takes all significant variables of the dropoff and
        # normalizes the curve to be comparable to ndfc
        # r = vector of radius values, in steps of pixels
        # rh = radius of half-intensity. Effectively the size of the radius of the dropoff
        # sigTOT = total width of the convolution kernel that recreates the width of the dropoff
        # between 85% and 15% illumination. Effectively just think of this as sigma
        # A = angmis/sigFRD, that is, the ratio between the angular misalignment
        # and the sigma due to only FRD. Usually this is on the order of 1-3.
        FitwithAngle = self.fcorr((r - rh) / sigTOT, A)
        return FitwithAngle

    def __call__(self):

        no_mis = self.FA(self.radiusvalues, self.imageradius, self.sigtotp, 0)
        with_mis = self.FA(self.radiusvalues, self.imageradius, self.sigtotp, self.misalign)
        dif_due_to_mis = with_mis - no_mis

        return dif_due_to_mis


class ZernikeFitterPFS(object):

    """Create a model images for PFS

    Despite its name, it does not actually ``fits'' the paramters describing the donuts,
    it ``just'' creates the images

    The final image is made by the convolution of
    1. an OpticalPSF (constructed using FFT)
    2. an input fiber image
    3. and other convolutions such as CCD charge diffusion

    The OpticalPSF part includes
    1.1. description of pupil
    1.2. specification of an arbitrary number of zernike wavefront aberrations

    This code uses lmfit to initalize the parameters.

    Calls Psf_position
    Calls Pupil classes (which ones?)

    Called by LN_PFS_Single (function constructModelImage_PFS_naturalResolution)
    """

    def __init__(self, image=np.ones((20, 20)), image_var=np.ones((20, 20)),
                 image_mask=None, pixelScale=20.76, wavelength=794,
                 diam_sic=139.5327e-3, npix=1536, pupilExplicit=None,
                 wf_full_Image=None,
                 ilum_Image=None, dithering=1, save=None,
                 pupil_parameters=None, use_pupil_parameters=None, use_optPSF=None, use_wf_grid=None,
                 zmaxInit=None, extraZernike=None, simulation_00=None, verbosity=None,
                 double_sources=None, double_sources_positions_ratios=None, test_run=None,
                 explicit_psf_position=None, use_only_chi=False, use_center_of_flux=False,
                 PSF_DIRECTORY=None, fiber_id=None, *args):
        """
        Parameters
        ----------
        image: `np.array`, (N, N)
            image that you wish to model
            if you do not pass the image that you wish to compare,
            the algorithm will default to creating 20x20 image that has
            value of '1' everywhere
        image_var: `np.array`, (N, N)
            variance image
            if you do not pass the variance image,
            the algorithm will default to creating 20x20 image that has
            value of '1' everywhere
        image_mask: `np.array`, (N, N)
            mask image
        pixelScale: `float`
            pixel scale in arcseconds
            This is size of the pixel in arcsec for PFS red arm in focus
            calculated with http://www.wilmslowastro.com/software/formulae.htm
            pixel size in microns/focal length in mm x 206.3
            pixel size = 15 microns, focal length = 149.2 mm
            (138 aperature x 1.1 f number)
        wavelength: `float`
            wavelength of the psf [nm]
            if you do not pass the value for wavelength it will default to 794 nm,
            which is roughly in the middle of the red detector
        diam_sic: `float`
            size of the exit pupil [m]
            Exit pupil size in focus, default is 139.5237e-3 meters
            (taken from Zemax)
        npix: `int`
            size of 2d array contaning exit pupil illumination
        pupilExplicit: `np.array`, (Np, Np)
            if avaliable, uses this image for pupil instead of
            creating it from supplied parameters
        wf_full_Image: `np.array`, (Np, Np)
            wavefront image
            if avaliable, uses this image for wavefront instead of
            creating it from supplied parameters
        dithering: `int`
            dithering scale (most likely 1 or 2)
        save: `int`
            if 1, save various intermediate results, for testing purposes
            needs to set up also PSF_DIRECTORY
        use_optPSF: `np.array`, (Np, Np)
            if provided skip creation of optical psf, only do postprocessing
        use_wf_grid: `np.array`, (Ny, Nx)
            if provided, use this explicit wavefront map
         zmaxInit: `int`
             highest Zernike order (11 or 22)
        extraZernike: `np.array`, (N)
            if provided, simulated Zernike orders higher than 22
        simulation_00: `np.array`, (2,)
            places optical center at the center of the final image
        verbosity: `int`
            verbosity during evaluations
        double_sources:
            is there a second source present in the image
        double_sources_positions_ratios: `np.arrray`, (2,)
            initial guess for the position and strength of the second source
        explicit_psf_position: `np.array`, (2,)
            explicit position where to place optical psf
        use_only_chi: `bool`
            if True, fit to minimize np.abs(chi), and not chi**2
        use_center_of_flux: `bool`
            if True, fit to minimize the distance between the center of flux
            for the model and the input image
        PSF_DIRECTORY: `str`
            where will intermediate outputs be saved for testing purposes
        Notes
        ----------
        Creates a model image that is fitted to the input sicence image
        The model image is made by the convolution of
        1. an OpticalPSF (constructed using FFT)
            created with _getOptPsf_naturalResolution
        The OpticalPSF part includes
            1.1. description of pupil
                 created with get_Pupil
            1.2. specification of an arbitrary number of
                zernike wavefront aberrations,
                which are input to galsim.phase_screens.OpticalScreen
        2. an input fiber image and other convolutions such as
            CCD charge diffusion created with _optPsf_postprocessing
        This code uses lmfit to initalize the parameters.
        Calls class PsfPosition
        Calls class PFSPupilFactory

        Examples
        ----------
        Simple exampe with initial parameters, changing only one parameter
        >>> zmax = 22
        >>> single_image_analysis = ZernikeFitterPFS(zmaxInit = zmax,
                                                     verbosity=1,
                                                     fiber_id =100)
        >>> single_image_analysis.initParams()
        >>> single_image_analysis.params['detFrac'] =\
            lmfit.Parameter(name='detFrac', value=0.70)
        >>> resulting_image, psf_pos =\
            single_image_analysis.constructModelImage_PFS_naturalResolution()
        """

        self.image = image
        self.image_var = image_var
        if image_mask is None:
            image_mask = np.zeros(image.shape)
        self.image_mask = image_mask
        self.wavelength = wavelength
        self.diam_sic = diam_sic
        self.npix = npix
        self.dithering = dithering
        self.pixelScale = pixelScale
        self.pixelScale_effective = self.pixelScale / dithering

        if save in (None, 0):
            save = None
        else:
            save = 1
        self.save = save
        self.use_optPSF = use_optPSF

        # puilExplicit can be used to pass explicitly the image of the pupil
        # instead of creating it from the supplied parameters
        if pupilExplicit is None:
            pupilExplicit is False
        self.pupilExplicit = pupilExplicit

        if pupil_parameters is None:
            self.pupil_parameters = pupil_parameters
        else:
            self.pupil_parameters = pupil_parameters

        if use_pupil_parameters is None:
            self.use_pupil_parameters = use_pupil_parameters
        else:
            self.use_pupil_parameters = use_pupil_parameters
            self.args = args

        self.use_wf_grid = use_wf_grid
        self.zmax = zmaxInit

        self.simulation_00 = simulation_00
        if self.simulation_00:
            self.simulation_00 = 1

        self.extraZernike = extraZernike
        self.verbosity = verbosity
        self.double_sources = double_sources
        self.double_sources_positions_ratios = double_sources_positions_ratios

        self.test_run = test_run

        self.explicit_psf_position = explicit_psf_position
        self.use_only_chi = use_only_chi
        self.use_center_of_flux = use_center_of_flux
        self.flux = float(np.sum(image))
        self.fiber_id = fiber_id

        try:
            if not explicit_psf_position:
                self.explicit_psf_position = None
        except BaseException:
            pass

        self.PSF_DIRECTORY = PSF_DIRECTORY
        ############################################################
        if self.PSF_DIRECTORY is None:
            # names of default directories where I often work
            if socket.gethostname() == 'IapetusUSA':
                self.PSF_DIRECTORY = '/Volumes/Saturn_USA/PFS/'
            elif socket.gethostname() == 'pfsa-usr01-gb.subaru.nao.ac.jp' or \
                    socket.gethostname() == 'pfsa-usr02-gb.subaru.nao.ac.jp':
                self.PSF_DIRECTORY = '/work/ncaplar/'
            else:
                self.PSF_DIRECTORY = '/tigress/ncaplar/PFS/'

        if self.PSF_DIRECTORY is not None:
            self.TESTING_FOLDER = self.PSF_DIRECTORY + 'Testing/'
            self.TESTING_PUPIL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Pupil_Images/'
            self.TESTING_WAVEFRONT_IMAGES_FOLDER = self.TESTING_FOLDER + 'Wavefront_Images/'
            self.TESTING_FINAL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Final_Images/'

        if self.verbosity == 1:
            # check the versions of the most important libraries
            logging.info('np.__version__' + str(np.__version__))
            logging.info('scipy.__version__' + str(scipy.__version__))

    def initParams(
            self,
            z4Init=None,
            detFracInit=None,
            strutFracInit=None,
            focalPlanePositionInit=None,
            slitFracInit=None,
            slitFrac_dy_Init=None,
            wide_0Init=None,
            wide_23Init=None,
            wide_43Init=None,
            radiometricEffectInit=None,
            radiometricExponentInit=None,
            x_ilumInit=None,
            y_ilumInit=None,
            pixel_effectInit=None,
            backgroundInit=None,
            x_fiberInit=None,
            y_fiberInit=None,
            effective_ilum_radiusInit=None,
            frd_sigmaInit=None,
            frd_lorentz_factorInit=None,
            misalignInit=None,
            det_vertInit=None,
            slitHolder_frac_dxInit=None,
            grating_linesInit=None,
            scattering_slopeInit=None,
            scattering_amplitudeInit=None,
            fiber_rInit=None,
            fluxInit=None):
        """Initialize lmfit Parameters object.


        Allows to set up all parameters describing the pupil and
        Zernike parameter (up to z22) explicitly. If any value is not passed,
        it will be substituted by a default value (specified below).
        Parameters
        ----------
        zmax: `int`
            Total number of Zernike aberrations used (11 or 22)
            Possible to add more with extra_zernike parameter
        z4Init: `float`
            Initial Z4 aberration value in waves (that is 2*np.pi*wavelengths)
        # pupil parameters
        detFracInit: `float`
            Value determining how much of the exit pupil obscured by the
             central obscuration(detector)
        strutFracInit: `float`
             Value determining how much of the exit pupil is obscured
             by a single strut
        focalPlanePositionInit: (`float`, `float`)
            2-tuple for position of the central obscuration(detector)
            in the focal plane
        slitFracInit: `float`
              Value determining how much of the exit pupil is obscured by slit
        slitFrac_dy_Init: `float`
              Value determining what is the vertical position of the slit
               in the exit pupil
        # parameters dsecribing individual struts
        wide_0Init: `float`
                Parameter describing widening of the strut at 0 degrees
        wide_23Init: `float`
               Parameter describing widening of the top-left strut
        wide_34Init: `float`
            Parameter describing widening of the bottom-left strut
        #non-uniform illumination
        radiometricEffectInit: `float`
            parameter describing non-uniform illumination of the pupil
            (1-params['radiometricEffect']**2*r**2)**\
            (params['radiometricExponent']) [DEPRECATED]
        radiometricExponentInit: `float`
            parameter describing non-uniform illumination of the pupil
            (1-params['radiometricEffect']**2*r**2)\
            **(params['radiometricExponent'])
        x_ilumInit: `float`
            x-position of the center of illumination
            of the exit pupil [DEPRECATED]
        y_ilumInit: `float`
             y-position of the center of illumination
             of the exit pupil [DEPRECATED]
        # illumination due to fiber, parameters
        x_fiberInit: `float`
              position of the fiber misaligment in the x direction
        y_fiberInit: `float`
             position of the fiber misaligment in the y direction
        effective_ilum_radiusInit: `float`
            fraction of the maximal radius of the illumination
            of the exit pupil that is actually illuminated
        frd_sigma: `float`
                 sigma of Gaussian convolving only outer edge, mimicking FRD
        frd_lorentz_factor: `float`
            strength of the lorentzian factor describing wings
            of the pupil illumination
        misalign: `float`
            amount of misaligment in the illumination
        # further pupil parameters
        det_vert: `float
            multiplicative factor determining vertical size
            of the detector obscuration
        slitHolder_frac_dx: `float`
            dx position of slit holder
        # convolving (postprocessing) parameters
        grating_lines: `int`
             number of effective lines in the grating
        scattering_slopeInit: `float`
            slope of scattering
        scattering_amplitudeInit: `float`
            amplitude of scattering compared to optical PSF
        pixel_effectInit: `float`
            sigma describing charge diffusion effect [in units of 15 microns]
        fiber_rInit: `float`
             radius of perfect tophat fiber, as seen on the detector
               [in units of 15 microns]
        fluxInit: `float`
            total flux in generated image compared to input image
            (needs to be 1 or very close to 1)
        """
        if self.verbosity == 1:
            logging.info(' ')
            logging.info('Initializing ZernikeFitterPFS')
            logging.info('Verbosity parameter is: ' + str(self.verbosity))
            logging.info('Highest Zernike polynomial is (zmax): ' + str(self.zmax))

        params = lmfit.Parameters()
        # Zernike parameters
        z_array = []

        if z4Init is None:
            params.add('z4', 0.0)
        else:
            params.add('z4', z4Init)

        for i in range(5, self.zmax + 1):
            params.add('z{}'.format(i), 0.0)

        # pupil parameters
        if detFracInit is None:
            params.add('detFrac', 0.65)
        else:
            params.add('detFrac', detFracInit)

        if strutFracInit is None:
            params.add('strutFrac', 0.07)
        else:
            params.add('strutFrac', strutFracInit)

        if focalPlanePositionInit is None:
            params.add('dxFocal', 0.0)
            params.add('dyFocal', 0.0)
        else:
            params.add('dxFocal', focalPlanePositionInit[0])
            params.add('dyFocal', focalPlanePositionInit[1])

        if slitFracInit is None:
            params.add('slitFrac', 0.05)
        else:
            params.add('slitFrac', slitFracInit)

        if slitFrac_dy_Init is None:
            params.add('slitFrac_dy', 0)
        else:
            params.add('slitFrac_dy', slitFrac_dy_Init)

        # parameters dsecribing individual struts
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

        # non-uniform illumination
        if radiometricExponentInit is None:
            params.add('radiometricExponent', 0.25)
        else:
            params.add('radiometricExponent', radiometricExponentInit)

        if radiometricEffectInit is None:
            params.add('radiometricEffect', 0)
        else:
            params.add('radiometricEffect', radiometricEffectInit)

        if x_ilumInit is None:
            params.add('x_ilum', 1)
        else:
            params.add('x_ilum', x_ilumInit)

        if y_ilumInit is None:
            params.add('y_ilum', 1)
        else:
            params.add('y_ilum', y_ilumInit)

        # illumination due to fiber, parameters
        if x_ilumInit is None:
            params.add('x_fiber', 0)
        else:
            params.add('x_fiber', x_fiberInit)

        if y_fiberInit is None:
            params.add('y_fiber', 0)
        else:
            params.add('y_fiber', y_fiberInit)

        if effective_ilum_radiusInit is None:
            params.add('effective_ilum_radius', 0.9)
        else:
            params.add('effective_ilum_radius', effective_ilum_radiusInit)

        if frd_sigmaInit is None:
            params.add('frd_sigma', 0.02)
        else:
            params.add('frd_sigma', frd_sigmaInit)

        if frd_lorentz_factorInit is None:
            params.add('frd_lorentz_factor', 0.5)
        else:
            params.add('frd_lorentz_factor', frd_lorentz_factorInit)

        if misalignInit is None:
            params.add('misalign', 0)
        else:
            params.add('misalign', misalignInit)

        # further pupil parameters
        if det_vertInit is None:
            params.add('det_vert', 1)
        else:
            params.add('det_vert', det_vertInit)

        if slitHolder_frac_dxInit is None:
            params.add('slitHolder_frac_dx', 0)
        else:
            params.add('slitHolder_frac_dx', slitHolder_frac_dxInit)

        # convolving (postprocessing) parameters
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

        if pixel_effectInit is None:
            params.add('pixel_effect', 0.35)
        else:
            params.add('pixel_effect', pixel_effectInit)

        if fiber_rInit is None:
            params.add('fiber_r', 1.8)
        else:
            params.add('fiber_r', fiber_rInit)

        if fluxInit is None:
            params.add('flux', 1)
        else:
            params.add('flux', fluxInit)

        self.params = params
        self.optPsf = None
        self.z_array = z_array

    def constructModelImage_PFS_naturalResolution(
            self,
            params=None,
            shape=None,
            pixelScale=None,
            use_optPSF=None,
            extraZernike=None,
            return_intermediate_images=False):
        """Construct model image given the set of parameters
        Parameters
        ----------
        params : `lmfit.Parameters` object or python dictionary
            Parameters describing model; None to use self.params
        shape : `(int, int)`
            Shape for model image; None to use the shape of self.maskedImage
        pixelScale : `float`
            Pixel scale in arcseconds to use for model image;
            None to use self.pixelScale.
        use_optPSF : `bool`
            If True, use previously generated optical PSF,
            skip _getOptPsf_naturalResolution, and conduct only postprocessing
        extraZernike : `np.array`, (N,)
            Zernike parameteres beyond z22
        return_intermediate_images : `bool`
             If True, return intermediate images created during the run
             This is in order to help with debugging and inspect
             the images created during the process
        Return
        ----------
        (if not return_intermediate_images)
        optPsf_final : `np.array`, (N, N)
            Final model image
        psf_position : np.array, (2,)
            Position where image is centered
        (if return_intermediate_images)
        optPsf_final : `np.array`, (N, N)
            Final model image
        ilum : `np.array`, (N, N)
            Illumination array
        wf_grid_rot : `np.array`, (N, N)
            Wavefront array
        psf_position : np.array, (2,)
            Position where image is centered
        Notes
        ----------
        Calls _getOptPsf_naturalResolution and optPsf_postprocessing
        """
        if self.verbosity == 1:
            logging.info(' ')
            logging.info('Entering constructModelImage_PFS_naturalResolution')

        if params is None:
            params = self.params
        if shape is None:
            shape = self.image.shape
        if pixelScale is None:
            pixelScale = self.pixelScale
            logging.info('pixelScale_1573'+str(pixelScale))
        try:
            parameter_values = params.valuesdict()
        except AttributeError:
            parameter_values = params
        use_optPSF = self.use_optPSF

        if extraZernike is None:
            pass
        else:
            extraZernike = list(extraZernike)
        self.extraZernike = extraZernike

        # if you did not pass pure optical psf image, create one here
        if use_optPSF is None:
            # change outputs depending on if you want intermediate results
            if not return_intermediate_images:
                optPsf = self._getOptPsf_naturalResolution(
                    parameter_values, return_intermediate_images=return_intermediate_images)
            else:
                optPsf, ilum, wf_grid_rot = self._getOptPsf_naturalResolution(
                    parameter_values, return_intermediate_images=return_intermediate_images)
        else:
            # if you claimed to have supplied optical psf image,
            # but none is provided still create one
            if self.optPsf is None:
                if not return_intermediate_images:
                    optPsf = self._getOptPsf_naturalResolution(
                        parameter_values, return_intermediate_images=return_intermediate_images)
                else:
                    optPsf, ilum, wf_grid_rot = self._getOptPsf_naturalResolution(
                        parameter_values, return_intermediate_images=return_intermediate_images)
                self.optPsf = optPsf
            else:
                optPsf = self.optPsf

        # at the moment, no difference in optPsf_postprocessing depending on return_intermediate_images
        optPsf_final, psf_position = self._optPsf_postprocessing(
            optPsf, return_intermediate_images=return_intermediate_images)

        if self.save == 1:
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf', optPsf)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_final', optPsf_final)
        else:
            pass

        if not return_intermediate_images:
            return optPsf_final, psf_position
        if return_intermediate_images:
            return optPsf_final, ilum, wf_grid_rot, psf_position

        if self.verbosity == 1:
            logging.info('Finished with constructModelImage_PFS_naturalResolution')
            logging.info(' ')

    def _optPsf_postprocessing(self, optPsf, return_intermediate_images=False):
        """Apply postprocessing to the pure optical psf image
        Parameters
        ----------
        optPsf : `np.array`, (N, N)
            Optical image, only psf
        return_intermediate_images : `bool`
             If True, return intermediate images created during the run
             This is potentially in order to help with debugging and inspect
             the images created during the process
        Returns
        ----------
        (At the moment, the output is the same no matter what
         return_intermediate_images is, but there is a possibility
         to add intermediate outputs)
        optPsf_final : `np.array`, (N, N)
            Final model image
        psf_position : `np.array`, (2,)
            Position where the image is centered
        Notes
        ----------
        Takes optical psf and ``postprocesses`` it to generate final image.
        The algorithm first reduces the oversampling and cuts the central part
        of the image. This is done to speed up the calculations.
        Then we apply various effects that are separate from
        the pure optical PSF considerations.
        We then finish with the centering algorithm to move our created image
        to fit the input science image, invoking PSFPosition class.
        The effects we apply are
        1. scattered light
            function apply_scattered_light
        2. convolution with fiber
            function convolve_with_fiber
        3. CCD difusion
            function convolve_with_CCD_diffusion
        4. grating effects
            function convolve_with_grating
        5. centering
            via class PsfPosition
        """
        time_start_single = time.time()
        if self.verbosity == 1:
            logging.info(' ')
            logging.info('Entering optPsf_postprocessing')

        params = self.params
        shape = self.image.shape

        # all of the parameters for the creation of the image
        # very stupidly called ``v'' without any reason whatsoever
        param_values = params.valuesdict()

        # how much is my generated image oversampled compared to final image
        oversampling_original = (self.pixelScale_effective) / self.scale_ModelImage_PFS_naturalResolution

        if self.verbosity == 1:
            logging.info('Shape of optPsf: ' + str(optPsf.shape))
            logging.info('Value of oversampling_original: ' + str(oversampling_original))

        # determine the size, so that from the huge generated image we can cut out
        # only the central portion (1.4 times larger than the size of actual
        # image)
        size_of_central_cut = int(oversampling_original * self.image.shape[0] * 1.4)

        if size_of_central_cut > optPsf.shape[0]:
            # if larger than size of image, cut the image
            # fail if not enough space
            size_of_central_cut = optPsf.shape[0]
            if self.verbosity == 1:
                logging.info('size_of_central_cut modified to ' + str(size_of_central_cut))
            assert int(oversampling_original * self.image.shape[0] * 1.0) < optPsf.shape[0]

        assert size_of_central_cut <= optPsf.shape[0]
        if self.verbosity == 1:
            logging.info('size_of_central_cut: ' + str(size_of_central_cut))

        # cut part which you need to form the final image
        # set oversampling to 1 so you are not resizing the image, and dx=0 and
        # dy=0 so that you are not moving around, i.e., you are cutting the
        # central region
        optPsf_cut = PsfPosition.cut_Centroid_of_natural_resolution_image(
            image=optPsf, size_natural_resolution=size_of_central_cut + 1, oversampling=1, dx=0, dy=0)
        if self.verbosity == 1:
            logging.info('optPsf_cut.shape' + str(optPsf_cut.shape))

        # we want to reduce oversampling to be roughly around 10 to make things computationaly easier
        # if oversamplign_original is smaller than 20 (in case of dithered images),
        # make res coarser by factor of 2
        # otherwise set it to 11
        if oversampling_original < 20:
            oversampling = np.round(oversampling_original / 2)
        else:
            oversampling = 11
        if self.verbosity == 1:
            logging.info('oversampling:' + str(oversampling))

        # what will be the size of the image after you resize it to the from
        # ``oversampling_original'' to ``oversampling'' ratio
        size_of_optPsf_cut_downsampled = np.int(
            np.round(size_of_central_cut / (oversampling_original / oversampling)))
        if self.verbosity == 1:
            logging.info('size_of_optPsf_cut_downsampled: ' + str(size_of_optPsf_cut_downsampled))

        # make sure that optPsf_cut_downsampled is an array which has an odd size
        # - increase size by 1 if needed
        if (size_of_optPsf_cut_downsampled % 2) == 0:
            im1 = galsim.Image(optPsf_cut, copy=True, scale=1)
            im1.setCenter(0, 0)
            interpolated_image = galsim._InterpolatedImage(im1, x_interpolant=galsim.Lanczos(5, True))
            optPsf_cut_downsampled = interpolated_image.\
                drawImage(nx=size_of_optPsf_cut_downsampled + 1, ny=size_of_optPsf_cut_downsampled + 1,
                          scale=(oversampling_original / oversampling), method='no_pixel').array
        else:
            im1 = galsim.Image(optPsf_cut, copy=True, scale=1)
            im1.setCenter(0, 0)
            interpolated_image = galsim._InterpolatedImage(im1, x_interpolant=galsim.Lanczos(5, True))
            optPsf_cut_downsampled = interpolated_image.\
                drawImage(nx=size_of_optPsf_cut_downsampled, ny=size_of_optPsf_cut_downsampled,
                          scale=(oversampling_original / oversampling), method='no_pixel').array

        if self.verbosity == 1:
            logging.info('optPsf_cut_downsampled.shape: ' + str(optPsf_cut_downsampled.shape))

        # gives middle point of the image to used for calculations of scattered light
        # mid_point_of_optPsf_cut_downsampled = int(optPsf_cut_downsampled.shape[0] / 2)

        # gives the size of one pixel in optPsf_downsampled in microns
        # one physical pixel is 15 microns
        # effective size is 15 / dithering
        # size_of_pixels_in_optPsf_cut_downsampled = (15 / self.dithering) / oversampling

        # size of the created optical PSF images in microns
        # size_of_optPsf_cut_in_Microns = size_of_pixels_in_optPsf_cut_downsampled * \
        #     (optPsf_cut_downsampled.shape[0])
        # if self.verbosity == 1:
        #     logging.info('size_of_optPsf_cut_in_Microns: ' + str(size_of_optPsf_cut_in_Microns))

        if self.verbosity == 1:
            logging.info('Postprocessing parameters are:')
            logging.info(str(['grating_lines', 'scattering_slope', 'scattering_amplitude',
                              'pixel_effect', 'fiber_r']))
            logging.info(str([param_values['grating_lines'], param_values['scattering_slope'],
                              param_values['scattering_amplitude'], param_values['pixel_effect'],
                              param_values['fiber_r']]))

        ##########################################
        # 1. scattered light
        optPsf_cut_downsampled_scattered = self.apply_scattered_light(optPsf_cut_downsampled,
                                                                      oversampling,
                                                                      param_values['scattering_slope'],
                                                                      param_values['scattering_amplitude'],
                                                                      dithering=self.dithering)

        ##########################################
        # 2. convolution with fiber
        optPsf_cut_fiber_convolved = self.convolve_with_fiber(optPsf_cut_downsampled_scattered,
                                                              oversampling,
                                                              param_values['fiber_r'],
                                                              dithering=self.dithering)

        ##########################################
        # 3. CCD difusion
        optPsf_cut_pixel_response_convolved = self.convolve_with_CCD_diffusion(optPsf_cut_fiber_convolved,
                                                                               oversampling,
                                                                               param_values['pixel_effect'],
                                                                               dithering=self.dithering)

        ##########################################
        # 4. grating effects
        optPsf_cut_grating_convolved = self.convolve_with_grating(optPsf_cut_pixel_response_convolved,
                                                                  oversampling,
                                                                  self.wavelength,
                                                                  param_values['grating_lines'],
                                                                  dithering=self.dithering)

        ##########################################
        # 5. centering
        # This is the part which creates the final image

        # the algorithm  finds the best downsampling combination automatically
        if self.verbosity == 1:
            logging.info('Are we invoking double sources (1 or True if yes): ' + str(self.double_sources))
            logging.info('Double source position/ratio is:' + str(self.double_sources_positions_ratios))

        # initialize the class which does the centering -
        # TODO: the separation between the class and the main function in the class,
        # ``find_single_realization_min_cut'', is a bit blurry and unsatisfactory
        # this needs to be improved
        single_Psf_position = PsfPosition(optPsf_cut_grating_convolved,
                                          int(round(oversampling)),
                                          shape[0],
                                          simulation_00=self.simulation_00,
                                          verbosity=self.verbosity,
                                          save=self.save,
                                          PSF_DIRECTORY=self.PSF_DIRECTORY)
        time_end_single = time.time()
        if self.verbosity == 1:
            logging.info('Time for postprocessing up to single_Psf_position protocol is: '
                         + str(time_end_single - time_start_single))

        #  run the code for centering
        time_start_single = time.time()
        optPsf_final, psf_position =\
            single_Psf_position.find_single_realization_min_cut(optPsf_cut_grating_convolved,
                                                                int(round(oversampling)),
                                                                shape[0],
                                                                self.image,
                                                                self.image_var,
                                                                self.image_mask,
                                                                v_flux=param_values['flux'],
                                                                double_sources=self.double_sources,
                                                                double_sources_positions_ratios=  # noqa: E251
                                                                self.double_sources_positions_ratios,
                                                                verbosity=self.verbosity,
                                                                explicit_psf_position=  # noqa: E251
                                                                self.explicit_psf_position,
                                                                use_only_chi=self.use_only_chi,
                                                                use_center_of_flux=self.use_center_of_flux)
        time_end_single = time.time()

        if self.verbosity == 1:
            logging.info('Time for single_Psf_position protocol is '
                         + str(time_end_single - time_start_single))

        if self.verbosity == 1:
            logging.info('Sucesfully created optPsf_final')

        if self.save == 1:
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut', optPsf_cut)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut_downsampled', optPsf_cut_downsampled)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut_downsampled_scattered',
                    optPsf_cut_downsampled_scattered)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut_fiber_convolved',
                    optPsf_cut_fiber_convolved)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut_pixel_response_convolved',
                    optPsf_cut_pixel_response_convolved)
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_cut_grating_convolved',
                    optPsf_cut_grating_convolved)

        if self.verbosity == 1:
            logging.info('Finished with optPsf_postprocessing')
            logging.info(' ')

        # TODO: at the moment, the output is the same but there is a possibility to add intermediate outputs
        if not return_intermediate_images:
            return optPsf_final, psf_position

        if return_intermediate_images:
            return optPsf_final, psf_position

    def apply_scattered_light(self, image, oversampling,
                              scattering_slope, scattering_amplitude, dithering):
        """Add scattered light to optical psf
        Parameters
        ----------
        image : `np.array`, (N, N)
            input image
        oversampling: `int`
            how oversampled is `image`
        scattering_slope: `float`
            slope of the scattered light
        scattering_amplitude: `float`
            amplitude of the scattered light
        dithering: `int`
            dithering
        Returns
        ----------
        image_scattered : `np.array`, (N, N)
            image convolved with the fiber image
        Notes
        ----------
        Assumes that one physical pixel is 15 microns
        so that effective size of the pixels is 15 / dithering
        """
        size_of_pixels_in_image = (15 / self.dithering) / oversampling

        # size of the created optical PSF images in microns
        size_of_image_in_Microns = size_of_pixels_in_image * \
            (image.shape[0])

        # create grid to apply scattered light
        pointsx = np.linspace(-(size_of_image_in_Microns - size_of_pixels_in_image) / 2,
                              (size_of_image_in_Microns - size_of_pixels_in_image) / 2,
                              num=image.shape[0],
                              dtype=np.float32)
        pointsy = np.linspace(-(size_of_image_in_Microns - size_of_pixels_in_image) / 2,
                              (size_of_image_in_Microns - size_of_pixels_in_image) / 2,
                              num=image.shape[0]).astype(np.float32)
        xs, ys = np.meshgrid(pointsx, pointsy)
        r0 = np.sqrt((xs - 0) ** 2 + (ys - 0) ** 2) + .01

        # creating scattered light
        scattered_light_kernel = (r0**(-scattering_slope))
        scattered_light_kernel[r0 < 7.5] = 7.5**(-scattering_slope)
        scattered_light_kernel[scattered_light_kernel == np.inf] = 0
        scattered_light_kernel = scattered_light_kernel * \
            (scattering_amplitude) / (10 * np.max(scattered_light_kernel))

        # convolve the psf with the scattered light kernel to create scattered light component
        scattered_light = signal.fftconvolve(image, scattered_light_kernel, mode='same')

        # add back the scattering to the image
        image_scattered = image + scattered_light

        return image_scattered

    def convolve_with_fiber(self, image, oversampling, fiber_r, dithering):
        """Convolve optical psf with a fiber
        Parameters
        ----------
        image : `np.array`, (N, N)
            input image
        oversampling: `int`
            how oversampled is `image`
        fiber_r: `float`
            radius of the fiber in pixel units
        dithering: `int`
            dithering
        Returns
        ----------
        image_fiber_convolved : `np.array`, (N, N)
            image convolved with the fiber image
        Notes
        ----------
        """
        fiber = Tophat2DKernel(oversampling * fiber_r * dithering,
                               mode='oversample').array
        # create array with zeros with size of the current image, which we will
        # fill with fiber array in the middle
        fiber_padded = np.zeros_like(image, dtype=np.float32)
        mid_point_of_image = int(image.shape[0] / 2)
        fiber_array_size = fiber.shape[0]
        # fill the zeroes image with fiber here
        fiber_padded[int(mid_point_of_image - fiber_array_size / 2) + 1:
                     int(mid_point_of_image + fiber_array_size / 2) + 1,
                     int(mid_point_of_image - fiber_array_size / 2) + 1:
                     int(mid_point_of_image + fiber_array_size / 2) + 1] = fiber

        # convolve with the fiber
        image_fiber_convolved = signal.fftconvolve(image, fiber_padded, mode='same')
        return image_fiber_convolved

    def convolve_with_CCD_diffusion(self, image, oversampling, pixel_effect, dithering):
        """Convolve optical psf with a ccd diffusion effect
        Parameters
        ----------
        image : `np.array`, (N, N)
            input image
        oversampling: `int`
            how oversampled is `image`
        pixel_effect: `float`
            sigma of gaussian kernel convolving image
        dithering: `int`
            dithering
        Returns
        ----------
        image_pixel_response_convolved : `np.array`, (N, N)
            image convolved with the ccd diffusion kernel
        Notes
        ----------
        Pixels are not perfect detectors
        Charge diffusion in our optical CCDs, can be well described with a Gaussian
        sigma that is around 7 microns (Jim Gunn - private communication).
        This is controled in our code by @param 'pixel_effect'
        """
        pixel_gauss = Gaussian2DKernel(oversampling * pixel_effect * dithering).array.astype(np.float32)
        pixel_gauss_padded = np.pad(pixel_gauss, int((len(image) - len(pixel_gauss)) / 2),
                                    'constant', constant_values=0)

        # assert that gauss_padded array did not produce empty array
        assert np.sum(pixel_gauss_padded) > 0

        image_pixel_response_convolved = signal.fftconvolve(image, pixel_gauss_padded, mode='same')
        return image_pixel_response_convolved

    def convolve_with_grating(self, image, oversampling, wavelength, grating_lines, dithering):
        """Convolve optical psf with a grating effect
        Parameters
        ----------
        image : `np.array`, (N, N)
            input image
        oversampling: `int`
            how oversampled is `image`
        wavelength: `float`
            central wavelength of the spot
        grating_lines: `int`
            effective number of grating lines in the spectrograph
        dithering: `int`
            dithering

        Returns
        ----------
        image_grating_convolved : `np.array`, (N, N)
            image convolved with the grating effect

        Notes
        ----------
        This code assumes that 15 microns covers wavelength range of 0.07907 nm
        (assuming that 4300 pixels in real detector uniformly covers 340 nm)
        """
        grating_kernel = np.ones((image.shape[0], 1), dtype=np.float32)
        for i in range(len(grating_kernel)):
            grating_kernel[i] = Ifun16Ne((i - int(image.shape[0] / 2)) * 0.07907 * 10**-9
                                         / (dithering * oversampling) + wavelength * 10**-9,
                                         wavelength * 10**-9, grating_lines)
        grating_kernel = grating_kernel / np.sum(grating_kernel)

        image_grating_convolved = signal.fftconvolve(image, grating_kernel, mode='same')
        return image_grating_convolved

    def _get_Pupil(self):
        """Create an image of the pupil

        Parameters
        ----------
        params : `lmfit.Parameters` object or python dictionary
            Parameters describing the pupil model

        Returns
        ----------
        pupil : `pupil`
            Instance of class PFSPupilFactory

        Notes
        ----------
        Calls PFSPupilFactory class
        """
        if self.verbosity == 1:
            logging.info(' ')
            logging.info('Entering _get_Pupil (function inside ZernikeFitterPFS)')

        if self.verbosity == 1:
            logging.info('Size of the pupil (npix): ' + str(self.npix))

        Pupil_Image = PFSPupilFactory(
            pupilSize=self.diam_sic,
            npix=self.npix,
            input_angle=np.pi / 2,
            detFrac=self.params['detFrac'].value,
            strutFrac=self.params['strutFrac'].value,
            slitFrac=self.params['slitFrac'].value,
            slitFrac_dy=self.params['slitFrac_dy'].value,
            x_fiber=self.params['x_fiber'].value,
            y_fiber=self.params['y_fiber'].value,
            effective_ilum_radius=self.params['effective_ilum_radius'].value,
            frd_sigma=self.params['frd_sigma'].value,  # noqa: E
            frd_lorentz_factor=self.params['frd_lorentz_factor'].value,
            det_vert=self.params['det_vert'].value,
            slitHolder_frac_dx=self.params['slitHolder_frac_dx'].value,
            wide_0=self.params['wide_0'].value,
            wide_23=self.params['wide_23'].value,
            wide_43=self.params['wide_43'].value,
            misalign=self.params['misalign'].value,
            verbosity=self.verbosity,
            fiber_id=self.fiber_id)

        point = [self.params['dxFocal'].value, self.params['dyFocal'].value]  # noqa: E
        pupil = Pupil_Image.getPupil(point)

        if self.save == 1:
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'pupil.illuminated',
                    pupil.illuminated.astype(np.float32))

        if self.verbosity == 1:
            logging.info('Finished with _get_Pupil')

        return pupil

    def _getOptPsf_naturalResolution(self, params, return_intermediate_images=False):
        """Returns optical PSF, given the initialized parameters

        Parameters
        ----------
        params : `lmfit.Parameters` object or python dictionary
            Parameters descrubing model
        return_intermediate_images : `bool`
             If True, return intermediate images created during the run
             This is in order to help with debugging and inspect
             the images created during the process

        Returns
        ----------
        (if not return_intermediate_images)
        img_apod : `np.array`
            Psf image, only optical components considred
        (if return_intermediate_images)
            # return the image, pupil, illumination applied to the pupil
        img_apod : `np.array`
            Psf image, only optical components considred
        ilum : `np.array`
            Image showing the illumination of the pupil
        wf_grid_rot : `np.array`
            Image showing the wavefront across the pupil

        Notes
        ----------
        called by constructModelImage_PFS_naturalResolution

        Illumination is handled in an unsatifactory manner at the moment.
        The complex process is as follows:
        1. get illumination from pupil.illuminated.
        2. This gets passed to galsim.Aperture class
        3. It gets extracted, !unchanged!, as ilum = aper.illuminated...
        4. Apply radiometric effect (change between exit and entrance pupil)
        and rename to ilum_radiometric - as we are currently not considering
        this effect this is again unchanged!!!
        5. Apply apodization - rename to ilum_radiometric_apodized
        6. Create additional array like ilum_radiometric_apodized but has
        boolean values -> ilum_radiometric_apodized_bool
        """

        if self.verbosity == 1:
            logging.info(' ')
            logging.info('Entering _getOptPsf_naturalResolution')

        ################################################################################
        # pupil and illumination of the pupil
        ################################################################################
        time_start_single_1 = time.time()
        if self.verbosity == 1:
            logging.info('use_pupil_parameters: ' + str(self.use_pupil_parameters))
            logging.info('pupil_parameters if you are explicity passing use_pupil_parameters: '
                         + str(self.pupil_parameters))

        # parmeters ``i'' just to precision in the construction of ``pupil_parameters'' array
        # not sure why linter is complaining here with
        # ('...'.format(...) has unused arguments at position(s): 0)
        i = 4
        if self.use_pupil_parameters is None:
            pupil_parameters = np.array([params['detFrac'.format(i)],  # noqa: E
                                         params['strutFrac'.format(i)],  # noqa: E
                                         params['dxFocal'.format(i)],  # noqa: E
                                         params['dyFocal'.format(i)],  # noqa: E
                                         params['slitFrac'.format(i)],  # noqa: E
                                         params['slitFrac_dy'.format(i)],  # noqa: E
                                         params['x_fiber'.format(i)],  # noqa: E
                                         params['y_fiber'.format(i)],  # noqa: E
                                         params['effective_ilum_radius'.format(i)],  # noqa: E
                                         params['frd_sigma'.format(i)],  # noqa: E
                                         params['frd_lorentz_factor'.format(i)],  # noqa: E
                                         params['det_vert'.format(i)],  # noqa: E
                                         params['slitHolder_frac_dx'.format(i)],  # noqa: E
                                         params['wide_0'.format(i)],  # noqa: E
                                         params['wide_23'.format(i)],  # noqa: E
                                         params['wide_43'.format(i)],  # noqa: E
                                         params['misalign'.format(i)]])  # noqa: E
            self.pupil_parameters = pupil_parameters
        else:
            pupil_parameters = np.array(self.pupil_parameters)

        diam_sic = self.diam_sic

        if self.verbosity == 1:
            logging.info(['detFrac', 'strutFrac', 'dxFocal', 'dyFocal', 'slitFrac', 'slitFrac_dy'])
            logging.info(['x_fiber', 'y_fiber', 'effective_ilum_radius', 'frd_sigma',
                          'frd_lorentz_factor', 'det_vert', 'slitHolder_frac_dx'])
            logging.info(['wide_0', 'wide_23', 'wide_43', 'misalign'])
            logging.info('set of pupil_parameters I. : ' + str([params['detFrac'], params['strutFrac'],
                                                                params['dxFocal'], params['dyFocal'],
                                                                params['slitFrac'], params['slitFrac_dy']]))
            logging.info('set of pupil_parameters II. : ' + str([params['x_fiber'], params['y_fiber'],
                                                                 params['effective_ilum_radius'],
                                                                 params['slitHolder_frac_dx'],
                                                                 params['frd_lorentz_factor'],
                                                                 params['det_vert'],
                                                                 params['slitHolder_frac_dx']]))
            logging.info('set of pupil_parameters III. : ' + str([params['wide_0'], params['wide_23'],
                                                                  params['wide_43'], params['misalign']]))
        time_start_single_2 = time.time()

        # initialize galsim.Aperature class
        pupil = self._get_Pupil()
        aper = galsim.Aperture(
            diam=pupil.size,
            pupil_plane_im=pupil.illuminated.astype(np.float32),
            pupil_plane_scale=pupil.scale,
            pupil_plane_size=None)

        if self.verbosity == 1:
            if self.pupilExplicit is None:
                logging.info('Requested pupil size is (pupil.size) [m]: ' + str(pupil.size))
                logging.info('One pixel has size of (pupil.scale) [m]: ' + str(pupil.scale))
                logging.info('Requested pupil has so many pixels (pupil_plane_im): '
                             + str(pupil.illuminated.astype(np.int16).shape))
            else:
                logging.info('Supplied pupil size is (diam_sic) [m]: ' + str(self.diam_sic))
                logging.info('One pixel has size of (diam_sic/npix) [m]: ' + str(self.diam_sic / self.npix))
                logging.info('Requested pupil has so many pixels (pupilExplicit): '
                             + str(self.pupilExplicit.shape))

        time_end_single_2 = time.time()
        if self.verbosity == 1:
            logging.info('Time for _get_Pupil function is ' + str(time_end_single_2 - time_start_single_2))

        time_start_single_3 = time.time()
        # create array with pixels=1 if the area is illuminated and 0 if it is obscured
        ilum = np.array(aper.illuminated, dtype=np.float32)
        assert np.sum(ilum) > 0, str(self.pupil_parameters)

        # gives size of the illuminated image
        lower_limit_of_ilum = int(ilum.shape[0] / 2 - self.npix / 2)
        higher_limit_of_ilum = int(ilum.shape[0] / 2 + self.npix / 2)
        if self.verbosity == 1:
            logging.info('lower_limit_of_ilum: ' + str(lower_limit_of_ilum))
            logging.info('higher_limit_of_ilum: ' + str(higher_limit_of_ilum))

        if self.pupilExplicit is None:
            ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                 lower_limit_of_ilum:higher_limit_of_ilum] = ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                                                                  lower_limit_of_ilum:higher_limit_of_ilum] *\
                pupil.illuminated
        else:
            ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                 lower_limit_of_ilum:higher_limit_of_ilum] = ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                                                                  lower_limit_of_ilum:higher_limit_of_ilum] *\
                self.pupilExplicit.astype(np.float32)

        if self.verbosity == 1:
            logging.info('Size after padding zeros to 2x size'
                         + 'and extra padding to get size suitable for FFT: '
                         + str(ilum.shape))

        # maximum extent of pupil image in units of radius of the pupil, needed for next step
        size_of_ilum_in_units_of_radius = ilum.shape[0] / self.npix

        if self.verbosity == 1:
            logging.info('size_of_ilum_in_units_of_radius: ' + str(size_of_ilum_in_units_of_radius))

        # do not caculate the ``radiometric effect (difference between entrance and exit pupil)
        # if parameters are too small to make any difference
        # if that is the case just declare the ``ilum_radiometric'' to be the same as ilum
        # i.e., the illumination of the exit pupil is the same as the illumination of the entrance pupil
        if params['radiometricExponent'] < 0.01 or params['radiometricEffect'] < 0.01:
            if self.verbosity == 1:
                logging.info('skiping ``radiometric effect\'\' ')
            ilum_radiometric = ilum
        else:
            if self.verbosity == 1:
                logging.info('radiometric parameters are: ')
                logging.info('x_ilum,y_ilum,radiometricEffect,radiometricExponent'
                             + str([params['x_ilum'], params['y_ilum'],
                                    params['radiometricEffect'], params['radiometricExponent']]))

            # add the change of flux between the entrance and exit pupil
            # end product is radiometricEffectArray
            points = np.linspace(-size_of_ilum_in_units_of_radius,
                                 size_of_ilum_in_units_of_radius, num=ilum.shape[0])
            xs, ys = np.meshgrid(points, points)
            _radius_coordinate = np.sqrt(
                (xs - params['x_ilum'] * params['dxFocal']) ** 2
                + (ys - params['y_ilum'] * params['dyFocal']) ** 2)

            # change in v_0.14
            # ilumination to which radiometric effet has been applied, describing
            # difference betwen entrance and exit pupil
            radiometricEffectArray = (1 + params['radiometricEffect']
                                      * _radius_coordinate**2)**(-params['radiometricExponent'])
            ilum_radiometric = np.nan_to_num(radiometricEffectArray * ilum, 0)

        # this is where you can introduce some apodization in the pupil image by using the line below
        # the apodization sigma is set to that in focus it is at 0.75
        # for larger images, scale according to the size of the input image which is to be FFT-ed
        # 0.75 is an arbitrary number
        apodization_sigma = ((len(ilum_radiometric)) / 1158)**0.875 * 0.75
        # apodization_sigma=0.75
        time_start_single_4 = time.time()

        # old code where I applied Gaussian to the whole ilum image
        # ilum_radiometric_apodized = gaussian_filter(ilum_radiometric, sigma=apodization_sigma)

        # cut out central region, apply Gaussian on the center region and return to the full size image
        # done to spped up the calculation
        # noqa: E128 in order to keep informative names
        ilum_radiometric_center_region =\
            ilum_radiometric[(lower_limit_of_ilum - int(np.ceil(3 * apodization_sigma))):
                             (higher_limit_of_ilum + int(np.ceil(3 * apodization_sigma))),
                             (lower_limit_of_ilum - int(np.ceil(3 * apodization_sigma))):
                             (higher_limit_of_ilum + int(np.ceil(3 * apodization_sigma)))]

        ilum_radiometric_center_region_apodized = gaussian_filter(
            ilum_radiometric_center_region, sigma=apodization_sigma)

        ilum_radiometric_apodized = np.copy(ilum_radiometric)
        ilum_radiometric_apodized[(lower_limit_of_ilum - int(np.ceil(3 * apodization_sigma))):
                                  (higher_limit_of_ilum + int(np.ceil(3 * apodization_sigma))),
                                  (lower_limit_of_ilum - int(np.ceil(3 * apodization_sigma))):
                                  (higher_limit_of_ilum + int(np.ceil(3 * apodization_sigma)))] =\
        ilum_radiometric_center_region_apodized  # noqa E:122

        time_end_single_4 = time.time()
        if self.verbosity == 1:
            logging.info('Time to apodize the pupil: ' + str(time_end_single_4 - time_start_single_4))
            logging.info('type(ilum_radiometric_apodized)' + str(type(ilum_radiometric_apodized[0][0])))
        # put pixels for which amplitude is less than 0.01 to 0
        r_ilum_pre = np.copy(ilum_radiometric_apodized)
        r_ilum_pre[ilum_radiometric_apodized > 0.01] = 1
        r_ilum_pre[ilum_radiometric_apodized < 0.01] = 0
        ilum_radiometric_apodized_bool = r_ilum_pre.astype(bool)

        # manual creation of aper.u and aper.v (mimicking steps which were automatically done in galsim)
        # this gives position information about each point in the exit pupil so we can apply wavefront to it

        # aperu_manual=[]
        # for i in range(len(ilum_radiometric_apodized_bool)):
        #    aperu_manual.append(np.linspace(-diam_sic*(size_of_ilum_in_units_of_radius/2),
        # diam_sic*(size_of_ilum_in_units_of_radius/2),len(ilum_radiometric_apodized_bool), endpoint=True))
        single_line_aperu_manual = np.linspace(-diam_sic * (size_of_ilum_in_units_of_radius / 2), diam_sic * (
            size_of_ilum_in_units_of_radius / 2), len(ilum_radiometric_apodized_bool), endpoint=True)
        aperu_manual = np.tile(
            single_line_aperu_manual,
            len(single_line_aperu_manual)).reshape(
            len(single_line_aperu_manual),
            len(single_line_aperu_manual))

        # full grid
        # u_manual=np.array(aperu_manual)
        u_manual = aperu_manual
        v_manual = np.transpose(aperu_manual)

        # select only parts of the grid that are actually illuminated
        u = u_manual[ilum_radiometric_apodized_bool]
        v = v_manual[ilum_radiometric_apodized_bool]

        time_end_single_3 = time.time()
        if self.verbosity == 1:
            logging.info('Time for postprocessing pupil after _get_Pupil '
                         + str(time_end_single_3 - time_start_single_3))

        time_end_single_1 = time.time()
        if self.verbosity == 1:
            logging.info('Time for pupil and illumination calculation is '
                         + str(time_end_single_1 - time_start_single_1))

        ################################################################################
        # wavefront
        ################################################################################
        # create wavefront across the exit pupil

        time_start_single = time.time()
        if self.verbosity == 1:
            logging.info('')
            logging.info('Starting creation of wavefront')

        aberrations_init = [0.0, 0, 0.0, 0.0]
        aberrations = aberrations_init
        # list of aberrations where we set z4, z11, z22 etc...
        # This is only for testing purposes to study behaviour of non-focus terms
        aberrations_0 = list(np.copy(aberrations_init))
        for i in range(4, self.zmax + 1):
            aberrations.append(params['z{}'.format(i)])
            if i in [4, 11, 22]:
                aberrations_0.append(0)
            else:
                aberrations_0.append(params['z{}'.format(i)])

        # if you have passed abberation above Zernike 22, join them with lower
        # order abberations here
        if self.extraZernike is None:
            pass
        else:
            aberrations_extended = np.concatenate((aberrations, self.extraZernike), axis=0)

        if self.verbosity == 1:
            logging.info('diam_sic [m]: ' + str(diam_sic))
            logging.info('aberrations: ' + str(aberrations))
            logging.info('aberrations moved to z4=0: ' + str(aberrations_0))
            logging.info('aberrations extra: ' + str(self.extraZernike))
            logging.info('wavelength [nm]: ' + str(self.wavelength))

        if self.extraZernike is None:
            optics_screen = galsim.phase_screens.OpticalScreen(
                diam=diam_sic, aberrations=aberrations, lam_0=self.wavelength)
            if self.save == 1:
                # only create fake with abberations 0 if we are going to save i.e., if we
                # presenting the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(
                    diam=diam_sic, aberrations=aberrations_0, lam_0=self.wavelength)
        else:
            optics_screen = galsim.phase_screens.OpticalScreen(
                diam=diam_sic, aberrations=aberrations_extended, lam_0=self.wavelength)
            if self.save == 1:
                # only create fake with abberations 0 if we are going to save i.e., if we
                # presenting the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(
                    diam=diam_sic, aberrations=aberrations_0, lam_0=self.wavelength)

        screens = galsim.PhaseScreenList(optics_screen)
        if self.save == 1:
            # only create fake with abberations 0 if we are going to save i.e., if we presenting the results
            screens_fake_0 = galsim.PhaseScreenList(optics_screen_fake_0)

        time_end_single = time.time()

        ################################################################################
        # combining pupil illumination and wavefront
        ################################################################################

        # apply wavefront to the array describing illumination
        # logging.info(self.use_wf_grid)

        if self.use_wf_grid is None:
            wf = screens.wavefront(u, v, None, 0)
            if self.save == 1:
                wf_full = screens.wavefront(u_manual, v_manual, None, 0)
            wf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.float32)
            wf_grid[ilum_radiometric_apodized_bool] = (wf / self.wavelength)
            wf_grid_rot = wf_grid
        else:
            # if you want to pass an explit wavefront, it goes here
            wf_grid = self.use_wf_grid
            wf_grid_rot = wf_grid

        if self.save == 1:
            # only create fake images with abberations set to 0 if we are going to save
            # i.e., if we are testing the results
            if self.verbosity == 1:
                logging.info('creating wf_full_fake_0')
            wf_full_fake_0 = screens_fake_0.wavefront(u_manual, v_manual, None, 0)

        # exponential of the wavefront - ilumination of the pupil enters here as ilum_radiometric_apodized
        expwf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.complex64)
        expwf_grid[ilum_radiometric_apodized_bool] =\
            ilum_radiometric_apodized[ilum_radiometric_apodized_bool] *\
            np.exp(2j * np.pi * wf_grid_rot[ilum_radiometric_apodized_bool])

        if self.verbosity == 1:
            logging.info('Time for wavefront and wavefront/pupil combining is '
                         + str(time_end_single - time_start_single))

        ################################################################################
        # execute the FFT
        ################################################################################

        time_start_single = time.time()
        ftexpwf = np.fft.fftshift(scipy.fftpack.fft2(np.fft.fftshift(expwf_grid)))
        img_apod = np.abs(ftexpwf)**2
        time_end_single = time.time()
        if self.verbosity == 1:
            logging.info('Time for FFT is ' + str(time_end_single - time_start_single))
        ######################################################################

        # size in arcseconds of the image generated by the code
        scale_ModelImage_PFS_naturalResolution = sky_scale(
            size_of_ilum_in_units_of_radius * self.diam_sic, self.wavelength)
        self.scale_ModelImage_PFS_naturalResolution = scale_ModelImage_PFS_naturalResolution

        if self.save == 1:
            if socket.gethostname() == 'IapetusUSA' or socket.gethostname() == 'tiger2-sumire.princeton.edu' \
                or socket.gethostname() == 'pfsa-usr01-gb.subaru.nao.ac.jp' or \
                    socket.gethostname() == 'pfsa-usr02-gb.subaru.nao.ac.jp':
                np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'aperilluminated', aper.illuminated)
                np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum', ilum)
                np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum_radiometric', ilum_radiometric)
                np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum_radiometric_apodized',
                        ilum_radiometric_apodized)
                np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum_radiometric_apodized_bool',
                        ilum_radiometric_apodized_bool)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'u_manual', u_manual)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'v_manual', v_manual)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'u', u)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'v', v)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'wf_grid', wf_grid)
                if self.use_wf_grid is None:
                    np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'wf_full', wf_full)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'wf_full_fake_0', wf_full_fake_0)
                np.save(self.TESTING_WAVEFRONT_IMAGES_FOLDER + 'expwf_grid', expwf_grid)

        if self.verbosity == 1:
            logging.info('Finished with _getOptPsf_naturalResolution')
            logging.info('Finished with _getOptPsf_naturalResolution')
            logging.info(' ')

        if not return_intermediate_images:
            return img_apod
        if return_intermediate_images:
            return img_apod, ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                                  lower_limit_of_ilum:higher_limit_of_ilum], wf_grid_rot


class LN_PFS_multi_same_spot(object):
    """!Class to compute quality of the multiple donut images,
    of the same spot taken at different defocuses

    Calls class LN_PFS_single, for example:
        model = LN_PFS_single(sci_image,var_image,pupil_parameters = pupil_parameters,
                          use_pupil_parameters=None,zmax=zmax,save=1)
        def model_return(allparameters_proposal):
            return model(allparameters_proposal,return_Image=True)

    Called by class Tokovinin_multi
    """

    def __init__(
            self,
            list_of_sci_images,
            list_of_var_images,
            list_of_mask_images=None,
            wavelength=None,
            dithering=None,
            save=None,
            verbosity=None,
            pupil_parameters=None,
            use_pupil_parameters=None,
            use_optPSF=None,
            list_of_wf_grid=None,
            zmax=None,
            extraZernike=None,
            pupilExplicit=None,
            simulation_00=None,
            double_sources=None,
            double_sources_positions_ratios=None,
            npix=None,
            list_of_defocuses=None,
            fit_for_flux=True,
            test_run=False,
            list_of_psf_positions=None,
            use_center_of_flux=False):
        """
        @param list_of_sci_images                       list of science images, list of 2d array
        @param list_of_var_images                       list of variance images, 2d arrays,
                                                        which are the same size as sci_image
        @param list_of_mask_images                      list of mask images, 2d arrays,
                                                        which are the same size as sci_image
        @param dithering                                dithering, 1=normal, 2=two times higher resolution,
                                                        3=not supported
        @param save                                     save intermediate result in the process
                                                        (set value at 1 for saving)
        @param verbosity                                verbosity of the process
                                                        (set value at 1 for full output)

        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF

        @param zmax                                     largest Zernike order used
                                                        (11 or 22, or larger than 22)
        @param extraZernike                             array consisting of higher order zernike
                                                        (if using higher order than 22)
        @param pupilExplicit

        @param simulation_00                            resulting image will be centered with optical center
                                                        in the center of the image
                                                        and not fitted acorrding to the sci_image
        @param double_sources                           1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /        arrray with parameters describing relative position\
                                                        and relative flux of the secondary source(s)
        @param npxix                                    size of the pupil (1536 reccomended)
        @param list_of_defocuses                        list of defocuses at which images are taken
                                                        (float or string?)

        @param fit_for_flux                             automatically fit for the best flux level
                                                        that minimizes the chi**2
        @param test_run                                 if True, skips the creation of model and
                                                        return science image - useful for testing
                                                        interaction of outputs of the module
                                                        in broader setting quickly
        @param explicit_psf_position                    gives position of the opt_psf
        """

        if verbosity is None:
            verbosity = 0

        if use_pupil_parameters is not None:
            assert pupil_parameters is not None

        # logging.info('double_sources in module: ' + str(double_sources))
        # logging.info('double_sources_positions_ratios in module: ' + str(double_sources_positions_ratios))
        # logging.info('list_of_psf_positions in LN_PFS_multi_same_spot '+str(list_of_psf_positions))
        if double_sources is not None and bool(double_sources) is not False:
            assert np.sum(np.abs(double_sources_positions_ratios)) > 0

        if zmax is None:
            zmax = 11

        if zmax == 11:
            self.columns = [
                'z4',
                'z5',
                'z6',
                'z7',
                'z8',
                'z9',
                'z10',
                'z11',
                'detFrac',
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
                'effective_ilum_radius',
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
        if zmax >= 22:
            self.columns = [
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
                'detFrac',
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
                'effective_ilum_radius',
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

        self.list_of_sci_images = list_of_sci_images
        self.list_of_var_images = list_of_var_images

        if list_of_mask_images is None:
            list_of_mask_images = []
            for i in range(len(list_of_sci_images)):
                mask_image = np.zeros(list_of_sci_images[i].shape)
                list_of_mask_images.append(mask_image)

        self.list_of_mask_images = list_of_mask_images

        # self.mask_image=mask_image
        # self.sci_image=sci_image
        # self.var_image=var_image
        self.wavelength = wavelength
        self.dithering = dithering
        self.save = save
        self.pupil_parameters = pupil_parameters
        self.use_pupil_parameters = use_pupil_parameters
        self.use_optPSF = use_optPSF
        self.pupilExplicit = pupilExplicit
        self.simulation_00 = simulation_00
        self.zmax = zmax
        self.extraZernike = extraZernike
        self.verbosity = verbosity
        self.double_sources = double_sources
        self.double_sources_positions_ratios = double_sources_positions_ratios
        self.npix = npix
        self.fit_for_flux = fit_for_flux
        self.list_of_defocuses = list_of_defocuses
        self.test_run = test_run
        if list_of_psf_positions is None:
            list_of_psf_positions = [None] * len(list_of_sci_images)
        self.list_of_psf_positions = list_of_psf_positions
        if list_of_wf_grid is None:
            list_of_wf_grid = [None] * len(list_of_sci_images)
        self.list_of_wf_grid = list_of_wf_grid

        # self.use_only_chi=use_only_chi
        self.use_center_of_flux = use_center_of_flux

    def move_parametrizations_from_1d_to_2d(self, allparameters_parametrizations_1d, zmax=None):
        """Reshape the parametrization from 1d array to 2d array

        Parameters
        ----------
        allparameters_parametrizations_1d : `np.array`
            Parametriztion to be reshaped
        zmax : `int`
            Highest order of Zernike parameters applied

        Returns
        ----------
        allparameters_parametrizations_2d : `np.array`
            Parametrization in 2d form
        """

        # 22 parameters has len of 61
        if zmax is None:
            zmax = int((len(allparameters_parametrizations_1d) - 61) / 2 + 22)

        assert len(allparameters_parametrizations_1d.shape) == 1

        z_parametrizations = allparameters_parametrizations_1d[:19 * 2].reshape(19, 2)
        g_parametrizations =\
            np.transpose(np.vstack((np.zeros(len(allparameters_parametrizations_1d[19 * 2:19 * 2 + 23])),
                                    allparameters_parametrizations_1d[19 * 2:19 * 2 + 23])))

        if zmax > 22:
            extra_Zernike_parameters_number = zmax - 22
            z_extra_parametrizations = allparameters_parametrizations_1d[19 * 2 + 23:].reshape(
                extra_Zernike_parameters_number, 2)

        if zmax <= 22:
            allparameters_parametrizations_2d = np.vstack((z_parametrizations, g_parametrizations))
        if zmax > 22:
            allparameters_parametrizations_2d = np.vstack(
                (z_parametrizations, g_parametrizations, z_extra_parametrizations))

        # logging.info('allparameters_parametrizations_2d[41]: '+ str(allparameters_parametrizations_2d[41]))
        # assert allparameters_parametrizations_2d[41][1] >= 0.98
        # assert allparameters_parametrizations_2d[41][1] <= 1.02

        return allparameters_parametrizations_2d

    def create_list_of_allparameters(self, allparameters_parametrizations, list_of_defocuses=None, zmax=None):
        """Create list of parameters at given defocuses

        Given the parametrizations (in either 1d or 2d ),
        create list_of_allparameters to be used in analysis of single images

        Parameters
        ----------
        allparameters_parametrizations : `np.array`
            Input parametrizations
        list_of_defocuses : `list`
            List contaning the strings of defoucses at which we are searching for parameters
        zmax : `int`
            Highest order of Zernike parameters applied

        Returns
        ----------
        list_of_allparameters : `list`
            List contaning the parameters for each defocus position
        """

        # logging.info('allparameters_parametrizations '+str(allparameters_parametrizations))

        if zmax is None:
            zmax = self.zmax

        # if you have passed parameterization in 1d, move to 2d
        # logging.info("allparameters_parametrizations.type: "+str(type(allparameters_parametrizations)))
        # logging.info("allparameters_parametrizations.len: "+str(+len(allparameters_parametrizations)))
        # logging.info("allparameters_parametrizations.shape: "+str(allparameters_parametrizations.shape))
        if len(allparameters_parametrizations.shape) == 1:
            allparameters_parametrizations = self.move_parametrizations_from_1d_to_2d(
                allparameters_parametrizations)

        list_of_allparameters = []

        # if this is only a single image, just return the input
        if list_of_defocuses is None:
            return allparameters_parametrizations
        else:
            list_of_defocuses_int = self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses)
            # logging.info(list_of_defocuses_int)
            # go through the list of defocuses, and create the allparameters array for each defocus
            for i in range(len(list_of_defocuses)):
                list_of_allparameters.append(
                    self.create_allparameters_single(
                        list_of_defocuses_int[i],
                        allparameters_parametrizations,
                        zmax))

            # logging.info(list_of_allparameters)

            return list_of_allparameters

    def value_at_defocus(self, mm, a, b=None):
        """Calculate linear fit to a value at a given defocus (in mm)

        Parameters
        ----------
        mm : `float`
            Slit defocus in mm
        a : `float`
            Linear parameter
        b :
            Contstant offset

        Returns
        ----------
            : `float`
            Result of linear fit
        """

        if b is None:
            return a
        else:
            return a * mm + b

    def create_allparameters_single(self, mm, array_of_polyfit_1_parameterizations, zmax=None):
        """ Given the defous, transform parametrization into parameters for that defocus

        This function ransforms 1d array of ``parametrizations'' into ``parameters''m i.e.,
        into form acceptable for creating single images.
        This is a  workhorse function used by function create_list_of_allparameters

        Parameters
        ----------
        mm : `float`
            defocus of the slit
        array_of_polyfit_1_parameterizations : `np.array`
            parametrization for linear fit for the parameters as a function of focus
        zmax : `int`
            Highest order of Zernike parameters applied

        Returns
        ----------
        allparameters_proposal_single : `np.array`
            Parameters that can be used to create single image
        """

        if zmax is None:
            # if len is 42, the zmax is 22
            zmax = array_of_polyfit_1_parameterizations.shape[0] - 42 + 22
            if zmax > 22:
                extra_Zernike_parameters_number = zmax - 22
        else:
            extra_Zernike_parameters_number = zmax - 22

        # for single case, up to z11
        if zmax == 11:
            z_parametrizations = array_of_polyfit_1_parameterizations[:8]
            g_parametrizations = array_of_polyfit_1_parameterizations[8:]

            allparameters_proposal_single = np.zeros((8 + len(g_parametrizations)))

            for i in range(0, 8, 1):
                allparameters_proposal_single[i] = self.value_at_defocus(
                    mm, z_parametrizations[i][0], z_parametrizations[i][1])

            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[i + 8] = g_parametrizations[i][1]

        if zmax >= 22:
            z_parametrizations = array_of_polyfit_1_parameterizations[:19]
            g_parametrizations = array_of_polyfit_1_parameterizations[19:19 + 23]

            if extra_Zernike_parameters_number > 0:
                z_extra_parametrizations = array_of_polyfit_1_parameterizations[42:]

            allparameters_proposal_single = np.zeros(
                (19 + len(g_parametrizations) + extra_Zernike_parameters_number))

            for i in range(0, 19, 1):
                # logging.info(str([i,mm,z_parametrizations[i]]))
                allparameters_proposal_single[i] = self.value_at_defocus(
                    mm, z_parametrizations[i][0], z_parametrizations[i][1])

            for i in range(len(g_parametrizations)):
                allparameters_proposal_single[19 + i] = g_parametrizations[i][1]

            for i in range(0, extra_Zernike_parameters_number, 1):
                # logging.info(str([i,mm,z_parametrizations[i]]))
                allparameters_proposal_single[19 + len(g_parametrizations) + i] = self.value_at_defocus(
                    mm, z_extra_parametrizations[i][0], z_extra_parametrizations[i][1])

        return allparameters_proposal_single

    def transform_list_of_defocuses_from_str_to_float(self, list_of_defocuses):
        """Transfroms list_of_defocuses from strings to float values

        Parameters
        ----------
        list_of_defocuses : `list`
            list of defocuses in string form (e.g., [m4,m25,0,p15,p4])

        Returns
        ----------
        list_of_defocuses_float : `list`
            list of defocuses in float form
        """

        list_of_defocuses_float = []
        for i in range(len(list_of_defocuses)):
            if list_of_defocuses[i][0] == '0':
                list_of_defocuses_float.append(0)
            else:
                if list_of_defocuses[i][0] == 'm':
                    sign = -1
                if list_of_defocuses[i][0] == 'p':
                    sign = +1
                if len(list_of_defocuses[i]) == 2:
                    list_of_defocuses_float.append(sign * float(list_of_defocuses[i][1:]))
                else:
                    list_of_defocuses_float.append(sign * float(list_of_defocuses[i][1:]) / 10)

        return list_of_defocuses_float

    def create_resonable_allparameters_parametrizations(
            self,
            array_of_allparameters,
            list_of_defocuses_input,
            zmax,
            remove_last_n=None):
        """Create ``parametrizations'' from list of ``parameters'' and defocuses

        Given parameters for single defocus images and their defocuses,
        create parameterizations (1d functions) for multi-image linear fit across various defocuses
        This is the inverse of function `create_list_of_allparameters`

        Parameters
        ----------
        array_of_allparameters : `np.array`
           Array with parameters of defocus, 2d array with shape
           [n(list_of_defocuses),number of parameters]
        list_of_defocuses_input : `list`
            List of strings at which defocuses are the data
            from array_of_allparameters
        zmax : `int`
            Highest order of Zernike parameters applied
        remove_last_n : `int`
            Do not do the fit for the last 'n' parameters
            If not specified, it defaults to 2

        Returns
        ----------
        array_of_polyfit_1_parameterizations : `np.array`
            Array contaning output 1d ``parameterizations
        """

        if remove_last_n is None:
            remove_last_n = 2

        list_of_defocuses_int = self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses_input)
        if remove_last_n > 0:
            array_of_allparameters = array_of_allparameters[:, :-remove_last_n]

        if zmax <= 22:
            len_of_iterations = array_of_allparameters.shape[1]
        else:
            len_of_iterations = 42 + zmax - 22

        list_of_polyfit_1_parameter = []
        for i in range(len_of_iterations):
            # logging.info([i,array_of_allparameters.shape[1]])
            if i < array_of_allparameters.shape[1]:
                # logging.info('i'+str(i)+' '+str(array_of_allparameters[:,i]))
                polyfit_1_parameter = np.polyfit(
                    x=list_of_defocuses_int, y=array_of_allparameters[:, i], deg=1)
            else:
                # logging.info('i'+str(i)+' '+'None')
                # if you have no input for such high level of Zernike, set it at zero
                polyfit_1_parameter = np.array([0, 0])

            # logging.info('i_polyfit'+str(i)+' '+str(polyfit_1_parameter))
            list_of_polyfit_1_parameter.append(polyfit_1_parameter)

        array_of_polyfit_1_parameterizations = np.array(list_of_polyfit_1_parameter)

        # list_of_defocuses_output_int=self.transform_list_of_defocuses_from_str_to_float(list_of_defocuses_input)
        # list_of_allparameters=[]
        # for i in list_of_defocuses_output_int:
        #    allparameters_proposal_single=self.create_allparameters_single(i,array_of_polyfit_1_parameterizations,zmax=self.zmax)
        #    list_of_allparameters.append(allparameters_proposal_single)

        return array_of_polyfit_1_parameterizations

    def lnlike_Neven_multi_same_spot(self, list_of_allparameters_input, return_Images=False,
                                     use_only_chi=False, multi_background_factor=3):
        """Create model images and estimate their quality

        Creates model images, and compares them to supplied data

        Parameters
        ----------
        list_of_allparameters_input : `list`
            List of parameteres to create image at each defocus
        return_Images : `bool`
            If True, return all the created images and auxiliary data
        use_only_chi : `bool`
            If True, use chi as the quality measure
            If False, use chi**2 as the quality measure
        multi_background_factor : `int`
            Only consider pixels with flux above this factor * background level

        Returns
        ----------
        (if return_Images is False):
        mean_res_of_multi_same_spot : `float`
            Mean quality of all images
        (if return_Images is True):
        mean_res_of_multi_same_spot [index 0] : `float`
            Mean quality of all images
        list_of_single_res [index 1] : `list`
            Quality per image
        list_of_single_model_image [index 2] : `list`
            List of created model images
        list_of_single_allparameters [index 3] : `list`
            List of parameters per image
        list_of_single_chi_results [index 4] : `list`
            List of arrays describing quality of fitting
            Each of these array contains
                0. chi2_max value, 1. Qvalue, 2. (chi or chi2)/d.o.f., 3. (chi2 or chi2_max)/d.o.f.
        array_of_psf_positions_output [index 5] : `np.array`
            Array showing the centering of images
        """
        self.use_only_chi = use_only_chi

        list_of_single_res = []
        if return_Images:
            list_of_single_model_image = []
            list_of_single_allparameters = []
            list_of_single_chi_results = []

        if len(self.list_of_sci_images) == len(list_of_allparameters_input):
            list_of_allparameters = np.copy(list_of_allparameters_input)

        else:
            allparametrization = list_of_allparameters_input

            # logging.info('self.list_of_defocuses: ' + str(self.list_of_defocuses))
            # logging.info('allparametrization.type: ' + str(allparametrization.type))
            list_of_allparameters = self.create_list_of_allparameters(
                allparametrization, list_of_defocuses=self.list_of_defocuses)

            if self.verbosity == 1:
                logging.info('Starting LN_PFS_multi_same_spot for parameters-hash '
                             + str(hash(str(allparametrization.data)))
                             + ' at ' + str(time.time()) + ' in thread '
                             + str(threading.get_ident()))

        assert len(self.list_of_sci_images) == len(list_of_allparameters)

        # logging.info(len(self.list_of_sci_images))
        # logging.info(len(list_of_allparameters))

        # use same weights, experiment
        # if use_only_chi==True:
        #    renormalization_of_var_sum=np.ones((len(self.list_of_sci_images)))*len(self.list_of_sci_images)
        #    central_index=int(len(self.list_of_sci_images)/2)
        #    renormalization_of_var_sum[central_index]=1

        # else:

        # find image with lowest variance - pressumably the one in focus
        # array_of_var_sum=np.array(list(map(np.sum,self.list_of_var_images)))
        # index_of_max_var_sum=np.where(array_of_var_sum==np.min(array_of_var_sum))[0][0]
        # find what variance selectes top 20% of pixels
        # this is done to weight more the images in focus and less the image out of focus in the
        # final likelihood result
        # quantile_08_focus=np.quantile(self.list_of_sci_images[index_of_max_var_sum],0.8)

        list_of_var_sums = []
        for i in range(len(list_of_allparameters)):
            # taking from create_chi_2_almost function in LN_PFS_single

            mask_image = self.list_of_mask_images[i]
            var_image = self.list_of_var_images[i]
            sci_image = self.list_of_sci_images[i]
            # array that has True for values which are good and False for bad values
            inverted_mask = ~mask_image.astype(bool)

            try:
                if sci_image.shape[0] == 20:
                    multi_background_factor = 3

                # logging.info('var_image.shape: '+str(var_image.shape))
                # logging.info('multi_background_factor: '+str(multi_background_factor))
                # logging.info('np.median(var_image[0]): '+str(np.median(var_image[0])))
                # logging.info('np.median(var_image[-1]): '+str(np.median(var_image[-1])))
                # logging.info('np.median(var_image[:,0]): '+str(np.median(var_image[:,0])))
                # logging.info('np.median(var_image[:,-1]): '+str(np.median(var_image[:,-1])))
                mean_value_of_background_via_var = np.mean([np.median(var_image[0]), np.median(
                    var_image[-1]), np.median(var_image[:, 0]),
                    np.median(var_image[:, -1])]) * multi_background_factor
                # logging.info('mean_value_of_background_via_var: '+str(mean_value_of_background_via_var))

                mean_value_of_background_via_sci = np.mean([np.median(sci_image[0]), np.median(
                    sci_image[-1]), np.median(sci_image[:, 0]),
                    np.median(sci_image[:, -1])]) * multi_background_factor
                # logging.info('mean_value_of_background_via_sci: '+str(mean_value_of_background_via_sci))
                mean_value_of_background = np.max(
                    [mean_value_of_background_via_var, mean_value_of_background_via_sci])
            except BaseException:
                pass

            # select only images with above 80% percentile of the image with max variance?
            var_image_masked = var_image * inverted_mask
            var_image_masked_without_nan = var_image_masked.ravel()[
                var_image_masked.ravel() > mean_value_of_background]

            if use_only_chi:
                # if you level is too high
                if len(var_image_masked_without_nan) == 0:
                    var_sum = -1
                else:
                    # var_sum=-(1)*(np.sum(np.sqrt(np.abs(var_image_masked_without_nan))))
                    var_sum = -1

            else:

                # if you level is too high
                if len(var_image_masked_without_nan) == 0:
                    var_sum = -(1)
                else:
                    var_sum = -(1) * (np.mean(np.abs(var_image_masked_without_nan)))
            list_of_var_sums.append(var_sum)

            # renormalization needs to be reconsidered?
            array_of_var_sum = np.array(list_of_var_sums)
            max_of_array_of_var_sum = np.max(array_of_var_sum)

            renormalization_of_var_sum = array_of_var_sum / max_of_array_of_var_sum
            # logging.info('renormalization_of_var_sum'+str(renormalization_of_var_sum))
        list_of_psf_positions_output = []

        for i in range(len(list_of_allparameters)):

            # if image is in focus which at this point is the size of image with 20

            if (self.list_of_sci_images[i].shape)[0] == 20:
                if self.use_center_of_flux:
                    use_center_of_flux = True
                else:
                    use_center_of_flux = False
            else:
                use_center_of_flux = False

            if self.verbosity == 1:
                logging.info('################################')
                logging.info('analyzing image ' + str(i + 1) + ' out of ' + str(len(list_of_allparameters)))
                logging.info(' ')

            # if this is the first image, do the full analysis, generate new pupil and illumination
            if i == 0:
                model_single = LN_PFS_single(
                    self.list_of_sci_images[i],
                    self.list_of_var_images[i],
                    self.list_of_mask_images[i],
                    wavelength=self.wavelength,
                    dithering=self.dithering,
                    save=self.save,
                    verbosity=self.verbosity,
                    pupil_parameters=self.pupil_parameters,
                    use_pupil_parameters=self.use_pupil_parameters,
                    use_optPSF=self.use_optPSF,
                    use_wf_grid=self.list_of_wf_grid[i],
                    zmax=self.zmax,
                    extraZernike=self.extraZernike,
                    pupilExplicit=self.pupilExplicit,
                    simulation_00=self.simulation_00,
                    double_sources=self.double_sources,
                    double_sources_positions_ratios=self.double_sources_positions_ratios,
                    npix=self.npix,
                    fit_for_flux=self.fit_for_flux,
                    test_run=self.test_run,
                    explicit_psf_position=self.list_of_psf_positions[i],
                    use_only_chi=self.use_only_chi,
                    use_center_of_flux=use_center_of_flux)

                res_single_with_intermediate_images = model_single(
                    list_of_allparameters[i],
                    return_Image=True,
                    return_intermediate_images=True,
                    use_only_chi=use_only_chi,
                    multi_background_factor=multi_background_factor)

                if res_single_with_intermediate_images == -np.inf:
                    return -np.inf
                if isinstance(res_single_with_intermediate_images, tuple):
                    if res_single_with_intermediate_images[0] == -np.inf:
                        return -np.inf
                likelihood_result = res_single_with_intermediate_images[0]
                model_image = res_single_with_intermediate_images[1]
                allparameters = res_single_with_intermediate_images[2]
                pupil_explicit_0 = res_single_with_intermediate_images[3]
                # wf_grid_rot = res_single_with_intermediate_images[4]
                chi_results = res_single_with_intermediate_images[5]
                psf_position = res_single_with_intermediate_images[6]

                list_of_single_res.append(likelihood_result)
                list_of_psf_positions_output.append(psf_position)
                if return_Images:
                    list_of_single_model_image.append(model_image)
                    list_of_single_allparameters.append(allparameters)
                    list_of_single_chi_results.append(chi_results)

            # and if this is not the first image, use the pupil and illumination used in the first image
            else:

                model_single = LN_PFS_single(
                    self.list_of_sci_images[i],
                    self.list_of_var_images[i],
                    self.list_of_mask_images[i],
                    wavelength=self.wavelength,
                    dithering=self.dithering,
                    save=self.save,
                    verbosity=self.verbosity,
                    pupil_parameters=self.pupil_parameters,
                    use_pupil_parameters=self.use_pupil_parameters,
                    use_optPSF=self.use_optPSF,
                    use_wf_grid=self.list_of_wf_grid[i],
                    zmax=self.zmax,
                    extraZernike=self.extraZernike,
                    pupilExplicit=pupil_explicit_0,
                    simulation_00=self.simulation_00,
                    double_sources=self.double_sources,
                    double_sources_positions_ratios=self.double_sources_positions_ratios,
                    npix=self.npix,
                    fit_for_flux=self.fit_for_flux,
                    test_run=self.test_run,
                    explicit_psf_position=self.list_of_psf_positions[i],
                    use_only_chi=self.use_only_chi,
                    use_center_of_flux=use_center_of_flux)
                if not return_Images:
                    res_single_without_intermediate_images = model_single(
                        list_of_allparameters[i],
                        return_Image=return_Images,
                        use_only_chi=use_only_chi,
                        multi_background_factor=multi_background_factor)

                    likelihood_result = res_single_without_intermediate_images[0]
                    psf_position = res_single_with_intermediate_images[-1]
                    # logging.info(likelihood_result)
                    list_of_single_res.append(likelihood_result)
                    list_of_psf_positions_output.append(psf_position)

                if return_Images:
                    res_single_with_an_image = model_single(
                        list_of_allparameters[i], return_Image=return_Images, use_only_chi=use_only_chi)
                    if res_single_with_an_image == -np.inf:
                        return -np.inf
                    likelihood_result = res_single_with_an_image[0]
                    model_image = res_single_with_an_image[1]
                    allparameters = res_single_with_an_image[2]
                    chi_results = res_single_with_an_image[3]
                    psf_position = res_single_with_an_image[-1]

                    list_of_single_res.append(likelihood_result)
                    list_of_single_model_image.append(model_image)
                    list_of_single_allparameters.append(allparameters)
                    list_of_single_chi_results.append(chi_results)
                    list_of_psf_positions_output.append(psf_position)
                    # possibly implement intermediate images here
        array_of_single_res = np.array(list_of_single_res)
        array_of_psf_positions_output = np.array(list_of_psf_positions_output)

        # renormalization
        if self.verbosity == 1:
            logging.info('################################')
            logging.info('Likelihoods returned per individual images are: ' + str(array_of_single_res))
            logging.info('Mean likelihood is ' + str(np.mean(array_of_single_res)))

        # mean_res_of_multi_same_spot=np.mean(array_of_single_res)
        mean_res_of_multi_same_spot = np.mean(array_of_single_res / renormalization_of_var_sum)

        if self.verbosity == 1:
            logging.info('################################')
            logging.info('Renormalized likelihoods returned per individual images are: '
                         + str(array_of_single_res / renormalization_of_var_sum))
            logging.info('Renormalization factors are: ' + str(renormalization_of_var_sum))
            logging.info('Mean renormalized likelihood is ' + str(mean_res_of_multi_same_spot))
            logging.info('array_of_psf_positions_output: ' + str(array_of_psf_positions_output))

        if self.verbosity == 1:
            # logging.info('Ending LN_PFS_multi_same_spot for parameters-hash '+
            # str(hash(str(allparametrization.data)))+' at '+str(time.time())+
            # ' in thread '+str(threading.get_ident()))
            logging.info('Ending LN_PFS_multi_same_spot at time '
                         + str(time.time()) + ' in thread ' + str(threading.get_ident()))
            logging.info(' ')

        if not return_Images:
            return mean_res_of_multi_same_spot
        if return_Images:
            # 0. mean_res_of_multi_same_spot - mean likelihood per images, renormalized
            # 1. list_of_single_res - likelihood per image, not renormalized
            # 2. list_of_single_model_image - list of created model images
            # 3. list_of_single_allparameters - list of parameters per image?
            # 4. list_of_single_chi_results - list of arrays describing quality of fitting
            #           1. chi2_max value, 2. Qvalue, 3. chi2/d.o.f., 4. chi2_max/d.o.f.
            # 5. array_of_psf_positions_output - list showing the centering of images

            return mean_res_of_multi_same_spot, list_of_single_res, list_of_single_model_image,\
                list_of_single_allparameters, list_of_single_chi_results, array_of_psf_positions_output

    def __call__(
            self,
            list_of_allparameters,
            return_Images=False,
            use_only_chi=False,
            multi_background_factor=3):

        return self.lnlike_Neven_multi_same_spot(list_of_allparameters, return_Images=return_Images,
                                                 use_only_chi=use_only_chi,
                                                 multi_background_factor=multi_background_factor)


class Tokovinin_multi(object):

    """

    # improvments possible - modify by how much to move parameters based on the previous step
    # in simplied H, take new model into account where doing changes


    outputs:
    initial_model_result,final_model_result,\
    list_of_initial_model_result,list_of_final_model_result,\
    out_images, pre_images, list_of_image_final,\
    allparameters_parametrization_proposal, allparameters_parametrization_proposal_after_iteration,\
    list_of_initial_input_parameters, list_of_finalinput_parameters,\
    list_of_pre_chi2,list_of_after_chi2,\
    list_of_psf_positions,list_of_final_psf_positions,\
    [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions]

    explanation of the results:
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

    def __init__(self, list_of_sci_images, list_of_var_images, list_of_mask_images=None,
                 wavelength=None, dithering=None, save=None, verbosity=None,
                 DIRECTORY=None,
                 pupil_parameters=None, use_pupil_parameters=None, use_optPSF=None, list_of_wf_grid=None,
                 zmax=None, extraZernike=None, pupilExplicit=None, simulation_00=None,
                 double_sources=None, double_sources_positions_ratios=None, npix=None,
                 list_of_defocuses=None, fit_for_flux=True, test_run=False, list_of_psf_positions=None,
                 num_iter=None, move_allparameters=None, pool=None):
        """
        @param list_of_sci_images                   list of science images, list of 2d array
        @param list_of_var_images                   list of variance images, 2d arrays,
                                                    which are the same size as sci_image
        @param list_of_mask_images                  list of mask images, 2d arrays,
                                                    which are the same size as sci_image
        @param wavelength                           wavelength in nm, to be passed to to module
        @param dithering                            dithering, 1=normal, 2=two times higher resolution,
                                                    3=not supported
        @param save                                 save intermediate result in the process
                                                    (set value at 1 for saving)
        @param verbosity                            verbosity of the process
                                                    (set value at 2 for full output,
                                                    1 only in Tokovinin, 0==nothing)

        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF

        @param zmax                                 largest Zernike order used (11 or 22, or larger than 22)
        @param extraZernike                         array consisting of higher order zernike
                                                    (if using higher order than 22)
        @param pupilExplicit

        @param simulation_00                        resulting image will be centered with optical center
                                                    in the center of the image
                                                    and not fitted acorrding to the sci_image
        @param double_sources                       1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /    arrray with parameters describing relative position\
                                                    and relative flux of the secondary source(s)
        @param npxix                                size of the pupil (1536 reccomended)
        @param list_of_defocuses                    list of defocuses at which images are taken
                                                    (float or string?)

        @param fit_for_flux                         automatically fit for the best flux level
                                                    that minimizes the chi**2
        @param test_run                             if True, skips the creation of model and
                                                    return science image - useful for testing
                                                    interaction of outputs of the module
                                                    in broader setting quickly

        @param list_of_psf_positions                gives position of the opt_psf
        @param num_iter                             number of iteration
        @param move_allparameters                   if True change all parameters i.e.,
                                                    also ``global'' parameters, i.e.,
                                                    not just wavefront parameters
        @param pool                                 pass pool of workers to calculate

        array of changes due to movement due to wavefront changes
        """

        if verbosity is None:
            verbosity = 0

        if use_pupil_parameters is not None:
            assert pupil_parameters is not None

        if double_sources is not None and double_sources is not False:
            assert np.sum(np.abs(double_sources_positions_ratios)) > 0

        if zmax is None:
            zmax = 22

        if zmax == 11:
            self.columns = [
                'z4',
                'z5',
                'z6',
                'z7',
                'z8',
                'z9',
                'z10',
                'z11',
                'detFrac',
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
                'effective_ilum_radius',
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
        if zmax >= 22:
            self.columns = [
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
                'detFrac',
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
                'effective_ilum_radius',
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

        self.list_of_sci_images = list_of_sci_images
        self.list_of_var_images = list_of_var_images

        if list_of_mask_images is None:
            list_of_mask_images = []
            for i in range(len(list_of_sci_images)):
                mask_image = np.zeros(list_of_sci_images[i].shape)
                list_of_mask_images.append(mask_image)

        self.list_of_mask_images = list_of_mask_images

        # implement custom variance image here

        # self.mask_image=mask_image
        # self.sci_image=sci_image
        # self.var_image=var_image
        self.wavelength = wavelength
        self.dithering = dithering
        self.save = save
        if DIRECTORY is None:
            DIRECTORY = '/tigress/ncaplar/'
        self.DIRECTORY = DIRECTORY
        self.pupil_parameters = pupil_parameters
        self.use_pupil_parameters = use_pupil_parameters
        self.use_optPSF = use_optPSF
        self.pupilExplicit = pupilExplicit
        self.simulation_00 = simulation_00
        self.zmax = zmax
        self.extraZernike = extraZernike
        self.verbosity = verbosity
        self.double_sources = double_sources
        self.double_sources_positions_ratios = double_sources_positions_ratios
        self.npix = npix
        self.fit_for_flux = fit_for_flux
        self.list_of_defocuses = list_of_defocuses
        self.test_run = test_run
        if list_of_psf_positions is None:
            list_of_psf_positions = [None] * len(list_of_sci_images)
        self.list_of_psf_positions = list_of_psf_positions
        if list_of_wf_grid is None:
            list_of_wf_grid = [None] * len(list_of_sci_images)
        self.list_of_wf_grid = list_of_wf_grid
        self.list_of_defocuses = list_of_defocuses
        self.move_allparameters = move_allparameters
        self.num_iter = num_iter
        self.pool = pool

        if self.verbosity >= 1:
            self.verbosity_model = self.verbosity - 1
        else:
            self.verbosity_model = self.verbosity

        # parameter that control if the intermediate outputs are saved to the hard disk
        save = False
        self.save = save

    def Tokovinin_algorithm_chi_multi(self, allparameters_parametrization_proposal,
                                      return_Images=False, num_iter=None, previous_best_result=None,
                                      use_only_chi=False, multi_background_factor=3, up_to_which_z=None):
        """ Apply Tokovinin algorithm to a set of images

        Parameters
        ----------
        allparameters_parametrization_proposal : `np.array`
            2d parametrization of variables
        return_Images : `bool`
            if True, also return created images
        num_iter : `int`
            number of iteration, used when creating save files
        previous_best_result : `np.array?`
            output from previous Tokovinin run
        use_only_chi : `bool`
            if True, optimize using chi, not chi**2
        multi_background_factor : `float`
            take into account only pixels with flux this many times above the background

        Returns
        ----------
        (if return_Images == False)
        final_model_result : `float`
            averaged ``likelihood'' over all images

        (if return_Images == True AND previous_best_result is None )
        initial_model_result :
            explanation
        final_model_result : `float`
            output with index 0 from model_multi - averaged ``likelihood'' over all input images
            if the proposed images has worse quality then input, reproduce the input value
        list_of_initial_model_result :
            explanation
        list_of_final_model_result :
            explanation
        allparameters_parametrization_proposal :
            explanation
        allparameters_parametrization_proposal_after_iteration :
            explanation
        list_of_initial_input_parameters :
            explanation
        list_of_finalinput_parameters :
           explanation
        list_of_pre_chi2 :
            explanation
        list_of_after_chi2 :
           explanation
        list_of_psf_positions :
            explanation
        list_of_final_psf_positions :
            explanation
        [uber_images_normalized, uber_M0_std, H_std,
         array_of_delta_z_parametrizations_None, list_of_final_psf_positions]:
            explanation

        (if return_Images == True AND previous_best_result is avaliable )





        """

        if self.verbosity >= 1:
            logging.info('###############################################################################')
            logging.info('###############################################################################')
            logging.info('Starting Tokovinin_algorithm_chi_multi with num_iter: ' + str(num_iter))
            logging.info('Tokovinin, return_Images: ' + str(return_Images))
            logging.info('Tokovinin, num_iter: ' + str(num_iter))
            logging.info('Tokovinin, use_only_chi: ' + str(use_only_chi))
            logging.info('Tokovinin, multi_background_factor: ' + str(multi_background_factor))

            logging.info('allparameters_parametrization_proposal'
                         + str(allparameters_parametrization_proposal))
            logging.info('allparameters_parametrization_proposal.shape'
                         + str(allparameters_parametrization_proposal.shape))

        list_of_sci_images = self.list_of_sci_images
        list_of_var_images = self.list_of_var_images
        list_of_mask_images = self.list_of_mask_images

        double_sources_positions_ratios = self.double_sources_positions_ratios
        list_of_defocuses_input_long = self.list_of_defocuses

        if num_iter is None:
            if self.num_iter is not None:
                num_iter = self.num_iter

        move_allparameters = self.move_allparameters

        # if you passed previous best result, set the list_of_explicit_psf_positions
        # by default it is put as the last element in the last cell in the previous_best_result output
        if previous_best_result is not None:
            # to be compatible with versions before 0.45
            if len(previous_best_result) == 5:
                self.list_of_psf_positions = previous_best_result[-1]
            else:
                self.list_of_psf_positions = previous_best_result[-1][-1]

        ##########################################################################
        # Create initial modeling as basis for future effort
        # the outputs of this section are 0. pre_model_result, 1. model_results, 2. pre_images,
        # 3. pre_input_parameters, 4. chi_2_before_iteration_array, 5. list_of_psf_positions
        if self.verbosity >= 1:
            logging.info('list_of_defocuses analyzed: ' + str(list_of_defocuses_input_long))

        # logging.info('list_of_sci_images'+str(list_of_sci_images))
        # logging.info('list_of_var_images'+str(list_of_var_images))
        # logging.info('list_of_mask_images'+str(list_of_mask_images))
        # logging.info('wavelength'+str(self.wavelength))
        # logging.info('dithering'+str(self.dithering))
        # logging.info('self.save'+str(self.save))
        # logging.info('self.zmax'+str(self.zmax))
        # logging.info('self.double_sources'+str(self.double_sources))
        # logging.info('self.double_sources_positions_ratios'+str(self.double_sources_positions_ratios))
        # logging.info('self.npix'+str(self.npix))
        # logging.info('self.list_of_defocuses_input_long'+str(list_of_defocuses_input_long))
        # logging.info('self.fit_for_flux'+str(self.fit_for_flux))
        # logging.info('self.test_run'+str(self.test_run))
        # logging.info('self.list_of_psf_positions'+str(self.list_of_psf_positions))

        model_multi = LN_PFS_multi_same_spot(
            list_of_sci_images,
            list_of_var_images,
            list_of_mask_images=list_of_mask_images,
            wavelength=self.wavelength,
            dithering=self.dithering,
            save=self.save,
            zmax=self.zmax,
            verbosity=self.verbosity_model,
            double_sources=self.double_sources,
            double_sources_positions_ratios=self.double_sources_positions_ratios,
            npix=self.npix,
            list_of_defocuses=list_of_defocuses_input_long,
            fit_for_flux=self.fit_for_flux,
            test_run=self.test_run,
            list_of_psf_positions=self.list_of_psf_positions)

        if self.verbosity >= 1:
            logging.info('****************************')
            logging.info('Starting Tokovinin procedure with num_iter: ' + str(num_iter))
            logging.info('Initial testing proposal is: ' + str(allparameters_parametrization_proposal))
        time_start_single = time.time()

        # create list of minchains, one per each image
        list_of_minchain = model_multi.create_list_of_allparameters(
            allparameters_parametrization_proposal,
            list_of_defocuses=list_of_defocuses_input_long,
            zmax=self.zmax)

        # if the parametrization is 2d array, move it into 1d shape
        if len(allparameters_parametrization_proposal.shape) == 2:
            allparameters_parametrization_proposal = move_parametrizations_from_2d_shape_to_1d_shape(
                allparameters_parametrization_proposal)

        if self.verbosity >= 1:
            logging.info('Starting premodel analysis with num_iter: ' + str(num_iter))

        # results from initial run, before running fitting algorithm
        # pre_model_result - mean likelihood across all images, renormalized
        # model_results - likelihood per image, not renormalized
        # pre_images - list of created model images
        # pre_input_parameters - list of parameters per image?
        # chi_2_before_iteration_array - list of lists describing quality of fitting
        # list_of_psf_positions -?
        try:
            # logging.info('len(list_of_minchain): '+str(len(list_of_minchain)))
            # logging.info('list_of_minchain[0] '+str(list_of_minchain[0]))
            # logging.info('multi_background_factor: '+str(multi_background_factor))
            # logging.info('type'+str(type(multi_background_factor)))
            # logging.info('up_to_which_z: '+str(up_to_which_z))
            # logging.info(str( list_of_minchain))
            # logging.info('use_only_chi: '+str( use_only_chi))
            # logging.info('list_of_minchain: '+str( list_of_minchain))

            pre_model_result, model_results, pre_images, pre_input_parameters, chi_2_before_iteration_array,\
                list_of_psf_positions =\
                model_multi(list_of_minchain, return_Images=True, use_only_chi=use_only_chi,
                            multi_background_factor=multi_background_factor)
            # modify variance image according to the models that have just been created
            # first time modifying variance image
            list_of_single_model_image = pre_images
            list_of_var_images_via_model = []
            for index_of_single_image in range(len(list_of_sci_images)):
                popt = create_popt_for_custom_var(self.list_of_sci_images[index_of_single_image],
                                                  self.list_of_var_images[index_of_single_image],
                                                  self.list_of_mask_images[index_of_single_image])
                single_var_image_via_model =\
                    create_custom_var_from_popt(list_of_single_model_image[index_of_single_image], popt)
                list_of_var_images_via_model.append(single_var_image_via_model)

            # replace the variance images provided with these custom variance images
            list_of_var_images = list_of_var_images_via_model
            # self.list_of_var_images = list_of_var_images

        except Exception as e:
            logging.info('Exception is: ' + str(e))
            logging.info('Exception type is: ' + str(repr(e)))
            logging.info(traceback.logging.info_exc())
            if self.verbosity >= 1:
                logging.info('Premodel analysis failed')
            # if the modelling failed
            # returning 7 nan values to be consistent with what would be the return if the algorithm passed
            # at position 0 return extremly likelihood to indicate failure
            # at position 3 return the input parametrization
            # return -9999999,np.nan,np.nan,allparameters_parametrization_proposal,np.nan,np.nan,np.nan
            return -9999999, -9999999, np.nan, np.nan, np.nan, np.nan, np.nan,
            allparameters_parametrization_proposal, allparameters_parametrization_proposal,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if self.verbosity >= 1:
            logging.info('list_of_psf_positions at the input stage: ' + str(np.array(list_of_psf_positions)))

        if self.save:
            np.save(self.DIRECTORY + 'Results/allparameters_parametrization_proposal_' + str(num_iter),
                    allparameters_parametrization_proposal)
            np.save(self.DIRECTORY + 'Results/pre_images_' + str(num_iter),
                    pre_images)
            np.save(self.DIRECTORY + 'Results/pre_input_parameters_' + str(num_iter),
                    pre_input_parameters)
            np.save(self.DIRECTORY + 'Results/list_of_sci_images_' + str(num_iter),
                    list_of_sci_images)
            np.save(self.DIRECTORY + 'Results/list_of_var_images_' + str(num_iter),
                    list_of_var_images)
            np.save(self.DIRECTORY + 'Results/list_of_mask_images_' + str(num_iter),
                    list_of_mask_images)

        # extract the parameters which will not change in this function, i.e., non-wavefront parameters
        nonwavefront_par = list_of_minchain[0][19:42]
        time_end_single = time.time()
        if self.verbosity >= 1:
            logging.info('Total time taken for premodel analysis with num_iter ' + str(num_iter)
                         + ' was  ' + str(time_end_single - time_start_single) + ' seconds')
            logging.info('chi_2_before_iteration is: ' + str(chi_2_before_iteration_array))

            logging.info('Ended premodel analysis ')
            logging.info('***********************')

        # import science images and determine the flux mask
        list_of_mean_value_of_background = []
        list_of_flux_mask = []
        list_of_sci_image_std = []
        for i in range(len(list_of_sci_images)):
            sci_image = list_of_sci_images[i]
            var_image = list_of_var_images[i]

            # do not use this for images in focus or near focus
            # probably needs to be done better than via shape measurment
            if sci_image.shape[0] == 20:
                multi_background_factor = 3

            mean_value_of_background_via_var = np.mean([np.median(var_image[0]), np.median(
                var_image[-1]), np.median(var_image[:, 0]),
                np.median(var_image[:, -1])]) * multi_background_factor

            mean_value_of_background_via_sci = np.mean([np.median(sci_image[0]), np.median(
                sci_image[-1]), np.median(sci_image[:, 0]),
                np.median(sci_image[:, -1])]) * multi_background_factor

            mean_value_of_background = np.max(
                [mean_value_of_background_via_var, mean_value_of_background_via_sci])
            if self.verbosity > 1:
                logging.info(
                    str(multi_background_factor) + 'x mean_value_of_background in image with index'
                    + str(i) + ' is estimated to be: ' + str(mean_value_of_background))

            list_of_mean_value_of_background.append(mean_value_of_background)

        list_of_flux_mask = []
        for i in range(len(list_of_sci_images)):
            sci_image = list_of_sci_images[i]
            var_image = list_of_var_images[i]
            flux_mask = sci_image > (list_of_mean_value_of_background[i])
            # normalized science image

            sci_image_std = sci_image / np.sqrt(var_image)
            list_of_sci_image_std.append(sci_image_std)
            list_of_flux_mask.append(flux_mask)

        # find postions for focus image in the raveled images
        if len(list_of_flux_mask) > 1:
            len_of_flux_masks = np.array(list(map(np.sum, list_of_flux_mask)))
            position_of_most_focus_image = np.where(len_of_flux_masks == np.min(len_of_flux_masks))[0][0]
            position_focus_1 = np.sum(len_of_flux_masks[:position_of_most_focus_image])
            position_focus_2 = np.sum(len_of_flux_masks[:position_of_most_focus_image + 1])

        self.list_of_flux_mask = list_of_flux_mask
        self.list_of_sci_image_std = list_of_sci_image_std
        ##########################################################################
        # masked science image
        list_of_I = []
        list_of_I_std = []
        list_of_std_image = []
        for i in range(len(list_of_sci_images)):

            sci_image = list_of_sci_images[i]
            sci_image_std = list_of_sci_image_std[i]
            flux_mask = list_of_flux_mask[i]
            std_image = np.sqrt(list_of_var_images[i][flux_mask]).ravel()

            # using variable name `I` to match the original source paper
            I = sci_image[flux_mask].ravel()  # noqa: E741
            # I=((sci_image[flux_mask])/np.sum(sci_image[flux_mask])).ravel()
            I_std = ((sci_image_std[flux_mask]) / 1).ravel()
            # I_std=((sci_image_std[flux_mask])/np.sum(sci_image_std[flux_mask])).ravel()

            list_of_I.append(I)
            list_of_std_image.append(std_image)
            list_of_I_std.append(I_std)

        # addition May22
        # array_of_sci_image_std = np.array(list_of_sci_image_std)
        list_of_std_sum = []
        for i in range(len(list_of_sci_image_std)):
            list_of_std_sum.append(np.sum(list_of_std_image[i]))

        array_of_std_sum = np.array(list_of_std_sum)
        array_of_std_sum = array_of_std_sum / np.min(array_of_std_sum)

        list_of_std_image_renormalized = []
        for i in range(len(list_of_std_image)):
            list_of_std_image_renormalized.append(list_of_std_image[i] * array_of_std_sum[i])
        #
        uber_std = [item for sublist in list_of_std_image_renormalized for item in sublist]

        # join all I,I_std from all individual images into one uber I,I_std
        uber_I = [item for sublist in list_of_I for item in sublist]
        # uber_std=[item for sublist in list_of_std_image for item in sublist]
        # uber_I_std=[item for sublist in list_of_I_std for item in sublist]

        uber_I = np.array(uber_I)
        uber_std = np.array(uber_std)

        uber_I_std = uber_I / uber_std

        if self.save:
            np.save(self.DIRECTORY + 'Results/list_of_sci_images_' + str(num_iter),
                    list_of_sci_images)
            np.save(self.DIRECTORY + 'Results/list_of_mean_value_of_background_' + str(num_iter),
                    list_of_mean_value_of_background)
            np.save(self.DIRECTORY + 'Results/list_of_flux_mask_' + str(num_iter),
                    list_of_flux_mask)
            np.save(self.DIRECTORY + 'Results/uber_std_' + str(num_iter),
                    uber_std)
            np.save(self.DIRECTORY + 'Results/uber_I_' + str(num_iter),
                    uber_I)

        # March 14, 2022, adding just pure avoid of the run
        if up_to_which_z is False:
            # 0. likelihood averaged over all images (before the function)
            # 1. likelihood averaged over all images (before the function)
            # 2. likelihood per image (output from model_multi) (before the function)
            # 3. likelihood per image (output from model_multi) (before the function)
            # 4. out_images
            # 5. list of initial model images
            # 6. list of initial model images
            # 7. parametrization before the function
            # 8. parametrization after the function
            # 9. list of parameters per image (before the function)
            # 10. list of parameters per image (after the function)
            # 11. list of chi2 per image (before the function)
            # 12. list of chi2 per image (after the function)
            # 13. list of psf position of image (before the function)
            # 14. list of psf position of image (after the function)

            initial_model_result, list_of_initial_model_result, list_of_image_0,\
                list_of_initial_input_parameters, list_of_pre_chi2, list_of_psf_positions =\
                pre_model_result, model_results, pre_images, pre_input_parameters,\
                chi_2_before_iteration_array, list_of_psf_positions

            if previous_best_result is None:
                return initial_model_result, initial_model_result,\
                    list_of_initial_model_result, list_of_initial_model_result,\
                    None, pre_images, pre_images,\
                    allparameters_parametrization_proposal,\
                    allparameters_parametrization_proposal,\
                    list_of_initial_input_parameters, list_of_initial_input_parameters,\
                    list_of_pre_chi2, list_of_pre_chi2,\
                    list_of_psf_positions, list_of_psf_positions,\
                    [None, None, None, None, None]
            else:
                return initial_model_result, initial_model_result,\
                    list_of_initial_model_result, list_of_initial_model_result,\
                    None, pre_images, pre_images,\
                    allparameters_parametrization_proposal,\
                    allparameters_parametrization_proposal,\
                    list_of_initial_input_parameters, list_of_initial_input_parameters,\
                    list_of_pre_chi2, list_of_pre_chi2,\
                    list_of_psf_positions, list_of_psf_positions

        # set number of extra Zernike
        # number_of_extra_zernike=0
        # twentytwo_or_extra=22
        # numbers that make sense are 11,22,37,56,79,106,137,172,211,254

        # if number_of_extra_zernike is None:
        #    number_of_extra_zernike=0
        # else:
        number_of_extra_zernike = self.zmax - 22

        ##########################################################################
        # Start of the iterative process

        number_of_non_decreses = [0]

        for iteration_number in range(1):

            if iteration_number == 0:

                # initial SVD treshold
                thresh0 = 0.02
            else:
                pass

            ##########################################################################
            # starting real iterative process here
            # create changes in parametrizations

            # list of how much to move Zernike coefficents
            # list_of_delta_z=[]
            # for z_par in range(3,22+number_of_extra_zernike):
            #    list_of_delta_z.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))

            # list of how much to move Zernike coefficents
            # possibly needs to me modified to be smarther and take into account that
            # every second parameter gets ``amplified'' in defocus
            # list_of_delta_z_parametrizations=[]
            # for z_par in range(0,19*2+2*number_of_extra_zernike):
            #    list_of_delta_z_parametrizations.append(0.5/((np.sqrt(8.*(z_par+1.)-6.)-1.)/2.))

            # this should produce reasonable changes in multi analysis
            list_of_delta_z_parametrizations = []
            for z_par in range(0, 19 * 2 + 2 * number_of_extra_zernike):
                z_par_i = z_par + 4
                # if this is the parameter that change
                if np.mod(z_par_i, 2) == 0:
                    list_of_delta_z_parametrizations.append(0.1 * 0.05 / np.sqrt(z_par_i))
                if np.mod(z_par_i, 2) == 1:
                    list_of_delta_z_parametrizations.append(0.05 / np.sqrt(z_par_i))

            array_of_delta_z_parametrizations = np.array(list_of_delta_z_parametrizations) * (1)

            if iteration_number == 0:
                pass
            else:
                # array_of_delta_z_parametrizations=first_proposal_Tokovnin/4
                array_of_delta_z_parametrizations = np.maximum(
                    array_of_delta_z_parametrizations, first_proposal_Tokovnin / 4)  # noqa

            # this code might work with global parameters?
            array_of_delta_global_parametrizations = np.array([0.1, 0.02, 0.1, 0.1, 0.1, 0.1,
                                                               0.3, 1, 0.1, 0.1,
                                                               0.15, 0.15, 0.1,
                                                               0.07, 0.05, 0.05, 0.4,
                                                               30000, 0.5, 0.001,
                                                               0.05, 0.05, 0.01])
            # array_of_delta_global_parametrizations=array_of_delta_global_parametrizations/1
            array_of_delta_global_parametrizations = array_of_delta_global_parametrizations / 10

            if move_allparameters:
                array_of_delta_all_parametrizations = np.concatenate(
                    (array_of_delta_z_parametrizations[0:19 * 2], array_of_delta_global_parametrizations,
                     array_of_delta_z_parametrizations[19 * 2:]))

            if self.save:
                np.save(self.DIRECTORY + 'Results/array_of_delta_z_parametrizations_'
                        + str(num_iter) + '_' + str(iteration_number), array_of_delta_z_parametrizations)
                np.save(self.DIRECTORY + 'Results/array_of_delta_global_parametrizations_'
                        + str(num_iter) + '_' + str(iteration_number), array_of_delta_global_parametrizations)
                if move_allparameters:
                    np.save(self.DIRECTORY + 'Results/array_of_delta_all_parametrizations_'
                            + str(num_iter) + '_' + str(iteration_number),
                            array_of_delta_all_parametrizations)

            # initialize
            # if this is the first iteration of the iterative algorithm
            if iteration_number == 0:

                thresh = thresh0
                all_global_parametrization_old = allparameters_parametrization_proposal[19 * 2:19 * 2 + 23]
                if number_of_extra_zernike == 0:
                    all_wavefront_z_parametrization_old = allparameters_parametrization_proposal[0:19 * 2]
                else:
                    # if you want more Zernike
                    if len(allparameters_parametrization_proposal) == 19 * 2 + 23:
                        # if you did not pass explicit extra Zernike, start with zeroes
                        all_wavefront_z_parametrization_old = np.concatenate(
                            (allparameters_parametrization_proposal[0:19 * 2],
                             np.zeros(2 * number_of_extra_zernike)))
                    else:
                        all_wavefront_z_parametrization_old = np.concatenate(
                            (allparameters_parametrization_proposal[0:19 * 2],
                             allparameters_parametrization_proposal[19 * 2 + 23:]))

                pass
            # if this is not a first iteration
            else:
                # errors in the typechecker for 10 lines below are fine
                if self.verbosity == 1:
                    logging.info('array_of_delta_z in ' + str(iteration_number) + ' '
                                 + str(array_of_delta_z_parametrizations))
                # code analysis programs might suggest that there is an error here, but everything is ok
                # chi_2_before_iteration=np.copy(chi_2_after_iteration)
                # copy wavefront from the end of the previous iteration

                all_wavefront_z_parametrization_old = np.copy(all_wavefront_z_parametrization_new)  # noqa
                if move_allparameters:
                    all_global_parametrization_old = np.copy(all_global_parametrization_new)  # noqa
                if self.verbosity >= 1:
                    if did_chi_2_improve == 1:  # noqa
                        logging.info('did_chi_2_improve: yes')
                    else:
                        logging.info('did_chi_2_improve: no')
                if did_chi_2_improve == 0:  # noqa
                    thresh = thresh0
                else:
                    thresh = thresh * 0.5

            ##########################################################################
            # create a model with input parameters from previous iteration

            list_of_all_wavefront_z_parameterization = []

            up_to_z22_parametrization_start = all_wavefront_z_parametrization_old[0:19 * 2]
            from_z22_parametrization_start = all_wavefront_z_parametrization_old[19 * 2:]
            global_parametrization_start = all_global_parametrization_old

            if self.verbosity >= 1:
                logging.info('up_to_z22_parametrization_start: ' + str(up_to_z22_parametrization_start))
                logging.info('nonwavefront_par: ' + str(nonwavefront_par))
                logging.info('from_z22_parametrization_start' + str(from_z22_parametrization_start))

            # logging.info('iteration '+str(iteration_number)+' shape of up_to_z22_parametrization_start is:
            #    '+str(up_to_z22_parametrization_start.shape))
            if move_allparameters:
                initial_input_parameterization = np.concatenate(
                    (up_to_z22_parametrization_start, global_parametrization_start,
                     from_z22_parametrization_start))
            else:
                initial_input_parameterization = np.concatenate(
                    (up_to_z22_parametrization_start, nonwavefront_par, from_z22_parametrization_start))

            if self.verbosity >= 1:
                logging.info(
                    'initial input parameters in iteration ' + str(iteration_number) + ' are: '
                    + str(initial_input_parameterization))
                logging.info(
                    'moving input wavefront parameters in iteration ' + str(iteration_number) + ' by: '
                    + str(array_of_delta_z_parametrizations))
            if move_allparameters:
                logging.info(
                    'moving global input parameters in iteration ' + str(iteration_number) + ' by: '
                    + str(array_of_delta_global_parametrizations))

            if self.save:
                np.save(self.DIRECTORY + 'Results/initial_input_parameterization_'
                        + str(num_iter) + '_' + str(iteration_number), initial_input_parameterization)

            # logging.info('len initial_input_parameterization '+str(len(initial_input_parameterization)))

            list_of_minchain = model_multi.create_list_of_allparameters(
                initial_input_parameterization, list_of_defocuses=list_of_defocuses_input_long,
                zmax=self.zmax)
            # list_of_minchain=model_multi.create_list_of_allparameters(allparameters_parametrization_proposal,list_of_defocuses=list_of_defocuses_input_long,zmax=56)

            # moved in under `else` statment
            # res_multi=model_multi(list_of_minchain,return_Images=True,use_only_chi=use_only_chi,\
            #                      multi_background_factor=multi_background_factor)

            # if this is the first iteration take over the results from premodel run
            if iteration_number == 0:
                initial_model_result, list_of_initial_model_result, list_of_image_0,\
                    list_of_initial_input_parameters, list_of_pre_chi2, list_of_psf_positions =\
                    pre_model_result, model_results, pre_images, pre_input_parameters,\
                    chi_2_before_iteration_array, list_of_psf_positions
            else:
                res_multi = model_multi(list_of_minchain, return_Images=True, use_only_chi=use_only_chi,
                                        multi_background_factor=multi_background_factor)
                # mean_res_of_multi_same_spot_proposal,list_of_single_res_proposal,list_of_single_model_image_proposal,\
                #            list_of_single_allparameters_proposal,list_of_single_chi_results_proposal=res_multi
                initial_model_result, list_of_initial_model_result, list_of_image_0,\
                    list_of_initial_input_parameters, list_of_pre_chi2, list_of_psf_positions = res_multi
                # modify variance image according to the models that have just been created
                # second time modifying variance image
                list_of_single_model_image = list_of_image_0
                list_of_var_images_via_model = []
                for index_of_single_image in range(len(list_of_sci_images)):
                    popt = create_popt_for_custom_var(self.list_of_sci_images[index_of_single_image],
                                                      self.list_of_var_images[index_of_single_image],
                                                      self.list_of_mask_images[index_of_single_image])
                    single_var_image_via_model =\
                        create_custom_var_from_popt(list_of_single_model_image[index_of_single_image], popt)

                    list_of_var_images_via_model.append(single_var_image_via_model)
                # replace the variance images provided with these custom variance images
                list_of_var_images = list_of_var_images_via_model
                # self.list_of_var_images = list_of_var_images

            # initial_model_result,image_0,initial_input_parameters,pre_chi2=model(initial_input_parameters,return_Image=True,return_intermediate_images=False)
            if self.save:
                np.save(self.DIRECTORY + 'Results/list_of_initial_model_result_'
                        + str(num_iter) + '_' + str(iteration_number), list_of_initial_model_result)
                np.save(self.DIRECTORY + 'Results/list_of_image_0_' + str(num_iter) + '_'
                        + str(iteration_number), list_of_image_0)
                np.save(self.DIRECTORY + 'Results/list_of_initial_input_parameters_'
                        + str(num_iter) + '_' + str(iteration_number), list_of_initial_input_parameters)
                np.save(self.DIRECTORY + 'Results/list_of_pre_chi2_' + str(num_iter) + '_'
                        + str(iteration_number), list_of_pre_chi2)
                np.save(self.DIRECTORY + 'Results/list_of_psf_positions_' + str(num_iter) + '_'
                        + str(iteration_number), list_of_psf_positions)

            ##########################################################################
            # divided model images by their standard deviations

            list_of_image_0_std = []
            for i in range(len(list_of_image_0)):
                # normalizing by standard deviation image
                # May 22 modification
                STD = np.sqrt(list_of_var_images[i]) * array_of_std_sum[i]
                image_0 = list_of_image_0[i]
                list_of_image_0_std.append(image_0 / STD)

            ##########################################################################
            # updated science images divided by std (given that we created new custom
            # variance images, via model)

            ##########################################################################
            # mask model images at the start of this iteration, before modifying parameters
            # create uber_M0

            list_of_M0 = []
            list_of_M0_std = []
            for i in range(len(list_of_image_0_std)):

                image_0 = list_of_image_0[i]
                image_0_std = list_of_image_0_std[i]
                flux_mask = list_of_flux_mask[i]
                # what is list_of_mask_images?

                M0 = image_0[flux_mask].ravel()
                # M0=((image_0[flux_mask])/np.sum(image_0[flux_mask])).ravel()
                M0_std = ((image_0_std[flux_mask]) / 1).ravel()
                # M0_std=((image_0_std[flux_mask])/np.sum(image_0_std[flux_mask])).ravel()

                list_of_M0.append(M0)
                list_of_M0_std.append(M0_std)

            # join all M0,M0_std from invidiual images into one uber M0,M0_std
            uber_M0 = [item for sublist in list_of_M0 for item in sublist]
            uber_M0_std = [item for sublist in list_of_M0_std for item in sublist]

            uber_M0 = np.array(uber_M0)
            uber_M0_std = np.array(uber_M0_std)

            # uber_M0=uber_M0/np.sum(uber_M0)
            # uber_M0_std=uber_M0_std/np.sum(uber_M0_std)

            self.uber_M0 = uber_M0
            self.uber_M0_std = uber_M0_std

            if self.save:
                np.save(self.DIRECTORY + 'Results/uber_M0_' + str(num_iter) + '_' + str(iteration_number),
                        uber_M0)
                np.save(self.DIRECTORY + 'Results/uber_M0_std_' + str(num_iter) + '_' + str(iteration_number),
                        uber_M0_std)

            ##########################################################################
            # difference between model (uber_M0) and science (uber_I) at start of this iteration

            # non-std version
            # not used, that is ok, we are at the moment using std version
            IM_start = np.sum(np.abs(np.array(uber_I) - np.array(uber_M0)))
            # std version
            IM_start_std = np.sum(np.abs(np.array(uber_I_std) - np.array(uber_M0_std)))

            if len(list_of_flux_mask) > 1:
                IM_start_focus = np.sum(
                    np.abs(np.array(uber_I) - np.array(uber_M0))[position_focus_1:position_focus_2])
                IM_start_std_focus = np.sum(
                    np.abs(np.array(uber_I_std) - np.array(uber_M0_std))[position_focus_1:position_focus_2])

            # mean of differences of our images - should we use mean?; probably not... needs to be normalized?
            unitary_IM_start = np.mean(IM_start)
            unitary_IM_start_std = np.mean(IM_start_std)

            # logging.info list_of_IM_start_std
            if self.verbosity == 1:
                logging.info('np.sum(np.abs(I-M0)) before iteration ' + str(num_iter)
                             + '_' + str(iteration_number) + ': ' + str(unitary_IM_start))
                logging.info('np.sum(np.abs(I_std-M0_std)) before iteration ' + str(num_iter)
                             + '_' + str(iteration_number) + ': ' + str(unitary_IM_start_std))
            # logging.info('np.sum(np.abs(I_std-M0_std)) before iteration '+str(iteration_number)+':
            # '+str(unitary_IM_start_std))

            ##########################################################################
            # create list of new parametrizations to be tested
            # combine the old wavefront parametrization with the delta_z_parametrization

            # create two lists:
            # 1. one contains only wavefront parametrizations
            # 2. second contains the whole parametrizations
            # logging.info('checkpoint 0')
            if move_allparameters:
                list_of_all_wavefront_z_parameterization = []
                list_of_input_parameterizations = []
                for z_par in range(19 * 2):
                    all_wavefront_z_parametrization_list = np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par] =\
                        all_wavefront_z_parametrization_list[z_par] + \
                        array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)

                    up_to_z22_start = all_wavefront_z_parametrization_list[0:19 * 2]
                    from_z22_start = all_wavefront_z_parametrization_list[19 * 2:]

                    parametrization_proposal = np.concatenate(
                        (up_to_z22_start, nonwavefront_par, from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)
                    # logging.info('checkpoint 1')
                for g_par in range(23):
                    all_global_parametrization_list = np.copy(all_global_parametrization_old)
                    all_global_parametrization_list[g_par] = all_global_parametrization_list[g_par] + \
                        array_of_delta_global_parametrizations[g_par]
                    # list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)

                    up_to_z22_start = all_wavefront_z_parametrization_old[0:19 * 2]
                    from_z22_start = all_wavefront_z_parametrization_old[19 * 2:]

                    parametrization_proposal = np.concatenate(
                        (up_to_z22_start, all_global_parametrization_list, from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)
                    # logging.info('checkpoint 2')
                for z_par in range(19 * 2, len(all_wavefront_z_parametrization_old)):
                    all_wavefront_z_parametrization_list = np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par] =\
                        all_wavefront_z_parametrization_list[z_par] + \
                        array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)

                    up_to_z22_start = all_wavefront_z_parametrization_list[0:19 * 2]
                    from_z22_start = all_wavefront_z_parametrization_list[19 * 2:]

                    parametrization_proposal = np.concatenate(
                        (up_to_z22_start, nonwavefront_par, from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)
                    # logging.info('checkpoint 3')

            else:
                list_of_all_wavefront_z_parameterization = []
                list_of_input_parameterizations = []
                for z_par in range(len(all_wavefront_z_parametrization_old)):
                    all_wavefront_z_parametrization_list = np.copy(all_wavefront_z_parametrization_old)
                    all_wavefront_z_parametrization_list[z_par] =\
                        all_wavefront_z_parametrization_list[z_par] + \
                        array_of_delta_z_parametrizations[z_par]
                    list_of_all_wavefront_z_parameterization.append(all_wavefront_z_parametrization_list)

                    up_to_z22_start = all_wavefront_z_parametrization_list[0:19 * 2]
                    from_z22_start = all_wavefront_z_parametrization_list[19 * 2:]

                    parametrization_proposal = np.concatenate(
                        (up_to_z22_start, nonwavefront_par, from_z22_start))
                    # actually it is parametrization
                    list_of_input_parameterizations.append(parametrization_proposal)
                    # logging.info('checkpoint 4')

            ##########################################################################
            # Starting testing new set of parameters
            # Creating new images

            out_ln = []
            out_ln_ind = []
            out_images = []
            out_parameters = []
            out_chi2 = []
            out_pfs_positions = []

            if self.verbosity >= 1:
                logging.info(
                    'We are now inside of the pool loop number ' + str(iteration_number)
                    + ' with num_iter: ' + str(num_iter))

            # actually it is parametrization
            # list of (56-3)*2 sublists, each one with (56-3)*2 + 23 values
            time_start = time.time()

            # This assume that Zernike parameters go up to 56
            # I need to pass each of 106 parametrization to model_multi BUT
            # model_multi actually takes list of parameters, not parametrizations
            # I need list that has 106 sublists, each one of those being 9x(53+23)
            # 9 == number of images
            # 53 == number of Zernike parameters (56-3)
            # 23 == number of global parameters
            uber_list_of_input_parameters = []
            for i in range(len(list_of_input_parameterizations)):

                list_of_input_parameters = model_multi.create_list_of_allparameters(
                    list_of_input_parameterizations[i],
                    list_of_defocuses=list_of_defocuses_input_long, zmax=self.zmax)
                uber_list_of_input_parameters.append(list_of_input_parameters)

            # save the uber_list_of_input_parameters
            if self.save:
                np.save(self.DIRECTORY + 'Results/uber_list_of_input_parameters_'
                        + str(num_iter) + '_' + str(iteration_number), uber_list_of_input_parameters)

            # pass new model_multi that has fixed pos (October 6, 2020)
            # should have same paramter as staring model_multi, apart from
            # list_of_psf_positions (maybe variance?, but prob not)
            model_multi_out = LN_PFS_multi_same_spot(
                list_of_sci_images,
                list_of_var_images,
                list_of_mask_images=list_of_mask_images,
                wavelength=self.wavelength,
                dithering=self.dithering,
                save=self.save,
                zmax=self.zmax,
                verbosity=self.verbosity_model,
                double_sources=self.double_sources,
                double_sources_positions_ratios=double_sources_positions_ratios,
                npix=self.npix,
                fit_for_flux=self.fit_for_flux,
                test_run=self.test_run,
                list_of_psf_positions=list_of_psf_positions)

            if move_allparameters:
                self.array_of_delta_all_parametrizations = array_of_delta_all_parametrizations
            else:
                self.array_of_delta_z_parametrizations = array_of_delta_z_parametrizations

            # start of creating H

            # H is normalized difference between pixels of the model image
            # that result from changing the j-th Zernike term compared to the original image
            # This is expensive because we have to generate new image for each Zernike term
            if previous_best_result is None:
                if self.verbosity >= 1:
                    logging.info('self.pool parameter is: ' + str(self.pool))

                # generate images
                if self.pool is None:
                    out1 = map(
                        partial(
                            model_multi_out,
                            return_Images=True,
                            use_only_chi=use_only_chi,
                            multi_background_factor=multi_background_factor),
                        uber_list_of_input_parameters)
                else:
                    out1 = self.pool.map(
                        partial(
                            model_multi_out,
                            return_Images=True,
                            use_only_chi=use_only_chi,
                            multi_background_factor=multi_background_factor),
                        uber_list_of_input_parameters)
                out1 = list(out1)
                time_end = time.time()
                if self.verbosity >= 1:
                    logging.info('time_end-time_start for creating model_multi_out '
                                 + str(time_end - time_start))

                # normalization of the preinput run? (what did I mean by that)
                pre_input_parameters = np.array(pre_input_parameters)
                if self.verbosity >= 1:
                    logging.info('pre_input_parameters.shape ' + str(pre_input_parameters.shape))
                    logging.info('pre_input_parameters[0][0:5] ' + str(pre_input_parameters[0][0:5]))

                # select the column specifying the flux normalization from the input images
                array_of_normalizations_pre_input = pre_input_parameters[:, 41]

                # out1=a_pool.map(model,input_parameters,repeat(True))
                for i in range(len(uber_list_of_input_parameters)):
                    # logging.info(i)

                    #    initial_model_result,list_of_initial_model_result,list_of_image_0,\
                    #                list_of_initial_input_parameters,list_of_pre_chi2

                    # outputs are
                    # 0. mean likelihood
                    # 1. list of individual res (likelihood)
                    # 2. list of science images
                    # 3. list of parameters used
                    # 4. list of quality measurments

                    out_images_pre_renormalization = np.array(out1[i][2])
                    out_parameters_single_move = np.array(out1[i][3])
                    # replace the normalizations in the output imags with the normalizations
                    # from the input images
                    array_of_normalizations_out = out_parameters_single_move[:, 41]
                    out_renormalization_parameters = array_of_normalizations_pre_input /\
                        array_of_normalizations_out

                    out_ln.append(out1[i][0])
                    out_ln_ind.append(out1[i][1])
                    # logging.info('out_images_pre_renormalization.shape: '+
                    # str(out_images_pre_renormalization.shape))
                    # logging.info('out_renormalization_parameters.shape: '+
                    # str(out_renormalization_parameters.shape))
                    # np.save('/tigress/ncaplar/Results/out_images_pre_renormalization',
                    # out_images_pre_renormalization)

                    out_images_step = []
                    for lv in range(len(out_renormalization_parameters)):
                        out_images_step.append(
                            out_images_pre_renormalization[lv]
                            * out_renormalization_parameters[lv])
                    out_images.append(out_images_step)

                    out_parameters.append(out1[i][3])
                    out_chi2.append(out1[i][4])
                    out_pfs_positions.append(out1[i][5])

                    # We use these out_images to study the differences due to changing parameters;
                    # We do not want the normalization to affect things (and position of optical center)
                    # so we renormalized to that multiplication constants are the same as in the input

                time_end = time.time()
                if self.verbosity >= 1:
                    logging.info('time_end-time_start for whole model_multi_out '
                                 + str(time_end - time_start))

                if self.save:
                    np.save(
                        self.DIRECTORY + 'Results/out_images_' + str(num_iter) + '_'
                        + str(iteration_number), out_images)
                    np.save(
                        self.DIRECTORY + 'Results/out_parameters_' + str(num_iter) + '_'
                        + str(iteration_number), out_parameters)
                    np.save(
                        self.DIRECTORY + 'Results/out_chi2_' + str(num_iter) + '_'
                        + str(iteration_number), out_chi2)

                ##########################################################################
                # Normalize created images

                # We created ((zmax-3)*2) x N images, where N is the number of defocused images

                # join all images together
                list_of_images_normalized_uber = []
                # list_of_images_normalized_std_uber = []
                # go over (zmax-3)*2 images
                for j in range(len(out_images)):
                    # two steps for what could have been achived in one, but to ease up
                    # transition from previous code
                    out_images_single_parameter_change = out_images[j]
                    optpsf_list = out_images_single_parameter_change

                    # flux image has to correct per image
                    # mask images that have been created in the fitting procedure with the
                    # appropriate flux mask
                    images_normalized = []
                    for i in range(len(optpsf_list)):
                        flux_mask = list_of_flux_mask[i]
                        images_normalized.append((optpsf_list[i][flux_mask]).ravel())

                    images_normalized_flat = [item for sublist in images_normalized for item in sublist]
                    images_normalized_flat = np.array(images_normalized_flat)

                    # list of (zmax-3)*2 raveled images
                    list_of_images_normalized_uber.append(images_normalized_flat)

                    # same but divided by STD
                    # images_normalized_std=[]
                    # for i in range(len(optpsf_list)):
                    # seems that I am a bit more verbose here with my definitions
                    # optpsf_list_i=optpsf_list[i]

                    # do I want to generate new STD images, from each image?
                    # May 22 modification
                    # STD=list_of_sci_image_std[i]*array_of_std_sum[i]
                    # optpsf_list_i_STD=optpsf_list_i/STD
                    # flux_mask=list_of_flux_mask[i]
                    # images_normalized_std.append((optpsf_list_i_STD[flux_mask]/np.sum(optpsf_list_i_STD[flux_mask])).ravel())

                    # join all images together
                    # images_normalized_std_flat=
                    # [item for sublist in images_normalized_std for item in sublist]
                    # normalize so that the sum is still one
                    # images_normalized_std_flat=np.array(images_normalized_std_flat)/len(optpsf_list)

                    # list_of_images_normalized_std_uber.append(images_normalized_std_flat)

                # create uber images_normalized,images_normalized_std
                # images that have zmax*2 rows and very large number of columns (number of
                # non-masked pixels from all N images)
                uber_images_normalized = np.array(list_of_images_normalized_uber)
                # uber_images_normalized_std=np.array(list_of_images_normalized_std_uber)

                if self.save:
                    np.save(
                        self.DIRECTORY + 'Results/uber_images_normalized_' + str(num_iter) + '_'
                        + str(iteration_number), uber_images_normalized)

                # np.save('/tigress/ncaplar/Results/uber_images_normalized_std_'+str(num_iter)+'_'+str(iteration_number),\
                #        uber_images_normalized_std)

                # single_wavefront_parameter_list=[]
                # for i in range(len(out_parameters)):
                #    single_wavefront_parameter_list.
                # append(np.concatenate((out_parameters[i][:19],out_parameters[i][42:])) )

                ##########################################################################
                # Core Tokovinin algorithm

                if self.verbosity >= 1:
                    logging.info('images_normalized (uber).shape: ' + str(uber_images_normalized.shape))
                    logging.info('array_of_delta_z_parametrizations[:,None].shape'
                                 + str(array_of_delta_z_parametrizations[:, None].shape))
                # equation A1 from Tokovinin 2006
                # new model minus old model
                if move_allparameters:
                    H = np.transpose(np.array((uber_images_normalized - uber_M0))
                                     / array_of_delta_all_parametrizations[:, None])
                    # H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/\
                    #    array_of_delta_z_parametrizations[:,None])
                    H_std = np.transpose(np.array((uber_images_normalized - uber_M0))
                                         / array_of_delta_all_parametrizations[:, None]) /\
                        uber_std.ravel()[:, None]
                else:
                    H = np.transpose(np.array((uber_images_normalized - uber_M0))
                                     / array_of_delta_z_parametrizations[:, None])
                    # H_std=np.transpose(np.array((uber_images_normalized_std-uber_M0_std))/array_of_delta_z_parametrizations[:,None])
                    H_std = np.transpose(np.array((uber_images_normalized - uber_M0))
                                         / array_of_delta_z_parametrizations[:, None]) /\
                        uber_std.ravel()[:, None]

                array_of_delta_z_parametrizations_None = np.copy(array_of_delta_z_parametrizations[:, None])
            else:
                H = self.create_simplified_H(previous_best_result)
                H_std = H / uber_std.ravel()[:, None]

            # end of creating H

            if self.save and previous_best_result is None:
                np.save(self.DIRECTORY + 'Results/array_of_delta_z_parametrizations_None_'
                        + str(num_iter) + '_' + str(iteration_number),
                        array_of_delta_z_parametrizations_None)

            if self.save:
                np.save(self.DIRECTORY + 'Results/H_' + str(num_iter) + '_' + str(iteration_number), H)
            if self.save:
                np.save(self.DIRECTORY + 'Results/H_std_' + str(num_iter) + '_' + str(iteration_number),
                        H_std)

            first_proposal_Tokovnin, first_proposal_Tokovnin_std = self.create_first_proposal_Tokovnin(
                H, H_std, uber_I, uber_M0, uber_std, up_to_which_z=up_to_which_z)

            """
            #logging.info('np.mean(H,axis=0).shape)'+str(np.mean(H,axis=0).shape))
            singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))<0.01]
            non_singlular_parameters=np.arange(H.shape[1])[np.abs((np.mean(H,axis=0)))>0.01]
            #logging.info('non_singlular_parameters.shape)'+str(non_singlular_parameters.shape))
            H=H[:,non_singlular_parameters]
            H_std=H_std[:,non_singlular_parameters]

            HHt=np.matmul(np.transpose(H),H)
            HHt_std=np.matmul(np.transpose(H_std),H_std)
            #logging.info('svd thresh is '+str(thresh))
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



            # if you have removed certain parameters because of the singularity,
            return them here, with no change
            if len(singlular_parameters)>0:
                for i in range(len(singlular_parameters)):
                    first_proposal_Tokovnin=np.insert(first_proposal_Tokovnin,singlular_parameters[i],0)
                    first_proposal_Tokovnin_std=np.insert(first_proposal_Tokovnin_std,singlular_parameters[i],0)
            #logging.info('first_proposal_Tokovnin_std'+str(first_proposal_Tokovnin_std.shape))
            #logging.info('invHHtHt_std.shape'+str(invHHtHt_std.shape))

            """

            if self.verbosity >= 1:
                logging.info('first_proposal_Tokovnin[:5] is: '
                             + str(first_proposal_Tokovnin[:8 * 2]))
                logging.info('first_proposal_Tokovnin_std[:5] is: '
                             + str(first_proposal_Tokovnin_std[:8 * 2]))
                try:
                    logging.info('ratio is of proposed to initial parameters (std) is: '
                                 + str(first_proposal_Tokovnin_std / array_of_delta_z_parametrizations))
                except BaseException:
                    pass

            # Tokovnin_proposal=0.7*first_proposal_Tokovnin
            if move_allparameters:
                Tokovnin_proposal = np.zeros((129,))
                # Tokovnin_proposal[non_singlular_parameters]=0.7*first_proposal_Tokovnin_std
                Tokovnin_proposal[non_singlular_parameters] = 1 * first_proposal_Tokovnin_std  # noqa

                all_parametrization_new = np.copy(initial_input_parameterization)
                allparameters_parametrization_proposal_after_iteration_before_global_check =\
                    all_parametrization_new + Tokovnin_proposal
                # tests if the global parameters would be out of bounds - if yes, reset
                # them to the limit values
                # noqa: E501 - breaking line limit in order to keep informative names
                global_parametrization_proposal_after_iteration_before_global_check =\
                    allparameters_parametrization_proposal_after_iteration_before_global_check[19 * 2:19 * 2 + 23]  # noqa: E501
                checked_global_parameters = check_global_parameters(
                    global_parametrization_proposal_after_iteration_before_global_check, test_print=1)

                allparameters_parametrization_proposal_after_iteration = np.copy(
                    allparameters_parametrization_proposal_after_iteration_before_global_check)
                allparameters_parametrization_proposal_after_iteration[19 * 2:19 * 2 + 23] =\
                    checked_global_parameters

            else:
                # Tokovnin_proposal=0.7*first_proposal_Tokovnin_std
                Tokovnin_proposal = 1 * first_proposal_Tokovnin_std

            if self.verbosity >= 1:
                logging.info('Tokovnin_proposal[:5] is: ' + str(Tokovnin_proposal[:5]))
                if self.zmax > 35:
                    logging.info('Tokovnin_proposal[38:43] is: ' + str(Tokovnin_proposal[38:43]))
            # logging.info('all_wavefront_z_parametrization_old in '+str(iteration_number)+' '+
            # str(all_wavefront_z_parametrization_old[:5]))
            # logging.info('Tokovnin_proposal[:5] is: '+str(Tokovnin_proposal[:5]))
            # logging.info('Tokovnin_proposal.shape '+str(Tokovnin_proposal.shape))

            # if the Tokovinin proposal is not made, return the initial result
            if len(Tokovnin_proposal) < 10:
                # return initial_model_result,list_of_initial_model_result,list_of_image_0,\
                #    allparameters_parametrization_proposal,list_of_initial_input_parameters,list_of_pre_chi2,list_of_psf_positions
                return initial_model_result, initial_model_result,\
                    list_of_initial_model_result, list_of_initial_model_result,\
                    out_images, list_of_image_0, list_of_image_0,\
                    allparameters_parametrization_proposal, allparameters_parametrization_proposal,\
                    list_of_initial_input_parameters, list_of_initial_input_parameters,\
                    list_of_pre_chi2, list_of_pre_chi2,\
                    list_of_psf_positions, list_of_psf_positions

                break

            # logging.info('std of Tokovnin_proposal is: '+str(np.std(Tokovnin_proposal)))
            if move_allparameters:
                # all_wavefront_z_parametrization_new=np.copy(all_wavefront_z_parametrization_old)
                # all_global_parametrization_new=np.copy(all_global_parametrization_old)
                # all_parametrization_new=np.copy(initial_input_parameterization)

                # allparameters_parametrization_proposal_after_iteration=all_parametrization_new+Tokovnin_proposal

                up_to_z22_end = allparameters_parametrization_proposal_after_iteration[:19 * 2]
                from_z22_end = allparameters_parametrization_proposal_after_iteration[19 * 2 + 23:]
                all_wavefront_z_parametrization_new = np.concatenate((up_to_z22_end, from_z22_end))

                # all_global_parametrization_new = allparameters_parametrization_proposal_after_iteration[
                #    19 * 2:19 * 2 + 23]

            else:
                all_wavefront_z_parametrization_new = np.copy(all_wavefront_z_parametrization_old)
                all_wavefront_z_parametrization_new = all_wavefront_z_parametrization_new + Tokovnin_proposal
                up_to_z22_end = all_wavefront_z_parametrization_new[:19 * 2]
                from_z22_end = all_wavefront_z_parametrization_new[19 * 2:]
                allparameters_parametrization_proposal_after_iteration = np.concatenate(
                    (up_to_z22_end, nonwavefront_par, from_z22_end))

            if self.save:
                np.save(
                    self.DIRECTORY + 'Results/first_proposal_Tokovnin' + str(num_iter) + '_'
                    + str(iteration_number), first_proposal_Tokovnin)
                np.save(
                    self.DIRECTORY + 'Results/first_proposal_Tokovnin_std' + str(num_iter) + '_'
                    + str(iteration_number), first_proposal_Tokovnin_std)
                np.save(
                    self.DIRECTORY + 'Results/allparameters_parametrization_proposal_after_iteration_'
                    + str(num_iter) + '_' + str(iteration_number),
                    allparameters_parametrization_proposal_after_iteration)

            #########################
            # Creating single exposure with new proposed parameters and seeing if there is improvment
            time_start_final = time.time()

            list_of_parameters_after_iteration = model_multi.create_list_of_allparameters(
                allparameters_parametrization_proposal_after_iteration,
                list_of_defocuses=list_of_defocuses_input_long,
                zmax=self.zmax)
            res_multi = model_multi(
                list_of_parameters_after_iteration,
                return_Images=True,
                use_only_chi=use_only_chi,
                multi_background_factor=multi_background_factor)

            if self.verbosity >= 1:
                logging.info('allparameters_parametrization_proposal_after_iteration '
                             + str(allparameters_parametrization_proposal_after_iteration[0:5]))
                logging.info('list_of_parameters_after_iteration[0][0:5] '
                             + str(list_of_parameters_after_iteration[0][0:5]))

            final_model_result, list_of_final_model_result, list_of_image_final,\
                list_of_finalinput_parameters, list_of_after_chi2, list_of_final_psf_positions = res_multi
            # third (last?) time modifying variance image
            list_of_single_model_image = list_of_image_final
            list_of_var_images_via_model = []
            for index_of_single_image in range(len(list_of_sci_images)):
                popt = create_popt_for_custom_var(self.list_of_sci_images[index_of_single_image],
                                                  self.list_of_var_images[index_of_single_image],
                                                  self.list_of_mask_images[index_of_single_image])
                single_var_image_via_model =\
                    create_custom_var_from_popt(list_of_single_model_image[index_of_single_image], popt)

                list_of_var_images_via_model.append(single_var_image_via_model)
            # replace the variance images provided with these custom variance images
            list_of_var_images = list_of_var_images_via_model
            # self.list_of_var_images = list_of_var_images

            time_end_final = time.time()
            if self.verbosity >= 1:
                logging.info('Total time taken for final iteration was ' + str(time_end_final
                                                                               - time_start_final)
                             + ' seconds with num_iter: ' + str(num_iter))

            if self.save:
                np.save(self.DIRECTORY + 'Results/list_of_final_model_result_'
                        + str(num_iter) + '_' + str(iteration_number), list_of_final_model_result)
                np.save(
                    self.DIRECTORY + 'Results/list_of_image_final_' + str(num_iter) + '_'
                    + str(iteration_number), list_of_image_final)
                np.save(self.DIRECTORY + 'Results/list_of_finalinput_parameters_'
                        + str(num_iter) + '_' + str(iteration_number), list_of_finalinput_parameters)
                np.save(self.DIRECTORY + 'Results/list_of_after_chi2_' + str(num_iter) + '_'
                        + str(iteration_number), list_of_after_chi2)
                np.save(self.DIRECTORY + 'Results/list_of_final_psf_positions_'
                        + str(num_iter) + '_' + str(iteration_number), list_of_final_psf_positions)

            if self.verbosity >= 1:
                logging.info('list_of_final_psf_positions : ' + str(list_of_psf_positions))

            ##########################################################################
            # divided model images by their standard deviations

            list_of_image_final_std = []
            for i in range(len(list_of_image_0)):
                # normalizing by standard deviation image
                # May 22 modification
                STD = np.sqrt(list_of_var_images[i]) * array_of_std_sum[i]
                image_final = list_of_image_final[i]
                list_of_image_final_std.append(image_final / STD)

            ##########################################################################
            #  masked model images after this iteration (mask by flux criteria)

            list_of_M_final = []
            list_of_M_final_std = []
            for i in range(len(list_of_image_final_std)):

                image_final = list_of_image_final[i]
                image_final_std = list_of_image_final_std[i]
                flux_mask = list_of_flux_mask[i]
                # what is list_of_mask_images?

                # M_final=((image_final[flux_mask])/np.sum(image_final[flux_mask])).ravel()
                M_final = (image_final[flux_mask]).ravel()
                # M_final_std=((image_final_std[flux_mask])/np.sum(image_final_std[flux_mask])).ravel()
                M_final_std = ((image_final_std[flux_mask]) / 1).ravel()

                list_of_M_final.append(M_final)
                list_of_M_final_std.append(M_final_std)

            # join all M0,M0_std from invidiual images into one uber M0,M0_std
            uber_M_final = [item for sublist in list_of_M_final for item in sublist]
            uber_M_final_std = [item for sublist in list_of_M_final_std for item in sublist]

            uber_M_final = np.array(uber_M_final)
            uber_M_final_std = np.array(uber_M_final_std)

            uber_M_final_linear_prediction = uber_M0 + \
                self.create_linear_aproximation_prediction(H, first_proposal_Tokovnin)
            uber_M_final_std_linear_prediction = uber_M0_std + \
                self.create_linear_aproximation_prediction(H_std, first_proposal_Tokovnin_std)

            if self.save:
                np.save(
                    self.DIRECTORY + 'Results/uber_M_final_' + str(num_iter) + '_'
                    + str(iteration_number), uber_M_final)
                np.save(
                    self.DIRECTORY + 'Results/uber_M_final_std_' + str(num_iter) + '_'
                    + str(iteration_number), uber_M_final_std)
            if self.save:
                np.save(self.DIRECTORY + 'Results/uber_M_final_linear_prediction_'
                        + str(num_iter) + '_' + str(iteration_number), uber_M_final_linear_prediction)
                np.save(self.DIRECTORY + 'Results/uber_M_final_std_linear_prediction_'
                        + str(num_iter) + '_' + str(iteration_number), uber_M_final_std_linear_prediction)

            ####
            # Seeing if there is an improvment
            # Quality measure is the sum of absolute differences of uber_I_std (all images/std)
            # and uber_M_final_std (all models / std)
            # how closely is that correlated with improvments in final_model_result?

            # non-std version
            # not used, that is ok, we are at the moment using std version
            IM_final = np.sum(np.abs(np.array(uber_I) - np.array(uber_M_final)))
            # std version
            IM_final_std = np.sum(np.abs(np.array(uber_I_std) - np.array(uber_M_final_std)))

            # linear prediction versions
            IM_final_linear_prediction = np.sum(
                np.abs(np.array(uber_I) - np.array(uber_M_final_linear_prediction)))
            # std version
            IM_final_std_linear_prediction = np.sum(
                np.abs(np.array(uber_I_std) - np.array(uber_M_final_std_linear_prediction)))

            # do a separate check on the improvment measure for the image in focus, when applicable
            if len(list_of_flux_mask) > 1:
                IM_final_focus = np.sum(
                    np.abs(np.array(uber_I) - np.array(uber_M_final))[position_focus_1:position_focus_2])
                IM_final_std_focus = np.sum(
                    np.abs(np.array(uber_I_std)
                           - np.array(uber_M_final_std))[position_focus_1:position_focus_2])

            if self.verbosity >= 1:
                logging.info('I-M_start before iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_start))
                logging.info('I-M_final after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final))
                logging.info('IM_final_linear_prediction after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final_linear_prediction))
                if len(list_of_flux_mask) > 1:
                    logging.info('I-M_start_focus before iteration ' + str(iteration_number)
                                 + ' with num_iter ' + str(num_iter) + ': ' + str(IM_start_focus))
                    logging.info('I-M_final_focus after iteration ' + str(iteration_number)
                                 + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final_focus))

                logging.info('I_std-M_start_std after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_start_std))
                logging.info('I_std-M_final_std after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final_std))
                logging.info('IM_final_std_linear_prediction after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final_std_linear_prediction))
                if len(list_of_flux_mask) > 1:
                    logging.info('I-M_start_focus_std before iteration ' + str(iteration_number)
                                 + ' with num_iter ' + str(num_iter) + ': ' + str(IM_start_std_focus))
                    logging.info('I-M_final_focus_std after iteration ' + str(iteration_number)
                                 + ' with num_iter ' + str(num_iter) + ': ' + str(IM_final_std_focus))

                logging.info('Likelihood before iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(initial_model_result))
                logging.info('Likelihood after iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(final_model_result))

                logging.info(
                    'Likelihood before iteration  '
                    + str(iteration_number)
                    + ' with num_iter '
                    + str(num_iter)
                    + ', per image: '
                    + str(list_of_initial_model_result))
                logging.info(
                    'Likelihood after iteration '
                    + str(iteration_number)
                    + ' with num_iter '
                    + str(num_iter)
                    + ', per image: '
                    + str(list_of_final_model_result))

                # logging.info('chi_2_after_iteration/chi_2_before_iteration '+
                # str(chi_2_after_iteration/chi_2_before_iteration ))
                logging.info('IM_final/IM_start with num_iter ' + str(num_iter)
                             + ': ' + str(IM_final / IM_start))
                logging.info('IM_final_std/IM_start_std with num_iter '
                             + str(num_iter) + ': ' + str(IM_final_std / IM_start_std))
                if len(list_of_flux_mask) > 1:
                    logging.info('IM_final_focus/IM_start_focus with num_iter '
                                 + str(num_iter) + ': ' + str(IM_final_focus / IM_start_focus))
                    logging.info('IM_final_std_focus/IM_start_std_focus with num_iter '
                                 + str(num_iter) + ': ' + str(IM_final_std_focus / IM_start_std_focus))

                logging.info('#########################################################')

            ##################
            # If improved take new parameters, if not dont

            # TEST, May18 2021
            # if more images, test that everything AND focus image has improved
            if len(list_of_flux_mask) > 1:
                if IM_final_std / IM_start_std < 1.0 and IM_final_std_focus / IM_start_std_focus < 1.25:
                    condition_for_improvment = True
                else:
                    condition_for_improvment = False
            else:
                # if you are having only one image
                if IM_final_std / IM_start_std < 1.0:
                    condition_for_improvment = True

            if self.verbosity >= 1:
                logging.info('condition_for_improvment in iteration ' + str(iteration_number)
                             + ' with num_iter ' + str(num_iter) + ': ' + str(condition_for_improvment))
            if condition_for_improvment:
                # when the quality measure did improve
                did_chi_2_improve = 1  # noqa
                number_of_non_decreses.append(0)
                if self.verbosity >= 1:
                    logging.info('number_of_non_decreses:' + str(number_of_non_decreses))
                    logging.info('current value of number_of_non_decreses is: '
                                 + str(np.sum(number_of_non_decreses)))
                    logging.info('#########################################')
                    logging.info('#########################################')
            else:
                # when the quality measure did not improve
                did_chi_2_improve = 0  # noqa
                # resetting all parameters
                if move_allparameters:
                    all_wavefront_z_parametrization_new = np.copy(all_wavefront_z_parametrization_old)
                    # all_global_parametrization_new = np.copy(all_global_parametrization_old)
                    allparameters_parametrization_proposal_after_iteration = initial_input_parameterization
                else:
                    all_wavefront_z_parametrization_new = np.copy(all_wavefront_z_parametrization_old)
                    # chi_2_after_iteration=chi_2_before_iteration
                    up_to_z22_end = all_wavefront_z_parametrization_new[:19 * 2]
                    from_z22_start = all_wavefront_z_parametrization_new[19 * 2:]
                    allparameters_parametrization_proposal_after_iteration = np.concatenate(
                        (up_to_z22_start, nonwavefront_par, from_z22_start))
                thresh = thresh0
                number_of_non_decreses.append(1)
                if self.verbosity >= 1:
                    logging.info('number_of_non_decreses:' + str(number_of_non_decreses))
                    logging.info('current value of number_of_non_decreses is: '
                                 + str(np.sum(number_of_non_decreses)))
                    logging.info('#######################################################')
                    logging.info('#######################################################')

                final_model_result = initial_model_result
                list_of_final_model_result = list_of_initial_model_result
                list_of_image_final = pre_images
                allparameters_parametrization_proposal_after_iteration =\
                    allparameters_parametrization_proposal
                list_of_finalinput_parameters = list_of_initial_input_parameters
                list_of_after_chi2 = list_of_pre_chi2
                list_of_final_psf_positions = list_of_psf_positions

            if np.sum(number_of_non_decreses) == 1:
                if not return_Images:
                    return final_model_result
                else:
                    if previous_best_result is None:
                        return initial_model_result, final_model_result,\
                            list_of_initial_model_result, list_of_final_model_result,\
                            out_images, pre_images, list_of_image_final,\
                            allparameters_parametrization_proposal,\
                            allparameters_parametrization_proposal_after_iteration,\
                            list_of_initial_input_parameters, list_of_finalinput_parameters,\
                            list_of_pre_chi2, list_of_after_chi2,\
                            list_of_psf_positions, list_of_final_psf_positions,\
                            [uber_images_normalized, uber_M0_std, H_std,
                             array_of_delta_z_parametrizations_None, list_of_final_psf_positions]
                    else:
                        return initial_model_result, final_model_result,\
                            list_of_initial_model_result, list_of_final_model_result,\
                            out_images, pre_images, list_of_image_final,\
                            allparameters_parametrization_proposal,\
                            allparameters_parametrization_proposal_after_iteration,\
                            list_of_initial_input_parameters, list_of_finalinput_parameters,\
                            list_of_pre_chi2, list_of_after_chi2,\
                            list_of_psf_positions, list_of_final_psf_positions

                break

        # if return_Images==False just return the mean likelihood
        if not return_Images:
            return final_model_result
        else:
            # if you return images, return full
            if previous_best_result is None:
                # 0. likelihood averaged over all images (before the function)
                # 1. likelihood averaged over all images (after the function)
                # 2. likelihood per image (output from model_multi) (before the function)
                # 3. likelihood per image (output from model_multi) (after the function)
                # 4. out_images
                # 5. list of initial model images
                # 6. list of final model images
                # 7. parametrization before the function
                # 8. parametrization after the function
                # 9. list of parameters per image (before the function)
                # 10. list of parameters per image (after the function)
                # 11. list of chi2 per image (before the function)
                # 12. list of chi2 per image (after the function)
                # 13. list of psf position of image (before the function)
                # 14. list of psf position of image (after the function)
                return initial_model_result, final_model_result,\
                    list_of_initial_model_result, list_of_final_model_result,\
                    out_images, pre_images, list_of_image_final,\
                    allparameters_parametrization_proposal,\
                    allparameters_parametrization_proposal_after_iteration,\
                    list_of_initial_input_parameters, list_of_finalinput_parameters,\
                    list_of_pre_chi2, list_of_after_chi2,\
                    list_of_psf_positions, list_of_final_psf_positions,\
                    [uber_images_normalized, uber_M0_std, H_std,
                     array_of_delta_z_parametrizations_None, list_of_final_psf_positions]
                # return final_model_result,list_of_final_model_result,out_images,list_of_image_final,\
                #    allparameters_parametrization_proposal_after_iteration,list_of_finalinput_parameters,\
                #        list_of_after_chi2,list_of_final_psf_positions,\
                #        [uber_images_normalized,uber_M0_std,H_std,array_of_delta_z_parametrizations_None,list_of_final_psf_positions]

            else:
                return initial_model_result, final_model_result,\
                    list_of_initial_model_result, list_of_final_model_result,\
                    out_images, pre_images, list_of_image_final,\
                    allparameters_parametrization_proposal,\
                    allparameters_parametrization_proposal_after_iteration,\
                    list_of_initial_input_parameters, list_of_finalinput_parameters,\
                    list_of_pre_chi2, list_of_after_chi2,\
                    list_of_psf_positions, list_of_final_psf_positions

    def create_simplified_H(self, previous_best_result, multi_background_factor=3):
        """create matrix `H` using the provided (previously made) images and changes

        The simplification comes from the assumtion that the changes are still
        the same in this iteration as in the previous iteration

        Parameters
        ----------
        previous_best_result: `np.array?`
            The arrays with the output from ?

        Returns
        ----------
        H : `np.array`
            normalized difference between pixel values
        """

        # to be compatable with version before 0.45 where previous_best_result
        # was actually only the last part of len=5
        #
        if len(previous_best_result) == 5:
            previous_best_result = previous_best_result
        else:
            # if you are passing the whole best result, separte the parts of the result
            main_body_of_best_result = previous_best_result[:-1]
            previous_best_result = previous_best_result[-1]

        # we need actual final model images from the previous best result
        # list_of_image_0_from_previous_best_result=main_body_of_best_result[6]
        # list_of_image_0=list_of_image_0_from_previous_best_result

        # we need actual initial model images from the previous best result
        # this will be used to evalute change of model due to changes in singel wavefront parameters
        # i.e., to estimate matrix H
        list_of_image_0_from_previous_best_result = main_body_of_best_result[5]
        list_of_image_0 = list_of_image_0_from_previous_best_result

        list_of_flux_mask = self.list_of_flux_mask

        ##########################################################################
        # divided model images by their standard deviations
        # list_of_image_0_std = []
        # for i in range(len(list_of_image_0)):
        # normalizing by standard deviation image
        # STD = np.sqrt(list_of_var_images[i])
        # image_0 = list_of_image_0[i]
        # list_of_image_0_std.append(image_0 / STD)

        ########################################################

        # mask model images at the start of this iteration, before modifying parameters
        # create uber_M0_previous_best - uber_M0 derived from previous best, but with current flux mask

        list_of_M0 = []
        # list_of_M0_std = []
        for i in range(len(list_of_image_0)):

            image_0 = list_of_image_0[i]
            # image_0_std = list_of_image_0_std[i]
            flux_mask = list_of_flux_mask[i]

            M0 = image_0[flux_mask].ravel()
            # M0=((image_0[flux_mask])/np.sum(image_0[flux_mask])).ravel()
            # M0_std = ((image_0_std[flux_mask]) / 1).ravel()
            # M0_std=((image_0_std[flux_mask])/np.sum(image_0_std[flux_mask])).ravel()

            list_of_M0.append(M0)
            # list_of_M0_std.append(M0_std)

        # join all M0,M0_std from invidiual images into one uber M0,M0_std
        uber_M0_previous_best = [item for sublist in list_of_M0 for item in sublist]
        # uber_M0_previous_best_std = [item for sublist in list_of_M0_std for item in sublist]

        uber_M0_previous_best = np.array(uber_M0_previous_best)
        # uber_M0_previous_best_std = np.array(uber_M0_previous_best_std)

        # uber_M0=uber_M0/np.sum(uber_M0)
        # uber_M0_std=uber_M0_std/np.sum(uber_M0_std)

        self.uber_M0_previous_best = uber_M0_previous_best
        # self.uber_M0_previous_best_std = uber_M0_previous_best_std

        ########################################################
        # uber_images_normalized_previous_best, but with current flux mask

        # previous uber images - not used
        # uber_images_normalized_previous_best_old_flux_mask=previous_best_result[0]
        # previous uber model - not used
        # uber_M0_previous_best_old_flux_mask=previous_best_result[1]

        # we need out images showing difference from original image due to
        # changing a single Zernike parameter
        out_images = main_body_of_best_result[4]

        # join all images together
        list_of_images_normalized_uber = []
        # list_of_images_normalized_std_uber = []
        # go over (zmax-3)*2 images
        for j in range(len(out_images)):
            # two steps for what could have been achived in one, but to ease up transition from previous code
            out_images_single_parameter_change = out_images[j]
            optpsf_list = out_images_single_parameter_change

            # flux image has to correct per image
            #  mask images that have been created in the fitting procedure with the appropriate flux mask
            images_normalized = []
            for i in range(len(optpsf_list)):

                flux_mask = list_of_flux_mask[i]
                if j == 0:

                    # logging.info('sum_flux_in images'+str([i,np.sum(flux_mask)]))
                    pass
                images_normalized.append((optpsf_list[i][flux_mask]).ravel())
                # !old double-normalizing code
                # !images_normalized.append((optpsf_list[i][flux_mask]/np.sum(optpsf_list[i][flux_mask])).ravel())

            images_normalized_flat = [item for sublist in images_normalized for item in sublist]
            images_normalized_flat = np.array(images_normalized_flat)
            # images_normalized_flat=np.array(images_normalized_flat)/len(optpsf_list)

            # list of (zmax-3)*2 raveled images
            list_of_images_normalized_uber.append(images_normalized_flat)

            # same but divided by STD
            # images_normalized_std = []
            for i in range(len(optpsf_list)):
                # seems that I am a bit more verbose here with my definitions
                # optpsf_list_i = optpsf_list[i]

                # do I want to generate new STD images, from each image?
                # STD = list_of_sci_image_std[i]
                # optpsf_list_i_STD = optpsf_list_i / STD
                flux_mask = list_of_flux_mask[i]
                # images_normalized_std.append((optpsf_list_i_STD[flux_mask]/np.sum(optpsf_list_i_STD[flux_mask])).ravel())

            # join all images together
            # images_normalized_std_flat=[item for sublist in images_normalized_std for item in sublist]
            # normalize so that the sum is still one
            # images_normalized_std_flat=np.array(images_normalized_std_flat)/len(optpsf_list)

            # list_of_images_normalized_std_uber.append(images_normalized_std_flat)

        # create uber images_normalized,images_normalized_std
        # images that have zmax*2 rows and very large number of columns (number of
        # non-masked pixels from all N images)
        uber_images_normalized_previous_best = np.array(list_of_images_normalized_uber)

        ########################################################
        # current model image
        # uber_M0 = self.uber_M0

        # current change of the parameters
        # if self.move_allparameters:
        # array_of_delta_all_parametrizations = self.array_of_delta_all_parametrizations
        # else:
        array_of_delta_parametrizations = self.array_of_delta_z_parametrizations

        # previous uber model
        # uber_M0_previous_best=previous_best_result[1]
        # previous H (not used)
        # H_previous_best=previous_best_result[2]
        # how much has delta parametrizations changed in the previous result
        array_of_delta_parametrizations_None_previous_best = previous_best_result[3]

        # ratio between current parametrization and the previous (provided) changed parametrization
        ratio_of_parametrizations = (
            array_of_delta_parametrizations[:, None] / array_of_delta_parametrizations_None_previous_best)

        # create the array of how wavefront changes the uber_model by multiply the changes with new ratios
        array_of_wavefront_changes = np.transpose(
            ratio_of_parametrizations
            * np.array(uber_images_normalized_previous_best - uber_M0_previous_best)
            / (array_of_delta_parametrizations_None_previous_best))

        # difference between current model image and previous model image

        # logging.info('uber_images_normalized_previous_best.shape'+str(uber_images_normalized_previous_best.shape))
        # logging.info('uber_M0_previous_best.shape'+str(uber_M0_previous_best.shape))
        # logging.info('uber_M0.shape'+str(uber_M0.shape))

        # change between the initial model in this step and the imported model
        # global_change = uber_M0 - uber_M0_previous_best
        # global_change_proposed
        # global_change_proposed = (uber_M0 - uber_M0_previous_best) /\
        # array_of_delta_parametrizations[:, None]

        # H is a change of wavefront
        H = array_of_wavefront_changes

        # H is a change of wavefront and change of model (is it?)
        # H=array_of_wavefront_changes + global_change[:,None]

        # np.save('/tigress/ncaplar/Results/global_change_'+str(2)+'_'+str(0),\
        #            global_change)

        # np.save('/tigress/ncaplar/Results/global_change_proposed_'+str(2)+'_'+str(0),\
        #            global_change_proposed)
        # np.save('/tigress/ncaplar/Results/array_of_wavefront_changes_'+str(2)+'_'+str(0),\
        #            array_of_wavefront_changes)
        # np.save('/tigress/ncaplar/Results/H_'+str(2)+'_'+str(0),\
        #            array_of_wavefront_changes)

        return H

    def create_first_proposal_Tokovnin(self, H, H_std, uber_I, uber_M0, uber_std, up_to_which_z=None):

        H_shape = H.shape

        # logging.info('H_shape'+str(H_shape))
        # logging.info('up_to_which_z:'+str(up_to_which_z))

        if up_to_which_z is not None:
            # H=H[:,1:(up_to_which_z-3)*2:2]
            # H_std=H_std[:,1:(up_to_which_z-3)*2:2]

            H = H[:, 0:(up_to_which_z - 3) * 2]
            H_std = H_std[:, 0:(up_to_which_z - 3) * 2]

        else:
            pass

        H = np.nan_to_num(H, 0)

        # logging.info('np.mean(H,axis=0).shape)'+str(np.mean(H,axis=0).shape))
        singlular_parameters = np.arange(H.shape[1])[np.abs((np.mean(H, axis=0))) < 0.001]
        non_singlular_parameters = np.arange(H.shape[1])[np.abs((np.mean(H, axis=0))) > 0.001]

        # logging.info('np.abs((np.mean(H,axis=0)))'+str(np.abs((np.mean(H,axis=0)))))
        # logging.info('non_singlular_parameters.shape)'+str(non_singlular_parameters.shape))
        # logging.info('singlular_parameters)'+str(singlular_parameters))

        H = H[:, non_singlular_parameters]
        H_std = H_std[:, non_singlular_parameters]

        HHt = np.matmul(np.transpose(H), H)
        HHt_std = np.matmul(np.transpose(H_std), H_std)
        # logging.info('svd thresh is '+str(thresh))
        # invHHt=svd_invert(HHt,thresh)
        # invHHt_std=svd_invert(HHt_std,thresh)
        invHHt = np.linalg.inv(HHt)
        invHHt_std = np.linalg.inv(HHt_std)

        invHHtHt = np.matmul(invHHt, np.transpose(H))
        invHHtHt_std = np.matmul(invHHt_std, np.transpose(H_std))

        # I is uber_I now (science images)
        # M0 is uber_M0 now (set of models before the iteration)
        first_proposal_Tokovnin = np.matmul(invHHtHt, uber_I - uber_M0)
        # first_proposal_Tokovnin_std=np.matmul(invHHtHt_std,uber_I_std-uber_M0_std)
        first_proposal_Tokovnin_std = np.matmul(invHHtHt_std, (uber_I - uber_M0) / uber_std.ravel())

        # logging.info('first_proposal_Tokovnin.shape before sing'+str(first_proposal_Tokovnin.shape))

        # if you have removed certain parameters because of the singularity, return them here, with no change
        if len(singlular_parameters) > 0:
            for i in range(len(singlular_parameters)):
                first_proposal_Tokovnin = np.insert(first_proposal_Tokovnin, singlular_parameters[i], 0)
                first_proposal_Tokovnin_std = np.insert(
                    first_proposal_Tokovnin_std, singlular_parameters[i], 0)

        # logging.info('first_proposal_Tokovnin.shape after sing'+str(first_proposal_Tokovnin.shape))

        if up_to_which_z is not None:
            # H=H[:,1:(up_to_which_z-3)*2:2]
            # H_std=H_std[:,1:(up_to_which_z-3)*2:2]

            first_proposal_Tokovnin_0 = np.zeros((H_shape[1]))
            first_proposal_Tokovnin_0_std = np.zeros((H_shape[1]))
            # logging.info('first_proposal_Tokovnin_0.shape'+str(first_proposal_Tokovnin_0.shape))

            # logging.info('up_to_which_z: '+str(up_to_which_z))
            # logging.info('first_proposal_Tokovnin:' +str(first_proposal_Tokovnin))

            # logging.info(first_proposal_Tokovnin_0[0:(up_to_which_z-3)*2].shape)
            # logging.info(first_proposal_Tokovnin.shape)
            first_proposal_Tokovnin_0[0:(up_to_which_z - 3) * 2] = first_proposal_Tokovnin
            first_proposal_Tokovnin_0_std[0:(up_to_which_z - 3) * 2] = first_proposal_Tokovnin_std

            first_proposal_Tokovnin = first_proposal_Tokovnin_0
            first_proposal_Tokovnin_std = first_proposal_Tokovnin_0_std
        else:
            pass

        return first_proposal_Tokovnin, first_proposal_Tokovnin_std

    def create_linear_aproximation_prediction(self, H, first_proposal_Tokovnin):
        return np.dot(H, first_proposal_Tokovnin)

    def __call__(self, allparameters_parametrization_proposal, return_Images=True, num_iter=None,
                 previous_best_result=None, use_only_chi=False,
                 multi_background_factor=3, up_to_which_z=None):

        return self.Tokovinin_algorithm_chi_multi(
            allparameters_parametrization_proposal,
            return_Images=return_Images,
            num_iter=num_iter,
            previous_best_result=previous_best_result,
            use_only_chi=use_only_chi,
            multi_background_factor=int(multi_background_factor),
            up_to_which_z=up_to_which_z)


class LN_PFS_single(object):

    """!

    Class to compute likelihood of the donut image, given the sci and var image
    Also the prinicpal way to get the images via ``return_Image'' option

    model = LN_PFS_single(sci_image,var_image,pupil_parameters=pupil_parameters,
                          use_pupil_parameters=None,zmax=zmax,save=1)
    def model_return(allparameters_proposal):
        return model(allparameters_proposal,return_Image=True)

    Calls ZernikeFitterPFS class (constructModelImage_PFS_naturalResolution function)
    in order to create images

    Called by LN_PFS_multi_same_spot

    """

    def __init__(self, sci_image, var_image,
                 mask_image=None,
                 wavelength=794, dithering=1, save=None, verbosity=None,
                 pupil_parameters=None, use_pupil_parameters=None, use_optPSF=None, use_wf_grid=None,
                 zmax=None, extraZernike=None, pupilExplicit=None, simulation_00=None,
                 double_sources=None, double_sources_positions_ratios=None, npix=None,
                 fit_for_flux=None, test_run=None, explicit_psf_position=None,
                 use_only_chi=False, use_center_of_flux=False,
                 PSF_DIRECTORY=None,
                 fiber_id=None):
        """
        @param sci_image                                science image, 2d array
        @param var_image                                variance image, 2d array,same size as sci_image
        @param mask_image                               mask image, 2d array,same size as sci_image
        @param dithering                                dithering, 1=normal, 2=two times higher resolution,
                                                        3=not supported
        @param save                                     save intermediate result in the process
                                                        (set value at 1 for saving)
        @param verbosity                                verbosity of the process
                                                        (set value at 1 for full output)
        @param pupil_parameters
        @param use_pupil_parameters
        @param use_optPSF
        @param zmax                                     largest Zernike order used (11 or 22)
        @param extraZernike                             array consistingin of higher order zernike
                                                        (if using higher order than 22)
        @param pupilExplicit
        @param simulation_00                            resulting image will be centered with optical center
                                                        in the center of the image
                                                        and not fitted acorrding to the sci_image
                                                        if use_center_of_flux==True, gives with center
                                                        of the flux in the center of the image
        @param double_sources                           1 if there are other secondary sources in the image
        @param double_sources_positions_ratios /        arrray with parameters describing relative position\
                                                        and relative flux of the secondary source(s)
        @param npxix                                    size of the pupil

        @param fit_for_flux                             automatically fit for the best flux level
                                                        that minimizes the chi**2
        @param test_run                                 if True, skips the creation of model
                                                        and return science image - useful for testing
                                                        interaction of outputs of the module
                                                        in broader setting quickly
        @param explicit_psf_position                    gives position of the opt_psf
        """

        if verbosity is None:
            verbosity = 0

        if use_pupil_parameters is not None:
            assert pupil_parameters is not None

        if double_sources is not None and bool(double_sources) is not False:
            assert np.sum(np.abs(double_sources_positions_ratios)) > 0

        if zmax is None:
            zmax = 11

        if zmax == 11:
            self.columns = [
                'z4',
                'z5',
                'z6',
                'z7',
                'z8',
                'z9',
                'z10',
                'z11',
                'detFrac',
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
                'effective_ilum_radius',
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
        if zmax >= 22:
            self.columns = [
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
                'detFrac',
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
                'effective_ilum_radius',
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

        if mask_image is None:
            mask_image = np.zeros(sci_image.shape)
        self.mask_image = mask_image
        self.sci_image = sci_image
        self.var_image = var_image

        popt_for_custom_var_image = self.create_popt_for_custom_var(sci_image, var_image, mask_image)
        self.popt_for_custom_var_image = popt_for_custom_var_image

        self.dithering = dithering
        self.pupil_parameters = pupil_parameters
        self.use_pupil_parameters = use_pupil_parameters
        self.use_optPSF = use_optPSF
        self.pupilExplicit = pupilExplicit
        self.simulation_00 = simulation_00
        if self.simulation_00:
            self.simulation_00 = 1

        self.zmax = zmax
        self.extraZernike = extraZernike
        self.verbosity = verbosity
        self.double_sources = double_sources
        self.double_sources_positions_ratios = double_sources_positions_ratios
        self.fit_for_flux = fit_for_flux
        if test_run is None:
            self.test_run = False
        else:
            self.test_run = test_run
        self.explicit_psf_position = explicit_psf_position

        # if npix is not specified automatically scale the image
        # this will create images which will have different pupil size for different sizes of science image
        # and this will influence the results
        if npix is None:
            if dithering is None or dithering == 1:
                npix = int(math.ceil(int(1024 * sci_image.shape[0] / (20 * 4))) * 2)
            else:
                npix = int(math.ceil(int(1024 * sci_image.shape[0] / (20 * 4 * self.dithering))) * 2)
        else:
            self.npix = npix

        if verbosity == 1:
            logging.info('Science image shape is: ' + str(sci_image.shape))
            logging.info('Top left pixel value of the science image is: ' + str(sci_image[0][0]))
            logging.info('Variance image shape is: ' + str(sci_image.shape))
            logging.info('Top left pixel value of the variance image is: ' + str(var_image[0][0]))
            logging.info('Mask image shape is: ' + str(sci_image.shape))
            logging.info('Sum of mask image is: ' + str(np.sum(mask_image)))
            logging.info('Dithering value is: ' + str(dithering))
            logging.info('')

            logging.info('explicit_psf_position in LN_PFS_single: '+str(explicit_psf_position))
            logging.info('supplied extra Zernike parameters (beyond zmax): ' + str(extraZernike))

        self.PSF_DIRECTORY = PSF_DIRECTORY

        self.fiber_id = fiber_id

        """
        parameters that go into ZernikeFitterPFS
        def __init__(self,image=None,image_var=None,image_mask=None,pixelScale=None,wavelength=None,
             diam_sic=None,npix=None,pupilExplicit=None,
             wf_full_Image=None,radiometricEffectArray_Image=None,
             ilum_Image=None,dithering=None,save=None,
             pupil_parameters=None,use_pupil_parameters=None,
             use_optPSF=None,use_wf_grid=None,
             zmaxInit=None,extraZernike=None,simulation_00=None,verbosity=None,
             double_sources=None,double_sources_positions_ratios=None,
             test_run=None,explicit_psf_position=None,
             fiber_id=None,
             *args):
        """

        # how are these two approaches different?
        if pupil_parameters is None:
            single_image_analysis = ZernikeFitterPFS(
                sci_image,
                var_image,
                image_mask=mask_image,
                wavelength=wavelength,
                npix=npix,
                pupilExplicit=pupilExplicit,
                wf_full_Image=None,
                ilum_Image=None,
                dithering=dithering,
                save=save,
                pupil_parameters=pupil_parameters,
                use_pupil_parameters=use_pupil_parameters,
                use_optPSF=use_optPSF,
                use_wf_grid=use_wf_grid,
                zmaxInit=zmax,
                extraZernike=extraZernike,
                simulation_00=self.simulation_00,
                verbosity=verbosity,
                double_sources=double_sources,
                double_sources_positions_ratios=double_sources_positions_ratios,
                test_run=test_run,
                explicit_psf_position=explicit_psf_position,
                use_only_chi=use_only_chi,
                use_center_of_flux=use_center_of_flux,
                PSF_DIRECTORY=PSF_DIRECTORY,
                fiber_id=fiber_id)
            single_image_analysis.initParams(zmax)
            self.single_image_analysis = single_image_analysis
        else:
            single_image_analysis = ZernikeFitterPFS(
                sci_image,
                var_image,
                image_mask=mask_image,
                npix=npix,
                dithering=dithering,
                save=save,
                pupil_parameters=pupil_parameters,
                use_pupil_parameters=use_pupil_parameters,
                extraZernike=extraZernike,
                simulation_00=self.simulation_00,
                verbosity=verbosity,
                double_sources=double_sources,
                double_sources_positions_ratios=double_sources_positions_ratios,
                test_run=test_run,
                explicit_psf_position=explicit_psf_position,
                use_only_chi=use_only_chi,
                use_center_of_flux=use_center_of_flux,
                PSF_DIRECTORY=PSF_DIRECTORY,
                fiber_id=fiber_id)

            single_image_analysis.initParams(
                zmax,
                detFracInit=pupil_parameters[0],
                strutFracInit=pupil_parameters[1],
                focalPlanePositionInit=(
                    pupil_parameters[2],
                    pupil_parameters[3]),
                slitFracInit=pupil_parameters[4],
                slitFrac_dy_Init=pupil_parameters[5],
                x_fiberInit=pupil_parameters[6],
                y_fiberInit=pupil_parameters[7],
                effective_ilum_radiusInit=pupil_parameters[8],
                frd_sigmaInit=pupil_parameters[9],
                det_vertInit=pupil_parameters[10],
                slitHolder_frac_dxInit=pupil_parameters[11],
                wide_0Init=pupil_parameters[12],
                wide_23Init=pupil_parameters[13],
                wide_43Init=pupil_parameters[14],
                misalignInit=pupil_parameters[15])

            self.single_image_analysis = single_image_analysis

    def create_popt_for_custom_var(self, sci_image, var_image, mask_image=None):
        """Create 2nd order poly fit; to be used in creation of custom var image

        TODO: same function in LN_PFS_Single... Very unsatifactory!

        The connection between variance and flux is determined from the provided science image
        and variance image.
        All of inputs have to be 2d np.arrays with same size.
        Introduced in 0.50 (PIPE2D-931)

        Called by Tokovinin_algorithm_chi_multi

        Parameters
        ----------
        sci_image : `np.array`
            Scientific array
        var_image : `np.array`
            Variance array
        mask_image : `np.array`
            Mask image

        Returns
        ----------
        custom_var_image : `np.array`
            Recreated variance map

        """
        if mask_image is None:
            sci_pixels = sci_image.ravel()
            var_pixels = var_image.ravel()
        else:
            sci_pixels = sci_image[mask_image == 0].ravel()
            var_pixels = var_image[mask_image == 0].ravel()
        # z = np.polyfit(sci_pixels, var_pixels, deg=2)
        # if z[0] < 0:
        #    z = np.polyfit(sci_pixels, var_pixels, deg=1)
        # p1 = np.poly1d(z)
        # custom_var_image = p1(sci_image)

        # I am using lambda expression to avoid superflous definition of quadratic function
        f = lambda x, *p: p[0] * x**2 + p[1] * x + p[2]  # noqa : E373
        popt, pcov = scipy.optimize.curve_fit(f, sci_pixels, var_pixels, [0, 0, np.min(var_pixels)],
                                              bounds=([-np.inf, -np.inf, np.min(var_pixels)],
                                                      [np.inf, np.inf, np.inf]))
        return popt

    def create_custom_var_from_popt(self, model_image, popt):
        """Creates variance map from the model image, given the 2nd poly fit parameters

        Introduced in 0.50 (PIPE2D-931)

        Parameters
        ----------
        modelImg : `np.array`
            Model image
        popt : `np.array`
            2d polyfit parameters
        Returns
        ----------
        custom_var_image : `np.array`
            Recreated variance map
        """

        # I am using lambda expression to avoid superflous definition of quadratic function
        f = lambda x, *p: p[0] * x**2 + p[1] * x + p[2]  # noqa : E373
        custom_var_image = f(model_image, *popt)
        return custom_var_image

    def create_custom_var(self, model_image, sci_image, var_image, mask_image):
        """Creates variance map from the model image, sci, var and mask_image

        Introduced in 0.50 (PIPE2D-931)

        Parameters
        ----------
        modelImg : `np.array`
            Model image
        ...
        Returns
        ----------
        custom_var_image : `np.array`
            Recreated variance map
        """

        popt_for_custom_var = self.create_popt_for_custom_var(sci_image, var_image, mask_image)
        custom_var_image = self.create_custom_var_from_popt(model_image, popt_for_custom_var)
        return custom_var_image

    def create_chi_2_almost(
            self,
            modelImg,
            sci_image,
            var_image,
            mask_image,
            use_only_chi=False,
            multi_background_factor=3):
        """Create values describing the quality of the fit

        Parameters
        ----------
        modelImg : `np.array`
            Model image
        sci_image : `np.array`
            Scientific image
        var_image : `np.array`
            Variance image
        mask_image : `np.array`
            Mask image
        use_only_chi : `bool`
            If True, the program is reporting np.abs(chi), not chi^2


        Returns
        ----------
        (5 values) : `list`
            0. normal chi**2
            1. what is 'instrinsic' chi**2, i.e., just sum((scientific image)**2/variance)
            2. 'Q' value = sum(abs(model - scientific image))/sum(scientific image)
            3. chi**2 reduced
            4. chi**2 reduced 'intrinsic'

        The descriptions below are applicable when use_only_chi = False
        """

        try:
            if sci_image.shape[0] == 20:
                multi_background_factor = 3

            mean_value_of_background_via_var = np.mean([np.median(var_image[0]), np.median(
                var_image[-1]), np.median(var_image[:, 0]), np.median(var_image[:, -1])])\
                * multi_background_factor

            mean_value_of_background_via_sci = np.mean([np.median(sci_image[0]), np.median(
                sci_image[-1]), np.median(sci_image[:, 0]), np.median(sci_image[:, -1])])\
                * multi_background_factor

            mean_value_of_background = np.max(
                [mean_value_of_background_via_var, mean_value_of_background_via_sci])

            flux_mask = sci_image > (mean_value_of_background)
            inverted_flux_mask = flux_mask.astype(bool)
        except BaseException:
            inverted_flux_mask = np.ones(sci_image.shape)

        # array that has True for values which are good and False for bad values
        inverted_mask = ~mask_image.astype(bool)

        # strengthen the mask by taking in the account only bright pixels, which have passed the flux cut
        inverted_mask = inverted_mask * inverted_flux_mask

        use_custom_var = True
        if use_custom_var:
            # logging.info('Test checkpoint 1')
            # logging.info('modelImg.shape'+str(modelImg.shape))
            # logging.info('modelImg[0][0:5]'+str(modelImg[0][0:5]))
            # logging.info('sci_image.shape'+str(sci_image.shape))
            # logging.info('sci_image[0][0:5]'+str(sci_image[0][0:5]))
            # logging.info('var_image.shape'+str(var_image.shape))
            # logging.info('var_image[0][0:5]'+str(var_image[0][0:5]))
            # logging.info('mask_image.shape'+str(mask_image.shape))
            # logging.info('mask_image[0][0:5]'+str(mask_image[0][0:5]))
            custom_var_image = self.create_custom_var_from_popt(modelImg, self.popt_for_custom_var_image)
            # logging.info('custom_var_image[0][0:5]'+str(custom_var_image[0][0:5]))
            # overload var_image with newly created image
            var_image = custom_var_image

        # apply the mask on all of the images (sci, var and model)
        var_image_masked = var_image * inverted_mask
        sci_image_masked = sci_image * inverted_mask
        modelImg_masked = modelImg * inverted_mask

        # logging.info('First 5 values are: '+str(var_image_masked[0:5]))

        # sigma values
        sigma_masked = np.sqrt(var_image_masked)

        # chi array
        chi = (sci_image_masked - modelImg_masked) / sigma_masked
        # chi intrinsic, i.e., without subtracting model
        chi_intrinsic = (sci_image_masked / sigma_masked)

        # ravel and remove bad values
        chi_without_nan = chi.ravel()[~np.isnan(chi.ravel())]
        chi_intrinsic_without_nan = chi_intrinsic.ravel()[~np.isnan(chi_intrinsic.ravel())]

        if not use_only_chi:
            # If you are computing chi**2, square it
            chi2_res = (chi_without_nan)**2
            chi2_intrinsic_res = (chi_intrinsic_without_nan)**2
        else:
            # If you are just using chi, do not square it
            # keep the names the same, but very careful as they are not squared quantities
            chi2_res = np.abs(chi_without_nan)**1
            chi2_intrinsic_res = np.abs(chi_intrinsic_without_nan)**1

        # logging.info('use_only_chi variable in create_chi_2_almost is: '+str(use_only_chi))
        # logging.info('chi2_res '+str(np.sum(chi2_res)))
        # logging.info('chi2_intrinsic_res '+str(np.sum(chi2_intrinsic_res)))

        # calculates 'Q' values
        Qlist = np.abs((sci_image_masked - modelImg_masked))
        Qlist_without_nan = Qlist.ravel()[~np.isnan(Qlist.ravel())]
        sci_image_without_nan = sci_image_masked.ravel()[~np.isnan(sci_image_masked.ravel())]
        Qvalue = np.sum(Qlist_without_nan) / np.sum(sci_image_without_nan)

        # return the result
        return [
            np.sum(chi2_res),
            np.sum(chi2_intrinsic_res),
            Qvalue,
            np.mean(chi2_res),
            np.mean(chi2_intrinsic_res)]

    def lnlike_Neven(
            self,
            allparameters,
            return_Image=False,
            return_intermediate_images=False,
            use_only_chi=False,
            multi_background_factor=3):
        """Report `likelihood` given the parameters of the model

        The algorithm gives -np.inf if one of the parameters is outside of the specified range
        (which are pecified below)

        Parameters
        ----------
        allparameters : `np.array`
            Model image
        return_Image : `bool
            explanation
        return_intermediate_images : `bool`
            explanation
        use_only_chi : `bool`
            explanation
        multi_background_factor : `float
            explanation

        Parameters
        ----------
        (if not return_Image)
        res : `float`
            `Likelihood` of the fit
        psf_position : `np.array`
            Position at which model has been centered
        (if return_Image==True)
        res : `float
            `Likelihood` of the fit
        modelImg : `np.array`
            Model image
        allparameters : `np.array`
            Parameters describing the model
        (quality) : `list`
            0. chi_2_almost : `float`
                chi2/chi total
            1. chi_2_almost_max : `float`
                Total chi2/chi without subtracting the model
            2. chi_2_almost_dof : `float
                chi2/chi total divided by the number of pixels
            3. chi_2_almost_max_dof : `float`
                Total chi2/chi without subtracting the model, divided by the number of pixels
        psf_position : `np.array`
            Position at which model has been centered
        (if return_intermediate_images)
        res : `float
            `Likelihood` of the fit
        modelImg : `np.array`
            Model image
        allparameters : `np.array`
            Parameters describing the model
        ilum : `np.array`
            Illumination of the pupil
        wf_grid_rot : `np.array`
            Wavfefront across the pupil
        (quality) : `list`
            0. chi_2_almost : `float`
                chi2/chi total
            1. chi_2_almost_max : `float`
                Total chi2/chi without subtracting the model
            2. chi_2_almost_dof : `float
                chi2/chi total divided by the number of pixels
            3. chi_2_almost_max_dof : `float`
                Total chi2/chi without subtracting the model, divided by the number of pixels
        psf_position : `np.array`
            Position at which model has been centered
        """

        time_lnlike_start = time.time()

        if self.verbosity == 1:
            logging.info('')
            logging.info('Entering lnlike_Neven')
            logging.info('allparameters ' + str(allparameters))

        if self.pupil_parameters is not None:
            if len(allparameters) < 25:
                allparameters = add_pupil_parameters_to_all_parameters(allparameters, self.pupil_parameters)
            else:
                allparameters = add_pupil_parameters_to_all_parameters(
                    remove_pupil_parameters_from_all_parameters(allparameters), self.pupil_parameters)

        if self.zmax <= 22:
            zmax_number = self.zmax - 3
        else:
            zmax_number = 19
        zparameters = allparameters[0:zmax_number]

        globalparameters = allparameters[len(zparameters):len(zparameters) + 23]

        # if self.fit_for_flux==True:
        #    globalparameters=np.concatenate((globalparameters,np.array([1])))

        # internal parameter for debugging change value to 1 to see which parameters are failling
        test_print = 0
        if self.verbosity == 1:
            test_print = 1

        # When running big fits these are limits which ensure
        # that the code does not wander off in totally non physical region
        # det frac
        if globalparameters[0] < 0.6 or globalparameters[0] > 0.8:
            logging.info('globalparameters[0] outside limits; value: '
                         + str(globalparameters[0])) if test_print == 1 else False
            return -np.inf

        # strut frac
        if globalparameters[1] < 0.07 or globalparameters[1] > 0.13:
            logging.info('globalparameters[1] outside limits') if test_print == 1 else False
            return -np.inf

        # slit_frac < strut frac
        # if globalparameters[4]<globalparameters[1]:
            # logging.info('globalparameters[1] not smaller than 4 outside limits')
            # return -np.inf

        # dx Focal
        if globalparameters[2] > 0.4:
            logging.info('globalparameters[2] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[2] < -0.4:
            logging.info('globalparameters[2] outside limits') if test_print == 1 else False
            return -np.inf

        # dy Focal
        if globalparameters[3] > 0.4:
            logging.info('globalparameters[3] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[3] < -0.4:
            logging.info('globalparameters[3] outside limits') if test_print == 1 else False
            return -np.inf

        # slitFrac
        if globalparameters[4] < 0.05:
            logging.info('globalparameters[4] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[4] > 0.09:
            logging.info('globalparameters[4] outside limits') if test_print == 1 else False
            return -np.inf

        # slitFrac_dy
        if globalparameters[5] < -0.5:
            logging.info('globalparameters[5] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[5] > 0.5:
            logging.info('globalparameters[5] outside limits') if test_print == 1 else False
            return -np.inf

        # wide_0
        if globalparameters[6] < 0:
            logging.info('globalparameters[6] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[6] > 1:
            logging.info('globalparameters[6] outside limits') if test_print == 1 else False
            return -np.inf

        # wide_23
        if globalparameters[7] < 0:
            logging.info('globalparameters[7] outside limits') if test_print == 1 else False
            return -np.inf
        # changed in w_23
        if globalparameters[7] > 1:
            logging.info('globalparameters[7] outside limits') if test_print == 1 else False
            return -np.inf

        # wide_43
        if globalparameters[8] < 0:
            logging.info('globalparameters[8] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[8] > 1:
            logging.info('globalparameters[8] outside limits') if test_print == 1 else False
            return -np.inf

        # misalign
        if globalparameters[9] < 0:
            logging.info('globalparameters[9] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[9] > 12:
            logging.info('globalparameters[9] outside limits') if test_print == 1 else False
            return -np.inf

        # x_fiber
        if globalparameters[10] < -0.4:
            logging.info('globalparameters[10] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[10] > 0.4:
            logging.info('globalparameters[10] outside limits') if test_print == 1 else False
            return -np.inf

        # y_fiber
        if globalparameters[11] < -0.4:
            logging.info('globalparameters[11] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[11] > 0.4:
            logging.info('globalparameters[11] outside limits') if test_print == 1 else False
            return -np.inf

        # effective_radius_illumination
        if globalparameters[12] < 0.7:
            logging.info('globalparameters[12] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[12] > 1.0:
            logging.info('globalparameters[12] outside limits with value '
                         + str(globalparameters[12])) if test_print == 1 else False
            return -np.inf

        # frd_sigma
        if globalparameters[13] < 0.01:
            logging.info('globalparameters[13] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[13] > .4:
            logging.info('globalparameters[13] outside limits') if test_print == 1 else False
            return -np.inf

        # frd_lorentz_factor
        if globalparameters[14] < 0.01:
            logging.info('globalparameters[14] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[14] > 1:
            logging.info('globalparameters[14] outside limits') if test_print == 1 else False
            return -np.inf

        # det_vert
        if globalparameters[15] < 0.85:
            logging.info('globalparameters[15] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[15] > 1.15:
            logging.info('globalparameters[15] outside limits') if test_print == 1 else False
            return -np.inf

        # slitHolder_frac_dx
        if globalparameters[16] < -0.8:
            logging.info('globalparameters[16] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[16] > 0.8:
            logging.info('globalparameters[16] outside limits') if test_print == 1 else False
            return -np.inf

        # grating_lines
        if globalparameters[17] < 1200:
            logging.info('globalparameters[17] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[17] > 120000:
            logging.info('globalparameters[17] outside limits') if test_print == 1 else False
            return -np.inf

        # scattering_slope
        if globalparameters[18] < 1.5:
            logging.info('globalparameters[18] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[18] > +3.0:
            logging.info('globalparameters[18] outside limits') if test_print == 1 else False
            return -np.inf

        # scattering_amplitude
        if globalparameters[19] < 0:
            logging.info('globalparameters[19] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[19] > +0.4:
            logging.info('globalparameters[19] outside limits') if test_print == 1 else False
            return -np.inf

        # pixel_effect
        if globalparameters[20] < 0.15:
            logging.info('globalparameters[20] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[20] > +0.8:
            logging.info('globalparameters[20] outside limits') if test_print == 1 else False
            return -np.inf

        # fiber_r
        if globalparameters[21] < 1.74:
            logging.info('globalparameters[21] outside limits') if test_print == 1 else False
            return -np.inf
        if globalparameters[21] > +1.98:
            logging.info('globalparameters[21] outside limits') if test_print == 1 else False
            return -np.inf

        # flux
        if self.fit_for_flux:
            globalparameters[22] = 1
        else:
            if globalparameters[22] < 0.98:
                logging.info('globalparameters[22] outside limits') if test_print == 1 else False
                return -np.inf
            if globalparameters[22] > 1.02:
                logging.info('globalparameters[22] outside limits') if test_print == 1 else False
                return -np.inf

        x = self.create_x(zparameters, globalparameters)

        for i in range(len(self.columns)):
            self.single_image_analysis.params[self.columns[i]].set(x[i])

        if len(allparameters) > len(self.columns):
            if self.verbosity == 1:
                logging.info('We are going higher than Zernike 22!')
            extra_Zernike_parameters = allparameters[len(self.columns):]
            if self.verbosity == 1:
                logging.info('extra_Zernike_parameters ' + str(extra_Zernike_parameters))
        else:
            extra_Zernike_parameters = None
            if self.verbosity == 1:
                logging.info('No extra Zernike (beyond zmax)')

        # if it is not a test run, run the actual code
        if not self.test_run:
            # this try statment avoids code crashing when code tries to analyze weird
            # combination of parameters which fail to produce an image
            try:
                if not return_intermediate_images:
                    modelImg, psf_position =\
                        self.single_image_analysis.constructModelImage_PFS_naturalResolution(
                            self.single_image_analysis.params, extraZernike=extra_Zernike_parameters,
                            return_intermediate_images=return_intermediate_images)
                if return_intermediate_images:
                    modelImg, ilum, wf_grid_rot, psf_position =\
                        self.single_image_analysis.constructModelImage_PFS_naturalResolution(
                            self.single_image_analysis.params, extraZernike=extra_Zernike_parameters,
                            return_intermediate_images=return_intermediate_images)
            except IndexError:
                return -np.inf, -np.inf
        else:
            randomizer_array = np.random.randn(self.sci_image.shape[0], self.sci_image.shape[1]) / 100 + 1
            if not return_intermediate_images:

                modelImg = self.sci_image * randomizer_array
                psf_position = [0, 0]

                if self.verbosity == 1:
                    logging.info('Careful - the model image is created in a test_run')
            else:
                # ilum_test=np.ones((3072,3072))
                ilum_test = np.ones((30, 30))

                # wf_grid_rot_test=np.ones((3072,3072))
                wf_grid_rot_test = np.ones((30, 30))

                psf_position_test = [0, 0]

                modelImg, ilum, wf_grid_rot, psf_position = self.sci_image * \
                    randomizer_array, ilum_test, wf_grid_rot_test, psf_position_test
                if self.verbosity == 1:
                    logging.info('Careful - the model image is created in a test_run')
                    logging.info('test run with return_intermediate_images==True!')

        # if image is in focus, which at the moment is size of post stamp image of 20 by 20
        # logging.info('self.sci_image.shape[0]'+str(self.sci_image.shape[0]))
        if self.sci_image.shape[0] == 20:
            # apply the procedure from
            # https://github.com/Subaru-PFS/drp_stella/blob/master/python/pfs/drp/stella/subtractSky2d.py
            # `image` from the pipeline is `sci_image` here
            # `psfImage` from the pipeline is `modelImg` here
            # `image.mask` from the pipeline is `mask_image` here
            # `image.variance` from the pipeline is `var_image` here

            inverted_mask = ~self.mask_image.astype(bool)

            modelDotModel = np.sum(modelImg[inverted_mask]**2)
            modelDotData = np.sum(modelImg[inverted_mask] * self.sci_image[inverted_mask])
            # modelDotModelVariance = np.sum(modelImg[inverted_mask]**2 * self.var_image[inverted_mask])
            flux = modelDotData / modelDotModel
            # fluxErr = np.sqrt(modelDotModelVariance) / modelDotModel

            modelImg = modelImg * flux
            if self.verbosity == 1:
                logging.info('Image in focus, using pipeline normalization;\
                      multiplying all values in the model by ' + str(flux))

        else:

            if self.fit_for_flux:
                if self.verbosity == 1:
                    logging.info('Internally fitting for flux; disregarding passed value for flux')

                def find_flux_fit(flux_fit):
                    return self.create_chi_2_almost(
                        flux_fit * modelImg,
                        self.sci_image,
                        self.var_image,
                        self.mask_image,
                        use_only_chi=use_only_chi)[0]

                flux_fitting_result = scipy.optimize.shgo(find_flux_fit, bounds=[(0.98, 1.02)], iters=6)
                flux = flux_fitting_result.x[0]
                if len(allparameters) == 42:
                    allparameters[-1] = flux
                if len(allparameters) == 41:
                    allparameters = np.concatenate((allparameters, np.array([flux])))
                else:
                    # logging.info('here')
                    # logging.info(allparameters[41])
                    if (allparameters[41] < 1.1) and (allparameters[41] > 0.9):
                        allparameters[41] = flux
                    else:
                        pass
                # logging.info('flux: '+str(flux))
                # logging.info(len(allparameters))
                # logging.info(allparameters)

                modelImg = modelImg * flux
                if self.verbosity == 1:
                    logging.info('Internally fitting for flux; multiplying all values in the model by '
                                 + str(flux))
            else:
                pass

        # returns 0. chi2 value, 1. chi2_max value, 2. Qvalue, 3. chi2/d.o.f., 4. chi2_max/d.o.f.
        chi_2_almost_multi_values = self.create_chi_2_almost(
            modelImg,
            self.sci_image,
            self.var_image,
            self.mask_image,
            use_only_chi=use_only_chi,
            multi_background_factor=multi_background_factor)
        chi_2_almost = chi_2_almost_multi_values[0]
        chi_2_almost_max = chi_2_almost_multi_values[1]
        chi_2_almost_dof = chi_2_almost_multi_values[3]
        chi_2_almost_max_dof = chi_2_almost_multi_values[4]

        # res stands for ``result''
        if not use_only_chi:
            # reporting likelihood in chi^2 case
            res = -(1 / 2) * (chi_2_almost + np.sum(np.log(2 * np.pi * self.var_image)))
        else:
            # reporting np.abs(chi) per d.o.f.
            res = -(1 / 1) * (chi_2_almost_dof)

        time_lnlike_end = time.time()
        if self.verbosity:
            logging.info('Finished with lnlike_Neven')
            if not use_only_chi:
                logging.info('chi_2_almost/d.o.f is ' + str(chi_2_almost_dof)
                             + '; chi_2_almost_max_dof is ' + str(chi_2_almost_max_dof)
                             + '; log(improvment) is '
                             + str(np.log10(chi_2_almost_dof / chi_2_almost_max_dof)))
            else:
                logging.info('chi_almost/d.o.f is ' + str(chi_2_almost_dof)
                             + '; chi_almost_max_dof is ' + str(chi_2_almost_max_dof)
                             + '; log(improvment) is '
                             + str(np.log10(chi_2_almost_dof / chi_2_almost_max_dof)))

            logging.info('The `likelihood` reported is: ' + str(res))
            # logging.info('multiprocessing.current_process() ' +
            #       str(current_process()) + ' thread ' + str(threading.get_ident()))
            # logging.info(str(platform.uname()))
            logging.info('Time for lnlike_Neven function in thread ' + str(threading.get_ident())
                         + ' is: ' + str(time_lnlike_end - time_lnlike_start) + str(' seconds'))
            logging.info(' ')

        if not return_Image:
            return res, psf_position
        else:
            # if return_Image==True return: 0. likelihood, 1. model image, 2.
            # parameters, 3. [0. chi**2, 1. chi**2_max, 2. chi**2/dof, 3.
            # chi**2_max/dof]
            if not return_intermediate_images:
                return res, modelImg, allparameters,\
                    [chi_2_almost, chi_2_almost_max, chi_2_almost_dof, chi_2_almost_max_dof], psf_position
            if return_intermediate_images:
                return res, modelImg, allparameters, ilum, wf_grid_rot,\
                    [chi_2_almost, chi_2_almost_max, chi_2_almost_dof, chi_2_almost_max_dof], psf_position

    def create_x(self, zparameters, globalparameters):
        """
        Given the zparameters and globalparameters separtly, this code moves them in a single array

        @param zparameters        Zernike coefficents
        @param globalparameters   other parameters describing the system
        """
        x = np.zeros((len(zparameters) + len(globalparameters)))
        for i in range(len(zparameters)):
            x[i] = zparameters[i]

        for i in range(len(globalparameters)):
            x[int(len(zparameters) / 1) + i] = globalparameters[i]

        return x

    def __call__(self, allparameters, return_Image=False, return_intermediate_images=False,
                 use_only_chi=False, multi_background_factor=3):
        return self.lnlike_Neven(allparameters, return_Image=return_Image,
                                 return_intermediate_images=return_intermediate_images,
                                 use_only_chi=use_only_chi,
                                 multi_background_factor=multi_background_factor)


class LNP_PFS(object):
    def __init__(self, image=None, image_var=None):
        self.image = image
        self.image_var = image_var

    def __call__(self, image=None, image_var=None):
        return 0.0


class PFSLikelihoodModule(object):
    """
    PFSLikelihoodModule class for calculating a likelihood for cosmoHammer.ParticleSwarmOptimizer
    """

    def __init__(self, model, explicit_wavefront=None):
        """

        Constructor of the PFSLikelihoodModule
        """
        self.model = model
        self.explicit_wavefront = explicit_wavefront

    def computeLikelihood(self, ctx):
        """
        Computes the likelihood using information from the context
        """
        # Get information from the context. This can be results from a core
        # module or the parameters coming from the sampler
        params = ctx.getParams()[0]
        return_Images_value = ctx.getParams()[1]

        logging.info('params' + str(params))

        # Calculate a likelihood up to normalization
        lnprob = self.model(params, return_Images=return_Images_value)

        # logging.info('current_process is: '+str(current_process())+str(lnprob))
        # logging.info(params)
        # logging.info('within computeLikelihood: parameters-hash '+\
        #   str(hash(str(params.data)))+'/threading: '+str(threading.get_ident()))

        # sys.stdout.flush()
        # Return the likelihood
        return lnprob

    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        # e.g. load data from files

        logging.info("PFSLikelihoodModule setup done")


class PsfPosition(object):
    """
    Class that deals with positioning the PSF model in respect to the data

    Function find_single_realization_min_cut enables the fit to the data
    """

    def __init__(self, image, oversampling, size_natural_resolution, simulation_00=False,
                 verbosity=0, save=None, PSF_DIRECTORY=None):
        """
        Parameters
        -----------------
        image: `np.array`, (N, N)
            oversampled model image
        oversampling: `int`
            by how much is the the oversampled image oversampled
        simulation_00: `bool`
            if True, put optical center of the model image in
            the center of the final image
        verbosity: `int`
            how verbose the procedure is (1 for full verbosity)
        save: `int`
            save intermediate images on hard drive (1 for save)
        PSF_DIRECTORY: `str`

        """
        self.image = image
        self.oversampling = oversampling
        self.size_natural_resolution = size_natural_resolution
        self.simulation_00 = simulation_00
        self.verbosity = verbosity
        if save is None:
            save = 0
        self.save = save
        self.PSF_DIRECTORY = PSF_DIRECTORY

        if self.PSF_DIRECTORY is not None:
            self.TESTING_FOLDER = self.PSF_DIRECTORY + 'Testing/'
            self.TESTING_PUPIL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Pupil_Images/'
            self.TESTING_WAVEFRONT_IMAGES_FOLDER = self.TESTING_FOLDER + 'Wavefront_Images/'
            self.TESTING_FINAL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Final_Images/'

    @staticmethod
    def cut_Centroid_of_natural_resolution_image(image, size_natural_resolution, oversampling, dx, dy):
        """Cut the central part from a larger oversampled image

        @param image                          input image
        @param size_natural_resolution        size of new image in natural units
        @param oversampling                   oversampling

        @returns                              central part of the input image
        """
        positions_from_where_to_start_cut = [int(len(image) / 2 - size_natural_resolution / 2
                                                 - dx * oversampling + 1),
                                             int(len(image) / 2 - size_natural_resolution / 2
                                                 - dy * oversampling + 1)]

        res = image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1]
                    + int(size_natural_resolution),
                    positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0]
                    + int(size_natural_resolution)]
        return res

    def find_single_realization_min_cut(
            self,
            input_image,
            oversampling,
            size_natural_resolution,
            sci_image,
            var_image,
            mask_image,
            v_flux,
            simulation_00=False,
            double_sources=None,
            double_sources_positions_ratios=[0, 0],
            verbosity=0,
            explicit_psf_position=None,
            use_only_chi=False,
            use_center_of_flux=False):
        """Move the image to find best position to downsample
        the oversampled image
        Parameters
        -----------------
        image: `np.array`, (N, N)
            model image to be analyzed
            (in our case this will be image of the
             optical psf convolved with fiber)
        oversampling: `int`
            oversampling
        size_natural_resolution: `int`
           size of final image (in the ``natural'' units, i.e., physical pixels
                                on the detector)
        sci_image_0: `np.array`, (N, N)
            science image
        var_image_0: `np.array`, (N, N)
            variance image
        v_flux: `float`
            flux normalization
        simulation_00: `bool`
            if True,do not move the center, for making fair comparisons between
            models - optical center in places in the center of the image
            if use_center_of_flux==True the behaviour changes
            and the result is the image with center of flux
            in the center of the image
        double_sources: `bool`
            if True, fit for two sources seen in the data
        double_sources_positions_ratios: `np.array`, (2,)
            2 values describing init guess for the relation between
            secondary and primary souces (offset, ratio)
        verbosity: `int`
            verbosity of the algorithm (1 for full verbosity)
        explicit_psf_position: `np.array`, (2,)
             x and y offset
        use_only_chi: `bool`
            quality of the centering is measured using chi, not chi**2
        use_center_of_flux: `bool`
                fit so that the center of flux of the model and
                the science image is as similar as possible
        Returns
        ----------
        model_image: `np.array`, (2,)
            returns model image in the size of the science image and
            centered to the science image
            (unless simulation_00=True or
             explicit_psf_position has been passed)
        Notes
        ----------
        Called by create_optPSF_natural in ZernikeFitterPFS
        Calls function create_complete_realization
        (many times in order to fit the best solution)
        """
        self.sci_image = sci_image
        self.var_image = var_image
        self.mask_image = mask_image
        self.v_flux = v_flux

        # if you are just asking for simulated image at (0,0) there is no possibility to create double sources
        if simulation_00 == 1:
            double_sources = None

        if double_sources is None or double_sources is False:
            double_sources_positions_ratios = [0, 0]

        shape_of_input_img = input_image.shape[0]
        shape_of_sci_image = sci_image.shape[0]

        self.shape_of_input_img = shape_of_input_img
        self.shape_of_sci_image = shape_of_sci_image

        if verbosity == 1:
            logging.info('Parameter use_only_chi in Psf_postion is set to: ' + str(use_only_chi))
            logging.info('Parameter use_center_of_flux in Psf_postion is set to: ' + str(use_center_of_flux))
            logging.info('Parameter simulation_00 in Psf_postion is set to: ' + str(simulation_00))

        # depending on if there is a second source in the image split here
        # double_sources should always be None when when creating centered images (simulation_00 = True)
        if double_sources is None or bool(double_sources) is False:
            # if simulation_00 AND using optical center just run the realization that is set at 0,0
            if simulation_00 == 1 and use_center_of_flux is False:
                if verbosity == 1:
                    logging.info('simulation_00 is set to 1 and use_center_of_flux==False -\
                        I am just returning the image at (0,0) coordinates ')

                # return the solution with x and y is zero, i.e., with optical center in
                # the center of the image
                mean_res, single_realization_primary_renormalized, single_realization_secondary_renormalized,\
                    complete_realization_renormalized \
                    = self.create_complete_realization([0, 0], return_full_result=True,
                                                       use_only_chi=use_only_chi,
                                                       use_center_of_light=use_center_of_flux,
                                                       simulation_00=simulation_00)

            # if you are fitting an actual image go through the full process
            else:
                # if you did not pass explict position search for the best position
                if explicit_psf_position is None:
                    # if creating the model so that the model is centered so
                    # that center of light of the model matches the center of the light
                    # of the scientific image, manually change values for centroid_of_sci_image here
                    if simulation_00 == 1 and use_center_of_flux:
                        if self.verbosity == 1:
                            logging.info('creating simulated image, center of light in center of the image')
                        shape_of_sci_image = 21
                        centroid_of_sci_image = [10.5, 10.5]
                    else:
                        # create one complete realization with default parameters - estimate
                        # centorids and use that knowledge to put fitting limits in the next step
                        centroid_of_sci_image = find_centroid_of_flux(sci_image)

                    time_1 = time.time()
                    initial_complete_realization = self.create_complete_realization(
                        [0, 0, double_sources_positions_ratios[0] * self.oversampling,
                            double_sources_positions_ratios[1]],
                        return_full_result=True,
                        use_only_chi=use_only_chi,
                        use_center_of_light=use_center_of_flux,
                        simulation_00=simulation_00)[-1]
                    time_2 = time.time()
                    if self.verbosity == 1:
                        logging.info('time_2-time_1 for initial_complete_realization: '
                                     + str(time_2 - time_1))

                    # center of the light for the first realization, set at optical center
                    centroid_of_initial_complete_realization = find_centroid_of_flux(
                        initial_complete_realization)

                    # determine offset between the initial guess and the data
                    offset_initial_and_sci = - \
                        ((np.array(find_centroid_of_flux(initial_complete_realization))
                          - np.array(find_centroid_of_flux(sci_image))))

                    if verbosity == 1:
                        logging.info('centroid_of_initial_complete_realization '
                                     + str(find_centroid_of_flux(initial_complete_realization)))
                        logging.info('centroid_of_sci_image '+str(find_centroid_of_flux(sci_image)))
                        logging.info('offset_initial_and_sci: ' + str(offset_initial_and_sci))
                        logging.info('[x_primary, y_primary, y_secondary,ratio_secondary] / chi2 output')
                    if self.save == 1:
                        np.save(self.TESTING_FINAL_IMAGES_FOLDER
                                + 'initial_complete_realization', initial_complete_realization)

                    # search for the best center using scipy ``shgo'' algorithm
                    # set the limits for the fitting procedure
                    y_2sources_limits = [
                        (offset_initial_and_sci[1] - 2) * self.oversampling,
                        (offset_initial_and_sci[1] + 2) * self.oversampling]
                    x_2sources_limits = [
                        (offset_initial_and_sci[0] - 1) * self.oversampling,
                        (offset_initial_and_sci[0] + 1) * self.oversampling]
                    # search for best positioning

                    # if use_center_of_flux==True, we use more direct approach to get to the center

                    if use_center_of_flux:
                        for i in range(5):
                            if verbosity == 1:
                                logging.info("###")

                            if i == 0:

                                x_i, y_i = offset_initial_and_sci * oversampling

                                x_offset, y_offset = 0, 0
                                x_offset = x_offset + x_i
                                y_offset = y_offset + y_i
                            else:
                                x_offset = x_offset + x_i
                                y_offset = y_offset + y_i
                            # complete_realization=self.create_complete_realization(x=[x_offset,y_offset,0,0,],\
                            #                                                      return_full_result=True,use_only_chi=True,use_center_of_light=True,simulation_00=False)[-1]
                            complete_realization = self.create_complete_realization(
                                x=[x_offset, y_offset, 0, 0, ], return_full_result=True, use_only_chi=True,
                                use_center_of_light=True, simulation_00=simulation_00)[-1]
                            offset_initial_and_sci = -((np.array(find_centroid_of_flux(complete_realization))
                                                        - np.array(find_centroid_of_flux(sci_image))))
                            if verbosity == 1:
                                logging.info('offset_initial_and_sci in step '
                                             + str(i) + ' ' + str(offset_initial_and_sci))
                                logging.info("###")
                            x_i, y_i = offset_initial_and_sci * oversampling

                        primary_position_and_ratio_x = [x_offset, y_offset]
                    # if use_center_of_flux=False, we have to optimize to find the best solution
                    else:
                        # implement try syntax for secondary too
                        try:
                            # logging.info('simulation_00 here is: '+str(simulation_00))
                            # logging.info('(False, use_only_chi,use_center_of_flux)' +
                            #       str((False, use_only_chi, use_center_of_flux)))
                            # logging.info('x_2sources_limits' + str(x_2sources_limits))
                            # logging.info('y_2sources_limits' + str(y_2sources_limits))
                            primary_position_and_ratio_shgo = scipy.optimize.shgo(
                                self.create_complete_realization,
                                args=(
                                    False,
                                    use_only_chi,
                                    use_center_of_flux,
                                    simulation_00),
                                bounds=[
                                    (x_2sources_limits[0],
                                     x_2sources_limits[1]),
                                    (y_2sources_limits[0],
                                     y_2sources_limits[1])],
                                n=10,
                                sampling_method='sobol',
                                options={
                                    'ftol': 1e-3,
                                    'maxev': 10})

                            # primary_position_and_ratio=primary_position_and_ratio_shgo
                            primary_position_and_ratio = scipy.optimize.minimize(
                                self.create_complete_realization,
                                args=(
                                    False,
                                    use_only_chi,
                                    use_center_of_flux,
                                    simulation_00),
                                x0=primary_position_and_ratio_shgo.x,
                                method='Nelder-Mead',
                                options={
                                    'xatol': 0.00001,
                                    'fatol': 0.00001})

                            primary_position_and_ratio_x = primary_position_and_ratio.x
                        except BaseException as e:
                            logging.info(e)
                            logging.info('search for primary position failed')
                            primary_position_and_ratio_x = [0, 0]

                    # return the best result, based on the result of the conducted search
                    mean_res, single_realization_primary_renormalized,\
                        single_realization_secondary_renormalized, complete_realization_renormalized \
                        = self.create_complete_realization(primary_position_and_ratio_x,
                                                           return_full_result=True,
                                                           use_only_chi=use_only_chi,
                                                           use_center_of_light=use_center_of_flux,
                                                           simulation_00=simulation_00)

                    if self.save == 1:
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER
                            + 'single_realization_primary_renormalized',
                            single_realization_primary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER
                            + 'single_realization_secondary_renormalized',
                            single_realization_secondary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER
                            + 'complete_realization_renormalized',
                            complete_realization_renormalized)

                    if self.verbosity == 1:
                        if simulation_00 != 1:
                            logging.info('We are fitting for only one source')
                            logging.info('One source fitting result is ' + str(primary_position_and_ratio_x))
                            logging.info('type(complete_realization_renormalized)'
                                         + str(type(complete_realization_renormalized[0][0])))

                            centroid_of_complete_realization_renormalized = find_centroid_of_flux(
                                complete_realization_renormalized)

                            # determine offset between the initial guess and the data
                            offset_final_and_sci = - \
                                (np.array(centroid_of_complete_realization_renormalized)
                                 - np.array(centroid_of_sci_image))

                            logging.info('offset_final_and_sci: ' + str(offset_final_and_sci))

                    return complete_realization_renormalized, primary_position_and_ratio_x

                # if you did pass explicit_psf_position for the solution evalute the code here
                else:
                    mean_res, single_realization_primary_renormalized,\
                        single_realization_secondary_renormalized, complete_realization_renormalized\
                        = self.create_complete_realization(explicit_psf_position,
                                                           return_full_result=True,
                                                           use_only_chi=use_only_chi,
                                                           use_center_of_light=use_center_of_flux)

                    if self.save == 1:
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER + 'single_realization_primary_renormalized',
                            single_realization_primary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER + 'single_realization_secondary_renormalized',
                            single_realization_secondary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER + 'complete_realization_renormalized',
                            complete_realization_renormalized)

                    if self.verbosity == 1:
                        if simulation_00 != 1:
                            logging.info('We are passing value for only one source')
                            logging.info('One source fitting result is ' + str(explicit_psf_position))
                            logging.info('type(complete_realization_renormalized)'
                                         + str(type(complete_realization_renormalized[0][0])))

                    return complete_realization_renormalized, explicit_psf_position

        else:
            # TODO: need to make possible that you can pass your own values for double source!!!!
            # create one complete realization with default parameters - estimate
            # centroids and use that knowledge to put fitting limits in the next step
            centroid_of_sci_image = find_centroid_of_flux(sci_image)
            initial_complete_realization = self.create_complete_realization([0,
                                                                             0,
                                                                             double_sources_positions_ratios[0]  # noqa: E501
                                                                             * self.oversampling,
                                                                             double_sources_positions_ratios[1]],  # noqa: E501
                                                                            return_full_result=True,
                                                                            use_only_chi=use_only_chi,
                                                                            use_center_of_light=  # noqa: E251
                                                                            use_center_of_flux,
                                                                            simulation_00=simulation_00)[-1]
            centroid_of_initial_complete_realization = find_centroid_of_flux(initial_complete_realization)

            # determine offset between the initial guess and the data
            offset_initial_and_sci = - \
                (np.array(centroid_of_initial_complete_realization) - np.array(centroid_of_sci_image))

            if verbosity == 1:

                logging.info('Evaulating double source psf positioning loop')
                logging.info('offset_initial_and_sci: ' + str(offset_initial_and_sci))
                logging.info('[x_primary, y_primary, y_secondary,ratio_secondary] / chi2 output')

            if self.save == 1:
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'sci_image', sci_image)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'initial_complete_realization',
                        initial_complete_realization)

            # implement that it does not search if second object far away while in focus
            # focus size is 20
            if shape_of_sci_image == 20 and np.abs(self.double_sources_positions_ratios[0]) > 15:
                if verbosity == 1:
                    logging.info('fitting second source, but assuming that second source is too far')

                # if the second spot is more than 15 pixels away
                # copying code from the non-double source part
                # search for the best center using scipy ``shgo'' algorithm
                # set the limits for the fitting procedure
                y_2sources_limits = [
                    (offset_initial_and_sci[1] - 2) * self.oversampling,
                    (offset_initial_and_sci[1] + 2) * self.oversampling]
                x_2sources_limits = [
                    (offset_initial_and_sci[0] - 1) * self.oversampling,
                    (offset_initial_and_sci[0] + 1) * self.oversampling]
                # search for best positioning
                # implement try for secondary too
                try:
                    # logging.info('(False,use_only_chi,use_center_of_flux)'+str((False,use_only_chi,use_center_of_flux)))
                    primary_position_and_ratio_shgo = scipy.optimize.shgo(
                        self.create_complete_realization,
                        args=(
                            False,
                            use_only_chi,
                            use_center_of_flux,
                            simulation_00),
                        bounds=[
                            (x_2sources_limits[0],
                             x_2sources_limits[1]),
                            (y_2sources_limits[0],
                             y_2sources_limits[1])],
                        n=10,
                        sampling_method='sobol',
                        options={
                            'ftol': 1e-3,
                            'maxev': 10})

                    if verbosity == 1:
                        logging.info('starting finer positioning')

                    # primary_position_and_ratio=primary_position_and_ratio_shgo
                    primary_position_and_ratio = scipy.optimize.minimize(
                        self.create_complete_realization,
                        args=(
                            False,
                            use_only_chi,
                            use_center_of_flux,
                            simulation_00),
                        x0=primary_position_and_ratio_shgo.x,
                        method='Nelder-Mead',
                        options={
                            'xatol': 0.00001,
                            'fatol': 0.00001})

                    primary_position_and_ratio_x = primary_position_and_ratio.x
                except BaseException:
                    logging.info('search for primary position failed')
                    primary_position_and_ratio_x = [0, 0]

                primary_secondary_position_and_ratio_x = np.array([0., 0., 0., 0.])
                primary_secondary_position_and_ratio_x[0] = primary_position_and_ratio_x[0]
                primary_secondary_position_and_ratio_x[1] = primary_position_and_ratio_x[1]

            else:

                # set the limits for the fitting procedure
                y_2sources_limits = [
                    (offset_initial_and_sci[1] - 2) * self.oversampling,
                    (offset_initial_and_sci[1] + 2) * self.oversampling]
                x_2sources_limits = [
                    (offset_initial_and_sci[0] - 1) * self.oversampling,
                    (offset_initial_and_sci[0] + 1) * self.oversampling]
                y_2sources_limits_second_source = [
                    (self.double_sources_positions_ratios[0] - 2) * oversampling,
                    (self.double_sources_positions_ratios[0] + 2) * oversampling]

                # search for best result
                # x position, y_position_1st, y_position_2nd, ratio

                primary_secondary_position_and_ratio = scipy.optimize.shgo(
                    self.create_complete_realization,
                    args=(
                        False,
                        use_only_chi,
                        use_center_of_flux,
                        simulation_00),
                    bounds=[
                        (x_2sources_limits[0],
                         x_2sources_limits[1]),
                        (y_2sources_limits[0],
                         y_2sources_limits[1]),
                        (y_2sources_limits_second_source[0],
                         y_2sources_limits_second_source[1]),
                        (self.double_sources_positions_ratios[1] / 2,
                         2 * self.double_sources_positions_ratios[1])],
                    n=10,
                    sampling_method='sobol',
                    options={
                        'ftol': 1e-3,
                        'maxev': 10})

                primary_secondary_position_and_ratio_x = primary_secondary_position_and_ratio.x

            # primary_secondary_position_and_ratio=scipy.optimize.shgo(self.create_complete_realization,(False,use_only_chi,use_center_of_flux),bounds=\
            #                                                         [(x_2sources_limits[0],x_2sources_limits[1]),(y_2sources_limits[0],y_2sources_limits[1]),\
            #                                                          (y_2sources_limits_second_source[0],y_2sources_limits_second_source[1]),\
            #                                                          (self.double_sources_positions_ratios[1]/2,2*self.double_sources_positions_ratios[1])],n=10,sampling_method='sobol',\
            #                                                          options={'maxev':10,'ftol':1e-3})

            # return best result
            # introduce best_result=True
            mean_res, single_realization_primary_renormalized,
            single_realization_secondary_renormalized, complete_realization_renormalized \
                = self.create_complete_realization(primary_secondary_position_and_ratio_x,
                                                   return_full_result=True, use_only_chi=use_only_chi,
                                                   use_center_of_light=use_center_of_flux,
                                                   simulation_00=simulation_00)

            if self.save == 1:
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER + 'single_realization_primary_renormalized',
                    single_realization_primary_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER + 'single_realization_secondary_renormalized',
                    single_realization_secondary_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER + 'complete_realization_renormalized',
                    complete_realization_renormalized)

            if self.verbosity == 1:
                logging.info('We are fitting for two sources')
                logging.info('Two source fitting result is ' + str(primary_secondary_position_and_ratio_x))
                logging.info('type(complete_realization_renormalized)'
                             + str(type(complete_realization_renormalized[0][0])))

        return complete_realization_renormalized, primary_secondary_position_and_ratio_x

    def create_complete_realization(
            self,
            x,
            return_full_result=False,
            use_only_chi=False,
            use_center_of_light=False,
            simulation_00=False):
        """Create one complete downsampled realization of the image,
        from the full oversampled image
        Parameters
        ----------
        x: `np.array`, (4,)
            array contaning x_primary, y_primary,
            offset in y to secondary source, \
            ratio in flux from secondary to primary;
            the units are oversampled pixels
        return_full_result: `bool`
            if True, returns the images itself (not just chi**2)
        use_only_chi: `bool`
                if True, minimize chi; if False, minimize chi^2
        use_center_of_light: `bool`
          if True, minimize distance to center of light, in focus
        simulation_00: `bool`
            if True,do not move the center, for making fair comparisons between
            models - optical center in places in the center of the image
            if use_center_of_light==True the behaviour changes
            and the result is the image with center of flux
            in the center of the image
        Returns
        ----------
        chi_2_almost_multi_values: `float`
            returns the measure of quality
            (chi**2, chi, or distance of center
             of light between science and model image)
            distance of center of light between science
            and model image is given in units of pixels
        single_primary_realization_renormalized: `np.array`, (N, N)
            image containg the model corresponding
            to the primary source in the science image
         single_secondary_realization_renormalized: `np.array`, (N, N)
            image containg the model corresponding
            to the secondary source in the science image
        complete_realization_renormalized: `np.array`, (N, N)
            image combining the primary
            and secondary source (if secondary source is needed)
        Notes
        ----------
        TODO: implement that you are able to call outside find_single_realization_min_cut
        Called by find_single_realization_min_cut
        Calls create_chi_2_almost_Psf_position
        """
        # oversampled input image
        image = self.image

        sci_image = self.sci_image
        var_image = self.var_image
        mask_image = self.mask_image
        shape_of_sci_image = self.size_natural_resolution

        oversampling = self.oversampling
        v_flux = self.v_flux

        # central position of the create oversampled image
        center_position = int(np.floor(image.shape[0] / 2))
        # to be applied on x-axis
        primary_offset_axis_1 = x[0]
        # to be applied on y-axis
        primary_offset_axis_0 = x[1]
        if simulation_00 == 1:
            simulation_00 = True

        # if you are only fitting for primary image
        # add zero values for secondary image
        if len(x) == 2:
            ratio_secondary = 0
        else:
            ratio_secondary = x[3]

        if len(x) == 2:
            secondary_offset_axis_1 = 0
            secondary_offset_axis_0 = 0
        else:
            secondary_offset_axis_1 = primary_offset_axis_1
            secondary_offset_axis_0 = x[2] + primary_offset_axis_0

        shape_of_oversampled_image = int(shape_of_sci_image * oversampling / 2)

        # from https://github.com/Subaru-PFS/drp_stella/blob/\
        #    6cceadfc8721fcb1c7eb1571cf4b9bc8472e983d/src/SpectralPsf.cc
        # // Binning by an odd factor requires the centroid at the center of a pixel.
        # // Binning by an even factor requires the centroid on the edge of a pixel.

        # the definitions used in primary image
        # we separate if the image shape is odd or even, but at the moment there is no difference
        if np.modf(shape_of_oversampled_image / 2)[0] == 0.0:
            # logging.info('shape is an even number')
            shift_x_mod = np.array(
                [-(np.round(primary_offset_axis_1) - primary_offset_axis_1),
                 -np.round(primary_offset_axis_1)])
            shift_y_mod = np.array(
                [-(np.round(primary_offset_axis_0) - primary_offset_axis_0),
                 -np.round(primary_offset_axis_0)])
        else:
            # logging.info('shape is an odd number')
            shift_x_mod = np.array(
                [-(np.round(primary_offset_axis_1) - primary_offset_axis_1),
                 -np.round(primary_offset_axis_1)])
            shift_y_mod = np.array(
                [-(np.round(primary_offset_axis_0) - primary_offset_axis_0),
                 -np.round(primary_offset_axis_0)])

        image_integer_offset = image[center_position
                                     + int(shift_y_mod[1]) - 1
                                     - shape_of_oversampled_image:center_position
                                     + int(shift_y_mod[1])
                                     + shape_of_oversampled_image + 1,
                                     center_position
                                     + int(shift_x_mod[1]) - 1
                                     - shape_of_oversampled_image: center_position
                                     + int(shift_x_mod[1])
                                     + shape_of_oversampled_image + 1]
        if simulation_00:
            image_integer_offset = image[center_position
                                         + int(shift_y_mod[1]) - 1
                                         - shape_of_oversampled_image:center_position
                                         + int(shift_y_mod[1])
                                         + shape_of_oversampled_image + 1 + 1,
                                         center_position
                                         + int(shift_x_mod[1]) - 1
                                         - shape_of_oversampled_image: center_position
                                         + int(shift_x_mod[1])
                                         + shape_of_oversampled_image + 1 + 1]
            logging.info('image_integer_offset shape: ' + str(image_integer_offset.shape))

        image_integer_offset_lsst = lsst.afw.image.image.ImageD(image_integer_offset.astype('float64'))

        oversampled_Image_LSST_apply_frac_offset = lsst.afw.math.offsetImage(
            image_integer_offset_lsst, shift_x_mod[0], shift_y_mod[0], algorithmName='lanczos5', buffer=5)

        single_primary_realization_oversampled = oversampled_Image_LSST_apply_frac_offset.array[1:-1, 1:-1]

        # logging.info('single_primary_realization_oversampled.shape[0]: '+
        # str(single_primary_realization_oversampled.shape[0]))
        # logging.info('shape_of_sci_image: '+str(shape_of_sci_image))
        # logging.info('oversampling: '+str(oversampling))

        assert single_primary_realization_oversampled.shape[0] == shape_of_sci_image * oversampling

        single_primary_realization = resize(
            single_primary_realization_oversampled, (shape_of_sci_image, shape_of_sci_image), ())

        # im1=  galsim.Image(image, copy=True,scale=1)
        # time_2=time.time()
        # interpolated_image = galsim._InterpolatedImage(im1,\
        #                     x_interpolant=galsim.Lanczos(5, True))
        # time_3=time.time()
        # time_3_1=time.time()
        # single_primary_realization_oversampled_1 =
        # interpolated_image.shift(primary_offset_axis_1,primary_offset_axis_0 )
        # time_3_2=time.time()
        # single_primary_realization_oversampled_2=single_primary_realization_oversampled_1.drawImage\
        # (nx=shape_of_sci_image*oversampling, ny=shape_of_sci_image*oversampling, scale=1, method='no_pixel')
        # time_3_3=time.time()
        # single_primary_realization_oversampled_3=single_primary_realization_oversampled_2.array
        # time_4=time.time()
        # single_primary_realization = resize(single_primary_realization_oversampled_3,\
        # (shape_of_sci_image,shape_of_sci_image),())
        # time_5=time.time()
        # if self.verbosity==1:
        #    logging.info('time_2-time_1 for shift and resize '+str(time_2-time_1))
        #    logging.info('time_3-time_2 for shift and resize '+str(time_3-time_2))
        #    logging.info('time_3_1-time_3 for shift and resize '+str(time_3_1-time_3))
        #    logging.info('time_3_2-time_3_1 for shift and resize '+str(time_3_2-time_3_1))
        #    logging.info('time_3_3-time_3_2 for shift and resize '+str(time_3_3-time_3_2))
        #    logging.info('time_4-time_3_3 for shift and resize '+str(time_4-time_3_3))
        #    logging.info('time_4-time_3 for shift and resize '+str(time_4-time_3))
        #    logging.info('time_5-time_4 for shift and resize '+str(time_5-time_4))
        #    logging.info('time_5-time_1 for shift and resize '+str(time_5-time_1))

        ###################
        # skip this part if only doing primary
        # go through secondary loop if the flux ratio is not zero
        # (needs to be implemented - if secondary too far outside the image, do not go through secondary)
        if ratio_secondary != 0:

            # overloading the definitions used in primary image
            if np.modf(shape_of_oversampled_image / 2)[0] == 0.0:
                # logging.info('shape is an even number')

                shift_x_mod = np.array(
                    [-(np.round(secondary_offset_axis_1) - secondary_offset_axis_1),
                     -np.round(secondary_offset_axis_1)])
                shift_y_mod = np.array(
                    [-(np.round(secondary_offset_axis_0) - secondary_offset_axis_0),
                     -np.round(secondary_offset_axis_0)])

            else:
                # logging.info('shape is an odd number')
                shift_x_mod = np.array(
                    [-(np.round(secondary_offset_axis_1) - secondary_offset_axis_1),
                     -np.round(secondary_offset_axis_1)])
                shift_y_mod = np.array(
                    [-(np.round(secondary_offset_axis_0) - secondary_offset_axis_0),
                     -np.round(secondary_offset_axis_0)])

            image_integer_offset = image[center_position
                                         + int(shift_y_mod[1]) - 1
                                         - shape_of_oversampled_image:center_position
                                         + int(shift_y_mod[1])
                                         + shape_of_oversampled_image + 2,
                                         center_position
                                         + int(shift_x_mod[1]) - 1
                                         - shape_of_oversampled_image: center_position
                                         + int(shift_x_mod[1])
                                         + shape_of_oversampled_image + 2]

            image_integer_offset_lsst = lsst.afw.image.image.ImageD(image_integer_offset.astype('float64'))

            oversampled_Image_LSST_apply_frac_offset = lsst.afw.math.offsetImage(
                image_integer_offset_lsst, shift_y_mod[0], shift_x_mod[0], algorithmName='lanczos5', buffer=5)

            single_secondary_realization_oversampled =\
                oversampled_Image_LSST_apply_frac_offset.array[1:-1, 1:-1]

            single_secondary_realization = resize(
                single_secondary_realization_oversampled, (shape_of_sci_image, shape_of_sci_image), ())

        inverted_mask = ~mask_image.astype(bool)

        ###################
        # create complete_realization which is just pimary if no secondary source
        # if there is secondary source, add two images together
        if ratio_secondary != 0:
            complete_realization = single_primary_realization + ratio_secondary * single_secondary_realization
            complete_realization_renormalized = complete_realization * \
                (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
        else:

            complete_realization = single_primary_realization
            complete_realization_renormalized = complete_realization * \
                (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))

        ###################
        # find chi values and save the results
        # logging.info('checkpoint in create_complete_realization')
        if not return_full_result:
            # time_1 = time.time()
            chi_2_almost_multi_values = self.create_chi_2_almost_Psf_position(
                complete_realization_renormalized,
                sci_image,
                var_image,
                mask_image,
                use_only_chi=use_only_chi,
                use_center_of_light=use_center_of_light,
                simulation_00=simulation_00)
            # time_2 = time.time()
            if self.verbosity == 1:
                logging.info(
                    'chi2 within shgo with use_only_chi ' + str(use_only_chi)
                    + ' and use_center_of_light ' + str(use_center_of_light) + ' ' + str(x) + ' / '
                    + str(chi_2_almost_multi_values))
                # logging.info('time_2-time_1 for create_chi_2_almost_Psf_position: '+str(time_2-time_1))
            return chi_2_almost_multi_values
        else:
            if ratio_secondary != 0:
                # logging.info('ratio_secondary 2nd loop: '+str(ratio_secondary))
                single_primary_realization_renormalized = single_primary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized = ratio_secondary * single_secondary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
            else:
                # logging.info('ratio_secondary 2nd loop 0: '+str(ratio_secondary))
                single_primary_realization_renormalized = single_primary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized = np.zeros(
                    single_primary_realization_renormalized.shape)

            if self.save == 1:
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'image', image)
                if ratio_secondary != 0:
                    np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'image_full_for_secondary', image)
                    np.save(self.TESTING_FINAL_IMAGES_FOLDER
                            + 'single_secondary_realization', single_secondary_realization)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER
                        + 'single_primary_realization', single_primary_realization)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER
                        + 'single_primary_realization_renormalized_within_create_complete_realization',
                        single_primary_realization_renormalized)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER
                        + 'single_secondary_realization_renormalized_within_create_complete_realization',
                        single_secondary_realization_renormalized)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER
                        + 'complete_realization_renormalized_within_create_complete_realization',
                        complete_realization_renormalized)

            # should I modify this function to remove distance from physcial center of
            # mass when using that option
            chi_2_almost_multi_values = self.create_chi_2_almost_Psf_position(
                complete_realization_renormalized,
                sci_image,
                var_image,
                mask_image,
                use_only_chi=use_only_chi,
                use_center_of_light=use_center_of_light,
                simulation_00=simulation_00)

            # if best, save oversampled image
            if simulation_00:
                if self.verbosity == 1:
                    logging.info('saving oversampled simulation_00 image')
                    # logging.info('I have to implement that again')
                    logging.info('saving at ' + self.TESTING_FINAL_IMAGES_FOLDER
                                 + 'single_primary_realization_oversampled')
                    np.save(self.TESTING_FINAL_IMAGES_FOLDER
                            + 'single_primary_realization_oversampled_to_save',
                            single_primary_realization_oversampled)
                    np.save(self.TESTING_FINAL_IMAGES_FOLDER
                            + 'complete_realization_renormalized_to_save',
                            single_primary_realization_oversampled)

            return chi_2_almost_multi_values,\
                single_primary_realization_renormalized, single_secondary_realization_renormalized,\
                complete_realization_renormalized

    def create_chi_2_almost_Psf_position(self, modelImg, sci_image, var_image, mask_image,
                                         use_only_chi=False, use_center_of_light=False, simulation_00=False):
        """Returns quality of the model's fit compared to the science image
        Parameters
        ----------
        modelImg: `np.array`, (N, N)
            model image
        sci_image: `np.array`, (N, N)
            science image
        var_image: `np.array`, (N, N)
            variance image
        mask_image: `np.array`, (N, N)
            mask image
        use_only_chi: `bool`
            if True, minimize chi; if False, minimize chi^2
        use_center_of_light: `bool`
            if True, minimizes distance of center of light between science
            and model image
        simulation_00: `bool`
            if True,do not move the center, for making fair comparisons between
            models - optical center in places in the center of the image
            if use_center_of_light==True the behaviour changes
            and the result is the image with center of flux
            in the center of the image
        Returns
        ----------
        measure_of_quality: `float`
            returns the measure of quality
            (chi**2, chi, or distance of center
             of light between science and model image)
            distance of center of light between science
            and model image is given in units of pixels
        Notes
        ----------
        Called by create_complete_realization
        """
        inverted_mask = ~mask_image.astype(bool)

        var_image_masked = var_image * inverted_mask
        sci_image_masked = sci_image * inverted_mask
        modelImg_masked = modelImg * inverted_mask

        # if you are minimizing chi or chi**2
        if not use_center_of_light:
            if not use_only_chi:
                chi2 = (sci_image_masked - modelImg_masked)**2 / var_image_masked
                chi2nontnan = chi2[~np.isnan(chi2)]
            if use_only_chi:
                chi2 = np.abs((sci_image_masked - modelImg_masked))**1 / np.sqrt(var_image_masked)
                chi2nontnan = chi2[~np.isnan(chi2)]
            return np.mean(chi2nontnan)
        else:
            if simulation_00 is False or simulation_00 is None:
                if self.verbosity == 1:
                    logging.info('sim00=False and center of light =true')

                distance_of_flux_center = np.sqrt(
                    np.sum((np.array(
                            find_centroid_of_flux(modelImg_masked))
                            - np.array(
                            find_centroid_of_flux(sci_image_masked)))**2))
            else:
                # if you pass both simulation_00 paramter and use_center_of_light=True,
                # center of light will be centered
                # in the downsampled image
                if self.verbosity == 1:
                    logging.info('sim00=True and center of light =true')

                distance_of_flux_center = np.sqrt(
                    np.sum((np.array(find_centroid_of_flux(modelImg_masked))
                            - np.array(np.array(np.ones((21, 21)).shape)
                                       / 2 - 0.5))**2))
            # logging.info('distance_of_flux_center: '+str(distance_of_flux_center))
            return distance_of_flux_center

    def fill_crop(self, img, pos, crop):
        '''
        Fills `crop` with values from `img` at `pos`,
        while accounting for the crop being off the edge of `img`.
        *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
        Taken from https://stackoverflow.com/questions/41153803/zero-padding-slice-past-end-of-array-in-numpy  # noqa:E501
        '''
        img_shape, pos, crop_shape = np.array(
            img.shape, dtype=int), np.array(
            pos, dtype=int), np.array(
            crop.shape, dtype=int)
        end = pos + crop_shape
        # Calculate crop slice positions
        crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
        crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
        crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
        # Calculate img slice positions
        pos = np.clip(pos, a_min=0, a_max=img_shape)
        end = np.clip(end, a_min=0, a_max=img_shape)
        img_slices = (slice(low, high) for low, high in zip(pos, end))
        try:
            crop[tuple(crop_slices)] = img[tuple(img_slices)]
        except TypeError:
            logging.info('TypeError in fill_crop function')
            pass


class Zernike_estimation_preparation(object):
    """
    Class that creates the inputs for the Zernike parameter estimation

    Parameters
    ----------
    list_of_obs : `list`
        list of observations
    list_of_spots :
        list of spots to be analyzed
    dataset : int

    list_of_arc : `list`
        list of arcs to be analyzed

        ...


    Returns
    ----------
    array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d : `np.array`
        array of the initial parametrization proposals
        input for ?
    array_of_sci_images_multi_spot : `np.array`
        array of science images
    array_of_var_images_multi_spot : `np.array`
        array of var images
    array_of_mask_images_multi_spot : `np.array`
        array of mask images

    """

    def __init__(self, list_of_labelInput, list_of_spots, dataset,
                 list_of_arc, eps, nsteps, analysis_type='defocus',
                 analysis_type_fiber=None,
                 DIRECTORY=None):

        self.list_of_labelInput = list_of_labelInput
        self.list_of_spots = list_of_spots
        self.dataset = dataset
        self.list_of_arc = list_of_arc
        self.eps = eps
        self.nsteps = nsteps
        self.analysis_type = analysis_type
        self.analysis_type_fiber = analysis_type_fiber
        if DIRECTORY is None:
            DIRECTORY = '/tigress/ncaplar/'
        self.DIRECTORY = DIRECTORY

        # TODO : make this as input or deduce from the data
        self.multi_var = True

        logging.info('Dataset analyzed is: ' + str(dataset))
        if dataset == 0 or dataset == 1:
            logging.info('ehm.... old data, not analyzed')

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
            if socket.gethostname() == 'pfsa-usr01-gb.subaru.nao.ac.jp' or \
               socket.gethostname() == 'pfsa-usr02-gb.subaru.nao.ac.jp':
                DATA_FOLDER = '/work/dev_2ddrp/ReducedData/Data_Nov_20/'
            else:
                DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_Nov_20/'

        # folder contaning the data taken with F/2.8 stop in June 2021, at LAM, on SM2
        if dataset == 7:
            DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_21_2021/'

        # folder contaning the data taken with F/2.8 stop in June 2021, at Subaru
        # (21 fibers)
        if dataset == 8:
            if 'subaru' in socket.gethostname():
                # DATA_FOLDER = '/work/ncaplar/ReducedData/Data_May_25_2021/'
                DATA_FOLDER = '/work/dev_2ddrp/ReducedData/Data_May_25_2021/'
            else:
                DATA_FOLDER = '/tigress/ncaplar/ReducedData/Data_May_25_2021/'

        STAMPS_FOLDER = DATA_FOLDER + 'Stamps_cleaned/'
        DATAFRAMES_FOLDER = DATA_FOLDER + 'Dataframes/'
        if 'subaru' in socket.gethostname():
            # RESULT_FOLDER = '/work/ncaplar/Results/'
            RESULT_FOLDER = self.DIRECTORY + 'Results/'
        else:
            RESULT_FOLDER = '/tigress/ncaplar/Results/'

        self.STAMPS_FOLDER = STAMPS_FOLDER
        self.DATAFRAMES_FOLDER = DATAFRAMES_FOLDER
        self.RESULT_FOLDER = RESULT_FOLDER

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
            options = [480, 2.793, 0.593]
        if eps == 6:
            options = [480, 2.793, 1.193]
        if eps == 7:
            options = [48, 2.793, 1.193]
        if eps == 8:
            options = [480, 2.793, 0.193]
        if eps == 9:
            options = [190, 1.193, 1.193]
            nsteps = int(2 * nsteps)
        if eps == 10:
            options = [390, 1.893, 2.893]

        particleCount = options[0]
        c1 = options[1]
        c2 = options[2]

        self.particleCount = particleCount
        self.options = options
        self.c1 = c1
        self.c2 = c2

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
            'detFrac',
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

        self.columns22 = columns22

    def return_auxiliary_info(self):
        """


        Parameters
        ----------

        Return
        ----------


        """

        return self.particleCount, self.c1, self.c2

    def create_list_of_obs_from_list_of_label(self):
        """


        Parameters
        ----------

        Return
        ----------


        """
        dataset = self.dataset
        list_of_labelInput = self.list_of_labelInput
        # arc = self.list_of_arc[0]

        logging.info('self.list_of_arc: '+str(self.list_of_arc))
        ################################################
        # (3.) import obs_pos & connect
        ################################################

        # What are the observations that can be analyzed
        # This information is used to associate observation with their input labels
        # (see definition of `label` below)
        # This is so that the initial parameters guess is correct

        list_of_obs_possibilites = []
        for arc in self.list_of_arc:

            # dataset 0, December 2017 data - possibly deprecated
            """
            if arc == 'HgAr':
                obs_possibilites = np.array([8552, 8555, 8558, 8561, 8564, 8567, 8570, 8573,
                                             8603, 8600, 8606, 8609, 8612, 8615, 8618, 8621, 8624, 8627])
            elif arc == 'Ne':
                logging.info('Neon?????')
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
                    obs_possibilites = np.array([16238 + 6,
                                                 16238 + 12,
                                                 16238 + 18,
                                                 16238 + 24,
                                                 16238 + 30,
                                                 16238 + 36,
                                                 16238 + 42,
                                                 16238 + 48,
                                                 16238 + 54,
                                                 16238 + 60,
                                                 16238 + 66,
                                                 16238 + 72,
                                                 16238 + 78,
                                                 16238 + 84,
                                                 16238 + 90,
                                                 16238 + 96,
                                                 16238 + 102,
                                                 16238 + 54])
                if arc == 'Kr':
                    obs_possibilites = np.array([17310 + 6,
                                                 17310 + 12,
                                                 17310 + 18,
                                                 17310 + 24,
                                                 17310 + 30,
                                                 17310 + 36,
                                                 17310 + 42,
                                                 17310 + 48,
                                                 17310 + 54,
                                                 17310 + 60,
                                                 17310 + 66,
                                                 17310 + 72,
                                                 17310 + 78,
                                                 17310 + 84,
                                                 17310 + 90,
                                                 17310 + 96,
                                                 17310 + 102,
                                                 17310 + 54])

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
                    obs_possibilites = np.array([19472 + 6,
                                                 19472 + 12,
                                                 19472 + 18,
                                                 19472 + 24,
                                                 19472 + 30,
                                                 19472 + 36,
                                                 19472 + 42,
                                                 19472 + 48,
                                                 19472 + 54,
                                                 19472 + 60,
                                                 19472 + 66,
                                                 19472 + 72,
                                                 19472 + 78,
                                                 19472 + 84,
                                                 19472 + 90,
                                                 19472 + 96,
                                                 19472 + 102,
                                                 19472 + 54])

            # F/2.8 July data
            if dataset == 4:
                if arc == 'HgAr':
                    obs_possibilites = np.array([21346 + 6,
                                                 21346 + 12,
                                                 21346 + 18,
                                                 21346 + 24,
                                                 21346 + 30,
                                                 21346 + 36,
                                                 21346 + 42,
                                                 21346 + 48,
                                                 21346 + 54,
                                                 21346 + 60,
                                                 21346 + 66,
                                                 21346 + 72,
                                                 21346 + 78,
                                                 21346 + 84,
                                                 21346 + 90,
                                                 21346 + 96,
                                                 21346 + 102,
                                                 21346 + 48])
                if arc == 'Ne':
                    obs_possibilites = np.array([21550 + 6,
                                                 21550 + 12,
                                                 21550 + 18,
                                                 21550 + 24,
                                                 21550 + 30,
                                                 21550 + 36,
                                                 21550 + 42,
                                                 21550 + 48,
                                                 21550 + 54,
                                                 21550 + 60,
                                                 21550 + 66,
                                                 21550 + 72,
                                                 21550 + 78,
                                                 21550 + 84,
                                                 21550 + 90,
                                                 21550 + 96,
                                                 21550 + 102,
                                                 21550 + 54])
                if arc == 'Kr':
                    obs_possibilites = np.array([21754 + 6,
                                                 21754 + 12,
                                                 21754 + 18,
                                                 21754 + 24,
                                                 21754 + 30,
                                                 21754 + 36,
                                                 21754 + 42,
                                                 21754 + 48,
                                                 21754 + 54,
                                                 21754 + 60,
                                                 21754 + 66,
                                                 21754 + 72,
                                                 21754 + 78,
                                                 21754 + 84,
                                                 21754 + 90,
                                                 21754 + 96,
                                                 21754 + 102,
                                                 21754 + 54])

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
                    obs_possibilites = np.array([27779,
                                                 - 999,
                                                 27683,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 27767,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 27698,
                                                 - 999,
                                                 27773,
                                                 - 999])
                if arc == 'Ne':
                    obs_possibilites = np.array([27713,
                                                 - 999,
                                                 27683,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 27677,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 - 999,
                                                 27698,
                                                 - 999,
                                                 27719,
                                                 - 999])
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

            logging.info('arc: '+str(arc))
            logging.info('obs_possibilites: '+str(obs_possibilites))
            list_of_obs_possibilites.append(obs_possibilites)

        logging.info('list_of_obs_possibilites: '+str(list_of_obs_possibilites))
        ##############################################

        # associates each observation with the label
        # describing movement of the hexapod and rough estimate of z4
        z4Input_possibilites = np.array([28, 24.5, 21, 17.5, 14, 10.5, 7, 3.5, 0,  # noqa F841
                                         -3.5, -7, -10.5, -14, -17.5, -21, -24.5, -28, 0])
        label = ['m4', 'm35', 'm3', 'm25', 'm2', 'm15', 'm1', 'm05', '0',
                 'p05', 'p1', 'p15', 'p2', 'p25', 'p3', 'p35', 'p4', '0p']

        list_of_obs_cleaned = []

        for a in range(len(self.list_of_arc)):
            obs_possibilites = list_of_obs_possibilites[a]
            list_of_obs = []
            for i in range(len(list_of_labelInput)):
                label_i = list_of_labelInput[i]
                obs_cleaned = obs_possibilites[list(label).index(label_i)]
                list_of_obs.append(obs_cleaned)
            list_of_obs_cleaned.append(list_of_obs)

        self.list_of_obs_cleaned = list_of_obs_cleaned
        # TODO: clean out this cheating here
        list_of_obs = list_of_obs_cleaned
        self.list_of_obs = list_of_obs
        self.list_of_obs_cleaned = list_of_obs_cleaned
        logging.info('self.list_of_obs:'+str(self.list_of_obs))
        logging.info('list_of_obs_cleaned:'+str(list_of_obs_cleaned))
        return list_of_obs_cleaned

    def get_sci_var_mask_data(self):
        """
        Get sci, var and mask data

        Parameters
        ----------

        Return
        ----------


        """

        STAMPS_FOLDER = self.STAMPS_FOLDER

        list_of_sci_images_multi_spot = []
        list_of_mask_images_multi_spot = []
        list_of_var_images_multi_spot = []
        list_of_obs_cleaned_multi_spot = []

        self.create_list_of_obs_from_list_of_label()

        logging.info('list_of_obs_cleaned'+str(self.list_of_obs_cleaned))
        for s in range(len(self.list_of_spots)):
            arc = self.list_of_arc[s]
            single_number = self.list_of_spots[s]

            list_of_sci_images = []
            list_of_mask_images = []
            list_of_var_images = []
            list_of_obs_cleaned = []
            # list_of_times = []

            # loading images for the analysis
            for obs in self.list_of_obs_cleaned[s]:
                try:
                    sci_image = np.load(
                        STAMPS_FOLDER + 'sci' + str(obs)
                        + str(single_number) + str(arc) + '_Stacked.npy')
                    mask_image = np.load(
                        STAMPS_FOLDER + 'mask' + str(obs)
                        + str(single_number) + str(arc) + '_Stacked.npy')
                    var_image = np.load(
                        STAMPS_FOLDER + 'var' + str(obs)
                        + str(single_number) + str(arc) + '_Stacked.npy')
                    logging.info(
                        'sci_image loaded from: ' + STAMPS_FOLDER + 'sci'
                        + str(obs) + str(single_number) + str(arc) + '_Stacked.npy')
                except Exception:
                    # change to that code does not fail and hang if the image is not found
                    # this will lead to pass statment in next step because
                    # np.sum(sci_image) = 0
                    logging.info('sci_image not found')
                    sci_image = np.zeros((20, 20))
                    var_image = np.zeros((20, 20))
                    mask_image = np.zeros((20, 20))
                    logging.info('not able to load image at: ' + str(STAMPS_FOLDER + 'sci'
                                                                     + str(obs) + str(single_number)
                                                                     + str(arc) + '_Stacked.npy'))

                # If there is no science image, do not add images
                if int(np.sum(sci_image)) == 0:
                    logging.info('No science image - passing')
                    pass
                else:
                    # do not analyze images where a large fraction of the image is masked
                    if np.mean(mask_image) > 0.1:
                        logging.info(str(np.mean(mask_image) * 100)
                                     + '% of image is masked... \
                              when it is more than 10% - exiting')
                        pass
                    else:
                        # the images ahs been found successfully
                        logging.info('adding images for obs: ' + str(obs))
                        list_of_sci_images.append(sci_image)
                        list_of_mask_images.append(mask_image)
                        list_of_var_images.append(var_image)

                        # observation which are of good enough quality to be analyzed get added here
                        list_of_obs_cleaned.append(obs)

            logging.info('for spot ' + str(self.list_of_spots[s]) + ' len of list_of_sci_images: '
                         + str(len(list_of_sci_images)))
            logging.info('len of accepted images ' + str(len(list_of_obs_cleaned))
                         + ' / len of asked images ' + str(len(self.list_of_obs_cleaned[s])))

            # If there is no valid images imported, exit
            if list_of_sci_images == []:
                logging.info('No valid images - exiting')
                sys.exit(0)

            # if you were able only to import only a fraction of images
            # if this fraction is too low - exit
            if (len(list_of_obs_cleaned) / len(self.list_of_obs_cleaned[s])) < 0.6:
                logging.info('Fraction of images imported is too low - exiting')
                sys.exit(0)

            list_of_sci_images_multi_spot.append(list_of_sci_images)
            list_of_mask_images_multi_spot.append(list_of_mask_images)
            list_of_var_images_multi_spot.append(list_of_var_images)
            list_of_obs_cleaned_multi_spot.append(list_of_obs_cleaned)

        self.list_of_obs_cleaned = list_of_obs_cleaned

        self.list_of_sci_images = list_of_sci_images
        self.list_of_var_images = list_of_var_images
        self.list_of_mask_images = list_of_mask_images

        array_of_sci_images_multi_spot = np.array(list_of_sci_images_multi_spot)
        array_of_mask_images_multi_spot = np.array(list_of_mask_images_multi_spot)
        array_of_var_images_multi_spot = np.array(list_of_var_images_multi_spot)
        array_of_obs_cleaned_multi_spot = np.array(list_of_obs_cleaned_multi_spot)

        self.array_of_obs_cleaned_multi_spot = array_of_obs_cleaned_multi_spot

        return array_of_sci_images_multi_spot, array_of_var_images_multi_spot,\
            array_of_mask_images_multi_spot, array_of_obs_cleaned_multi_spot

    def create_output_names(self, date_of_output):
        """
        Get sci, var and mask data

        Parameters
        ----------

        Return
        ----------


        """

        eps = self.eps

        list_of_NAME_OF_CHAIN = []
        list_of_NAME_OF_LIKELIHOOD_CHAIN = []

        for s in range(len(self.list_of_spots)):
            arc = self.list_of_arc[s]
            # to be consistent with previous versions of the code, use the last obs avalible in the name
            obs_for_naming = self.array_of_obs_cleaned_multi_spot[s][-1]
            # give it invidual name here, just to make srue that by accident we do not
            # overload the variable and cause errors downstream
            single_number_str = self.list_of_spots[s]
            NAME_OF_CHAIN = 'chain' + str(date_of_output) + '_Single_P_' + \
                str(obs_for_naming) + str(single_number_str) + str(eps) + str(arc)
            NAME_OF_LIKELIHOOD_CHAIN = 'likechain' + str(date_of_output) + '_Single_P_' +\
                str(obs_for_naming) + str(single_number_str) + str(eps) + str(arc)

            list_of_NAME_OF_CHAIN.append(NAME_OF_CHAIN)
            list_of_NAME_OF_LIKELIHOOD_CHAIN.append(NAME_OF_LIKELIHOOD_CHAIN)

        return list_of_NAME_OF_CHAIN, list_of_NAME_OF_LIKELIHOOD_CHAIN

    def get_finalArc(self, arc, date_of_input='Sep0521', direct_or_interpolation='direct'):
        # TODO make it so you ont need to specify date of input if you only want finalArc
        DATAFRAMES_FOLDER = self.DATAFRAMES_FOLDER

        dataset = self.dataset
        ################################################
        # (3.) import dataframes
        ################################################

        # where are the dataframes located
        # these files give auxiliary information which enables us to connect spot number with other properties
        # such as the position on the detector, wavelength, etc...
        # Ar (Argon)
        if str(arc) == 'Ar' or arc == 'HgAr':
            with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation)
                      + '_Ar_from_' + str(date_of_input) + '.pkl', 'rb') as f:
                results_of_fit_input_HgAr = pickle.load(f)
                logging.info('results_of_fit_input_Ar is taken from: ' + str(f))
            # if before considering all fibers
            if dataset < 8:
                with open(DATAFRAMES_FOLDER + 'finalAr_Feb2020', 'rb') as f:
                    finalAr_Feb2020_dataset = pickle.load(f)
            else:
                with open(DATAFRAMES_FOLDER + 'finalAr_Jul2021.pkl', 'rb') as f:
                    finalAr_Feb2020_dataset = pickle.load(f)

        # Ne (Neon)
        if str(arc) == 'Ne':
            with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation)
                      + '_Ne_from_' + str(date_of_input) + '.pkl', 'rb') as f:
                results_of_fit_input_Ne = pickle.load(f)
            logging.info('results_of_fit_input_Ne is taken from: ' + str(f))
            if dataset < 8:
                with open(DATAFRAMES_FOLDER + 'finalNe_Feb2020', 'rb') as f:
                    finalNe_Feb2020_dataset = pickle.load(f)
            else:
                with open(DATAFRAMES_FOLDER + 'finalNe_Jul2021.pkl', 'rb') as f:
                    finalNe_Feb2020_dataset = pickle.load(f)

        # Kr (Krypton)
        if str(arc) == 'Kr':
            with open(DATAFRAMES_FOLDER + 'results_of_fit_many_' + str(direct_or_interpolation)
                      + '_Kr_from_' + str(date_of_input) + '.pkl', 'rb') as f:
                results_of_fit_input_Kr = pickle.load(f)
            logging.info('results_of_fit_input_Kr is taken from: ' + str(f))
            if dataset < 8:
                with open(DATAFRAMES_FOLDER + 'finalKr_Feb2020', 'rb') as f:
                    finalKr_Feb2020_dataset = pickle.load(f)
            else:
                with open(DATAFRAMES_FOLDER + 'finalKr_Jul2021.pkl', 'rb') as f:
                    finalKr_Feb2020_dataset = pickle.load(f)

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

        self.results_of_fit_input = results_of_fit_input
        self.finalArc = finalArc

        return finalArc

    def create_array_of_wavelengths(self):

        list_of_spots = self.list_of_spots

        list_of_wavelengths = []
        for s in range(len(list_of_spots)):
            arc = self.list_of_arc[s]
            finalArc = self.get_finalArc(arc)
            single_number = list_of_spots[s]
            wavelength = float(finalArc.iloc[int(single_number)]['wavelength'])
            logging.info("wavelength used for spot "+str(s)+" [nm] is: " + str(wavelength))
            list_of_wavelengths.append(wavelength)
        array_of_wavelengths = np.array(list_of_wavelengths)

        return array_of_wavelengths

    def create_parametrization_proposals(self, date_of_input,
                                         direct_or_interpolation='direct',
                                         twentytwo_or_extra=56):
        """


        Parameters
        ----------
        date_of_input: `str`
            Date desription of the input dataframe
        direct_or_interpolation: `str`
            Mode description of the input dataframe
        twentytwo_or_extra: `int`
            Highest Zernike to go to

        Return
        ----------

        Notes
        ----------
        """

        list_of_spots = self. list_of_spots
        # dataset = self.dataset
        # DATAFRAMES_FOLDER = self.DATAFRAMES_FOLDER

        list_of_array_of_polyfit_1_parameterizations_proposal_shape_2d = []

        for s in range(len(list_of_spots)):
            arc = self.list_of_arc[s]
            finalArc = self.get_finalArc(arc, date_of_input=date_of_input,
                                         direct_or_interpolation=direct_or_interpolation)
            results_of_fit_input = self.results_of_fit_input

            single_number = list_of_spots[s]

            # you are passing multiple images, so allparameters and defocuses need to be passed into a list
            list_of_allparameters = []
            list_of_defocuses = []
            # search for the previous avaliable results
            # add the ones that you found in array_of_allparameters and for which
            # labels are avaliable in list_of_defocuses
            for label in ['m4', 'm35', 'm3', 'm05', '0', 'p05', 'p3', 'p35', 'p4']:

                # check if your single_number is avaliable

                logging.info('adding label ' + str(label) + ' with single_number '
                             + str(int(single_number)) + ' for creation of array_of_allparameters')
                try:
                    if int(single_number) < 999:
                        logging.info(results_of_fit_input[label].index.astype(int))
                        # if your single_number is avaliable go ahead
                        if int(single_number) in results_of_fit_input[label].index.astype(
                                int):
                            logging.info('Solution for this spot is avaliable')
                            if isinstance(results_of_fit_input[label].index[0], str) or str(
                                    type(results_of_fit_input[label].index[0])) == "<class 'numpy.str_'>":
                                list_of_allparameters.append(
                                    results_of_fit_input[label].loc[str(single_number)].values)
                                logging.info('results_of_fit_input[' + str(label) + '].loc['
                                             + str(int(single_number)) + '].values' + str(
                                    results_of_fit_input[label].loc[str(single_number)].values))
                            else:
                                # logging.info('results_of_fit_input[label]'+str(results_of_fit_input[label]))
                                list_of_allparameters.append(
                                    results_of_fit_input[label].loc[int(single_number)].values)
                                logging.info('results_of_fit_input[' + str(label) + '].loc['
                                             + str(int(single_number)) + '].values' + str(
                                    results_of_fit_input[label].loc[int(single_number)].values))
                            list_of_defocuses.append(label)

                        else:
                            # if the previous solution is not avaliable,
                            # find the closest avaliable, right?
                            logging.info(
                                'Solution for this spot is not avaliable, reconstructing from nearby spot')

                            # positions of all avaliable spots
                            x_positions = finalArc.loc[results_of_fit_input[label].index.astype(
                                int)]['xc_effective']
                            y_positions = finalArc.loc[results_of_fit_input[label].index.astype(
                                int)]['yc']
                            logging.info('checkpoint 1')
                            logging.info(label)
                            # logging.info(results_of_fit_input[labelInput].index)
                            # position of the input spot
                            position_x_single_number = finalArc['xc_effective'].loc[int(
                                single_number)]
                            position_y_single_number = finalArc['yc'].loc[int(
                                single_number)]
                            logging.info('checkpoint 2')
                            logging.info(position_x_single_number)
                            distance_of_avaliable_spots = np.abs(
                                (x_positions - position_x_single_number)**2
                                + (y_positions - position_y_single_number)**2)
                            single_number_input =\
                                distance_of_avaliable_spots[distance_of_avaliable_spots ==  # noqa W504
                                                            np.min(distance_of_avaliable_spots)].index[0]
                            logging.info(
                                'Nearest spot avaliable is: ' + str(single_number_input))
                            if isinstance(results_of_fit_input[label].index[0], str) or str(
                                    type(results_of_fit_input[label].index[0])) == "<class 'numpy.str_'>":
                                list_of_allparameters.append(
                                    results_of_fit_input[label].loc[str(single_number_input)].values)
                            else:
                                list_of_allparameters.append(
                                    results_of_fit_input[label].loc[int(single_number_input)].values)
                            list_of_defocuses.append(label)
                            logging.info('results_of_fit_input[' + str(label) + '].loc['
                                         + str(int(single_number_input)) + '].values' + str(
                                results_of_fit_input[label].loc[int(single_number_input)].values))

                            pass

                except BaseException:
                    logging.info('not able to add label ' + str(label))
                    pass

            array_of_allparameters = np.array(list_of_allparameters)

            # based on the information from the previous step (results at list_of_defocuses),
            # generate singular array_of_allparameters at list_of_labelInput positions
            # has shape 2xN, N = number of parameters
            logging.info('Variable twentytwo_or_extra: ' + str(twentytwo_or_extra))
            analysis_type = 'defocus'
            if analysis_type == 'defocus':
                logging.info('Variable array_of_allparameters.shape: '
                             + str(array_of_allparameters.shape))

                # model_multi is only needed to create reasonable parametrizations and
                # could possibly be avoided in future versions?
                model_multi = LN_PFS_multi_same_spot(
                    list_of_sci_images=self.list_of_sci_images,
                    list_of_var_images=self.list_of_var_images,
                    list_of_mask_images=self.list_of_mask_images,
                    wavelength=800,
                    dithering=1,
                    save=0,
                    verbosity=0,
                    npix=1536,
                    list_of_defocuses=self.list_of_labelInput,
                    zmax=twentytwo_or_extra,
                    double_sources=False,
                    double_sources_positions_ratios=[0, 0],
                    test_run=False)

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

        self.array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d = \
            array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d

        return array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d

    def create_init_parameters_for_particles(self, zmax_input=56, analysis_type='defocus',
                                             analysis_type_fiber=None):
        """
        Create initial parameters for all particles

        Parameters
        ----------
        zmax_input: `int`
            Highest Zernike order in input?

        analysis_type: `str`
            fiber_par
            Zernike_par?
            ?
            ?


        Return
        ----------


        """
        logging.info('analysis_type '+str(analysis_type))
        options = self.options

        array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d = \
            self.array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d
        list_of_spots = self.list_of_spots
        multi_var = self.multi_var
        logging.info('analysis_type_fiber: ' + str(analysis_type_fiber))
        logging.info(type(analysis_type_fiber))
        if analysis_type_fiber == "fiber_par" or analysis_type_fiber == "fixed_fiber_par":
            # x_fiber, y_fiber, effective_radius_illumination, frd_sigma, frd_lorentz_factor
            unified_index = [10, 11, 12, 13, 14]

        ################################################
        # (8.) Create init parameters for particles
        ################################################

        # TODO: ellimate either columns or columns22 variable
        # columns = columns22

        #############
        # First swarm

        # if working with defocus, but many images
        # z4, z5, z6, z7, z8, z9, , z11
        # z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22
        # detFrac, strutFrac, dxFocal, dyFocal, slitFrac, slitFrac_dy
        # wide_0, wide_23, wide_43, misalign
        # x_fiber, y_fiber, effective_radius_illumination, frd_sigma, frd_lorentz_factor,
        # det_vert, slitHolder_frac_dx, grating_lines,
        # scattering_slope, scattering_amplitude
        # pixel_effect, fiber_r
        logging.info('analysis_type_fiber is: '+str(analysis_type_fiber))
        if analysis_type_fiber == "fiber_par":
            # dont vary global illumination paramters or Zernike
            stronger_array_01 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,
                                          1.2, 1.2, 1.2, 1.2, 1.2,
                                          0, 0, 1,
                                          1, 1,
                                          1, 0.6, 1])
        elif analysis_type_fiber == "fixed_fiber_par":
            logging.info('inside analysis_type_fiber')
            stronger_array_01 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1,
                                          0, 0, 0, 0, 0,
                                          1, 1, 1,
                                          1, 1,
                                          1, 0.6, 1])
        else:
            stronger_array_01 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                                          1.2, 1.2, 1.2, 1.2,
                                          1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                                          1.2, 1.2,
                                          1.2, 1.2, 1.2, 1.2, 1])

        # fix all of the properties that describe the fiber illumination
        # v051.b change
        # if self.analysis_type == 'fixed_single_spot':
        #    stronger_array_01[19*2:][unified_index] = 0

        # we are passing the parametrizations, which need to be translated to parameters for each image
        # from Zernike_Module we imported the function `check_global_parameters'
        list_of_global_parameters = []
        for s in range(len(self.list_of_spots)):
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
        # 'x_fiber', 'y_fiber', 'effective_radius_illumination', 'frd_sigma',
        # 'frd_lorentz_factor'

        for s in range(len(list_of_spots)):
            for i in unified_index:
                array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s][:, 1][19:19 + 23][i] =\
                    checked_global_parameters[i]

        # list contaning parInit1 for each spot
        list_of_parInit1 = []
        for s in range(len(list_of_spots)):

            array_of_polyfit_1_parameterizations_proposal_shape_2d =\
                array_of_array_of_polyfit_1_parameterizations_proposal_shape_2d[s]
            logging.info(
                'array_of_polyfit_1_parameterizations_proposal_shape_2d: '
                + str(array_of_polyfit_1_parameterizations_proposal_shape_2d))
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

        # Standard deviation of parameters (for control only?).
        # One array is enough, because we can use it for both spots (right?).
        parInit1_std = []
        for i in range(parInit1.shape[1]):
            parInit1_std.append(np.std(parInit1[:, i]))
        parInit1_std = np.array(parInit1_std)
        logging.info('parInit1_std: ' + str(parInit1_std))

        # Number of particles and number of parameters
        particleCount = options[0]
        paramCount = len(parInit1[0])

        # initialize the particles
        particle_likelihood = np.array([-9999999])
        # particle_position = np.zeros(paramCount)
        # particle_velocity = np.zeros(paramCount)
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

        return list_of_array_of_particle_position_proposal,\
            list_of_array_of_particle_velocity_proposal, list_of_best_particle_likelihood, paramCount


# ***********************
# 'free' (not inside a class) definitions below
# ***********************

def create_popt_for_custom_var(sci_image, var_image, mask_image=None):
    """Create 2nd order poly fit; to be used in creation of custom var image

    TODO: same function in LN_PFS_Single... Very unsatifactory!

    The connection between variance and flux is determined from the provided science image
    and variance image.
    All of inputs have to be 2d np.arrays with same size.
    Introduced in 0.50 (PIPE2D-931)

    Called by Tokovinin_algorithm_chi_multi

    Parameters
    ----------
    sci_image : `np.array`
        Scientific array
    var_image : `np.array`
        Variance array
    mask_image : `np.array`
        Mask image

    Returns
    ----------
    custom_var_image : `np.array`
        Recreated variance map

    """
    if mask_image is None:
        sci_pixels = sci_image.ravel()
        var_pixels = var_image.ravel()
    else:
        sci_pixels = sci_image[mask_image == 0].ravel()
        var_pixels = var_image[mask_image == 0].ravel()
    # z = np.polyfit(sci_pixels, var_pixels, deg=2)
    # if z[0] < 0:
    #    z = np.polyfit(sci_pixels, var_pixels, deg=1)
    # p1 = np.poly1d(z)
    # custom_var_image = p1(sci_image)

    # I am using lambda expression to avoid superflous definition of quadratic function
    f = lambda x, *p: p[0] * x**2 + p[1] * x + p[2]  # noqa : E373
    popt, pcov = scipy.optimize.curve_fit(f, sci_pixels, var_pixels, [0, 0, np.min(var_pixels)],
                                          bounds=([-np.inf, -np.inf, np.min(var_pixels)],
                                                  [np.inf, np.inf, np.inf]))
    return popt


def create_custom_var_from_popt(model_image, popt):
    """Creates variance map from the model image, given the 2nd poly fit parameters

    Introduced in 0.50 (PIPE2D-931)

    Parameters
    ----------
    modelImg : `np.array`
        Model image
    popt : `np.array`
        2d polyfit parameters
    Returns
    ----------
    custom_var_image : `np.array`
        Recreated variance map
    """
    # I am using lambda expression to avoid superflous definition of quadratic function
    f = lambda x, *p: p[0] * x**2 + p[1] * x + p[2]  # noqa : E373
    custom_var_image = f(model_image, *popt)
    return custom_var_image


def svd_invert(matrix, threshold):
    '''
    :param matrix:
    :param threshold:
    :return:SCD-inverted matrix
    '''
    # logging.info 'MATRIX:',matrix
    u, ws, v = svd(matrix, full_matrices=True)

    # invw = inv(np.identity(len(ws))*ws)
    # return ws

    ww = np.max(ws)
    n = len(ws)
    invw = np.identity(n)
    ncount = 0

    for i in range(n):
        if ws[i] < ww * threshold:
            # log.info('SVD_INVERT: Value %i=%.2e rejected (threshold=%.2e).'%(i,ws[i],ww*threshold))
            invw[i][i] = 0.
            ncount += 1
        else:
            # logging.info 'WS[%4i] %15.9f'%(i,ws[i])
            invw[i][i] = 1. / ws[i]

    # log.info('%i singular values rejected in inversion'%ncount)
    # fixed error on September 18, 2020 - before it was missing one transpose, see below
    # inv_matrix = np.dot(u , np.dot( invw, v))
    inv_matrix = np.dot(u, np.dot(np.transpose(invw), v))

    return inv_matrix


def find_centroid_of_flux(image, mask=None):
    """
    function giving the tuple of the position of weighted average of the flux in a square image
    indentical result as calculateCentroid from drp_stella.images

    @input image    poststamp image for which to find center
    @input mask     mask, same size as the image

    returns tuple with x and y center, in units of pixels
    """
    if mask is None:
        mask = np.ones(image.shape)

    x_center = []
    y_center = []

    # if there are nan values (most likely cosmics), replace them with max value in the rest of the image
    # careful, this can seriously skew the results if not used for this purpose
    max_value_image = np.max(image[~np.isnan(image)])
    image[np.isnan(image)] = max_value_image

    I_x = []
    for i in range(len(image)):
        I_x.append([i, np.mean(image[:, i] * mask[:, i])])

    I_x = np.array(I_x)

    I_y = []
    for i in range(len(image)):
        I_y.append([i, np.mean(image[i] * mask[i])])

    I_y = np.array(I_y)

    x_center = (np.sum(I_x[:, 0] * I_x[:, 1]) / np.sum(I_x[:, 1]))
    y_center = (np.sum(I_y[:, 0] * I_y[:, 1]) / np.sum(I_y[:, 1]))

    return (x_center, y_center)


def create_parInit(allparameters_proposal, multi=None, pupil_parameters=None, allparameters_proposal_err=None,
                   stronger=None, use_optPSF=None, deduced_scattering_slope=None, zmax=None):
    """!given the suggested parametrs create array with randomized starting values to supply to fitting code

    @param allparameters_proposal            array contaning suggested starting values for a model
    @param multi                             set to True when you want to analyze more images at once
    @param pupil_parameters                  fix parameters describing the pupil
    @param allparameters_proposal_err        uncertantity on proposed parameters
    @param stronger                          factors which increases all uncertanties by a constant value
    @param use_optPFS                        fix all parameters that give pure optical PSF, except z4
    (allowing change in ['z4', 'scattering_slope', 'scattering_amplitude', 'pixel_effect', 'fiber_r', 'flux'])
    @param deduced_scattering_slope
    @param zmax

    """

    np.random.seed(101)

    if multi:
        #
        if len(allparameters_proposal.shape) == 2:
            # if you have passed 2d parametrization you have to move it to one 1d
            array_of_polyfit_1_parameterizations = np.copy(allparameters_proposal)
            if zmax == 11:
                # not implemented
                pass
            if zmax == 22:
                # logging.info('zmax is 22, right: ' +str(zmax))
                # logging.info('len(array_of_polyfit_1_parameterizations[19:]) ' +
                # str(len(array_of_polyfit_1_parameterizations[19:]) ))

                # if you have passed the parametrization that goes to the zmax=22,
                # depending if you passed value for flux
                if len(array_of_polyfit_1_parameterizations[19:]) == 23:
                    allparameters_proposal = np.concatenate(
                        (array_of_polyfit_1_parameterizations[:19].ravel(),
                         array_of_polyfit_1_parameterizations[19:-1][:, 1]))
                if len(array_of_polyfit_1_parameterizations[19:]) == 22:
                    allparameters_proposal = np.concatenate(
                        (array_of_polyfit_1_parameterizations[:19].ravel(),
                         array_of_polyfit_1_parameterizations[19:][:, 1]))

                # if you have passed too many
                if len(array_of_polyfit_1_parameterizations[19:]) > 23:
                    allparameters_proposal = np.concatenate(
                        (array_of_polyfit_1_parameterizations[:19].ravel(),
                         array_of_polyfit_1_parameterizations[19:][:, 1]))

            if zmax > 22:
                # will fail if you ask for z larger than 22 and you have not provided it
                allparameters_proposal = np.concatenate((array_of_polyfit_1_parameterizations[:19].ravel(
                ), array_of_polyfit_1_parameterizations[19:19 + 23][:, 1],
                    array_of_polyfit_1_parameterizations[42:].ravel()))

        # if you have passed 1d parametrizations just copy
        else:

            allparameters_proposal = np.copy(allparameters_proposal)

    # if you are passing explicit estimate for uncertantity of parameters,
    # make sure length is the same as of the parameters
    if allparameters_proposal_err is not None:
        assert len(allparameters_proposal) == len(allparameters_proposal_err)

    # default value for multiplying all uncertantity values (making them stronger) is 1
    if stronger is None:
        stronger = 1

    # default value for zmax, if None is passed it is set at 22
    if zmax is None:
        zmax = 22

    # if you are passing fixed scattering slope at number deduced from larger defocused image
    # does not work with multi!!!! (really? - I think it should)
    if zmax == 11:
        if deduced_scattering_slope is not None:
            allparameters_proposal[26] = np.abs(deduced_scattering_slope)
    if zmax == 22:
        if deduced_scattering_slope is not None:
            allparameters_proposal[26 + 11] = np.abs(deduced_scattering_slope)

    if zmax == 11:
        if allparameters_proposal_err is None:
            if multi is None:
                # 8 values describing z4-z11
                allparameters_proposal_err = stronger * np.array([2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                                                                  0.1, 0.02, 0.1, 0.1, 0.1, 0.1,
                                                                  0.3, 1, 0.1, 0.1,
                                                                  0.15, 0.15, 0.1,
                                                                  0.07, 0.2, 0.05, 0.4,
                                                                  30000, 0.5, 0.01,
                                                                  0.1, 0.05, 0.01])
                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26] = 0
            else:
                # 16 values describing z4-z11
                allparameters_proposal_err = stronger * np.array([2,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.1,
                                                                  0.1,
                                                                  0.1,
                                                                  0.1,
                                                                  0.05,
                                                                  0.1,
                                                                  0.2,
                                                                  0.4,
                                                                  0.1,
                                                                  0.1,
                                                                  0.1,
                                                                  0.1,
                                                                  0.02,
                                                                  0.02,
                                                                  0.5,
                                                                  0.2,
                                                                  0.1,
                                                                  30000,
                                                                  0.5,
                                                                  0.01,
                                                                  0.1,
                                                                  0.05,
                                                                  0.01])
    if zmax >= 22:

        extra_Zernike_parameters_number = zmax - 22
        # logging.info('extra_Zernike_parameters_number in parInit:' +str(extra_Zernike_parameters_number))
        if allparameters_proposal_err is None:
            if multi is None or multi is False:
                # 19 values describing z4-z22
                # smaller values for z12-z22
                # ['z4','z5','z6','z7','z8','z9','z10','z11',
                #          'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
                # 'detFrac','strutFrac','dxFocal','dyFocal','slitFrac','slitFrac_dy',
                # 'wide_0','wide_23','wide_43','misalign',
                # 'x_fiber','y_fiber','effective_ilum_radius',
                # 'frd_sigma','frd_lorentz_factor','det_vert','slitHolder_frac_dx',
                # 'grating_lines','scattering_slope','scattering_amplitude',
                # 'pixel_effect','fiber_r','flux']

                allparameters_proposal_err = stronger * np.array([2,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.08,
                                                                  0.03,
                                                                  0.1,
                                                                  0.1,
                                                                  0.016,
                                                                  0.05,
                                                                  0.3,
                                                                  0.3,
                                                                  0.3,
                                                                  10,
                                                                  0.15,
                                                                  0.15,
                                                                  0.1,
                                                                  0.1,
                                                                  0.64,
                                                                  0.05,
                                                                  0.2,
                                                                  60000,
                                                                  0.95,
                                                                  0.014,
                                                                  0.2,
                                                                  0.14,
                                                                  0.015])
                if extra_Zernike_parameters_number > 0:
                    extra_Zernike_proposal = 0.0 * np.ones((extra_Zernike_parameters_number,))
                    allparameters_proposal_err = np.concatenate(
                        (allparameters_proposal_err, extra_Zernike_proposal))

                # fixed scattering slope at number deduced from larger defocused image
                if deduced_scattering_slope is not None:
                    allparameters_proposal_err[26 + 11] = 0
            else:
                # determined from results_of_fit_input_HgAr

                allparameters_proposal_err = stronger * np.array([0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.25,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.15,
                                                                  0.035,
                                                                  0.02,
                                                                  0.1,
                                                                  0.1,
                                                                  0.008,
                                                                  0.05,
                                                                  0.3,
                                                                  0.3,
                                                                  0.3,
                                                                  10,
                                                                  0.1,
                                                                  0.1,
                                                                  0.1,
                                                                  0.08,
                                                                  0.2,
                                                                  0.05,
                                                                  0.1,
                                                                  60000,
                                                                  0.4,
                                                                  0.006,
                                                                  0.2,
                                                                  0.04,
                                                                  0.015])

                # at the moment zero because I do not want these to actually move around,
                # but perhaps needs to be reconsidered in the future
                extra_Zernike_proposal = 0.0 * np.ones((extra_Zernike_parameters_number * 2,))
                allparameters_proposal_err = np.concatenate(
                    (allparameters_proposal_err, extra_Zernike_proposal))

    if pupil_parameters is None:
        number_of_par = len(allparameters_proposal_err)
    else:
        number_of_par = len(allparameters_proposal_err) - len(pupil_parameters)

    walkers_mult = 6
    nwalkers = number_of_par * walkers_mult

    if zmax == 11:
        if multi is None or multi is False:
            zparameters_flatten = allparameters_proposal[0:8]
            zparameters_flatten_err = allparameters_proposal_err[0:8]
            globalparameters_flatten = allparameters_proposal[8:]
            globalparameters_flatten_err = allparameters_proposal_err[8:]

        else:
            zparameters_flatten = allparameters_proposal[0:8 * 2]
            zparameters_flatten_err = allparameters_proposal_err[0:8 * 2]
            globalparameters_flatten = allparameters_proposal[8 * 2:]
            globalparameters_flatten_err = allparameters_proposal_err[8 * 2:]
    # if we have 22 or more
    if zmax >= 22:
        if multi is None or multi is False:
            zparameters_flatten = allparameters_proposal[0:8 + 11]
            zparameters_flatten_err = allparameters_proposal_err[0:8 + 11]
            globalparameters_flatten = allparameters_proposal[8 + 11:8 + 11 + 23]
            globalparameters_flatten_err = allparameters_proposal_err[8 + 11:8 + 11 + 23]
            zparameters_extra_flatten = allparameters_proposal[(8 + 11) * 1 + 23:]
            zparameters_extra_flatten_err = allparameters_proposal_err[(8 + 11) * 1 + 23:]
        else:
            zparameters_flatten = allparameters_proposal[0:(8 + 11) * 2]
            zparameters_flatten_err = allparameters_proposal_err[0:(8 + 11) * 2]
            globalparameters_flatten = allparameters_proposal[(8 + 11) * 2:(8 + 11) * 2 + 23]
            globalparameters_flatten_err = allparameters_proposal_err[(8 + 11) * 2:(8 + 11) * 2 + 23]
            zparameters_extra_flatten = allparameters_proposal[(8 + 11) * 2 + 23:]
            zparameters_extra_flatten_err = allparameters_proposal_err[(8 + 11) * 2 + 23:]
            # logging.info('zparameters_flatten '+str(zparameters_flatten))

    if zmax == 11:
        if multi is None:
            try:
                for i in range(8):
                    if i == 0:
                        zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]],
                                                                      np.random.normal(
                            zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))
                    else:
                        zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]],
                                                                      np.random.normal(
                            zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))
                    if i == 0:
                        zparameters_flat = zparameters_flat_single_par
                    else:
                        zparameters_flat = np.column_stack((zparameters_flat, zparameters_flat_single_par))
            except NameError:
                logging.info('NameError!')
        else:
            try:
                for i in range(8 * 2):
                    zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]], np.random.normal(
                        zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))
                    if i == 0:
                        zparameters_flat = zparameters_flat_single_par
                    else:
                        zparameters_flat = np.column_stack((zparameters_flat, zparameters_flat_single_par))
            except NameError:
                logging.info('NameError!')

    # if we have 22 or more
    if zmax >= 22:
        if multi is None or multi is False:
            try:

                for i in range(8 + 11):
                    if i == 0:
                        zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]],
                                                                      np.random.normal(
                            zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))
                    else:
                        zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]],
                                                                      np.random.normal(
                            zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))
                    if i == 0:
                        zparameters_flat = zparameters_flat_single_par
                    else:
                        zparameters_flat = np.column_stack((zparameters_flat, zparameters_flat_single_par))

                # if you are going for extra Zernike parameters
                # copied the same code from multi
                for i in range(extra_Zernike_parameters_number):

                    zparameters_extra_flat_single_par = np.concatenate(([zparameters_extra_flatten[i]],
                                                                        np.random.normal(
                        zparameters_extra_flatten[i], zparameters_extra_flatten_err[i], nwalkers - 1)))

                    # zparameters_extra_flat_single_par=np.random.normal(0,0.05,nwalkers)
                    # logging.info(zparameters_extra_flat_single_par.shape)
                    if i == 0:
                        zparameters_extra_flat = zparameters_extra_flat_single_par
                    else:
                        zparameters_extra_flat = np.column_stack(
                            (zparameters_extra_flat, zparameters_extra_flat_single_par))

            except NameError:
                logging.info('NameError!')

        # in case that multi variable is turned on:
        else:
            try:
                for i in range((8 + 11) * 2):
                    # logging.info('i'+str(i))
                    # logging.info('zparameters_flatten[i]: '+str(zparameters_flatten[i]))
                    # logging.info('zparameters_flatten_err[i]: '+str(zparameters_flatten_err[i]))
                    # logging.info('nwalkers-1: '+str(nwalkers-1))
                    # logging.info(np.random.normal(zparameters_flatten[i],zparameters_flatten_err[i],nwalkers-1))
                    zparameters_flat_single_par = np.concatenate(([zparameters_flatten[i]], np.random.normal(
                        zparameters_flatten[i], zparameters_flatten_err[i], nwalkers - 1)))

                    if i == 0:
                        zparameters_flat = zparameters_flat_single_par
                    else:
                        zparameters_flat = np.column_stack((zparameters_flat, zparameters_flat_single_par))

                # if you are going for extra Zernike parameters
                if zmax > 22:
                    for i in range(extra_Zernike_parameters_number * 2):
                        zparameters_extra_flat_single_par = np.concatenate(([zparameters_extra_flatten[i]],
                                                                            np.random.normal(
                            zparameters_extra_flatten[i], zparameters_extra_flatten_err[i], nwalkers - 1)))

                        # zparameters_extra_flat_single_par=np.random.normal(0,0.05,nwalkers)
                        # logging.info(zparameters_extra_flat_single_par.shape)
                        if i == 0:
                            zparameters_extra_flat = zparameters_extra_flat_single_par
                        else:
                            zparameters_extra_flat = np.column_stack(
                                (zparameters_extra_flat, zparameters_extra_flat_single_par))
                        # logging.info(zparameters_extra_flat.shape)

            except NameError:
                logging.info('NameError!')

    try:
        div_same = 10

        # detFrac always positive
        globalparameters_flat_0 = np.abs(
            np.random.normal(
                globalparameters_flatten[0],
                globalparameters_flatten_err[0],
                nwalkers * 20))
        globalparameters_flat_0[np.random.choice(len(globalparameters_flat_0),
                                                 size=int(len(globalparameters_flat_0)/div_same),
                                                 replace=False)] = globalparameters_flatten[0]
        globalparameters_flat_0 = np.concatenate(([globalparameters_flatten[0]],
                                                  globalparameters_flat_0[np.all(
                                                      (globalparameters_flat_0 >= 0.6,
                                                       globalparameters_flat_0 <= 0.8),
                                                      axis=0)][0:nwalkers - 1]))
        # strutFrac always positive
        globalparameters_flat_1_long = np.abs(
            np.random.normal(
                globalparameters_flatten[1],
                globalparameters_flatten_err[1],
                nwalkers * 200))
        globalparameters_flat_1 = globalparameters_flat_1_long
        globalparameters_flat_1[np.random.choice(len(globalparameters_flat_1),
                                                 size=int(len(globalparameters_flat_1)/div_same),
                                                 replace=False)] = globalparameters_flatten[1]
        globalparameters_flat_1 = np.concatenate(([globalparameters_flatten[1]],
                                                  globalparameters_flat_1[np.all(
                                                      (globalparameters_flat_1 >= 0.07,
                                                       globalparameters_flat_1 <= 0.13),
                                                      axis=0)][0:nwalkers - 1]))
        # dxFocal
        globalparameters_flat_2 = np.random.normal(
            globalparameters_flatten[2],
            globalparameters_flatten_err[2],
            nwalkers * 20)
        globalparameters_flat_2[np.random.choice(len(globalparameters_flat_2),
                                                 size=int(len(globalparameters_flat_2)/div_same),
                                                 replace=False)] = globalparameters_flatten[2]
        globalparameters_flat_2 = np.concatenate(([globalparameters_flatten[2]],
                                                  globalparameters_flat_2[np.all(
                                                      (globalparameters_flat_2 >= -0.4,
                                                       globalparameters_flat_2 <= 0.4),
                                                      axis=0)][0:nwalkers - 1]))
        # dyFocal
        globalparameters_flat_3 = np.random.normal(
            globalparameters_flatten[3],
            globalparameters_flatten_err[3],
            nwalkers * 20)
        globalparameters_flat_3[np.random.choice(len(globalparameters_flat_3),
                                                 size=int(len(globalparameters_flat_3)/div_same),
                                                 replace=False)] = globalparameters_flatten[3]
        globalparameters_flat_3 = np.concatenate(([globalparameters_flatten[3]],
                                                  globalparameters_flat_3[np.all(
                                                      (globalparameters_flat_3 >= -0.4,
                                                       globalparameters_flat_3 <= 0.4),
                                                      axis=0)][0:nwalkers - 1]))
        # slitFrac
        globalparameters_flat_4 = np.abs(
            np.random.normal(
                globalparameters_flatten[4],
                globalparameters_flatten_err[4],
                nwalkers * 20))
        # logging.info(globalparameters_flatten_err[4])
        globalparameters_flat_4[np.random.choice(len(globalparameters_flat_4),
                                                 size=int(len(globalparameters_flat_4)/div_same),
                                                 replace=False)] = globalparameters_flatten[4]
        globalparameters_flat_4 = np.concatenate(([globalparameters_flatten[4]],
                                                  globalparameters_flat_4[np.all(
                                                      (globalparameters_flat_4 >= 0.05,
                                                       globalparameters_flat_4 <= 0.09),
                                                      axis=0)][0:nwalkers - 1]))
        # slitFrac_dy
        globalparameters_flat_5 = np.abs(
            np.random.normal(
                globalparameters_flatten[5],
                globalparameters_flatten_err[5],
                nwalkers * 20))
        globalparameters_flat_5[np.random.choice(len(globalparameters_flat_5),
                                                 size=int(len(globalparameters_flat_5)/div_same),
                                                 replace=False)] = globalparameters_flatten[5]
        globalparameters_flat_5 = np.concatenate(([globalparameters_flatten[5]],
                                                  globalparameters_flat_5[np.all(
                                                      (globalparameters_flat_5 >= -0.5,
                                                       globalparameters_flat_5 <= 0.5),
                                                      axis=0)][0:nwalkers - 1]))
        # wide_0
        globalparameters_flat_6 = np.abs(
            np.random.normal(
                globalparameters_flatten[6],
                globalparameters_flatten_err[6],
                nwalkers * 20))
        globalparameters_flat_6[np.random.choice(len(globalparameters_flat_6),
                                                 size=int(len(globalparameters_flat_6)/div_same),
                                                 replace=False)] = globalparameters_flatten[6]
        globalparameters_flat_6 = np.concatenate(([globalparameters_flatten[6]],
                                                  globalparameters_flat_6[np.all(
                                                      (globalparameters_flat_6 >= 0,
                                                       globalparameters_flat_6 <= 1),
                                                      axis=0)][0:nwalkers - 1]))
        # wide_23
        globalparameters_flat_7 = np.random.normal(
            globalparameters_flatten[7],
            globalparameters_flatten_err[7],
            nwalkers * 20)
        globalparameters_flat_7[np.random.choice(len(globalparameters_flat_7),
                                                 size=int(len(globalparameters_flat_7)/div_same),
                                                 replace=False)] = globalparameters_flatten[7]
        globalparameters_flat_7 = np.concatenate(([globalparameters_flatten[7]],
                                                  globalparameters_flat_7[np.all(
                                                      (globalparameters_flat_7 >= 0.0,
                                                       globalparameters_flat_7 <= 1),
                                                      axis=0)][0:nwalkers - 1]))
        # wide_43
        globalparameters_flat_8 = np.abs(
            np.random.normal(
                globalparameters_flatten[8],
                globalparameters_flatten_err[8],
                nwalkers * 20))
        globalparameters_flat_8[np.random.choice(len(globalparameters_flat_8),
                                                 size=int(len(globalparameters_flat_8)/div_same),
                                                 replace=False)] = globalparameters_flatten[8]
        globalparameters_flat_8 = np.concatenate(([globalparameters_flatten[8]],
                                                  globalparameters_flat_8[np.all(
                                                      (globalparameters_flat_8 >= 0,
                                                       globalparameters_flat_8 <= 1),
                                                      axis=0)][0:nwalkers - 1]))
        # misalign
        globalparameters_flat_9 = np.abs(
            np.random.normal(
                globalparameters_flatten[9],
                globalparameters_flatten_err[9],
                nwalkers * 20))
        globalparameters_flat_9[np.random.choice(len(globalparameters_flat_9),
                                                 size=int(len(globalparameters_flat_9)/div_same),
                                                 replace=False)] = globalparameters_flatten[9]
        globalparameters_flat_9 = np.concatenate(([globalparameters_flatten[9]],
                                                  globalparameters_flat_9[np.all(
                                                      (globalparameters_flat_9 >= 0,
                                                       globalparameters_flat_9 <= 12),
                                                      axis=0)][0:nwalkers - 1]))
        # x_fiber
        globalparameters_flat_10 = np.random.normal(
            globalparameters_flatten[10],
            globalparameters_flatten_err[10],
            nwalkers * 20)
        globalparameters_flat_10[np.random.choice(len(globalparameters_flat_10),
                                                  size=int(len(globalparameters_flat_10)/div_same),
                                                  replace=False)] = globalparameters_flatten[10]
        globalparameters_flat_10 = np.concatenate(([globalparameters_flatten[10]],
                                                   globalparameters_flat_10[np.all(
                                                       (globalparameters_flat_10 >= -0.4,
                                                        globalparameters_flat_10 <= 0.4),
                                                       axis=0)][0:nwalkers - 1]))
        # y_fiber
        globalparameters_flat_11 = np.random.normal(
            globalparameters_flatten[11],
            globalparameters_flatten_err[11],
            nwalkers * 20)
        globalparameters_flat_11[np.random.choice(len(globalparameters_flat_11),
                                                  size=int(len(globalparameters_flat_11)/div_same),
                                                  replace=False)] = globalparameters_flatten[11]
        globalparameters_flat_11 = np.concatenate(([globalparameters_flatten[11]],
                                                   globalparameters_flat_11[np.all(
                                                       (globalparameters_flat_11 >= -0.4,
                                                        globalparameters_flat_11 <= 0.4),
                                                       axis=0)][0:nwalkers - 1]))

        # effective_radius_illumination
        globalparameters_flat_12 = np.random.normal(
            globalparameters_flatten[12],
            globalparameters_flatten_err[12],
            nwalkers * 20)
        globalparameters_flat_12[np.random.choice(len(globalparameters_flat_12),
                                                  size=int(len(globalparameters_flat_12)/div_same),
                                                  replace=False)] = globalparameters_flatten[12]
        globalparameters_flat_12 = np.concatenate(([globalparameters_flatten[12]],
                                                   globalparameters_flat_12[np.all(
                                                       (globalparameters_flat_12 >= 0.7,
                                                        globalparameters_flat_12 <= 1.0),
                                                       axis=0)][0:nwalkers - 1]))

        if globalparameters_flatten[13] < 0.01:
            globalparameters_flatten[13] = 0.01
        # frd_sigma
        globalparameters_flat_13 = np.random.normal(
            globalparameters_flatten[13],
            globalparameters_flatten_err[13],
            nwalkers * 20)
        globalparameters_flat_13[np.random.choice(len(globalparameters_flat_13),
                                                  size=int(len(globalparameters_flat_13)/div_same),
                                                  replace=False)] = globalparameters_flatten[13]
        globalparameters_flat_13 = np.concatenate(([globalparameters_flatten[13]],
                                                   globalparameters_flat_13[np.all(
                                                       (globalparameters_flat_13 >= 0.01,
                                                        globalparameters_flat_13 <= 0.4),
                                                       axis=0)][0:nwalkers - 1]))

        # frd_lorentz_factor
        globalparameters_flat_14 = np.random.normal(
            globalparameters_flatten[14],
            globalparameters_flatten_err[14],
            nwalkers * 20)
        globalparameters_flat_14[np.random.choice(len(globalparameters_flat_14),
                                                  size=int(len(globalparameters_flat_14)/div_same),
                                                  replace=False)] = globalparameters_flatten[14]
        globalparameters_flat_14 = np.concatenate(([globalparameters_flatten[14]],
                                                   globalparameters_flat_14[np.all(
                                                       (globalparameters_flat_14 >= 0.01,
                                                        globalparameters_flat_14 <= 1),
                                                       axis=0)][0:nwalkers - 1]))

        # det_vert
        globalparameters_flat_15 = np.random.normal(
            globalparameters_flatten[15],
            globalparameters_flatten_err[15],
            nwalkers * 20)
        globalparameters_flat_15[np.random.choice(len(globalparameters_flat_15),
                                                  size=int(len(globalparameters_flat_15)/div_same),
                                                  replace=False)] = globalparameters_flatten[15]
        globalparameters_flat_15 = np.concatenate(([globalparameters_flatten[15]],
                                                   globalparameters_flat_15[np.all(
                                                       (globalparameters_flat_15 >= 0.85,
                                                        globalparameters_flat_15 <= 1.15),
                                                       axis=0)][0:nwalkers - 1]))

        # slitHolder_frac_dx
        globalparameters_flat_16 = np.random.normal(
            globalparameters_flatten[16],
            globalparameters_flatten_err[16],
            nwalkers * 20)
        globalparameters_flat_16[np.random.choice(len(globalparameters_flat_16),
                                                  size=int(len(globalparameters_flat_16)/div_same),
                                                  replace=False)] = globalparameters_flatten[16]
        globalparameters_flat_16 = np.concatenate(([globalparameters_flatten[16]],
                                                   globalparameters_flat_16[np.all(
                                                       (globalparameters_flat_16 >= -0.8,
                                                        globalparameters_flat_16 <= 0.8),
                                                       axis=0)][0:nwalkers - 1]))

        # grating lines
        globalparameters_flat_17 = np.random.normal(
            globalparameters_flatten[17],
            globalparameters_flatten_err[17],
            nwalkers * 20)
        globalparameters_flat_17[np.random.choice(len(globalparameters_flat_17),
                                                  size=int(len(globalparameters_flat_17)/div_same),
                                                  replace=False)] = globalparameters_flatten[17]
        globalparameters_flat_17 = np.concatenate(([globalparameters_flatten[17]],
                                                   globalparameters_flat_17[np.all(
                                                       (globalparameters_flat_17 >= 1200,
                                                        globalparameters_flat_17 <= 120000),
                                                       axis=0)][0:nwalkers - 1]))

        # scattering_slope
        globalparameters_flat_18 = np.random.normal(
            globalparameters_flatten[18],
            globalparameters_flatten_err[18],
            nwalkers * 20)
        globalparameters_flat_18[np.random.choice(len(globalparameters_flat_18),
                                                  size=int(len(globalparameters_flat_18)/div_same),
                                                  replace=False)] = globalparameters_flatten[18]
        globalparameters_flat_18 = np.concatenate(([globalparameters_flatten[18]],
                                                   globalparameters_flat_18[np.all(
                                                       (globalparameters_flat_18 >= 1.5,
                                                        globalparameters_flat_18 <= 3.0),
                                                       axis=0)][0:nwalkers - 1]))
        # scattering_amplitude
        globalparameters_flat_19 = np.random.normal(
            globalparameters_flatten[19],
            globalparameters_flatten_err[19],
            nwalkers * 20)
        globalparameters_flat_19[np.random.choice(len(globalparameters_flat_19),
                                                  size=int(len(globalparameters_flat_19)/div_same),
                                                  replace=False)] = globalparameters_flatten[19]
        globalparameters_flat_19 = np.concatenate(([globalparameters_flatten[19]],
                                                   globalparameters_flat_19[np.all(
                                                       (globalparameters_flat_19 >= 0.0,
                                                        globalparameters_flat_19 <= 0.4),
                                                       axis=0)][0:nwalkers - 1]))
        # pixel_effect
        globalparameters_flat_20 = np.random.normal(
            globalparameters_flatten[20],
            globalparameters_flatten_err[20],
            nwalkers * 20)
        globalparameters_flat_20[np.random.choice(len(globalparameters_flat_20),
                                                  size=int(len(globalparameters_flat_20)/div_same),
                                                  replace=False)] = globalparameters_flatten[20]
        globalparameters_flat_20 = np.concatenate(([globalparameters_flatten[20]],
                                                   globalparameters_flat_20[np.all(
                                                       (globalparameters_flat_20 >= 0.15,
                                                        globalparameters_flat_20 <= 0.8),
                                                       axis=0)][0:nwalkers - 1]))

        # fiber_r
        if globalparameters_flatten[21] < 1.74:
            globalparameters_flatten[21] = 1.8

        globalparameters_flat_21 = np.random.normal(
            globalparameters_flatten[21],
            globalparameters_flatten_err[21],
            nwalkers * 20)
        globalparameters_flat_21[np.random.choice(len(globalparameters_flat_21),
                                                  size=int(len(globalparameters_flat_21)/div_same),
                                                  replace=False)] = globalparameters_flatten[21]
        globalparameters_flat_21 = np.concatenate(([globalparameters_flatten[21]],
                                                   globalparameters_flat_21[np.all(
                                                       (globalparameters_flat_21 >= 1.74,
                                                        globalparameters_flat_21 <= 1.98),
                                                       axis=0)][0:nwalkers - 1]))

        if len(globalparameters_flatten) == 23:
            # flux
            globalparameters_flat_22 = np.random.normal(
                globalparameters_flatten[22], globalparameters_flatten_err[22], nwalkers * 20)
            globalparameters_flat_22 = np.concatenate(([globalparameters_flatten[22]],
                                                       globalparameters_flat_22[np.all(
                                                           (globalparameters_flat_22 >= 0.98,
                                                            globalparameters_flat_22 <= 1.02),
                                                           axis=0)][0:nwalkers - 1]))
        else:
            pass

        # uncomment in order to troubleshoot and show many parameters generated for each parameter
        """
        for i in [globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,
                  globalparameters_flat_3,
                                                   globalparameters_flat_4,globalparameters_flat_5,
                                                   globalparameters_flat_6,globalparameters_flat_7,
                                                  globalparameters_flat_8,globalparameters_flat_9,
                                                  globalparameters_flat_10,
                                                   globalparameters_flat_11,globalparameters_flat_12,
                                                   globalparameters_flat_13,
                                                   globalparameters_flat_14,globalparameters_flat_15,
                                                   globalparameters_flat_16,
                                                   globalparameters_flat_17,globalparameters_flat_18,
                                                   globalparameters_flat_19,
                                                   globalparameters_flat_20,globalparameters_flat_21,
                                                   globalparameters_flat_22]:
            logging.info(str(i[0])+': '+str(len(i)))
        """
        if pupil_parameters is None:
            if len(globalparameters_flatten) == 23:
                # logging.info('considering globalparameters_flatten 23 ')
                # logging.info(globalparameters_flat_0.shape)
                # logging.info(globalparameters_flat_3.shape)
                # logging.info(globalparameters_flat_6.shape)
                # logging.info(globalparameters_flat_9.shape)
                # logging.info(globalparameters_flat_12.shape)
                # logging.info(globalparameters_flat_15.shape)
                # logging.info(globalparameters_flat_18.shape)
                # logging.info(globalparameters_flat_21.shape)
                # logging.info(globalparameters_flat_22.shape)
                globalparameters_flat = np.column_stack(
                    (globalparameters_flat_0,
                     globalparameters_flat_1,
                     globalparameters_flat_2,
                     globalparameters_flat_3,
                     globalparameters_flat_4,
                     globalparameters_flat_5,
                     globalparameters_flat_6,
                     globalparameters_flat_7,
                     globalparameters_flat_8,
                     globalparameters_flat_9,
                     globalparameters_flat_10,
                     globalparameters_flat_11,
                     globalparameters_flat_12,
                     globalparameters_flat_13,
                     globalparameters_flat_14,
                     globalparameters_flat_15,
                     globalparameters_flat_16,
                     globalparameters_flat_17,
                     globalparameters_flat_18,
                     globalparameters_flat_19,
                     globalparameters_flat_20,
                     globalparameters_flat_21,
                     globalparameters_flat_22))
            else:
                logging.info('not considering globalparameters_flatten 23 !!! ')
                globalparameters_flat = np.column_stack(
                    (globalparameters_flat_0,
                     globalparameters_flat_1,
                     globalparameters_flat_2,
                     globalparameters_flat_3,
                     globalparameters_flat_4,
                     globalparameters_flat_5,
                     globalparameters_flat_6,
                     globalparameters_flat_7,
                     globalparameters_flat_8,
                     globalparameters_flat_9,
                     globalparameters_flat_10,
                     globalparameters_flat_11,
                     globalparameters_flat_12,
                     globalparameters_flat_13,
                     globalparameters_flat_14,
                     globalparameters_flat_15,
                     globalparameters_flat_16,
                     globalparameters_flat_17,
                     globalparameters_flat_18,
                     globalparameters_flat_19,
                     globalparameters_flat_20,
                     globalparameters_flat_21))

        else:
            globalparameters_flat = np.column_stack(
                (globalparameters_flat_6,
                 globalparameters_flat_7,
                 globalparameters_flat_8,
                 globalparameters_flat_9,
                 globalparameters_flat_16,
                 globalparameters_flat_17,
                 globalparameters_flat_18,
                 globalparameters_flat_19,
                 globalparameters_flat_20,
                 globalparameters_flat_21,
                 globalparameters_flat_22))

    except NameError:
        logging.info("NameError")

    # logging.info('globalparameters_flat.shape'+str(zparameters_flat.shape) )
    # logging.info('globalparameters_flat.shape'+str(globalparameters_flat.shape) )

    if zmax <= 22:
        allparameters = np.column_stack((zparameters_flat, globalparameters_flat))
    if zmax > 22:
        # logging.info('globalparameters_flat.shape'+str(zparameters_extra_flat.shape) )
        allparameters = np.column_stack((zparameters_flat, globalparameters_flat, zparameters_extra_flat))

    parInit = allparameters.reshape(nwalkers, number_of_par)

    # hm..... relic of some older code, needs cleaning
    if use_optPSF is not None:
        if zmax == 11:
            for i in range(1, 25):
                # for i in np.concatenate((range(1,7),range(8,25))):
                # for i in range(8,25):
                parInit[:, i] = np.full(len(parInit[:, i]), allparameters_proposal[i])
        else:
            for i in range(1, 25 + 11):
                # for i in np.concatenate((range(1,7),range(8,25))):
                # for i in range(8,25):
                parInit[:, i] = np.full(len(parInit[:, i]), allparameters_proposal[i])
    else:
        pass

    return parInit


def Ifun16Ne(lambdaV, lambda0, Ne):
    """Construct Lorentizan scattering kernel
        Parameters
        ----------
        lambdaV: `float`
           wavelength at which compute the grating effect
        lambda0: `float`
           reference wavelength
        Ne: `int`
                number of effective grating lines of the spectrograph
        Returns
        ----------
        value_of_scatter: `float`
            strenth of the kernel at lambdaV wavelength
    """
    return (lambda0 / (Ne * np.pi * np.sqrt(2)))**2 / \
        ((lambdaV - lambda0)**2 + (lambda0 / (Ne * np.pi * np.sqrt(2)))**2)


def custom_fftconvolve(array1, array2):
    assert array1.shape == array2.shape

    fft_result = np.fft.fftshift(
        np.real(
            np.fft.irfft2(
                np.fft.rfft2(array1) * np.fft.rfft2(array2),
                s=np.array(
                    array1.shape))))
    # ensure that the resulting shape is an odd nubmer, needed for fft convolutions later
    if array1.shape[0] % 2 == 0:
        # if the size of an array is even number
        fft_result = fft_result[:fft_result.shape[0] - 1, :fft_result.shape[1] - 1]
    else:
        # if the size of an array is an odd number
        fft_result = fft_result[:fft_result.shape[0] - 2, :fft_result.shape[1] - 2]
        fft_result = np.pad(fft_result, 1, 'constant', constant_values=0)

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
    return 2 * np.pi * pupil_plane_scale / (lam * 1e-9) * scale_unit / galsim.radians


def maxK(pupil_plane_size, lam, scale_unit=galsim.arcsec):
    """Return the Fourier grid half-size for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Inverse units in which to return result [default: galsim.arcsec]
    @returns           Fourier grid half-size.
    """
    return np.pi * pupil_plane_size / (lam * 1e-9) * scale_unit / galsim.radians


def sky_scale(pupil_plane_size, lam, scale_unit=galsim.arcsec):
    """Return the image scale for this aperture at given wavelength

    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image scale.
    """
    return (lam * 1e-9) / pupil_plane_size * galsim.radians / scale_unit


def sky_size(pupil_plane_scale, lam, scale_unit=galsim.arcsec):
    """Return the image size for this aperture at given wavelength.
    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image size.
    """
    return (lam * 1e-9) / pupil_plane_scale * galsim.radians / scale_unit


def remove_pupil_parameters_from_all_parameters(parameters):
    lenpar = len(parameters)
    return np.concatenate((parameters[:lenpar - 23],
                           parameters[lenpar - 17:lenpar - 13],
                           parameters[lenpar - 7:]))


def add_pupil_parameters_to_all_parameters(parameters, pupil_parameters):
    lenpar = len(parameters)
    return np.concatenate((parameters[:lenpar - 11],
                           pupil_parameters[:6],
                           parameters[lenpar - 11:lenpar - 7],
                           pupil_parameters[6:],
                           parameters[lenpar - 7:]),
                          axis=0)


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

# @lru_cache()


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
        old_breaks = np.linspace(0, old_size, num=old_size + 1, dtype=np.float32)
        new_breaks = np.linspace(0, old_size, num=new_size + 1, dtype=np.float32)
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


def check_global_parameters(globalparameters, test_print=None, fit_for_flux=None):
    # When running big fits these are limits which ensure that the code does
    # not wander off in totally non physical region

    globalparameters_output = np.copy(globalparameters)

    # det frac
    if globalparameters[0] <= 0.6 or globalparameters[0] >= 0.8:
        logging.info('globalparameters[0] outside limits; value: '
                     + str(globalparameters[0])) if test_print == 1 else False
    if globalparameters[0] <= 0.6:
        globalparameters_output[0] = 0.6
    if globalparameters[0] > 0.8:
        globalparameters_output[0] = 0.8

    # strut frac
    if globalparameters[1] < 0.07 or globalparameters[1] > 0.13:
        logging.info('globalparameters[1] outside limits') if test_print == 1 else False
    if globalparameters[1] <= 0.07:
        globalparameters_output[1] = 0.07
    if globalparameters[1] > 0.13:
        globalparameters_output[1] = 0.13

    # slit_frac < strut frac
    # if globalparameters[4]<globalparameters[1]:
        # logging.info('globalparameters[1] not smaller than 4 outside limits')
        # return -np.inf

    # dx Focal
    if globalparameters[2] < -0.4 or globalparameters[2] > 0.4:
        logging.info('globalparameters[2] outside limits') if test_print == 1 else False
    if globalparameters[2] < -0.4:
        globalparameters_output[2] = -0.4
    if globalparameters[2] > 0.4:
        globalparameters_output[2] = 0.4

    # dy Focal
    if globalparameters[3] > 0.4:
        logging.info('globalparameters[3] outside limits') if test_print == 1 else False
        globalparameters_output[3] = 0.4
    if globalparameters[3] < -0.4:
        logging.info('globalparameters[3] outside limits') if test_print == 1 else False
        globalparameters_output[3] = -0.4

    # slitFrac
    if globalparameters[4] < 0.05:
        logging.info('globalparameters[4] outside limits') if test_print == 1 else False
        globalparameters_output[4] = 0.05
    if globalparameters[4] > 0.09:
        logging.info('globalparameters[4] outside limits') if test_print == 1 else False
        globalparameters_output[4] = 0.09

    # slitFrac_dy
    if globalparameters[5] < -0.5:
        logging.info('globalparameters[5] outside limits') if test_print == 1 else False
        globalparameters_output[5] = -0.5
    if globalparameters[5] > 0.5:
        logging.info('globalparameters[5] outside limits') if test_print == 1 else False
        globalparameters_output[5] = +0.5

    # radiometricEffect / wide_0
    if globalparameters[6] < 0:
        logging.info('globalparameters[6] outside limits') if test_print == 1 else False
        globalparameters_output[6] = 0
    if globalparameters[6] > 1:
        logging.info('globalparameters[6] outside limits') if test_print == 1 else False
        globalparameters_output[6] = 1

    # radiometricExponent / wide_23
    if globalparameters[7] < 0:
        logging.info('globalparameters[7] outside limits') if test_print == 1 else False
        globalparameters_output[7] = 0
    # changed in v0.42
    if globalparameters[7] > 1:
        logging.info('globalparameters[7] outside limits') if test_print == 1 else False
        globalparameters_output[7] = 1

    # x_ilum /wide_43
    if globalparameters[8] < 0:
        logging.info('globalparameters[8] outside limits') if test_print == 1 else False
        globalparameters_output[8] = 0
    # changed in v0.42
    if globalparameters[8] > 1:
        logging.info('globalparameters[8] outside limits') if test_print == 1 else False
        globalparameters_output[8] = 1

    # y_ilum / misalign
    if globalparameters[9] < 0:
        logging.info('globalparameters[9] outside limits') if test_print == 1 else False
        globalparameters_output[9] = 0
    if globalparameters[9] > 12:
        logging.info('globalparameters[9] outside limits') if test_print == 1 else False
        globalparameters_output[9] = 12

    # x_fiber
    if globalparameters[10] < -0.4:
        logging.info('globalparameters[10] outside limits') if test_print == 1 else False
        globalparameters_output[10] = -0.4
    if globalparameters[10] > 0.4:
        logging.info('globalparameters[10] outside limits') if test_print == 1 else False
        globalparameters_output[10] = 0.4

    # y_fiber
    if globalparameters[11] < -0.4:
        logging.info('globalparameters[11] outside limits') if test_print == 1 else False
        globalparameters_output[11] = -0.4
    if globalparameters[11] > 0.4:
        logging.info('globalparameters[11] outside limits') if test_print == 1 else False
        globalparameters_output[11] = 0.4

    # effective_radius_illumination
    if globalparameters[12] < 0.7:
        logging.info('globalparameters[12] outside limits') if test_print == 1 else False
        globalparameters_output[12] = 0.7
    if globalparameters[12] > 1.0:
        logging.info('globalparameters[12] outside limits') if test_print == 1 else False
        globalparameters_output[12] = 1

    # frd_sigma
    if globalparameters[13] < 0.01:
        logging.info('globalparameters[13] outside limits') if test_print == 1 else False
        globalparameters_output[13] = 0.01
    if globalparameters[13] > .4:
        logging.info('globalparameters[13] outside limits') if test_print == 1 else False
        globalparameters_output[13] = 0.4

    # frd_lorentz_factor
    if globalparameters[14] < 0.01:
        logging.info('globalparameters[14] outside limits') if test_print == 1 else False
        globalparameters_output[14] = 0.01
    if globalparameters[14] > 1:
        logging.info('globalparameters[14] outside limits') if test_print == 1 else False
        globalparameters_output[14] = 1

    # det_vert
    if globalparameters[15] < 0.85:
        logging.info('globalparameters[15] outside limits') if test_print == 1 else False
        globalparameters_output[15] = 0.85
    if globalparameters[15] > 1.15:
        logging.info('globalparameters[15] outside limits') if test_print == 1 else False
        globalparameters_output[15] = 1.15

    # slitHolder_frac_dx
    if globalparameters[16] < -0.8:
        logging.info('globalparameters[16] outside limits') if test_print == 1 else False
        globalparameters_output[16] = -0.8
    if globalparameters[16] > 0.8:
        logging.info('globalparameters[16] outside limits') if test_print == 1 else False
        globalparameters_output[16] = 0.8

    # grating_lines
    if globalparameters[17] < 1200:
        logging.info('globalparameters[17] outside limits') if test_print == 1 else False
        globalparameters_output[17] = 1200
    if globalparameters[17] > 120000:
        logging.info('globalparameters[17] outside limits') if test_print == 1 else False
        globalparameters_output[17] = 120000

    # scattering_slope
    if globalparameters[18] < 1.5:
        logging.info('globalparameters[18] outside limits') if test_print == 1 else False
        globalparameters_output[18] = 1.5
    if globalparameters[18] > +3.0:
        logging.info('globalparameters[18] outside limits') if test_print == 1 else False
        globalparameters_output[18] = 3

    # scattering_amplitude
    if globalparameters[19] < 0:
        logging.info('globalparameters[19] outside limits') if test_print == 1 else False
        globalparameters_output[19] = 0
    if globalparameters[19] > +0.4:
        logging.info('globalparameters[19] outside limits') if test_print == 1 else False
        globalparameters_output[19] = 0.4

    # pixel_effect
    if globalparameters[20] < 0.15:
        logging.info('globalparameters[20] outside limits') if test_print == 1 else False
        globalparameters_output[20] = 0.15
    if globalparameters[20] > +0.8:
        logging.info('globalparameters[20] outside limits') if test_print == 1 else False
        globalparameters_output[20] = 0.8

    # fiber_r
    if globalparameters[21] < 1.74:
        logging.info('globalparameters[21] outside limits') if test_print == 1 else False
        globalparameters_output[21] = 1.74
    if globalparameters[21] > +1.98:
        logging.info('globalparameters[21] outside limits') if test_print == 1 else False
        globalparameters_output[21] = 1.98

    # flux
    if fit_for_flux:
        globalparameters_output[22] = 1
    else:
        if globalparameters[22] < 0.98:
            logging.info('globalparameters[22] outside limits') if test_print == 1 else False
            globalparameters_output[22] = 0.98
        if globalparameters[22] > 1.02:
            logging.info('globalparameters[22] outside limits') if test_print == 1 else False
            globalparameters_output[22] = 1.02

    return globalparameters_output


def move_parametrizations_from_2d_shape_to_1d_shape(allparameters_best_parametrization_shape_2d):
    """
    change the linear parametrization array in 2d shape to parametrization array in 1d

    @param allparameters_best_parametrization_shape_2d        linear parametrization, 2d array

    """

    if allparameters_best_parametrization_shape_2d.shape[0] > 42:
        #  if you are using above Zernike above 22
        # logging.info('we are creating new result with Zernike above 22')
        allparameters_best_parametrization_shape_1d = np.concatenate((
            allparameters_best_parametrization_shape_2d[:19].ravel(),
            allparameters_best_parametrization_shape_2d[19:19 + 23][:, 1],
            allparameters_best_parametrization_shape_2d[19 + 23:].ravel()))

    else:
        # logging.info('we are creating new result with Zernike at 22')
        allparameters_best_parametrization_shape_1d = np.concatenate((
            allparameters_best_parametrization_shape_2d[:19].ravel(),
            allparameters_best_parametrization_shape_2d[19:-1][:, 1]))

    return allparameters_best_parametrization_shape_1d  # noqa: W292
