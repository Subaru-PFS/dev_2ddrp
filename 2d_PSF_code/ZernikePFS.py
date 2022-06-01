#!/usr/bin/env python3

# numpy and scipy
import numpy as np
import scipy.fftpack
import scipy.misc
from scipy.special import erf
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

# galsim and lmfit
import galsim
import lmfit

# astropy
# TODO: replace with afw equivalents
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import Tophat2DKernel

# lsst
import lsst.afw.math
import lsst.afw.image

# auxiliary imports
import socket
import time


class ZernikeFitterPFS(object):
    """Create a model image for Prime Focus Spectrograph
    """

    def __init__(self, image=np.ones((20, 20)), image_var=np.ones((20, 20)), image_mask=None,
                 pixelScale=20.76, wavelength=794,
                 diam_sic=139.5327e-3, npix=1536,
                 pupilExplicit=None, wf_full_Image=None,
                 dithering=None, save=None,
                 use_optPSF=None, use_wf_grid=None,
                 zmaxInit=None, extraZernike=None, simulation_00=None, verbosity=None,
                 double_sources=None, double_sources_positions_ratios=None,
                 explicit_psf_position=None, use_only_chi=False, use_center_of_flux=False,
                 PSF_DIRECTORY=None, *args):
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

        Calls class Psf_position
        Calls class PFSPupilFactory

        Examples
        ----------
        Simple exampe with initial parameters, changing only one parameter

        >>> zmax = 22
        >>> single_image_analysis = ZernikeFitterPFS(zmaxInit = zmax,
                                                     verbosity=1)
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
        self.pixelScale = pixelScale
        self.diam_sic = diam_sic
        self.npix = npix

        if pupilExplicit is None:
            pupilExplicit is False
        self.pupilExplicit = pupilExplicit

        # effective size of pixels, which can be differ from physical size of pixels due to dithering
        if dithering is None:
            dithering = 1
        self.dithering = dithering
        self.pixelScale_effective = self.pixelScale / dithering

        if save in (None, 0):
            save = None
        else:
            save = 1
            assert PSF_DIRECTORY is not None
        self.save = save
        self.use_optPSF = use_optPSF
        self.use_wf_grid = use_wf_grid
        self.zmax = zmaxInit

        self.simulation_00 = simulation_00
        if self.simulation_00:
            self.simulation_00 = 1

        self.extraZernike = extraZernike
        self.verbosity = verbosity
        self.double_sources = double_sources
        self.double_sources_positions_ratios = double_sources_positions_ratios
        self.explicit_psf_position = explicit_psf_position
        self.use_only_chi = use_only_chi
        self.use_center_of_flux = use_center_of_flux

        # flux = number of counts in the image
        self.flux = float(np.sum(image))

        try:
            if not explicit_psf_position:
                self.explicit_psf_position = None
        except BaseException:
            pass

        self.PSF_DIRECTORY = PSF_DIRECTORY
        if PSF_DIRECTORY is not None:
            self.TESTING_FOLDER = PSF_DIRECTORY + 'Testing/'
            self.TESTING_PUPIL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Pupil_Images/'
            self.TESTING_WAVEFRONT_IMAGES_FOLDER = self.TESTING_FOLDER + 'Wavefront_Images/'
            self.TESTING_FINAL_IMAGES_FOLDER = self.TESTING_FOLDER + 'Final_Images/'

        if self.verbosity == 1:
            # check the versions of the most important libraries
            print('np.__version__' + str(np.__version__))
            print('scipy.__version__' + str(scipy.__version__))

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
        """Initialize `lmfit.Parameters` object

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
            print(' ')
            print('Initializing ZernikeFitterPFS')
            print('Verbosity parameter is: ' + str(self.verbosity))
            print('Highest Zernike polynomial is (zmax): ' + str(self.zmax))

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
            params.add('x_fiber', 1)
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
            # if you claimed to supply optical psf image, but none is provided
            # still create one
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
            np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'optPsf_final',
                    optPsf_final)
        else:
            pass

        if not return_intermediate_images:
            return optPsf_final, psf_position
        if return_intermediate_images:
            return optPsf_final, ilum, wf_grid_rot, psf_position

        if self.verbosity == 1:
            print('Finished with constructModelImage_PFS_naturalResolution')
            print(' ')

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
        3. CCD difusion
        4. grating effects
        5. centering
        """
        time_start_single = time.time()
        if self.verbosity == 1:
            print(' ')
            print('Entering optPsf_postprocessing')

        params = self.params
        shape = self.image.shape

        # all of the parameters for the creation of the image
        param_values = params.valuesdict()

        # how much is my generated image oversampled compared to final image
        oversampling_original = (self.pixelScale_effective) / self.scale_ModelImage_PFS_naturalResolution

        if self.verbosity == 1:
            print('optPsf.shape: ' + str(optPsf.shape))
            print('oversampling_original: ' + str(oversampling_original))
            # print('type(optPsf) '+str(type(optPsf[0][0])))

        # determine the size of the central cut, so that from the huge generated
        # image we can cut out only the central portion (1.4 times larger
        #  than the size of actual final image)
        size_of_central_cut = int(oversampling_original * self.image.shape[0] * 1.4)

        if size_of_central_cut > optPsf.shape[0]:
            # if larger than size of image, cut the image
            # fail if not enough space to cut out the image
            size_of_central_cut = optPsf.shape[0]
            if self.verbosity == 1:
                print('size_of_central_cut modified to ' + str(size_of_central_cut))
            assert int(oversampling_original * self.image.shape[0] * 1.0) < optPsf.shape[0]

        assert size_of_central_cut <= optPsf.shape[0]
        if self.verbosity == 1:
            print('size_of_central_cut: ' + str(size_of_central_cut))

        # cut part which you need to form the final image
        # set oversampling to 1 so you are not resizing the image, and dx=0 and
        # dy=0 so that you are not moving around, i.e., you are just cutting the
        # central region
        optPsf_cut = PsfPosition.cut_Centroid_of_natural_resolution_image(
            image=optPsf, size_natural_resolution=size_of_central_cut + 1, oversampling=1, dx=0, dy=0)
        if self.verbosity == 1:
            print('optPsf_cut.shape' + str(optPsf_cut.shape))

        # we want to reduce oversampling to be roughly around 10 to make things computationaly easier
        # if oversamplign_original is smaller than 20 (in case of dithered images),
        # make resolution coarser by factor of 2
        # otherwise set it to 11
        if oversampling_original < 20:
            oversampling = np.round(oversampling_original / 2)
        else:
            oversampling = 11
        if self.verbosity == 1:
            print('oversampling:' + str(oversampling))

        # what will be the size of the image after you resize it to the from
        # ``oversampling_original'' to ``oversampling'' ratio
        size_of_optPsf_cut_downsampled = np.int(
            np.round(size_of_central_cut / (oversampling_original / oversampling)))
        if self.verbosity == 1:
            print('size_of_optPsf_cut_downsampled: ' + str(size_of_optPsf_cut_downsampled))

        # make sure that optPsf_cut_downsampled is an array which has an odd size
        # - increase size by 1 if needed
        if (size_of_optPsf_cut_downsampled % 2) == 0:
            im1 = galsim.Image(optPsf_cut, copy=True, scale=1)
            interpolated_image = galsim._InterpolatedImage(im1, x_interpolant=galsim.Lanczos(5, True))
            optPsf_cut_downsampled = interpolated_image.\
                drawImage(nx=size_of_optPsf_cut_downsampled + 1, ny=size_of_optPsf_cut_downsampled + 1,
                          scale=(oversampling_original / oversampling), method='no_pixel').array
        else:
            im1 = galsim.Image(optPsf_cut, copy=True, scale=1)
            interpolated_image = galsim._InterpolatedImage(im1, x_interpolant=galsim.Lanczos(5, True))
            optPsf_cut_downsampled = interpolated_image.\
                drawImage(nx=size_of_optPsf_cut_downsampled, ny=size_of_optPsf_cut_downsampled,
                          scale=(oversampling_original / oversampling), method='no_pixel').array

        if self.verbosity == 1:
            print('optPsf_cut_downsampled.shape: ' + str(optPsf_cut_downsampled.shape))

        if self.verbosity == 1:
            print('Postprocessing parameters are:')
            print(str(['grating_lines', 'scattering_slope', 'scattering_amplitude',
                       'pixel_effect', 'fiber_r']))
            print(str([param_values['grating_lines'], param_values['scattering_slope'],
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
            print('Are we invoking double sources (1 or True if yes): ' + str(self.double_sources))
            print('Double source position/ratio is:' + str(self.double_sources_positions_ratios))

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
            print('Time for postprocessing up to single_Psf_position protocol is ' +
                  str(time_end_single - time_start_single))

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
                                                                double_sources_positions_ratios= #noqa: E251
                                                                self.double_sources_positions_ratios,
                                                                verbosity=self.verbosity,
                                                                explicit_psf_position= #noqa: E251
                                                                self.explicit_psf_position,
                                                                use_only_chi=self.use_only_chi,
                                                                use_center_of_flux=self.use_center_of_flux)
        time_end_single = time.time()

        if self.verbosity == 1:
            print('Time for single_Psf_position protocol is ' + str(time_end_single - time_start_single))

        if self.verbosity == 1:
            print('Sucesfully created optPsf_final')

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
            print('Finished with optPsf_postprocessing')
            print(' ')

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
        so that effective size is 15 / dithering
        """
        size_of_pixels_in_image = (15 / self.dithering) / oversampling

        # size of the created optical PSF images in microns
        size_of_image_in_Microns = size_of_pixels_in_image * \
            (image.shape[0])
        if self.verbosity == 1:
            print('image: ' + str(image))

        ##########################################
        # 1. scattered light

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
            grating_kernel[i] = Ifun16Ne((i - int(image.shape[0] / 2)) * 0.07907 * 10**-9 /
                                         (dithering * oversampling) + wavelength * 10**-9,
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
            print(' ')
            print('Entering _get_Pupil (function inside ZernikeFitterPFS)')

        if self.verbosity == 1:
            print('Size of the pupil (npix): ' + str(self.npix))

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
            frd_sigma=self.params['frd_sigma'].value,#noqa: E
            frd_lorentz_factor=self.params['frd_lorentz_factor'].value,
            det_vert=self.params['det_vert'].value,
            slitHolder_frac_dx=self.params['slitHolder_frac_dx'].value,
            wide_0=self.params['wide_0'].value,
            wide_23=self.params['wide_23'].value,
            wide_43=self.params['wide_43'].value,
            misalign=self.params['misalign'].value,
            verbosity=self.verbosity)

        point = [self.params['dxFocal'].value, self.params['dyFocal'].value]#noqa: E
        pupil = Pupil_Image.getPupil(point)

        if self.save == 1:
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'pupil.illuminated',
                    pupil.illuminated.astype(np.float32))

        if self.verbosity == 1:
            print('Finished with _get_Pupil')

        return pupil

    def _getOptPsf_naturalResolution(self, params, return_intermediate_images=False):
        """Returns optical PSF, given the initialized parameters

        called by constructModelImage_PFS_naturalResolution

        Parameters
        ----------
        params : `lmfit.Parameters` object or python dictionary
            Parameters describing the model
        return_intermediate_images : `bool`
             If True, return intermediate images created during the run
             This is in order to help with debugging and inspect
             the images created during the process

        Returns
        ----------
        (if not return_intermediate_images)
        img_apod : `np.array`, (N, N)
            Psf image, only optical components considered

        (if return_intermediate_images)
            # return the image, pupil, illumination applied to the pupil
        img_apod : `np.array`, (N, N)
            Psf image, only optical components considred
        ilum : `np.array`, (N, N)
            Image showing the illumination of the pupil
        wf_grid_rot : `np.array`, (N, N)
            Image showing the wavefront across the pupil

        Notes
        ----------
        called by constructModelImage_PFS_naturalResolution
        """
        if self.verbosity == 1:
            print(' ')
            print('Entering _getOptPsf_naturalResolution')

        ################################################################################
        # pupil and illumination of the pupil
        ################################################################################
        time_start_single_1 = time.time()

        diam_sic = self.diam_sic

        if self.verbosity == 1:
            print(['detFrac', 'strutFrac', 'dxFocal', 'dyFocal', 'slitFrac', 'slitFrac_dy'])
            print(['x_fiber', 'y_fiber', 'effective_ilum_radius', 'frd_sigma',
                  'frd_lorentz_factor', 'det_vert', 'slitHolder_frac_dx'])
            print(['wide_0', 'wide_23', 'wide_43', 'misalign'])
            # print('set of pupil_parameters I. : ' + str(self.pupil_parameters[:6]))
            # print('set of pupil_parameters II. : ' + str(self.pupil_parameters[6:6 + 7]))
            # print('set of pupil_parameters III. : ' + str(self.pupil_parameters[13:]))
        time_start_single_2 = time.time()

        # initialize galsim.Aperature class
        # the output will be the size of pupil.illuminated

        pupil = self._get_Pupil()
        aper = galsim.Aperture(
            diam=pupil.size,
            pupil_plane_im=pupil.illuminated.astype(np.float32),
            pupil_plane_scale=pupil.scale,
            pupil_plane_size=None)

        if self.verbosity == 1:
            if self.pupilExplicit is None:
                print('Requested pupil size is (pupil.size) [m]: ' + str(pupil.size))
                print('One pixel has size of (pupil.scale) [m]: ' + str(pupil.scale))
                print('Requested pupil has so many pixels (pupil_plane_im): ' +
                      str(pupil.illuminated.astype(np.int16).shape))
            else:
                print('Supplied pupil size is (diam_sic) [m]: ' + str(self.diam_sic))
                print('One pixel has size of (diam_sic/npix) [m]: ' + str(self.diam_sic / self.npix))
                print('Requested pupil has so many pixels (pupilExplicit): ' + str(self.pupilExplicit.shape))

        time_end_single_2 = time.time()
        if self.verbosity == 1:
            print('Time for _get_Pupil function is ' + str(time_end_single_2 - time_start_single_2))

        time_start_single_3 = time.time()
        # create array with pixels=1 if the area is illuminated and 0 if it is obscured
        ilum = np.array(aper.illuminated, dtype=np.float32)
        assert np.sum(ilum) > 0, str(self.pupil_parameters)

        # gives size of the illuminated image
        lower_limit_of_ilum = int(ilum.shape[0] / 2 - self.npix / 2)
        higher_limit_of_ilum = int(ilum.shape[0] / 2 + self.npix / 2)
        if self.verbosity == 1:
            print('lower_limit_of_ilum: ' + str(lower_limit_of_ilum))
            print('higher_limit_of_ilum: ' + str(higher_limit_of_ilum))

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
            print('Size after padding zeros to 2x size and extra padding to get size suitable for FFT: ' +
                  str(ilum.shape))

        # maximum extent of pupil image in units of radius of the pupil, needed for next step
        size_of_ilum_in_units_of_radius = ilum.shape[0] / self.npix

        if self.verbosity == 1:
            print('size_of_ilum_in_units_of_radius: ' + str(size_of_ilum_in_units_of_radius))

        # do not caculate the ``radiometric effect (difference between entrance and exit pupil)
        # if paramters are too small to make any difference
        # if that is the case just declare the ``ilum_radiometric'' to be the same as ilum
        # i.e., the illumination of the exit pupil is the same as the illumination of the entrance pupil
        if params['radiometricExponent'] < 0.01 or params['radiometricEffect'] < 0.01:
            if self.verbosity == 1:
                print('skiping ``radiometric effect\'\' ')
            ilum_radiometric = ilum
        else:
            if self.verbosity == 1:
                print('radiometric parameters are: ')
                print('x_ilum,y_ilum,radiometricEffect,radiometricExponent' +
                      str([params['x_ilum'], params['y_ilum'],
                           params['radiometricEffect'], params['radiometricExponent']]))

            # add the change of flux between the entrance and exit pupil
            # end product is radiometricEffectArray
            points = np.linspace(-size_of_ilum_in_units_of_radius,
                                 size_of_ilum_in_units_of_radius, num=ilum.shape[0])
            xs, ys = np.meshgrid(points, points)
            _radius_coordinate = np.sqrt(
                (xs - params['x_ilum'] * params['dxFocal']) ** 2 +
                (ys - params['y_ilum'] * params['dyFocal']) ** 2)

            # ilumination to which radiometric effet has been applied, describing
            # difference betwen entrance and exit pupil
            radiometricEffectArray = (1 + params['radiometricEffect'] *
                                      _radius_coordinate**2)**(-params['radiometricExponent'])
            ilum_radiometric = np.nan_to_num(radiometricEffectArray * ilum, 0)

        # this is where you can introduce some apodization in the pupil image by using the line below
        # for larger images, scale according to the size of the input image which is to be FFT-ed
        # 0.75 is an arbitrary number
        apodization_sigma = ((len(ilum_radiometric)) / 1158)**0.875 * 0.75
        time_start_single_4 = time.time()

        # cut out central region, apply Gaussian on the center region and return to the full size image
        # thus is done to spped up the calculation
        # noqa: E128 in order to keep informative names
        ilum_radiometric_center_region = ilum_radiometric[(lower_limit_of_ilum -
                                         int(np.ceil(3 * apodization_sigma))):(higher_limit_of_ilum + # noqa: E128
                                         int(np.ceil(3 * apodization_sigma))),
                                         (lower_limit_of_ilum - int(np.ceil(3 * apodization_sigma))):
                                         (higher_limit_of_ilum + int(np.ceil(3 * apodization_sigma)))]

        ilum_radiometric_center_region_apodized = gaussian_filter(
            ilum_radiometric_center_region, sigma=apodization_sigma)

        ilum_radiometric_apodized = np.copy(ilum_radiometric)
        ilum_radiometric_apodized[(lower_limit_of_ilum -
                                   int(np.ceil(3 * apodization_sigma))):(higher_limit_of_ilum +
                                   int(np.ceil(3 * apodization_sigma))), (lower_limit_of_ilum - # noqa: E128
                                   int(np.ceil(3 * apodization_sigma))):(higher_limit_of_ilum +
                                   int(np.ceil(3 * apodization_sigma)))] =\
                                   ilum_radiometric_center_region_apodized

        time_end_single_4 = time.time()
        if self.verbosity == 1:
            print('Time to apodize the pupil: ' + str(time_end_single_4 - time_start_single_4))
            print('type(ilum_radiometric_apodized)' + str(type(ilum_radiometric_apodized[0][0])))

        # set pixels for which amplitude is less than 0.01 to 0
        r_ilum_pre = np.copy(ilum_radiometric_apodized)
        r_ilum_pre[ilum_radiometric_apodized > 0.01] = 1
        r_ilum_pre[ilum_radiometric_apodized < 0.01] = 0
        ilum_radiometric_apodized_bool = r_ilum_pre.astype(bool)

        # manual creation of aper.u and aper.v (mimicking steps which were automatically done in galsim)
        # this gives position information about each point in the exit pupil so we can apply wavefront to it
        single_line_aperu_manual = np.linspace(-diam_sic * (size_of_ilum_in_units_of_radius / 2), diam_sic * (
            size_of_ilum_in_units_of_radius / 2), len(ilum_radiometric_apodized_bool), endpoint=True)
        aperu_manual = np.tile(
            single_line_aperu_manual,
            len(single_line_aperu_manual)).reshape(
            len(single_line_aperu_manual),
            len(single_line_aperu_manual))

        # full grid
        u_manual = aperu_manual
        v_manual = np.transpose(aperu_manual)

        # select only parts of the grid that are actually illuminated
        u = u_manual[ilum_radiometric_apodized_bool]
        v = v_manual[ilum_radiometric_apodized_bool]

        time_end_single_3 = time.time()
        if self.verbosity == 1:
            print('Time for postprocessing pupil after _get_Pupil ' +
                  str(time_end_single_3 - time_start_single_3))

        time_end_single_1 = time.time()
        if self.verbosity == 1:
            print('Time for pupil and illumination calculation is ' +
                  str(time_end_single_1 - time_start_single_1))

        ################################################################################
        # wavefront
        ################################################################################
        # create wavefront across the exit pupil

        time_start_single = time.time()
        if self.verbosity == 1:
            print('')
            print('Starting creation of wavefront')

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
            print('diam_sic [m]: ' + str(diam_sic))
            print('aberrations: ' + str(aberrations))
            print('aberrations moved to z4=0: ' + str(aberrations_0))
            print('aberrations extra: ' + str(self.extraZernike))
            print('wavelength [nm]: ' + str(self.wavelength))

        if self.extraZernike is None:
            optics_screen = galsim.phase_screens.OpticalScreen(
                diam=diam_sic, aberrations=aberrations, lam_0=self.wavelength)
            if self.save == 1:
                # only create fake images with abberations set to 0 if we are going to save
                # i.e., if we testing the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(
                    diam=diam_sic, aberrations=aberrations_0, lam_0=self.wavelength)
        else:
            optics_screen = galsim.phase_screens.OpticalScreen(
                diam=diam_sic, aberrations=aberrations_extended, lam_0=self.wavelength)
            if self.save == 1:
                # only create fake images with abberations set to 0 if we are going to save
                # i.e., if we are testing the results
                optics_screen_fake_0 = galsim.phase_screens.OpticalScreen(
                    diam=diam_sic, aberrations=aberrations_0, lam_0=self.wavelength)

        screens = galsim.PhaseScreenList(optics_screen)
        if self.save == 1:
            # only create fake images with abberations set to 0 if we are going to save
            # i.e., if we are testing the results
            screens_fake_0 = galsim.PhaseScreenList(optics_screen_fake_0)

        time_end_single = time.time()

        ################################################################################
        # combining the pupil illumination and the wavefront
        ################################################################################

        # apply wavefront to the array describing illumination
        if self.use_wf_grid is None:
            wf = screens.wavefront(u, v, None, 0)
            if self.save == 1:
                wf_full = screens.wavefront(u_manual, v_manual, None, 0)
            wf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.float32)
            wf_grid[ilum_radiometric_apodized_bool] = (wf / self.wavelength)
            wf_grid_rot = wf_grid
        else:
            # if you want to pass an explit wavefront, it is applied here
            wf_grid = self.use_wf_grid
            wf_grid_rot = wf_grid

        if self.save == 1:
            # only create fake images with abberations set to 0 if we are going to save
            # i.e., if we are testing the results
            if self.verbosity == 1:
                print('creating wf_full_fake_0')
            wf_full_fake_0 = screens_fake_0.wavefront(u_manual, v_manual, None, 0)

        # exponential of the wavefront
        expwf_grid = np.zeros_like(ilum_radiometric_apodized_bool, dtype=np.complex64)
        expwf_grid[ilum_radiometric_apodized_bool] =\
            ilum_radiometric_apodized[ilum_radiometric_apodized_bool] *\
            np.exp(2j * np.pi * wf_grid_rot[ilum_radiometric_apodized_bool])

        if self.verbosity == 1:
            print('Time for wavefront and wavefront/pupil combining is ' +
                  str(time_end_single - time_start_single))

        ################################################################################
        # execute the FFT
        ################################################################################
        time_start_single = time.time()
        ftexpwf = np.fft.fftshift(scipy.fftpack.fft2(np.fft.fftshift(expwf_grid)))
        img_apod = np.abs(ftexpwf)**2
        time_end_single = time.time()
        if self.verbosity == 1:
            print('Time for FFT is ' + str(time_end_single - time_start_single))
        ######################################################################

        # size in arcseconds of the image generated by the code
        scale_ModelImage_PFS_naturalResolution = sky_scale(
            size_of_ilum_in_units_of_radius * self.diam_sic, self.wavelength)
        self.scale_ModelImage_PFS_naturalResolution = scale_ModelImage_PFS_naturalResolution

        if self.save == 1:
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'aperilluminated', aper.illuminated)
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum', ilum)
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum_radiometric', ilum_radiometric)
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER + 'ilum_radiometric_apodized', ilum_radiometric_apodized)
            np.save(self.TESTING_PUPIL_IMAGES_FOLDER +
                    'ilum_radiometric_apodized_bool', ilum_radiometric_apodized_bool)
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
            print('Finished with _getOptPsf_naturalResolution')
            print(' ')

        if not return_intermediate_images:
            return img_apod
        if return_intermediate_images:
            return img_apod, ilum[lower_limit_of_ilum:higher_limit_of_ilum,
                                  lower_limit_of_ilum:higher_limit_of_ilum], wf_grid_rot


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
            wide_0,
            wide_23,
            wide_43,
            misalign,
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
            print('Entering PupilFactory class')

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

        @param[in,out] pupil   Pupil to modify in place
        @param[in] p0          2-tuple indicating region center
        @param[in] r           Ellipse region radius = major axis
        @param[in] b           Ellipse region radius = minor axis
        @param[in] thetarot    Ellipse region rotation
        """
        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        theta = np.arctan(self.u / self.v) + thetarot

        pupil.illuminated[r2 > r**2 * b**2 / (b**2 * (np.cos(theta))**2 + r**2 * (np.sin(theta))**2)] = False

    def _cutSquare(self, pupil, p0, r, angle, det_vert):
        """Cut out the interior of a circular region from a Pupil.

        It is not necesarilly square, because of the det_vert parameter

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

        camX_value_for_f_multiplier = p0[0]
        camY_value_for_f_multiplier = p0[1]

        camY_Max = 0.02
        f_multiplier_factor = (-camX_value_for_f_multiplier * 100 / 3) * \
            (np.abs(camY_value_for_f_multiplier) / camY_Max) + 1

        if self.verbosity == 1:
            print('f_multiplier_factor for size of detector triangle is: ' + str(f_multiplier_factor))

        pupil_illuminated_only0_in_only1 = np.zeros((i_y_max - i_y_min, i_x_max - i_x_min))

        u0 = self.u[i_y_min:i_y_max, i_x_min:i_x_max]
        v0 = self.v[i_y_min:i_y_max, i_x_min:i_x_max]

        # factors that are controling how big is the triangle in the corner of the detector
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
        triangle21 = [[p0[0] + x22, p0[1] + y21], [p0[0] + x22, p0[1] +
                                                   y21 - y21 * f_lr], [p0[0] + x22 - x22 * f_lr, p0[1] + y21]]

        p21 = triangle21[0]
        y22 = (triangle21[1][1] - triangle21[0][1]) / np.sqrt(2)
        y21 = 0
        x21 = (triangle21[2][0] - triangle21[0][0]) / np.sqrt(2)
        x22 = -(triangle21[2][0] - triangle21[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((v0 - p21[1]) * np.cos(-angleRad21) -
                                          (u0 - p21[0]) * np.sin(-angleRad21) < y22)] = True
        ###########################################################
        # Upper left corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert

        f_ul = np.copy(f) * (1 / f_multiplier)

        triangle12 = [[p0[0] + x21, p0[1] + y22], [p0[0] + x21, p0[1] +
                                                   y22 - y22 * f_ul], [p0[0] + x21 - x21 * f_ul, p0[1] + y22]]

        p21 = triangle12[0]
        y22 = 0
        y21 = (triangle12[1][1] - triangle12[0][1]) / np.sqrt(2)
        x21 = -(triangle12[2][0] - triangle12[0][0]) / np.sqrt(2)
        x22 = +(triangle12[2][0] - triangle12[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((v0 - p21[1]) * np.cos(-angleRad21) -
                                          (u0 - p21[0]) * np.sin(-angleRad21) > y21)] = True
        ###########################################################
        # Upper right corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert
        f_ur = np.copy(f) * f_multiplier

        triangle22 = [[p0[0] + x22, p0[1] + y22], [p0[0] + x22, p0[1] +
                                                   y22 - y22 * f_ur], [p0[0] + x22 - x22 * f_ur, p0[1] + y22]]

        p21 = triangle22[0]
        y22 = -0
        y21 = +(triangle22[1][1] - triangle22[0][1]) / np.sqrt(2)
        x21 = +(triangle22[2][0] - triangle22[0][0]) / np.sqrt(2)
        x22 = -(triangle22[2][0] - triangle22[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((u0 - p21[0]) * np.cos(-angleRad21) +
                                          (v0 - p21[1]) * np.sin(-angleRad21) > x21)] = True
        ###########################################################
        # Lower left corner
        x21 = -r / 2 * 1
        x22 = +r / 2 * 1
        y21 = -r / 2 * det_vert
        y22 = +r / 2 * det_vert
        f_ll = np.copy(f) * f_multiplier

        triangle11 = [[p0[0] + x21, p0[1] + y21], [p0[0] + x21, p0[1] +
                                                   y21 - y21 * f_ll], [p0[0] + x21 - x21 * f_ll, p0[1] + y21]]

        p21 = triangle11[0]
        y22 = -(triangle11[1][1] - triangle11[0][1]) / np.sqrt(2)
        y21 = 0
        x21 = +(triangle11[2][0] - triangle11[0][0]) / np.sqrt(2)
        x22 = +(triangle11[2][0] - triangle11[0][0]) / np.sqrt(2)

        pupil_illuminated_only0_in_only1[((u0 - p21[0]) * np.cos(-angleRad21) +
                                          (v0 - p21[1]) * np.sin(-angleRad21) < x22)] = True

        pupil_illuminated_only1[i_y_min:i_y_max, i_x_min:i_x_max] = pupil_illuminated_only0_in_only1

        pupil.illuminated = pupil.illuminated * pupil_illuminated_only1
        time_end_single_square = time.time()

        if self.verbosity == 1:
            print('Time for cutting out the central square is ' +
                  str(time_end_single_square - time_start_single_square))

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

        pupil.illuminated[(d < 0.5 * thickness * (1 + wide * radial_distance)) &
                          ((self.u - p0[0]) * np.cos(angleRad) +
                           (self.v - p0[1]) * np.sin(angleRad) >= 0)] = False

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
        pupil.illuminated[(d < 0.5 * thickness) &
                          ((self.u - p0[0]) * np.cos(angleRad) +
                           (self.v - p0[1]) * np.sin(angleRad) >= 0)] = True


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
            wide_0,
            wide_23,
            wide_43,
            misalign,
            verbosity=0):
        """Construct a PFS PupilFactory.

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
            print('Entering PFSPupilFactory class')

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

        self.wide_0 = wide_0
        self.wide_23 = wide_23
        self.wide_43 = wide_43
        self.misalign = misalign

    def getPupil(self, point):
        """Calculate a Pupil at a given point in the focal plane.

        @param point  Point2D indicating focal plane coordinates.
        @returns      Pupil
        """
        if self.verbosity == 1:
            print('Entering getPupil (function inside PFSPupilFactory)')

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

        # creating FRD effects
        single_element = np.linspace(-1, 1, len(pupil.illuminated), endpoint=True, dtype=np.float32)
        u_manual = np.tile(single_element, (len(single_element), 1))
        v_manual = np.transpose(u_manual)
        center_distance = np.sqrt((u_manual - self.x_fiber * hscRate * hscPlateScale * 12)
                                  ** 2 + (v_manual - self.y_fiber * hscRate * hscPlateScale * 12)**2)
        frd_sigma = self.frd_sigma
        sigma = 2 * frd_sigma
        pupil_frd = (1 / 2 * (scipy.special.erf((-center_distance + self.effective_ilum_radius) / sigma) +
                              scipy.special.erf((center_distance + self.effective_ilum_radius) / sigma)))

        ################
        # Adding misaligment in this section
        time_misalign_start = time.time()

        position_of_center_0 = np.where(center_distance == np.min(center_distance))
        position_of_center = [position_of_center_0[1][0], position_of_center_0[0][0]]

        position_of_center_0_x = position_of_center_0[0][0]
        position_of_center_0_y = position_of_center_0[1][0]

        distances_to_corners = np.array([np.sqrt(position_of_center[0]**2 + position_of_center[1]**2),
                                         np.sqrt((len(pupil_frd) - position_of_center[0])**2 +
                                         position_of_center[1]**2), np.sqrt((position_of_center[0])**2 +
                                         (len(pupil_frd) - position_of_center[1])**2),
                                         np.sqrt((len(pupil_frd) - position_of_center[0])**2 +
                                         (len(pupil_frd) - position_of_center[1])**2)])

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

        radiusvalues = np.linspace(0, int(np.ceil(max_distance_to_corner)),
                                   int(np.ceil(max_distance_to_corner)) + 1)

        sigtotp = sigma * 550

        dif_due_to_mis_class = Pupil_misalign(radiusvalues, imageradius, sigtotp, self.misalign)
        dif_due_to_mis = dif_due_to_mis_class()

        scaling_factor_pixel_to_physical = max_distance_to_corner / np.max(center_distance)
        distance_int = np.round(center_distance * scaling_factor_pixel_to_physical).astype(int)

        pupil_frd_with_mis = pupil_frd + dif_due_to_mis[distance_int]
        pupil_frd_with_mis[pupil_frd_with_mis > 1] = 1

        time_misalign_end = time.time()

        if self.verbosity == 1:
            print('Time to execute illumination considerations due to misalignment ' +
                  str(time_misalign_end - time_misalign_start))

        ####
        pupil_lorentz = (np.arctan(2 * (self.effective_ilum_radius - center_distance) / (4 * sigma)) +
                         np.arctan(2 * (self.effective_ilum_radius + center_distance) / (4 * sigma))) /\
                        (2 * np.arctan((2 * self.effective_ilum_radius) / (4 * sigma)))

        pupil.illuminated = (pupil_frd + 1 * self.frd_lorentz_factor *
                             pupil_lorentz) / (1 + self.frd_lorentz_factor)

        pupil_lorentz = (np.arctan(2 * (self.effective_ilum_radius - center_distance) / (4 * sigma)) +
                         np.arctan(2 * (self.effective_ilum_radius + center_distance) / (4 * sigma))) /\
                        (2 * np.arctan((2 * self.effective_ilum_radius) / (4 * sigma)))

        pupil_frd = np.copy(pupil_frd_with_mis)
        pupil.illuminated = (pupil_frd + 1 * self.frd_lorentz_factor *
                             pupil_lorentz) / (1 + self.frd_lorentz_factor)

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
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_0)
            if angle == np.pi * 2 / 3:
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_23)
            if angle == np.pi * 4 / 3:
                self._cutRay(pupil, (x, y), angle, subaruStrutThick, 'rad', self.wide_43)

        # cut out slit shadow
        self._cutRay(pupil, (2, slitFrac_dy / 18), -np.pi, subaruSlit * 1.05, 'rad')

        # cut out slit holder shadow
        # subaruSlit/3 is roughly the width of the holder
        self._cutRay(pupil, (self.slitHolder_frac_dx / 18, 1), -np.pi / 2, subaruSlit * 0.3, 'rad')

        if self.verbosity == 1:
            print('Finished with getPupil')

        return pupil


class Pupil(object):
    """Pupil obscuration function.
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
        # print(len(x))
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


class PsfPosition(object):
    """Class that deals with positioning of the PSF model

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
        """
        self.image = image
        self.oversampling = oversampling
        self.size_natural_resolution = size_natural_resolution
        self.simulation_00 = simulation_00
        self.verbosity = verbosity
        if save is None:
            save = 0
        self.save = save

        if PSF_DIRECTORY is not None:
            self.PSF_DIRECTORY = PSF_DIRECTORY
            self.TESTING_FOLDER = PSF_DIRECTORY + 'Testing/'
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
        positions_from_where_to_start_cut = [int(len(image) / 2 -
                                                 size_natural_resolution / 2 -
                                                 dx * oversampling + 1),
                                             int(len(image) / 2 -
                                                 size_natural_resolution / 2 -
                                                 dy * oversampling + 1)]
        res = image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1] +
                    int(size_natural_resolution),
                    positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0] +
                    int(size_natural_resolution)]

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
        simulation_00 = self.simulation_00

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
            print('parameter use_only_chi in Psf_postion is set to: ' + str(use_only_chi))
            print('parameter use_center_of_flux in Psf_postion is set to: ' + str(use_center_of_flux))
            print('parameter simulation_00 in Psf_postion is set to: ' + str(simulation_00))

        # depending on if there is a second source in the image the algoritm splits here
        # double_sources should always be None when when creating centered images (simulation_00 = True)
        if double_sources is None or bool(double_sources) is False:
            # if simulation_00 AND using optical center just run the realization that is set at 0,0
            if simulation_00 == 1 and use_center_of_flux is False:
                if verbosity == 1:
                    print('simulation_00 is set to 1 and use_center_of_flux==False -\
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
                            print('creating simulated image, center of light in center of the image')
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
                        print('time_2-time_1 for initial_complete_realization: ' + str(time_2 - time_1))

                    # center of the light for the first realization, set at optical center
                    centroid_of_initial_complete_realization = find_centroid_of_flux(
                        initial_complete_realization)

                    # determine offset between the initial guess and the data
                    offset_initial_and_sci = - \
                        ((np.array(find_centroid_of_flux(initial_complete_realization)) -
                          np.array(find_centroid_of_flux(sci_image))))

                    if verbosity == 1:
                        print('centroid_of_initial_complete_realization ' +
                              str(find_centroid_of_flux(initial_complete_realization)))
                        print('centroid_of_sci_image '+str(find_centroid_of_flux(sci_image)))
                        print('offset_initial_and_sci: ' + str(offset_initial_and_sci))
                        print('[x_primary, y_primary, y_secondary,ratio_secondary] / chi2 output')
                    if self.save == 1:
                        np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'initial_complete_realization',
                                initial_complete_realization)

                    # search for the best center using scipy ``shgo'' algorithm
                    # set the limits for the fitting procedure
                    y_2sources_limits = [
                        (offset_initial_and_sci[1] - 2) * self.oversampling,
                        (offset_initial_and_sci[1] + 2) * self.oversampling]
                    x_2sources_limits = [
                        (offset_initial_and_sci[0] - 1) * self.oversampling,
                        (offset_initial_and_sci[0] + 1) * self.oversampling]

                    # search for best positioning
                    if use_center_of_flux:
                        for i in range(5):
                            if verbosity == 1:
                                print("###")

                            if i == 0:
                                x_i, y_i = offset_initial_and_sci * oversampling
                                x_offset, y_offset = 0, 0
                                x_offset = x_offset + x_i
                                y_offset = y_offset + y_i
                            else:
                                x_offset = x_offset + x_i
                                y_offset = y_offset + y_i

                                complete_realization = self.create_complete_realization(
                                    x=[x_offset, y_offset, 0, 0, ],
                                    return_full_result=True, use_only_chi=True,
                                    use_center_of_light=True, simulation_00=simulation_00)[-1]
                            offset_initial_and_sci = -((np.array(find_centroid_of_flux(complete_realization))
                                                        - np.array(find_centroid_of_flux(sci_image))))
                            if verbosity == 1:
                                print('offset_initial_and_sci in step ' +
                                      str(i) + ' ' + str(offset_initial_and_sci))
                                print("###")
                            x_i, y_i = offset_initial_and_sci * oversampling
                        primary_position_and_ratio_x = [x_offset, y_offset]

                    # if use_center_of_flux=False, we have to optimize to find the best solution
                    else:
                        # implement try syntax for secondary too
                        try:
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
                            print(e)
                            print('search for primary position failed')
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
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'single_realization_primary_renormalized',
                            single_realization_primary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'single_realization_secondary_renormalized',
                            single_realization_secondary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'complete_realization_renormalized',
                            complete_realization_renormalized)

                    if self.verbosity == 1:
                        if simulation_00 != 1:
                            print('We are fitting for only one source')
                            print('One source fitting result is ' + str(primary_position_and_ratio_x))
                            print('type(complete_realization_renormalized)' +
                                  str(type(complete_realization_renormalized[0][0])))

                            centroid_of_complete_realization_renormalized = find_centroid_of_flux(
                                complete_realization_renormalized)

                            # determine offset between the initial guess and the data
                            offset_final_and_sci = - \
                                (np.array(centroid_of_complete_realization_renormalized) -
                                 np.array(centroid_of_sci_image))

                            print('offset_final_and_sci: ' + str(offset_final_and_sci))

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
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'single_realization_primary_renormalized',
                            single_realization_primary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'single_realization_secondary_renormalized',
                            single_realization_secondary_renormalized)
                        np.save(
                            self.TESTING_FINAL_IMAGES_FOLDER +
                            'complete_realization_renormalized',
                            complete_realization_renormalized)

                    if self.verbosity == 1:
                        if simulation_00 != 1:
                            print('We are passing value for only one source')
                            print('One source fitting result is ' + str(explicit_psf_position))
                            print('type(complete_realization_renormalized)' +
                                  str(type(complete_realization_renormalized[0][0])))

                    return complete_realization_renormalized, explicit_psf_position

        else:
            # TODO: need to make possible that you can pass your own values for double source

            # create one complete realization with default parameters - estimate
            # centroids and use that knowledge to put fitting limits in the next step
            centroid_of_sci_image = find_centroid_of_flux(sci_image)
            initial_complete_realization = self.create_complete_realization([0,
                                                                             0,
                                                                             double_sources_positions_ratios[0] #noqa: E501
                                                                             * self.oversampling,
                                                                             double_sources_positions_ratios[1]], #noqa: E501
                                                                            return_full_result=True,
                                                                            use_only_chi=use_only_chi,
                                                                            use_center_of_light= #noqa: E251
                                                                            use_center_of_flux,
                                                                            simulation_00=simulation_00)[-1]
            centroid_of_initial_complete_realization = find_centroid_of_flux(initial_complete_realization)

            # determine offset between the initial guess and the data
            offset_initial_and_sci = - \
                (np.array(centroid_of_initial_complete_realization) - np.array(centroid_of_sci_image))

            if verbosity == 1:
                print('Evaulating double source psf positioning loop')
                print('offset_initial_and_sci: ' + str(offset_initial_and_sci))
                print('[x_primary, y_primary, y_secondary,ratio_secondary] / chi2 output')

            if self.save == 1:
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'sci_image', sci_image)
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'initial_complete_realization',
                        initial_complete_realization)

            # implement that it does not search if second object far away while in focus
            # focus size is 20 - do not search if second pfs is more than 15 pixels away
            if shape_of_sci_image == 20 and np.abs(self.double_sources_positions_ratios[0]) > 15:
                if verbosity == 1:
                    print('fitting second source, but assuming that second source is too far')

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
                    # print('(False,use_only_chi,use_center_of_flux)'+str((False,use_only_chi,use_center_of_flux)))
                    primary_position_and_ratio_shgo = scipy.optimize.shgo(
                        self.create_complete_realization,
                        args=(False, use_only_chi,
                              use_center_of_flux, simulation_00),
                        bounds=[(x_2sources_limits[0], x_2sources_limits[1]),
                                (y_2sources_limits[0], y_2sources_limits[1])],
                        n=10, sampling_method='sobol',
                        options={'ftol': 1e-3, 'maxev': 10})

                    if verbosity == 1:
                        print('starting finer positioning')

                    primary_position_and_ratio = scipy.optimize.minimize(
                        self.create_complete_realization,
                        args=(False, use_only_chi,
                              use_center_of_flux, simulation_00),
                        x0=primary_position_and_ratio_shgo.x,
                        method='Nelder-Mead',
                        options={'xatol': 0.00001, 'fatol': 0.00001})

                    primary_position_and_ratio_x = primary_position_and_ratio.x
                except BaseException:
                    print('search for primary position failed')
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

                primary_secondary_position_and_ratio = scipy.optimize.shgo(
                    self.create_complete_realization,
                    args=(False, use_only_chi,
                          use_center_of_flux, simulation_00),
                    bounds=[
                        (x_2sources_limits[0],
                         x_2sources_limits[1]),
                        (y_2sources_limits[0],
                         y_2sources_limits[1]),
                        (y_2sources_limits_second_source[0],
                         y_2sources_limits_second_source[1]),
                        (self.double_sources_positions_ratios[1] / 2,
                         2 * self.double_sources_positions_ratios[1])],
                    n=10, sampling_method='sobol',
                    options={'ftol': 1e-3, 'maxev': 10})

                primary_secondary_position_and_ratio_x = primary_secondary_position_and_ratio.x

            # return best result
            mean_res, single_realization_primary_renormalized,
            single_realization_secondary_renormalized, complete_realization_renormalized \
                = self.create_complete_realization(primary_secondary_position_and_ratio_x,
                                                   return_full_result=True, use_only_chi=use_only_chi,
                                                   use_center_of_light=use_center_of_flux,
                                                   simulation_00=simulation_00)

            if self.save == 1:
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'single_realization_primary_renormalized',
                    single_realization_primary_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'single_realization_secondary_renormalized',
                    single_realization_secondary_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'complete_realization_renormalized',
                    complete_realization_renormalized)

            if self.verbosity == 1:
                print('We are fitting for two sources')
                print('Two source fitting result is ' + str(primary_secondary_position_and_ratio_x))
                print('type(complete_realization_renormalized)' +
                      str(type(complete_realization_renormalized[0][0])))

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
        # oversampled input model image
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
            # 'shape is an even number'

            shift_x_mod = np.array(
                [-(np.round(primary_offset_axis_1) - primary_offset_axis_1),
                 -np.round(primary_offset_axis_1)])
            shift_y_mod = np.array(
                [-(np.round(primary_offset_axis_0) - primary_offset_axis_0),
                 -np.round(primary_offset_axis_0)])
        else:
            # 'shape is an odd number'
            shift_x_mod = np.array(
                [-(np.round(primary_offset_axis_1) - primary_offset_axis_1),
                 -np.round(primary_offset_axis_1)])
            shift_y_mod = np.array(
                [-(np.round(primary_offset_axis_0) - primary_offset_axis_0),
                 -np.round(primary_offset_axis_0)])

        image_integer_offset = image[center_position +
                                     int(shift_y_mod[1]) - 1 -
                                     shape_of_oversampled_image:center_position +
                                     int(shift_y_mod[1]) +
                                     shape_of_oversampled_image + 1,
                                     center_position +
                                     int(shift_x_mod[1]) - 1 -
                                     shape_of_oversampled_image: center_position +
                                     int(shift_x_mod[1]) +
                                     shape_of_oversampled_image + 1]
        if simulation_00:
            image_integer_offset = image[center_position +
                                         int(shift_y_mod[1]) - 1 -
                                         shape_of_oversampled_image:center_position +
                                         int(shift_y_mod[1]) +
                                         shape_of_oversampled_image + 1 + 1,
                                         center_position +
                                         int(shift_x_mod[1]) - 1 -
                                         shape_of_oversampled_image: center_position +
                                         int(shift_x_mod[1]) +
                                         shape_of_oversampled_image + 1 + 1]
            print('image_integer_offset shape: ' + str(image_integer_offset.shape))

        image_integer_offset_lsst = lsst.afw.image.image.ImageD(image_integer_offset.astype('float64'))
        oversampled_Image_LSST_apply_frac_offset = lsst.afw.math.offsetImage(
            image_integer_offset_lsst, shift_x_mod[0], shift_y_mod[0], algorithmName='lanczos5', buffer=5)
        single_primary_realization_oversampled = oversampled_Image_LSST_apply_frac_offset.array[1:-1, 1:-1]
        assert single_primary_realization_oversampled.shape[0] == shape_of_sci_image * oversampling
        single_primary_realization = resize(
            single_primary_realization_oversampled, (shape_of_sci_image, shape_of_sci_image), ())

        ###################
        # This part is skipped if there is only primary source in the image
        # go through secondary loop if the flux ratio is not zero
        # (TODO: if secondary too far outside the image, do not go through secondary)
        if ratio_secondary != 0:
            # overloading the definitions used in primary image
            if np.modf(shape_of_oversampled_image / 2)[0] == 0.0:
                # print('shape is an even number')
                shift_x_mod = np.array(
                    [-(np.round(secondary_offset_axis_1) - secondary_offset_axis_1),
                     -np.round(secondary_offset_axis_1)])
                shift_y_mod = np.array(
                    [-(np.round(secondary_offset_axis_0) - secondary_offset_axis_0),
                     -np.round(secondary_offset_axis_0)])

            else:
                # print('shape is an odd number')
                shift_x_mod = np.array(
                    [-(np.round(secondary_offset_axis_1) - secondary_offset_axis_1),
                     -np.round(secondary_offset_axis_1)])
                shift_y_mod = np.array(
                    [-(np.round(secondary_offset_axis_0) - secondary_offset_axis_0),
                     -np.round(secondary_offset_axis_0)])

            image_integer_offset = image[center_position +
                                         int(shift_y_mod[1]) - 1 - shape_of_oversampled_image:
                                         center_position + int(shift_y_mod[1]) +
                                         shape_of_oversampled_image + 2,
                                         center_position + int(shift_x_mod[1]) -
                                         1 - shape_of_oversampled_image:
                                         center_position + int(shift_x_mod[1]) +
                                         shape_of_oversampled_image + 2]

            image_integer_offset_lsst = lsst.afw.image.image.ImageD(image_integer_offset.astype('float64'))

            oversampled_Image_LSST_apply_frac_offset = lsst.afw.math.offsetImage(
                image_integer_offset_lsst, shift_y_mod[0], shift_x_mod[0], algorithmName='lanczos5', buffer=5)

            single_secondary_realization_oversampled =\
                oversampled_Image_LSST_apply_frac_offset.array[1:-1, 1:-1]

            single_secondary_realization = resize(
                single_secondary_realization_oversampled, (shape_of_sci_image, shape_of_sci_image), ())

        inverted_mask = ~mask_image.astype(bool)

        ###################
        # create complete_realization which is just primary if no secondary source
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
        if not return_full_result:
            chi_2_almost_multi_values = self.create_chi_2_almost_Psf_position(
                complete_realization_renormalized,
                sci_image,
                var_image,
                mask_image,
                use_only_chi=use_only_chi,
                use_center_of_light=use_center_of_light,
                simulation_00=simulation_00)
            if self.verbosity == 1:
                print(
                    'chi2 within shgo with use_only_chi ' +
                    str(use_only_chi) +
                    ' and use_center_of_light ' +
                    str(use_center_of_light) +
                    ' ' + str(x) + ' / ' + str(chi_2_almost_multi_values))
            return chi_2_almost_multi_values
        else:
            if ratio_secondary != 0:
                # print('ratio_secondary 2nd loop: '+str(ratio_secondary))
                single_primary_realization_renormalized = single_primary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized = ratio_secondary * single_secondary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
            else:
                # print('ratio_secondary 2nd loop 0: '+str(ratio_secondary))
                single_primary_realization_renormalized = single_primary_realization * \
                    (np.sum(sci_image[inverted_mask]) * v_flux / np.sum(complete_realization[inverted_mask]))
                single_secondary_realization_renormalized = np.zeros(
                    single_primary_realization_renormalized.shape)

            if self.save == 1:
                np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'image', image)
                if ratio_secondary != 0:
                    np.save(self.TESTING_FINAL_IMAGES_FOLDER + 'image_full_for_secondary', image)
                    np.save(
                        self.TESTING_FINAL_IMAGES_FOLDER +
                        'single_secondary_realization',
                        single_secondary_realization)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'single_primary_realization',
                    single_primary_realization)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'single_primary_realization_renormalized_within_create_complete_realization',
                    single_primary_realization_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'single_secondary_realization_renormalized_within_create_complete_realization',
                    single_secondary_realization_renormalized)
                np.save(
                    self.TESTING_FINAL_IMAGES_FOLDER +
                    'complete_realization_renormalized_within_create_complete_realization',
                    complete_realization_renormalized)

            # TODO: should I modify this function to remove distance from physcial center of
            # mass when using that option
            chi_2_almost_multi_values = self.create_chi_2_almost_Psf_position(
                complete_realization_renormalized,
                sci_image,
                var_image,
                mask_image,
                use_only_chi=use_only_chi,
                use_center_of_light=use_center_of_light,
                simulation_00=simulation_00)

            # save the best oversampled image
            if simulation_00:
                if self.verbosity == 1:
                    print('saving oversampled simulation_00 image')
                    # print('I have to implement that again')
                    print(
                        'saving at ' +
                        self.TESTING_FINAL_IMAGES_FOLDER +
                        'single_primary_realization_oversampled')
                    np.save(
                        self.TESTING_FINAL_IMAGES_FOLDER +
                        'single_primary_realization_oversampled_to_save',
                        single_primary_realization_oversampled)
                    np.save(
                        self.TESTING_FINAL_IMAGES_FOLDER +
                        'complete_realization_renormalized_to_save',
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

        # if you are minimizes chi or chi**2
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
                    print('sim00=False and center of light =true')

                distance_of_flux_center = np.sqrt(
                    np.sum(
                        (np.array(
                            find_centroid_of_flux(modelImg_masked)) -
                            np.array(
                            find_centroid_of_flux(sci_image_masked)))**2))
            else:
                # if you pass both simulation_00 paramter and use_center_of_light=True,
                # center of light will be centered in the downsampled image
                if self.verbosity == 1:
                    print('sim00 = True and center of light = True')

                distance_of_flux_center = np.sqrt(
                    np.sum(
                        (np.array(find_centroid_of_flux(modelImg_masked)) -
                            np.array(np.array(
                                np.ones((21, 21)).shape) / 2 - 0.5))**2))
            return distance_of_flux_center

    def fill_crop(self, img, pos, crop):
        '''Fills `crop` with values from `img` at `pos`
        while accounting for the crop being off the edge of `img`.
        *Note:* negative values in `pos` are interpreted as-is,
        not as "from the end".
        Taken from https://stackoverflow.com/questions/41153803/zero-padding-slice-past-end-of-array-in-numpy #noqa:E501
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
            print('TypeError in fill_crop function')
            pass


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


def sky_scale(pupil_plane_size, lam, scale_unit=galsim.arcsec):
    """Return the image scale for this aperture at given wavelength.

    @param lam         Wavelength in nanometers.
    @param scale_unit  Units in which to return result [default: galsim.arcsec]
    @returns           Image scale
    """
    return (lam * 1e-9) / pupil_plane_size * galsim.radians / scale_unit


def find_centroid_of_flux(image, mask=None):
    """Find center of flux in an image

    function giving the tuple of the position of weighted average of
    the flux in a square image
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

    return(x_center, y_center)

# Resizing functions below
# Function below taken from https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
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
      first_breaks: breaks between entries in the first set of intervals,
      with shape (N+1,). Must be a non-decreasing sequence.
      second_breaks: breaks between entries in the second set of intervals,
      with shape (M+1,). Must be a non-decreasing sequence.

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


def resize(array, shape,
           reflect_axes=()) -> np.ndarray:
    """Resize an array with the local mean / bilinear scaling.

    Works for both upsampling and downsampling in a fashion equivalent to
    block_mean and zoom, but allows for resizing by non-integer multiples.
    Prefer block_mean and zoom when possible,
    as this implementation is probably slower.

    Args:
      array: array to resize.
      shape: shape of the resized array.
      reflect_axes: iterable of axis numbers with
      reflecting boundary conditions, mirrored over the center
      of the first and last cell.

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
    return output #noqa: W292