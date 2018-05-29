#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on April 5 2018

@author: ncaplar@princeton.edu

"""

from __future__ import absolute_import, division, print_function

import socket
import lmfit
import galsim
galsim.GSParams.maximum_fft_size=12000
import numpy as np
np.set_printoptions(suppress=True)
import emcee
import time
import sys

from emcee.utils import MPIPool
import mpi4py

from scipy.ndimage import gaussian_filter
import scipy.misc
import skimage.transform

import astropy
import astropy.convolution
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.io import fits

import lsst.afw
from lsst.afw.cameraGeom import PupilFactory
from lsst.afw.geom import Angle, degrees
from lsst.afw import geom
from lsst.afw.geom import Point2D

np.seterr(divide='ignore', invalid='ignore')

__all__ = ['PupilFactory', 'Pupil']


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

    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy):
        """!Construct a PupilFactory.

        @params others
        @param[in] npix       Constructed Pupils will be npix x npix.
        """
        self.pupilSize = pupilSize
        self.npix = npix
        self.input_angle=input_angle
        self.hscFrac=hscFrac
        self.strutFrac=strutFrac
        #self.illumminatedFrac=illumminatedFrac
        self.pupilScale = pupilSize/npix
        self.slitFrac=slitFrac
        self.slitFrac_dy=slitFrac_dy
        u = (np.arange(npix, dtype=np.float64) - (npix - 1)/2) * self.pupilScale
        self.u, self.v = np.meshgrid(u, u)

    def getPupil(self, point):
        """!Calculate a Pupil at a given point in the focal plane.

        @param point  Point2D indicating focal plane coordinates.
        @returns      Pupil
        """
        raise NotImplementedError(
            "PupilFactory not implemented for this camera")

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
        
        illuminated = np.ones(self.u.shape, dtype=np.bool)
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
        
    """def _cutSquare(self,pupil, p0, r,angle):
        Cut out the interior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half lenght of the length of square side
        @param[in] angle      angle that the camera is rotated
      
        x21 = p0[0]-r/2
        x22 = p0[0]+r/2
        y21 = p0[1]-r/2
        y22 = p0[1]+r/2
        print("I am not sure that central square moves properly when moving and rotating on focal plane!!!!!")
        #pupil.illuminated[np.logical_and((self.u<x22) & (self.u>x21),(self.v<y22) & (self.v>y21))] = False
        angleRad = angle
        pupil.illuminated[np.logical_and((self.u*np.cos(-angle)+self.v*np.sin(-angleRad)<x22) & \
                          (self.u*np.cos(-angleRad)+self.v*np.sin(-angleRad)>x21),\
                          (self.v*np.cos(-angleRad)-self.u*np.sin(-angleRad)<y22) & \
                          (self.v*np.cos(-angleRad)-self.u*np.sin(-angleRad)>y21))] = False
  """
    def _cutSquare(self,pupil, p0, r,angle):
        """Cut out the interior of a circular region from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating region center
        @param[in] r          half lenght of the length of square side
        @param[in] angle      angle that the camera is rotated
        """
        x21 = -r/2
        x22 = +r/2
        y21 = -r/2
        y22 = +r/2
        #print("We are using HSC parameters for movement on focal plane!!!")
        #pupil.illuminated[np.logical_and((self.u<x22) & (self.u>x21),(self.v<y22) & (self.v>y21))] = False
        angleRad = angle
        pupil.illuminated[np.logical_and(((self.u-p0[0])*np.cos(-angle)+(self.v-p0[1])*np.sin(-angleRad)<x22) & \
                          ((self.u-p0[0])*np.cos(-angleRad)+(self.v-p0[1])*np.sin(-angleRad)>x21),\
                          ((self.v-p0[1])*np.cos(-angleRad)-(self.u-p0[0])*np.sin(-angleRad)<y22) & \
                          ((self.v-p0[1])*np.cos(-angleRad)-(self.u-p0[0])*np.sin(-angleRad)>y21))] = False    
        
        
    def _cutRay(self, pupil, p0, angle, thickness):
        """Cut out a ray from a Pupil.

        @param[in,out] pupil  Pupil to modify in place
        @param[in] p0         2-tuple indicating ray starting point
        @param[in] angle      Ray angle measured CCW from +x.
        @param[in] thickness  Thickness of cutout
        """
        angleRad = angle.asRadians()
        # the 1 is arbitrary, just need something to define another point on
        # the line
        p1 = (p0[0] + 1, p0[1] + np.tan(angleRad))
        d = PupilFactory._pointLineDistance((self.u, self.v), p0, p1)
        pupil.illuminated[(d < 0.5*thickness) &
                          ((self.u - p0[0])*np.cos(angleRad) +
                           (self.v - p0[1])*np.sin(angleRad) >= 0)] = False        

class PFSPupilFactory(PupilFactory):
    """!Pupil obscuration function factory for PFS 
    """
    def __init__(self, pupilSize, npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy):
        """!Construct a PupilFactory.

        @param[in] visitInfo  VisitInfo object for a particular exposure.
        @param[in] pupilSize  Size in meters of constructed Pupils.
        @param[in] npix       Constructed Pupils will be npix x npix.
        """
        PupilFactory.__init__(self, pupilSize,npix,input_angle,hscFrac,strutFrac,slitFrac,slitFrac_dy)
        
        hra = self._horizonRotAngle()
        hraRad = hra.asRadians()
        rot = np.array([[np.cos(hraRad), np.sin(hraRad)],
                        [-np.sin(hraRad), np.cos(hraRad)]])

        # Compute spider shadow parameters accounting for rotation angle.
        # Location where pairs of struts meet near prime focus.
        unrotStartPos = [np.array([0., 0]),
                         np.array([0., 0.]),
                         np.array([0, 0])]
        # Half angle between pair of struts that meet at Subaru prime focus
        # ring.
        strutAngle =60*degrees
        alpha = strutAngle - 60.0*degrees
        unrotAngles = [90*degrees + alpha,
                       210*degrees - alpha,
                       330*degrees + alpha]
        # Apply rotation and save the results
        self._spiderStartPos = []
        self._spiderAngles = []
        for pos, angle in zip(unrotStartPos, unrotAngles):
            self._spiderStartPos.append(np.dot(rot, pos))
            self._spiderAngles.append(angle - hra)

    def _horizonRotAngle(self):
        """!Compute rotation angle of camera with respect to horizontal
        coordinates from self.visitInfo.

        @returns horizon rotation angle.
        
        observatory = self.visitInfo.getObservatory()
        lat = observatory.getLatitude()
        lon = observatory.getLongitude()
        radec = self.visitInfo.getBoresightRaDec()
        ra = radec.getRa()
        dec = radec.getDec()
        era = self.visitInfo.getEra()
        ha = (era + lon - ra).wrap()
        alt = self.visitInfo.getBoresightAzAlt().getLatitude()

        # parallactic angle
        sinParAng = (np.cos(lat.asRadians()) * np.sin(ha.asRadians()) /
                     np.cos(alt.asRadians()))
        cosParAng = np.sqrt(1 - sinParAng*sinParAng)
        if dec > lat:
            cosParAng = -cosParAng
        parAng = Angle(np.arctan2(sinParAng, cosParAng))

        bra = self.visitInfo.getBoresightRotAngle()
        #return (bra - parAng).wrap()
        """
        parAng = Angle(self.input_angle)
        return parAng.wrap()

    def getPupil(self, point):
        """!Calculate a Pupil at a given point in the focal plane.

        @param point  Point2D indicating focal plane coordinates.
        @returns      Pupil
        """
        #subaruRadius = self.telescopeDiameter/2
        # subaruRadius = (self.pupilSize/2)*self.illumminatedFrac
        subaruRadius = (self.pupilSize/2)*1

        hscFrac = self.hscFrac  # linear fraction
        # radius of PSF camera shadow in meters - deduced from Figure 9 in Smee et al. (2014)
        hscRadius = hscFrac * subaruRadius

        slitFrac = self.slitFrac 
        subaruSlit = slitFrac*subaruRadius
        # meters - current value is basically random. As I am not 
        #sure what is the size that I should be using (sic!) I am using fraction 
        strutFrac = self.strutFrac 
        subaruStrutThick = strutFrac*subaruRadius
        
        slitFrac_dy = self.slitFrac_dy 


        
        # See DM-8589 for more detailed description of following parameters
        # d(lensCenter)/d(theta) in meters per degree
        #lensRate = 0.0276 * 3600 / 128.9 * subaruRadius
        # d(cameraCenter)/d(theta) in meters per degree
        hscRate = 2.62 / 1000 * subaruRadius
        # Projected radius of lens obstruction in meters
        #lensRadius = subaruRadius * 138./128.98


        hscPlateScale = 380  
        thetaX = point.getX() * hscPlateScale 
        thetaY = point.getY() * hscPlateScale 

        pupil = self._fullPupil()
        # Cut out primary mirror exterior
        self._cutCircleExterior(pupil, (0.0, 0.0), subaruRadius)
        # Cut out camera shadow
        camX = thetaX * hscRate
        camY = thetaY * hscRate
        #self._cutCircleInterior(pupil, (camX, camY), hscRadius)
        self._cutSquare(pupil, (camX, camY), hscRadius,self.input_angle)
        # Cut outer edge where L1 is too small
        #lensX = thetaX * lensRate
        #lensY = thetaY * lensRate
        
        #No vignetting for the spectroscope 
        #self._cutCircleExterior(pupil, (lensX, lensY), lensRadius)
        # Cut out spider shadow
        for pos, angle in zip(self._spiderStartPos, self._spiderAngles):
            x = pos[0] + camX
            y = pos[1] + camY
            self._cutRay(pupil, (x, y), angle, subaruStrutThick)
            
        self._cutRay(pupil, (2,slitFrac_dy/18), Angle(-np.pi),subaruSlit) 
        #self._cutRay(pupil, (0.6,2), Angle(-np.pi/2),0.2*subaruSlit) 
        return pupil

class ZernikeFitter_PFS(object):
    
    """!Class to create  donut images in PFS
    The model is constructed using GalSim, and consists of the convolution of
    an OpticalPSF and an input fiber image.  The OpticalPSF part includes the
    specification of an arbitrary number of zernike wavefront aberrations. 
    The centroid parameters are also free parameters.
    Note that to create the images, the parameters must be initialized with the
    `.initParams` method.
    
    This code uses lmfit to initalize the parameters. This is a relic of the code
    in which this class was used to actually fit the parameters
    """
    def __init__(self, image=None,image_var=None,pixelScale=None,wavelength=None,
                 jacobian=None,diam_sic=None,npix=None,pupilExplicit=None,**kwargs):
        """
        @param image        image to analyze
        @param image_var    variance image
        @param pixelScale   pixel scale in arcseconds 
        @param jacobian     An optional 2x2 Jacobian distortion matrix to apply
                            to the forward model.  Note that this is relative to
                            the pixelScale above.  Default is the identity matrix.
        @param wavelength   wavelenght      
        @param npix         number of pixels describign the pupil
        @param illumminatedFrac ?
        @param pupilExplicit if you want to pass explicit image of the exit pupil
        """
        

        
        if image is None:
            image=np.ones((41,41))
            self.image = image
        else:
            self.image = image
            
        if image_var is None:
            image_var=np.ones((41,41))
            self.image_var=image_var
        else:
            self.image_var = image_var

        #flux has to be declared after image 
        flux = float(np.sum(image))
        self.flux=flux    
            
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float64)
        else:
            self.jacobian = jacobian
        
        if wavelength is None:
            wavelength=794
            self.wavelength=wavelength
        else:
            self.wavelength=wavelength       
        
        # This is scale for PFS red arm in focus
        if pixelScale is None:
            pixelScale=20.598753
            self.pixelScale=pixelScale
        else:
            self.pixelScale=pixelScale
        
        # This is scale for PFS red arm in focus, 10x oversampled
        if diam_sic is None:
            diam_sic=136.93774e-3
            self.diam_sic=diam_sic
        else:
            self.diam_sic=diam_sic
        
        if npix is None:
            npix=2048
            self.npix=npix
        else:
            self.npix=npix   
            
        if pupilExplicit is None:
            pupilExplicit==False
            self.pupilExplicit=pupilExplicit
        else:
            self.pupilExplicit=pupilExplicit
                     
        self.kwargs = kwargs
    
    def initParams(self, zmax=22, z4Init=None, dxInit=None,dyInit=None,hscFracInit=None,strutFracInit=None,
                   focalPlanePositionInit=None,fiber_rInit=None,
                  slitFracInit=None,slitFrac_dy_Init=None,apodizationInit=None,radiometricEffectInit=None,trace_valueInit=None,fiber_NA_effectInit=None,backgroundInit=None):
        """Initialize lmfit Parameters object.
        @param zmax                 Total number of Zernike aberrations used
        @param z4Init               Initial Z4 aberration value in waves (that is 2*np.pi*wavelengths).
        @param centroidInit         2-tuple for the position of the center of the spot in the image.
        @param hscFracInit          Fraction of the pupil obscured by the central obscuration(camera) 
        @param strutFracInit        Fraction of the pupil obscured by a single strut
        @param focalPlanePositionInit 2-tuple for position of the central obscuration(camera) in the focal plane
        @param diam_sicInit         Diameters of the telescope in meters
        @param fiber_rInit          Multiplicative factor that determines the size of the input fiber
        @param trace_valueInit      Which fraction of the flux spot goes in the trace
        """

        self.zmax=zmax
        params = lmfit.Parameters()

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
            params.add('hscFrac', hscFracInit)        

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
            
        if fiber_rInit is None:
            params.add('fiber_r', 1)
        else:
            params.add('fiber_r', fiber_rInit)  
    
        if slitFracInit is None:
            params.add('slitFrac', 0)
        else:
            params.add('slitFrac', slitFracInit)     
            
        if slitFrac_dy_Init is None:
            params.add('slitFrac_dy', 0)
        else:
            params.add('slitFrac_dy', slitFrac_dy_Init)   
            
        if apodizationInit is None:
            params.add('apodization', 10)
        else:
            params.add('apodization', apodizationInit)  
            
        if radiometricEffectInit is None:
            params.add('radiometricEffect', 0)
        else:
            params.add('radiometricEffect', radiometricEffectInit)    
            
        if trace_valueInit is None:
            params.add('trace_value', 0)
        else:
            params.add('trace_value', trace_valueInit)  

        if fiber_NA_effectInit is None:
            params.add('fiber_NA_effect', 1)
        else:
            params.add('fiber_NA_effect', fiber_NA_effectInit)   
            
        if backgroundInit is None:
            params.add('background', 0)
        else:
            params.add('background', backgroundInit)           
            
        nyquistscale=nyQuistScale(self.diam_sic,self.wavelength)
        self.nyquistscale=nyquistscale
        
        self.params = params
        
    def _getOptPsf(self,params):

        aberrations_init=[0.0,0,0.0,0.0]
        aberrations = aberrations_init

        diam_sic=self.diam_sic
        npix=self.npix
        illumminatedFrac=self.illumminatedFrac     
        
        for i in range(4, self.zmax + 1):
            aberrations.append(params['z{}'.format(i)])
            
        Pupil_Image=PFSPupilFactory(diam_sic,npix,
                                    np.pi/2,
                                  params['hscFrac'.format(i)],params['strutFrac'.format(i)],
                                  params['slitFrac'.format(i)],
                                    params['slitFrac_dy'.format(i)])
        point=Point2D(params['dxFocal'.format(i)],params['dyFocal'.format(i)])
        pupil=Pupil_Image.getPupil(point)
        
        # relic code if you want to see how the pupil looks
        #np.save('/Users/nevencaplar/Documents/PFS/FromZemax/pupil',pupil.illuminated.astype(np.int16))    
     
        #pupilW=np.load('/Users/nevencaplar/Documents/PFS/FromZemax/resZemaxNormalizedAll5.npy')
        #pupilW=np.transpose(pupilW)
        #aper = galsim.Aperture(
        #    diam = pupil.size,
        #    pupil_plane_im = pupil.illuminated.astype(np.int16),
        #    pupil_plane_scale = pupil.scale,
        #    pupil_plane_size = None)  
        
        # this code allows the user the specify the exact pupil 
        if self.pupilExplicit is None:
            aper = galsim.Aperture(
                diam = pupil.size,
                pupil_plane_im = pupil.illuminated.astype(np.int16),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None) 
        else:
            aper = galsim.Aperture(
                diam =  pupil.size,
                pupil_plane_im = self.pupilExplicit.astype(np.int16),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None)                 
            
        # pad_factor=2.0,oversampling=2 do nothing here as we are providing the pupil already
        big_fft_params = galsim.GSParams(maximum_fft_size=18240) 
        return galsim.OpticalPSF(lam=self.wavelength,diam=aper.diam, 
                                 aberrations = aberrations,aper=aper,gsparams=big_fft_params,pad_factor=2.0,oversampling=2)

    def _getOptPsf_naturalResolution(self,params):
        "returns array in natural resolution"
        aberrations_init=[0.0,0,0.0,0.0]
        aberrations = aberrations_init

        diam_sic=self.diam_sic
        npix=self.npix   

        for i in range(4, self.zmax + 1):
            aberrations.append(params['z{}'.format(i)])
            
        Pupil_Image=PFSPupilFactory(diam_sic,npix,
                                    np.pi/2,
                                  params['hscFrac'.format(i)],params['strutFrac'.format(i)],params['slitFrac'.format(i)],
                                    params['slitFrac_dy'.format(i)])
        point=Point2D(params['dxFocal'.format(i)],params['dyFocal'.format(i)])
        pupil=Pupil_Image.getPupil(point)
        
        if self.pupilExplicit is None:
            aper = galsim.Aperture(
                diam = pupil.size,
                pupil_plane_im = pupil.illuminated.astype(np.int16),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None) 
        else:
            aper = galsim.Aperture(
                diam =  pupil.size,
                pupil_plane_im = self.pupilExplicit.astype(np.int16),
                pupil_plane_scale = pupil.scale,
                pupil_plane_size = None)   
            
        optics_screen = galsim.OpticalScreen(diam=diam_sic,aberrations=aberrations,lam_0=self.wavelength)
        screens = galsim.PhaseScreenList(optics_screen)    
        
        ilum=np.array(aper.illuminated, dtype=np.float64)
        r = gaussian_filter(ilum, sigma=params['apodization'.format(i)])
        
        points = np.linspace(-2, 2,num=npix*2)
        xs, ys = np.meshgrid(points, points)
        r = np.sqrt(xs ** 2 + ys ** 2)
        radiometricEffectArray=(1-params['radiometricEffect']**2*r**2)**(1/4)
        
        ilum_radiometric=np.nan_to_num(radiometricEffectArray*ilum,0)
        
        r = gaussian_filter(ilum_radiometric, sigma=params['apodization'.format(i)])
        
        r_ilum_pre=np.copy(r)
        r_ilum_pre[r>0.01]=1
        r_ilum_pre[r<0.01]=0
        r_ilum=r_ilum_pre.astype(bool)

        u = aper.u[r_ilum]
        v = aper.v[r_ilum]

        wf = screens.wavefront(u, v, None, 0)
        wf_grid = np.zeros_like(r_ilum, dtype=np.float64)
        wf_grid[r_ilum] = (wf/self.wavelength)

        wf_grid_rot=wf_grid

        expwf_grid = np.zeros_like(r_ilum, dtype=np.complex128)
        expwf_grid[r_ilum] = r[r_ilum]*np.exp(2j*np.pi * wf_grid_rot[r_ilum])

        ftexpwf = galsim.fft.fft2(expwf_grid,shift_in=True,shift_out=True)
        img_apod = np.abs(ftexpwf)**2
        return img_apod
    
    def create_trace(self,best_img,norm_of_trace):
        # I messes around with the defintion - probably only works on rectangualr images
        data_shifted_right=np.zeros(np.shape(best_img))
        for shift in range(1,best_img.shape[0]):
            data_shifted = np.concatenate((norm_of_trace*best_img[shift:best_img.shape[0],:], np.zeros((shift,best_img.shape[1]))), 0) 
            data_shifted_right += data_shifted

        data_shifted_left=np.zeros(np.shape(best_img))
        for shift in range(1,best_img.shape[0]):
            data_shifted = np.concatenate((np.zeros((shift,best_img.shape[0])),norm_of_trace*best_img[0:best_img.shape[1]-shift,:]),0) 
            data_shifted_left += data_shifted
        
        #this is taken from the data
        norm_of_trace_up_down=norm_of_trace*10/22
        
        data_shifted_up=np.zeros(np.shape(best_img))
        for shift in range(1,best_img.shape[0]):
            data_shifted = np.concatenate((norm_of_trace_up_down*best_img[:,shift:best_img.shape[0]], np.zeros((best_img.shape[1],shift))), 1) 
            data_shifted_up += data_shifted

        data_shifted_down=np.zeros(np.shape(best_img))
        for shift in range(1,best_img.shape[0]):
            data_shifted = np.concatenate((np.zeros((best_img.shape[0],shift)),norm_of_trace_up_down*best_img[:,0:best_img.shape[1]-shift]),1) 
            data_shifted_down += data_shifted            
            
        return data_shifted_right+data_shifted_left+best_img+data_shifted_up+data_shifted_down
    
    def constructModelImage_PFS(self,params=None,shape=None,pixelScale=None,jacobian=None):
        """Construct model image from parameters
        @param params      lmfit.Parameters object or python dictionary with
                           param values to use, or None to use self.params
        @param pixelScale  pixel scale in arcseconds to use for model image,
                           or None to use self.pixelScale.
        @param jacobian    An optional 2x2 Jacobian distortion matrix to apply
                           to the forward model.  Note that this is relative to
                           the pixelScale above.  Use self.jacobian if this is
                           None.
        @param shape       (nx, ny) shape for model image, or None to use
                           the shape of self.maskedImage
        @returns           numpy array image with the same flux as the input image
        """
        
        if params is None:
            params = self.params
        if shape is None:
            shape = self.image.shape
        if pixelScale is None:
            pixelScale = self.pixelScale
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float64)
        else:
            jacobian = self.jacobian    
            
        try:
            v = params.valuesdict()
        except AttributeError:
            v = params

        #This creates opticalpsf
        optPsf=self._getOptPsf(v)
        optPsf = optPsf.shift(pixelScale*v['dx'], pixelScale*v['dy'])
        print(optPsf)
        wcs = galsim.JacobianWCS(*list(pixelScale*jacobian.ravel()))

        # fiber smearing here
        fiber = galsim.TopHat(flux=1, radius=pixelScale*1/2*v['fiber_r'])
        optPsfConvolvedwithFiber= galsim.Convolve([optPsf,fiber])

        # create image
        modelImg = optPsfConvolvedwithFiber.drawImage(nx = shape[0],ny = shape[1],wcs = wcs)
        
        # adding trace
        print(modelImg)
        return_img_before_trace=modelImg.array
        return_img_after_trace=self.create_trace(return_img_before_trace,v['trace_value'])

        # normalizing flux in the image
        multiplicative_factor=np.sum(self.image)/np.sum(return_img_after_trace)
        finalImg=multiplicative_factor*return_img_after_trace
        
        return finalImg

    def constructModelImage_PFS_naturalResolution(self,params=None,shape=None,pixelScale=None,jacobian=None):
        """Construct model image from parameters
        @param params      lmfit.Parameters object or python dictionary with
                           param values to use, or None to use self.params
        @param pixelScale  pixel scale in arcseconds to use for model image,
                           or None to use self.pixelScale.
        @param jacobian    An optional 2x2 Jacobian distortion matrix to apply
                           to the forward model.  Note that this is relative to
                           the pixelScale above.  Use self.jacobian if this is
                           None.
        @param shape       (nx, ny) shape for model image, or None to use
                           the shape of self.maskedImage
        @returns           numpy array image with the same flux as the input image
        """
        
        if params is None:
            params = self.params
        if shape is None:
            shape = self.image.shape
        if pixelScale is None:
            pixelScale = self.pixelScale
        if jacobian is None:
            jacobian = np.eye(2, dtype=np.float64)
        else:
            jacobian = self.jacobian    
            
        try:
            v = params.valuesdict()
        except AttributeError:
            v = params
        
        # This give image in nyquist resolution
        optPsf=self._getOptPsf_naturalResolution(v)
        
        
        # what is that scale
        nyquistscale=self.nyquistscale
        oversampling=self.pixelScale/nyquistscale
        #print('nyquistscale: '+str(nyquistscale))
        # how big does the image have to be 
        SizeOfImageBeforeResizing=int(round(self.pixelScale/nyquistscale*len(self.image))) 
        
        # moving the image to the center and cutting the image to the size needed for resizing       
        #find_centroid_of_sci_image=map(lambda x: x/len(self.image), find_centroid_of_flux(self.image))
        #optPsf_cut=find_Centroid_of_natural_resolution_image(optPsf,SizeOfImageBeforeResizing,find_centroid_of_sci_image)
        
        optPsf_cut=cut_Centroid_of_natural_resolution_image(optPsf,SizeOfImageBeforeResizing,oversampling,v['dx'],v['dy'])
        # fiber smearing here       
        # first (re)compute what is one pixel of the decetor in the natural scale
        SinglePixelResizing=self.pixelScale/nyquistscale
        
        #pixels are not perfect, sigma is around 7 microns Jim claims
        optPsf_cut_pixel_response_convolved=scipy.signal.fftconvolve(optPsf_cut, Gaussian2DKernel(SinglePixelResizing*7/15).array, mode = 'full')
        
        #create fiber kernel
        fiber = astropy.convolution.Tophat2DKernel(SinglePixelResizing*2*v['fiber_r']).array

        # fiber convolved to accomodate possible numerical aperature effects
        if v['fiber_NA_effect']==0:
            fiber_convolved=fiber
        else:
            fiber_convolved=scipy.signal.fftconvolve(fiber, Gaussian2DKernel(SinglePixelResizing*2*v['fiber_r']*v['fiber_NA_effect']).array, mode = 'full')
            
        optPsf_cut_fiber_convolved=scipy.signal.fftconvolve(optPsf_cut_pixel_response_convolved, fiber_convolved, mode = 'same')
        
        # downsampling image 
        optPsf_cut_fiber_convolved_downsampled=skimage.transform.resize(optPsf_cut_fiber_convolved,(shape[0],shape[1]),order=5)
            
        # adding trace
        optPsf_cut_fiber_convolved_downsampled_trace=self.create_trace(optPsf_cut_fiber_convolved_downsampled,v['trace_value'])

        # normalizing flux in the image
        multiplicative_factor=np.sum(self.image-v['background'])/np.sum(optPsf_cut_fiber_convolved_downsampled_trace)
        finalImg=multiplicative_factor*optPsf_cut_fiber_convolved_downsampled_trace+v['background']
        
        return finalImg
    
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

class LN_PFS_single(object):
        
    def __init__(self,sci_image,var_image):    
                
        #print("Checkpoint 1")      
        
        DATA_FOLDER='/tigress/ncaplar/Data/JanData/'
        self.DATA_FOLDER=DATA_FOLDER
        RESULT_FOLDER='/Users/nevencaplar/Documents/PFS/TigerAnalysis/Results/'
        self.RESULT_FOLDER=RESULT_FOLDER
        NAME_OF_CHAIN='NAME_OF_CHAIN'
        self.NAME_OF_CHAIN=NAME_OF_CHAIN
        
        self.columns=['z4','z5','z6','z7','z8','z9','z10','z11',
                      'z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
                      'hscFrac','strutFrac','dxFocal','dyFocal',
                      'fiber_r','fiber_NA_effect','slitFrac','slitFrac_dy','trace_value','radiometricEffect','apodization',
                      'dx','dy','background']        

        #self.columns=['z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22',
        #         'hscFrac','strutFrac','dxFocal','dyFocal','fiber_r','fiber_NA_effect','slitFrac','slitFrac_dy','trace_value']
        
        self.sci_image=sci_image
        self.var_image=var_image
        zmax=22
        
        single_image_analysis=ZernikeFitter_PFS(sci_image,var_image,npix=2048)
        single_image_analysis.initParams(zmax) 
        self.single_image_analysis=single_image_analysis

    def create_chi_2_almost(self,modelImg,sci_image,var_image):
       
        # this creates most normal chi2
        sigma = np.sqrt(var_image)
        chi = (sci_image - modelImg)/sigma
        chi2_intrinsic=np.sum(sci_image**2/var_image)
        
        chi_without_nan=[]
        chi_without_nan = chi.ravel()[~np.isnan(chi.ravel())]
        res=(chi_without_nan)**2
        
        # calculates q vaues
        Qlist=np.abs((sci_image - modelImg))
        Qlist_without_nan=Qlist.ravel()[~np.isnan(Qlist.ravel())]
        sci_image_without_nan=sci_image.ravel()[~np.isnan(sci_image.ravel())]
        Qvalue = np.sum(Qlist_without_nan)/np.sum(sci_image_without_nan)
        
        return [np.sum(res),chi2_intrinsic,Qvalue]
    
    def lnlike_Neven(self,allparameters):
        
        # this is now optimized to go to z22
        #print(len(allparameters))
        zparameters=allparameters[0:8+11]
        globalparameters=allparameters[8+11:(8+11+14)]

         # hsc frac
        if globalparameters[0]<=0.5:
            return -np.inf
        if globalparameters[0]>1.0:
            return -np.inf
        
         #strut frac
        if globalparameters[1]<0:
            return -np.inf
        if globalparameters[1]>0.2:
            return -np.inf
        
         #dx Focal
        if globalparameters[2]>0.8:
            return -np.inf
        if globalparameters[2]<-0.8:
            return -np.inf
        
        # dy Focal
        if globalparameters[3]>0.8:
            return -np.inf
        if globalparameters[3]<-0.8:
            return -np.inf
        
        # fiber_r
        if globalparameters[4]<0.2:
            return -np.inf
        if globalparameters[4]>1.5:
            return -np.inf
       
        #fiber_NA_effect
        if globalparameters[5]<0:
            return -np.inf
        if globalparameters[5]>1:
            return -np.inf
        
        # slit_frac
        if globalparameters[6]<0:
            return -np.inf
        if globalparameters[6]>0.2:
            return -np.inf  
        
        # slitFrac_dy
        if globalparameters[7]<-0.5:
            return -np.inf
        if globalparameters[7]>0.5:
            return -np.inf 
        
        # trace_value
        if globalparameters[8]<0:
            return -np.inf
        if globalparameters[8]>0.02:
            return -np.inf

        # radiometricEffect
        if globalparameters[9]<0:
            return -np.inf
        if globalparameters[9]>2:
            return -np.inf      
        
        # apodization
        if globalparameters[10]<0:
            return -np.inf
        if globalparameters[10]>20:
            return -np.inf      
        
        # dx
        if globalparameters[11]<-5:
            return -np.inf
        if globalparameters[11]>+5:
            return -np.inf      
        
        # dy
        if globalparameters[12]<-5:
            return -np.inf
        if globalparameters[12]>+5:
            return -np.inf     
        
        # background
        if globalparameters[13]<-100:
            return -np.inf
        if globalparameters[13]>+100:
            return -np.inf  
        
        x=self.create_x(zparameters,globalparameters)
        for i in range(len(self.columns)):
            self.single_image_analysis.params[self.columns[i]].set(x[i])
        modelImg = self.single_image_analysis.constructModelImage_PFS_naturalResolution(self.single_image_analysis.params)  
        
        #np.save(RESULT_FOLDER+NAME_OF_CHAIN+'x',x)
        #np.save(RESULT_FOLDER+NAME_OF_CHAIN+'modelImg',modelImg)            
        #np.save(self.RESULT_FOLDER+self.NAME_OF_CHAIN+'zparameters',zparameters)
        #np.save(self.RESULT_FOLDER+self.NAME_OF_CHAIN+'globalparameters',globalparameters)       
        #np.save(self.RESULT_FOLDER+'modelImg',modelImg)

        chi_2_almost_multi_values=self.create_chi_2_almost(modelImg,self.sci_image,self.var_image)
        chi_2_almost=chi_2_almost_multi_values[0]
        chi_2_max=chi_2_almost_multi_values[1]     
        chi_2_Q=chi_2_almost_multi_values[2]


        chi_2_almost_reduced=chi_2_almost/(self.sci_image.shape[0]**2)
        #print(chi_2_almost)
        #print(chi_2_almost_reduced)
        #print(chi_2_Q)

        res=-(1/2)*(chi_2_almost+np.log(2*np.pi*np.sum(self.var_image)))
        return res

    def create_x(self,zparameters,globalparameters):
        #for single case, up to z11
        x=np.zeros((21+12))
        for i in range(len(zparameters)):
            x[i]=zparameters[i]     

        #hscFrac
        x[8+11]=globalparameters[0] 

        #strutFrac
        x[9+11]=globalparameters[1] 

        #dxFocal
        x[10+11]=globalparameters[2] 

        #dyFocal
        x[11+11]=globalparameters[3] 

        #fiber_r
        x[12+11]=globalparameters[4]

        #fiber_NA_effect
        x[13+11]=globalparameters[5]  

        #slitFrac
        x[14+11]=globalparameters[6]  

        #slitFrac_dy
        x[15+11]=globalparameters[7]  

        #trace_value
        x[16+11]=globalparameters[8] 
        
        #radiometricEffect
        x[17+11]=globalparameters[9]  

        #apodization
        x[18+11]=globalparameters[10] 
        
        #dx
        x[19+11]=globalparameters[11] 

        #dy
        x[20+11]=globalparameters[12] 

        #background
        x[21+11]=globalparameters[13]  
        
        return x       
     
    def __call__(self, allparameters):
        return self.lnlike_Neven(allparameters)

class LNP_PFS(object):
        
    def __call__(self, allparameters):
        return 0.0
    
#free definitions below
def create_parInit(allparameters_proposal=None):
    #if zmax=11
    number_of_par=21+12
    walkers_mult=12
    nwalkers=number_of_par*walkers_mult
    
    zparameters_flatten=allparameters_proposal[0:8+11]
    globalparameters_flatten=allparameters_proposal[8+11:(8+11+14)]
    
    try: 
        for i in range(8+11):
            zparameters_flat_single_par=np.random.normal(zparameters_flatten[i],0.05,nwalkers)
            if i==0:
                zparameters_flat=zparameters_flat_single_par
            else:
                zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
    except NameError:
        for i in range(8+11):
            zparameters_flat_single_par=np.random.normal(0,0.05,nwalkers)
            if i==0:
                zparameters_flat=zparameters_flat_single_par
            else:
                zparameters_flat=np.column_stack((zparameters_flat,zparameters_flat_single_par))
        
        
    #print(zparameters_flat.shape)       
    
    try:
        # hscFrac always positive
        globalparameters_flat_0=np.abs(np.random.normal(globalparameters_flatten[0],0.1,nwalkers))
        # strutFrac always positive
        globalparameters_flat_1=np.abs(np.random.normal(globalparameters_flatten[1],0.1,nwalkers))
        # dyFocal
        globalparameters_flat_2=np.random.normal(globalparameters_flatten[2],0.1,nwalkers)
        # dyFocal
        globalparameters_flat_3=np.random.normal(globalparameters_flatten[3],0.1,nwalkers)
        # fiber_r always positive
        globalparameters_flat_4=np.abs(np.random.normal(globalparameters_flatten[4],0.1,nwalkers))
        # fiber_NA_effect
        globalparameters_flat_5=np.abs(np.random.normal(globalparameters_flatten[5],0.1,nwalkers))        
        # slit Frac always postive
        globalparameters_flat_6=np.abs(np.random.normal(globalparameters_flatten[6],0.1,nwalkers))
        # slit Frac_dy
        globalparameters_flat_7=np.random.normal(globalparameters_flatten[7],0.05,nwalkers)
        #special treatment for trace value as when it goes below 0 it destoys the code
        globalparameters_flat_8=np.abs(np.random.normal(globalparameters_flatten[8],globalparameters_flatten[8]/10,nwalkers))
        # radiometricEffect
        globalparameters_flat_9=np.random.normal(globalparameters_flatten[9],0.1,nwalkers)
        # apodization
        globalparameters_flat_10=np.abs(np.random.normal(globalparameters_flatten[10],1,nwalkers))
        # dx
        globalparameters_flat_11=np.random.uniform(-1,1,nwalkers)
        # dy
        globalparameters_flat_12=np.random.uniform(-1,1,nwalkers)
        # background
        globalparameters_flat_13=np.random.normal(globalparameters_flatten[13],5,nwalkers)        
        globalparameters_flat=np.column_stack((globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3
                                               ,globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                              globalparameters_flat_8,globalparameters_flat_9,globalparameters_flat_10,globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13))
    except NameError:
        globalparameters_flat_0=np.random.normal(0.8,0.1,nwalkers)
        globalparameters_flat_1=np.random.normal(0.2,0.1,nwalkers)
        globalparameters_flat_2=np.random.normal(0,0.1,nwalkers)
        globalparameters_flat_3=np.random.normal(0,0.1,nwalkers)
        globalparameters_flat_4=np.random.normal(1,0.1,nwalkers)
        globalparameters_flat_5=np.random.normal(0.1,0.1,nwalkers)
        globalparameters_flat_6=np.random.normal(0,0.1,nwalkers)
        globalparameters_flat_7=np.random.normal(0.2,0.05,nwalkers)
        globalparameters_flat_8=np.random.normal(0.01,0.005,nwalkers)
        globalparameters_flat_9=np.random.normal(0.01,0.005,nwalkers)
        globalparameters_flat_10=np.random.normal(0.01,0.005,nwalkers)
        globalparameters_flat_11=np.random.normal(0,0.05,nwalkers)
        globalparameters_flat_12=np.random.normal(0,0.05,nwalkers)
        globalparameters_flat_13=np.random.normal(0,5,nwalkers)
        globalparameters_flat=np.column_stack((globalparameters_flat_0,globalparameters_flat_1,globalparameters_flat_2,globalparameters_flat_3
                                               ,globalparameters_flat_4,globalparameters_flat_5,globalparameters_flat_6,globalparameters_flat_7,
                                              globalparameters_flat_8, globalparameters_flat_9, globalparameters_flat_10,globalparameters_flat_11,globalparameters_flat_12,globalparameters_flat_13))
    
    #print(globalparameters_flat.shape) 
    
    allparameters=np.column_stack((zparameters_flat,globalparameters_flat))
    
    parInit=allparameters.reshape(nwalkers,number_of_par) 
    
    return parInit







def find_centroid_of_flux(image):
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
#old version
def find_Centroid_of_natural_resolution_image(image,size_natural_resolution,centroid_of_sci_image):
    "input: image in natural units, how many natural units to create new image, centroid of target image"
    I_x=[]
    for i in range(len(image)):
        I_x.append([i,np.sum(image[:,i])])
    I_x=np.array(I_x)
    
    # 512 below is stop gap solution for now
    x_center_array=[]
    for x_index in range(int(len(I_x)/2)-512,int(len(I_x)/2)+512):
        x_center=(np.sum(I_x[0:0+size_natural_resolution,0]*I_x[x_index:x_index+size_natural_resolution,1])/np.sum(I_x[x_index:x_index+size_natural_resolution,1]))
        x_center_array.append([x_index,x_center/size_natural_resolution])

    x_center_array=np.array(x_center_array)

    I_y=[]
    for i in range(len(image)):
        I_y.append([i,np.sum(image[i])])
    I_y=np.array(I_y)

    y_center_array=[]
    for y_index in range(int(len(I_y)/2)-512,int(len(I_y)/2)+512):
        y_center=(np.sum(I_y[0:0+size_natural_resolution,0]*I_y[y_index:y_index+size_natural_resolution,1])/np.sum(I_y[y_index:y_index+size_natural_resolution,1]))
        y_center_array.append([y_index,y_center/size_natural_resolution])

    y_center_array=np.array(y_center_array)
    
    #int(len(I_x)/2-size_natural_resolution) this factors takes only central part of images - perhaps we can do a better job
    positions_from_where_to_start_cut=[int(len(I_x)/2)-512+find_nearest(x_center_array[:,1],centroid_of_sci_image[0]),int(len(I_y)/2)-512+find_nearest(y_center_array[:,1],centroid_of_sci_image[1])]
    
    return image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1]+size_natural_resolution,
                 positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0]+size_natural_resolution]


def cut_Centroid_of_natural_resolution_image(image,size_natural_resolution,oversampling,dx,dy):
    "input: image in natural units, how many natural units to create new image, what is oversampling factor, dx and dy movement"

    #int(len(I_x)/2-size_natural_resolution) this factors takes only central part of images - perhaps we can do a better job
    positions_from_where_to_start_cut=[int(len(image)/2-size_natural_resolution/2-dx*oversampling),
                                       int(len(image)/2-size_natural_resolution/2-dy*oversampling)]

    res=image[positions_from_where_to_start_cut[1]:positions_from_where_to_start_cut[1]+size_natural_resolution,
                 positions_from_where_to_start_cut[0]:positions_from_where_to_start_cut[0]+size_natural_resolution]
    
    return res

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def nyQuistScale(diam,lamda):
    "gives nyquist scale in arcsec, provide diam(exit pupil size) in meters and lamda in nanometers, calculated with 2048 pixels"
    multiplicativefactor=0.474674269198*(136.88e-3)/(630*10**-9)
    return multiplicativefactor*(lamda*10**-9)/diam

def Exit_pupil_size(x):
    return (656.0581848529348+ 0.7485514705882259*x)*10**-3

# after fitting in Wolfram Mathematica function for f number as a function of position of the slit
def F_number_size(x):
    return -1.178009972058799 - 0.00328027941176467* x

# after fitting in Wolfram Mathematica function for pixelsize in arcsec  as a function of position of the slit
def Pixel_size(x):
    return 206265 *np.arctan(0.015/((-1.178009972058799 - 0.00328027941176467* x)*(656.0581848529348 + 0.7485514705882259* x)))

############################################################################################################
#Start analysis here
############################################################################################################

if __name__ == "__main__":
         
         
    pool = MPIPool(loadbalance=True)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)   
    
    print('Name of machine is '+socket.gethostname())
    print('Start time is: '+time.ctime())  
    time_start=time.time()  

    DATA_FOLDER='/tigress/ncaplar/Data/JanData/'
    RESULT_FOLDER='/tigress/ncaplar/Results/'

    #name of the spot which we will analyze

    obs= sys.argv[1]
    xycoordinates=sys.argv[2]
    single_number= xycoordinates
	
    sci_image =np.load(DATA_FOLDER+'sci'+str(obs)+str(single_number)+'.npy')
    var_image =np.load(DATA_FOLDER+'var'+str(obs)+str(single_number)+'.npy')

    NAME_OF_CHAIN='chainApril14MPI_PT_z22_'+str(obs)+str(xycoordinates)
    NAME_OF_FLATCHAIN='flatchainApril14MPI_PT_z22_'+str(obs)+str(xycoordinates)
    NAME_OF_LIKELIHOOD_CHAIN='likechainApril14MPI_PT_z22_'+str(obs)+str(xycoordinates)
    NAME_OF_LIKELIHOOD_FLATCHAIN='likeflatchainApril14MPI_PT_z22_'+str(obs)+str(xycoordinates)
    NAME_OF_PROBABILITY_CHAIN='probchainApril14MPI_PT_z22_'+str(obs)+str(xycoordinates)

    # how many parallel processes to run - not used - really?
    nthreads=int(sys.argv[4])
    # number of steps each walkers will take 
    nsteps = int(sys.argv[3])


    # Telescope size
    diam_sic=Exit_pupil_size(-693.5)
    
    # This is 'normal' orientation of the system
    obs_int=int(obs)
    if obs_int==8561:
        z4Input=15
    if obs_int==8564:
        z4Input=12
    if obs_int==8567:
        z4Input=9
    if obs_int==8570:
        z4Input=6
    if obs_int==8573:
        z4Input=3
        
    if obs_int==8603:
        z4Input=0  
    
    if obs_int==8606:
        z4Input=-3
    if obs_int==8609:
        z4Input=-6
    if obs_int==8612:
        z4Input=-9
    if obs_int==8615:
        z4Input=-12
    if obs_int==8618:
        z4Input=-15        

    #allparameters_proposal=np.array([z4Input,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
   #                                  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    #                                 0.69,0.05,0.0,0.0,0.7,0.4,0.01,0,0.002,0.5,10,0,0,0])
    
    allparameters_proposal=np.array([z4Input,-0.15,-0.17,0.017,0.07,-0.049,-0.07,0.009,
                                     0.025,0.12,-0.035,-0.0065,-0.068,0.007,-0.04,-0.05,-0.059,-0.1,0.32,
                                     0.767,0.08,-0.18,0.06,1.12,0.14,0.09,0.21,0.0013,0.53,1.4,-0.075,-0.386,0])
    nT=4
    parInit1=create_parInit(allparameters_proposal)
    parInit2=create_parInit(allparameters_proposal)
    parInit3=create_parInit(allparameters_proposal)
    parInit4=create_parInit(allparameters_proposal)
    parInitnT=np.array([parInit1,parInit2,parInit3,parInit4])

    zmax=22
    zmax_input=zmax
    print('Spot coordinates are: '+str(xycoordinates))  
    print('steps: '+str(nsteps)) 
    print('name: '+str(NAME_OF_CHAIN)) 
    
    print('Starting calculation at: '+time.ctime())

    model = LN_PFS_single(sci_image,var_image)
    modelP =LNP_PFS()
    print(allparameters_proposal)
    print(model(allparameters_proposal))
    print(parInit1[0])
    print(model(parInit1[0]))

    sampler = emcee.PTSampler(parInitnT.shape[0],parInitnT.shape[1], parInitnT.shape[2], model,modelP,
                    pool=pool)
    
    for i, result in enumerate(sampler.sample(parInitnT, iterations=nsteps)):
        print('z22: '+"{0:5.1%}\r".format(float(i+1) / nsteps)+' time '+str(time.time())),     
    
    #Export this file
    np.save(RESULT_FOLDER+NAME_OF_CHAIN,sampler.chain)
    np.save(RESULT_FOLDER+NAME_OF_FLATCHAIN,sampler.flatchain)
    #np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN,sampler.lnprobability) 
    #np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_FLATCHAIN,sampler.flatlnprobability) 
    np.save(RESULT_FOLDER+NAME_OF_LIKELIHOOD_CHAIN,sampler.lnlikelihood) 
    np.save(RESULT_FOLDER+NAME_OF_PROBABILITY_CHAIN,sampler.lnprobability)    
    
    
               
    print('Time when z22 par finished was: '+time.ctime())     
    time_end=time.time()   
    print('Time taken for z22 part was '+str(time_end-time_start)+' seconds')
     
    pool.close()
    sys.exit(0)