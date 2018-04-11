"""
Point spread functions
"""
import math

import numpy as np
from scipy import ndimage

from astropy import units as u
from astropy.convolution import discretize_model
from astropy.modeling import Fittable2DModel
from astropy.modeling.functional_models import Moffat2D

from gunagala import utils


class PSF():
    """
    Abstract base class representing a 2D point spread function.

    Used to calculate pixelated version of the PSF and associated
    parameters useful for point source signal to noise and saturation
    limit calculations.
    """
    @property
    def pixel_scale(self):
        """
        Pixel scale used when calculating pixellated point spread
        functions or related parameters.

        Returns
        -------
        pixel_scale : astropy.units.Quantity
            Pixel scale in angle on the sky per pixel units.
        """
        try:
            return self._pixel_scale
        except AttributeError:
            return None

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale):
        pixel_scale = utils.ensure_unit(pixel_scale, (u.arcsecond / u.pixel))
        if pixel_scale <= 0 * u.arcsecond / u.pixel:
            raise ValueError("Pixel scale should be > 0, got {}!".format(pixel_scale))
        else:
            self._pixel_scale = pixel_scale
        # When pixel scale is set/changed need to update model parameters:
        self._update_model()

    @property
    def n_pix(self):
        """
        The PSF's effective number of pixels for the worse case where the
        PSF is centred on the corner of a pixel.

        The effective number of pixels is for signal to noise calculations
        for PSF fitting photometry. The signal to noise for PSF fitting
        photometry is the same as if the signal were evenly distributed
        over this many pixels.

        Returns
        -------
        n_pix : astropy.units.Quantity
            Effective number of pixels
        """
        try:
            return self._n_pix
        except AttributeError:
            return None

    @property
    def peak(self):
        """
        The maximum fraction of the total signal that can fall in a single
        pixel.

        This is simply the central pixel value of the PSF when it is
        perfectly centred on a pixel centre. This is useful for saturation
        limit calculations.

        Returns
        -------
        peak : astropy.units.Quantity
            Maximum fraction of the total signal that can fall in a single
            pixel, in 1/pixel units.
        """
        try:
            return self._peak
        except AttributeError:
            return None

    def _get_peak(self):
        """
        Calculate the peak pixel value (as a fraction of total counts) for
        a PSF centred on a pixel. This is useful for calculating
        saturation limits for point sources.
        """
        # Odd number of pixels (1) so offsets = (0, 0) is centred on a pixel
        centred_psf = self.pixellated(size=1, offsets=(0, 0))
        return centred_psf[0, 0] / u.pixel

    def _get_n_pix(self, size=20):
        """
        Calculate the effective number of pixels for PSF fitting
        photometry with this PSF, in the worse case where the PSF is
        centred on the corner of a pixel.
        """
        # Want a even number of pixels.
        size = size + size % 2
        # Even number of pixels so offsets = (0, 0) is centred on pixel corners
        corner_psf = self.pixellated(size, offsets=(0, 0))
        return 1 / ((corner_psf**2).sum()) * u.pixel

    def _update_model(self):
        raise NotImplementedError


class FittablePSF(PSF, Fittable2DModel):
    """
    Base class representing a 2D point spread function based on a
    Fittable2DModel from astropy.modelling.

    Used to calculate pixelated version of the PSF and associated
    parameters useful for point source signal to noise and saturation
    limit calculations.

    Parameters
    ----------
    FWHM : astropy.units.Quantity
        Full Width at Half-Maximum of the PSF in angle on the sky units.
    pixel_scale : astropy.units.Quantity, optional
        Pixel scale (angle/pixel) to use when calculating pixellated point
        spread functions or related parameters. Does not need to be set on
        object creation but must be set before before pixellation function
        can be used.
    """
    def __init__(self, FWHM, pixel_scale=None, **kwargs):
        self._FWHM = utils.ensure_unit(FWHM, u.arcsecond)

        if pixel_scale is not None:
            self.pixel_scale = pixel_scale

        super().__init__(**kwargs)

    @property
    def FWHM(self):
        """
        Full Width at Half-Maximum of the PSF.

        Returns
        -------
        FWHM : astropy.units.Quantity
            Full Width at Half-Maximum in angle on the sky units.
        """
        return self._FWHM

    @FWHM.setter
    def FWHM(self, FWHM):
        FWHM = utils.ensure_unit(FWHM, u.arcsecond)
        if FWHM <= 0 * u.arcsecond:
            raise ValueError("FWHM should be > 0, got {}!".format(FWHM))
        else:
            self._FWHM = FWHM
        # If a pixel scale has already been set should update model parameters when FWHM changes.
        if self.pixel_scale:
            self._update_model()

    def pixellated(self, size=21, offsets=(0.0, 0.0)):
        """
        Calculates a pixellated version of the PSF.

        The pixel values are calculated using 10x oversampling, i.e. by
        evaluating the continuous PSF model at a 10 x 10 grid of positions
        within each pixel and averaging the results.

        Parameters
        ----------
        size : int, optional
            Size of the pixellated PSF to calculate, the returned image
            will have `size` x `size` pixels. Default value 21.
        offset : tuple of floats, optional
            y and x axis offsets of the centre of the PSF from the centre
            of the returned image, in pixels.

        Returns
        -------
        pixellated : numpy.array
            Pixellated PSF image with `size` by `size` pixels. The PSF
            is normalised to an integral of 1 however the sum of the
            pixellated PSF will be somewhat less due to truncation of the
            PSF wings by the edge of the image.
        """
        size = int(size)
        if size <= 0:
            raise ValueError("`size` must be > 0, got {}!".format(size))

        # Update PSF centre coordinates
        self.x_0 = offsets[0]
        self.y_0 = offsets[1]

        xrange = (-(size - 1) / 2, (size + 1) / 2)
        yrange = (-(size - 1) / 2, (size + 1) / 2)

        return discretize_model(self, xrange, yrange, mode='oversample', factor=10)


class MoffatPSF(FittablePSF, Moffat2D):
    """
    Class representing a 2D Moffat profile point spread function.

    Used to calculate pixelated version of the PSF and associated
    parameters useful for point source signal to noise and saturation
    limit calculations.

    Parameters
    ----------
    FWHM : astropy.units.Quantity
        Full Width at Half-Maximum of the PSF in angle on the sky units
    shape : float, optional
        Shape parameter of the Moffat function, must be > 1, default 2.5
    pixel_scale : astropy.units.Quantity, optional
        Pixel scale (angle/pixel) to use when calculating pixellated point
        spread functions or related parameters. Does not need to be set
        on object creation but must be set before before pixellation
        function can be used.

    Attributes
    ----------
    n_pix: astropy.units.Quantity
            Effective number of pixels

    Notes
    -----
    Smaller values of the shape parameter correspond to 'wingier'
    profiles. A value of 4.765 would give the best fit to pure Kolmogorov
    atmospheric turbulence. When instrumental effects are added a lower
    value is appropriate. IRAF uses a default of 2.5.
    """
    def __init__(self, model=None, shape=2.5, **kwargs):

        if shape <= 1.0:
            raise ValueError('shape must be greater than 1!')

        super().__init__(alpha=shape, **kwargs)

    @property
    def shape(self):
        """
        Shape parameter of the Moffat function, see Notes.

        Returns
        -------
        shape : float
            Shape parameter value.
        """
        return self.alpha

    @shape.setter
    def shape(self, alpha):
        if alpha <= 1.0:
            raise ValueError('shape must be greater than 1!')

        self.alpha = alpha
        # If a pixel scale has already been set should update model parameters when alpha changes.
        if self.pixel_scale:
            self._update_model()

    def _update_model(self):
        # Convert FWHM from arcsecond to pixel units
        self._FWHM_pix = self.FWHM / self.pixel_scale
        # Convert to FWHM to Moffat profile width parameter in pixel units
        gamma = self._FWHM_pix / (2 * np.sqrt(2**(1 / self.alpha) - 1))
        # Calculate amplitude required for normalised PSF
        amplitude = (self.alpha - 1) / (np.pi * gamma**2)
        # Update model parameters
        self.gamma = gamma.to(u.pixel).value
        self.amplitude = amplitude.to(u.pixel**-2).value

        self._n_pix = self._get_n_pix()
        self._peak = self._get_peak()


class PixellatedPSF(PSF):
    """
    Class representing a 2D point spread function based on an already
    pixellated data, e.g. a PSF calculated with optical design software.

    Used to calculate pixelated version of the PSF and associated
    parameters useful for point source signal to noise and saturation
    limit calculations.

    Parameters
    ----------

    psf_data: numpy.array
        Pixellated PSF data.
    psf_sampling: astropy.units.Quantity
        Pixel scale (angle/pixel) of psf_data.
    psf_centre: (float, float), optional
        Pixel coordinates of the PSF centre within psf_data (zero based, (y, x)).
        If not given psf_data.shape / 2 will be assumed.
    oversampling : integer, optional
        Oversampling factor used when shifting & resampling the PSF, default 10.
    pixel_scale : astropy.units.Quantity, optional
        Pixel scale (angle/pixel) to use when calculating pixellated point
        spread functions or related parameters. Does not need to be set on
        object creation but must be set before before pixellation function
        can be used.
    renormalise: bool, optional
        Whether to renormalise the PSF to a total of 1 during initialisation,
        default True. Only set to False if the psf_data is already correctly
        normalised.
    """
    def __init__(self,
                 psf_data,
                 psf_sampling,
                 psf_centre=None,
                 oversampling=10,
                 pixel_scale=None,
                 renormalise=True,
                 **kwargs):
        if renormalise:
            self._psf_data = psf_data / psf_data.sum()
        else:
            self._psf_data = psf_data
        self._psf_sampling = utils.ensure_unit(psf_sampling, u.arcsecond / u.pixel)
        if psf_centre is None:
            self._psf_centre = np.array(psf_data.shape) / 2
        else:
            self._psf_centre = np.array(psf_centre)
        self._oversampling = int(oversampling)

        if pixel_scale is not None:
            # This will also call _update_model()
            self.pixel_scale = pixel_scale


    def pixellated(self, size=21, offsets=(0.0, 0.0)):
        """
        Calculates a pixellated version of the PSF.

        The pixel values are calculated by shifting and resampling the
        original psf_data, then binning by the oversampling factor.

        Parameters
        ----------
        size : int, optional
            Size of the pixellated PSF to calculate, the returned image
            will have `size` x `size` pixels. Default value 21.
        offsets : tuple of floats, optional
            y and x axis offsets of the centre of the PSF from the centre
            of the returned image, in pixels.

        Returns
        -------
        pixellated : numpy.array
            Pixellated PSF image with `size` by `size` pixels. The PSF
            is normalised to an integral of 1 however the sum of the
            pixellated PSF will be somewhat less due to truncation of the
            PSF wings by the edge of the image.
        """
        resampled_size = size * self._oversampling
        # Arrays of pixel coordinates relative to array centre
        resampled_coordinates = np.mgrid[-(resampled_size - 1) / 2:resampled_size / 2,
                                         -(resampled_size - 1) / 2:resampled_size / 2]
        # Apply offsets so origin is at desired PSF centre
        resampled_offsets = np.reshape(np.array(offsets) * self._oversampling, (2, 1, 1))
        resampled_coordinates = resampled_coordinates - resampled_offsets
        # Convert from resampled PSF pixel units to original PSF pixel units
        resampled_coordinates = resampled_coordinates / self._resampling_factor
        # Add position of PSF centre in psf_data so origin is the same as the origin of psf_data
        resampled_coordinates = resampled_coordinates + np.reshape(self._psf_centre, (2, 1, 1))
        # Calculate resampled PSF using cubic spline interpolation
        resampled_psf = ndimage.map_coordinates(self._psf_data, resampled_coordinates)
        # Rebin to the correct pixel scale
        pixellated = utils.bin_array(resampled_psf, self._oversampling)
        # Renormalise to correct for the effect of resampling
        return pixellated / self._resampling_factor**2

    def _get_n_pix(self):
        # For accurate results want the calculation to include the whole PSF.
        size = int(math.ceil(max(self._psf_data.shape) * self._psf_sampling / self.pixel_scale))
        return super()._get_n_pix(size=size)

    def _update_model(self):
        self._resampled_scale = self.pixel_scale / self._oversampling
        self._resampling_factor = self._psf_sampling / self._resampled_scale

        self._n_pix = self._get_n_pix()
        self._peak = self._get_peak()
