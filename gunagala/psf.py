"""
Point spread functions
"""
import numpy as np

from astropy import units as u
from astropy.convolution import discretize_model
from astropy.modeling import Fittable2DModel
from astropy.modeling.functional_models import Moffat2D

from .utils import ensure_unit


class PSF(Fittable2DModel):

    def __init__(self, FWHM, pixel_scale=None, **kwargs):
        """
        Abstract base class representing a 2D point spread function.

        Used to calculate pixelated version of the PSF and associated parameters useful for
        point source signal to noise and saturation limit calculations.

        Args:
            FWHM (Quantity): Full Width at Half-Maximum of the PSF in angle on the sky units
            pixel_scale (Quantity, optional): pixel scale (angle/pixel) to use when calculating pixellated point
                spread functions or related parameters. Does not need to be set on object creation but must be set
                before before pixellation function can be used.
        """
        self._FWHM = ensure_unit(FWHM, u.arcsecond)

        if pixel_scale:
            self.pixel_scale = pixel_scale

        super().__init__(**kwargs)

    @property
    def FWHM(self):
        return self._FWHM

    @FWHM.setter
    def FWHM(self, FWHM):
        self._FWHM = ensure_unit(FWHM, u.arcsecond)
        # If a pixel scale has already been set should update model parameters when FWHM changes.
        if self.pixel_scale:
            self._update_model()

    @property
    def pixel_scale(self):
        try:
            return self._pixel_scale
        except AttributeError:
            return None

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale):
        self._pixel_scale = ensure_unit(pixel_scale, (u.arcsecond / u.pixel))
        # When pixel scale is set/changed need to update model parameters:
        self._update_model()

    @property
    def n_pix(self):
        try:
            return self._n_pix
        except AttributeError:
            return None

    @property
    def peak(self):
        try:
            return self._peak
        except AttributeError:
            return None

    def pixellated(self, pixel_scale=None, size=21, offsets=(0.0, 0.0)):
        """
        Calculates a pixellated version of the PSF for a given pixel scale
        """
        if not pixel_scale:
            pixel_scale = self.pixel_scale

        # Update PSF centre coordinates
        self.x_0 = offsets[0]
        self.y_0 = offsets[1]

        xrange = (-(size - 1) / 2, (size + 1) / 2)
        yrange = (-(size - 1) / 2, (size + 1) / 2)

        return discretize_model(self, xrange, yrange, mode='oversample', factor=10)

    def _get_peak(self, pixel_scale=None):
        """
        Calculate the peak pixel value (as a fraction of total counts) for a PSF centred
        on a pixel. This is useful for calculating saturation limits for point sources.
        """
        # Odd number of pixels (1) so offsets = (0, 0) is centred on a pixel
        centred_psf = self.pixellated(pixel_scale, 1, offsets=(0, 0))
        return centred_psf[0, 0] / u.pixel

    def _get_n_pix(self, pixel_scale=None, size=20):
        """
        Calculate the effective number of pixels for PSF fitting photometry with this
        PSF, in the worse case where the PSF is centred on the corner of a pixel.
        """
        # Want a even number of pixels.
        size = size + size % 2
        # Even number of pixels so offsets = (0, 0) is centred on pixel corners
        corner_psf = self.pixellated(pixel_scale, size, offsets=(0, 0))
        return 1 / ((corner_psf**2).sum()) * u.pixel

    def _update_model(self):
        raise NotImplementedError


class Moffat_PSF(PSF, Moffat2D):

    def __init__(self, model=None, shape=2.5, **kwargs):
        """
        Class representing a 2D Moffat profile point spread function.

        Used to calculate pixelated version of the PSF and associated parameters useful for
        point source signal to noise and saturation limit calculations.

        Args:
            FWHM (Quantity): Full Width at Half-Maximum of the PSF in angle on the sky units
            shape (optional, default 2.5): shape parameter of the Moffat function, must be > 1
            pixel_scale (Quantity, optional): pixel scale (angle/pixel) to use when calculating pixellated point
                spread functions or related parameters. Does not need to be set on object creation but must be set
                before before pixellation function can be used.

        Smaller values of the shape parameter correspond to 'wingier' profiles.
        A value of 4.765 would give the best fit to pure Kolmogorov atmospheric turbulence.
        When instrumental effects are added a lower value is appropriate.
        IRAF uses a default of 2.5.
        """
        if shape <= 1.0:
            raise ValueError('shape must be greater than 1!')

        super().__init__(alpha=shape, **kwargs)

    @property
    def shape(self):
        return self.alpha

    @shape.setter
    def shape(self, alpha):
        if shape <= 1.0:
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
