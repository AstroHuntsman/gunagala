import numpy as np
from scipy.interpolate import RectSphereBivariateSpline, SmoothBivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import poisson, norm, lognorm
from scipy.special import eval_chebyt
import astropy.io.fits as fits
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, get_sun, Angle
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table
import ccdproc


class Imager:
    """
    Class representing an imaging instrument.
    """
    def __init__(self, npix_x, npix_y, pixel_scale, aperture_area, throughput, filters, QE, gain, read_noise, temperature, zl):

        self.pixel_scale = pixel_scale

        # Construct a simple template WCS to store the focal plane configuration parameters
        self.wcs = WCS(naxis=2)
        self.wcs._naxis1 = npix_x
        self.wcs._naxis2 = npix_y
        self.wcs.wcs.crpix = [(npix_x + 1)/2, (npix_y + 1)/2]
        self.wcs.wcs.cdelt = [pixel_scale.to(u.degree).value, pixel_scale.to(u.degree).value]
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        # Store throughput related parameters
        self.aperture_area = aperture_area
        self.throughput = throughput
        self.filters = filters
        self.QE = QE

        self.gain = gain
        self.read_noise = read_noise

        self.zl = zl

        # Pre-calculate effective aperture areas. pivot wavelengths and sensitivity integrals
        self._eff_areas = self._effective_areas()
        self._pivot_waves = self._pivot_wavelengths()
        self._sensitivities = self._sensitivity_integral()

        # Pre-calculate normalisation for observed ZL
        self._zl_ep = self._zl_obs_ep()

        # Precalculate dark frame
        self.dark_current, self.dark_frame = self._make_dark_frame(temperature)



    def _effective_areas(self):
        """
        Utility function to calculate the effective aperture area of for each filter as a function of
        wavelength, i.e. aperture area * optical throughput * image sensor QE
        """
        eff_areas = {}

        for (f_name, f_data) in self.filters.items():
            # Interpolate throughput data at same wavelengths as filter transmission data
            t = np.interp(f_data['Wavelength'], self.throughput['Wavelength'], \
                          self.throughput['Throughput']) *  self.throughput['Throughput'].unit
            # Interpolate QE data at same wavelengths as filter transmission data
            q = np.interp(f_data['Wavelength'], self.QE['Wavelength'], self.QE['QE']) *  self.QE['QE'].unit
            eff_areas[f_name] = Table(names = ('Wavelength', 'Effective Area'), \
                                      data = (f_data['Wavelength'], self.aperture_area * t * f_data['Transmission'] * q))

        return eff_areas

    def _pivot_wavelengths(self):
        """
        Utility function to calculate the pivot wavelengths for each of the filters using the
        effective area data (must be pre-calculated before calling this function).
        """
        pivot_waves = {}

        # Generally this is definied in terms of system efficiency instead of aperture
        # effective aperture area but they're equivalent as the aperture area factor
        # cancels.
        for (f_name, eff_data) in self._eff_areas.items():
            p1 = np.trapz(eff_data['Effective Area'] * eff_data['Wavelength'], x=eff_data['Wavelength'])
            p2 = np.trapz(eff_data['Effective Area'] / eff_data['Wavelength'], x=eff_data['Wavelength'])
            pivot_waves[f_name] = (p1/p2)**0.5 * eff_data['Wavelength'].unit

        return pivot_waves

    def _sensitivity_integral(self):
        """
        Utility function to calculate the sensitivity integral for each of the filters,
        i.e. the factor to convert a constant spectral flux density in F_lambda units
        to a count rate in electrons per second.
        """
        # Need to make sure units get preserved here.
        sensitivities = {}

        for (f_name, eff_data) in self._eff_areas.items():
            s = np.trapz(eff_data['Wavelength'] * eff_data['Effective Area'], x=eff_data['Wavelength'])
            s = s * eff_data['Effective Area'].unit * eff_data['Wavelength'].unit**2 * u.photon / (c.h * c.c)
            sensitivities[f_name] = s.to((u.electron/u.second) / (u.Watt / (u.m**2 * u.micron)))

        return sensitivities

    def _zl_obs_ep(self):
        """
        Utility function to pre-calculate observed ecliptic pole zodiacal light count rates
        """
        # Integrate product of zodiacal light photon SFD and effective aperture area
        # over wavelength to get observed ecliptic pole surface brightness for each filter.
        # Note, these are constant with time so can and should precalculate once.
        zl_ep = {}

        for f in self.filters.keys():
            electrons = np.zeros((self.wcs._naxis2, self.wcs._naxis1)) * u.electron / u.second
            eff_area_interp = np.interp(self.zl.waves, self._eff_areas[f]['Wavelength'], \
                                        self._eff_areas[f]['Effective Area']) * \
                                        self._eff_areas[f]['Effective Area'].unit
            zl_ep[f] = (np.trapz(self.zl.photon_sfd * eff_area_interp, x=self.zl.waves))

        return zl_ep
