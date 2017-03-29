import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import eval_chebyt

from astropy import units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from .utils import ensure_unit


data_dir = 'data/performance_data'


class Filter:

    def __init__(self,
                 transmission_filename=None,
                 chebyshev_params=None,
                 butterworth_params=None,
                 sky_mu=None,
                 *kwargs):
        """Class representing an optical bandpass filter. The filter bandpass can be defined either by a table of
        transmission versus wavelength data or by one of the included analytic functions: Butterworth function or
        Chebyshev Type I.

        Args:
            transmission_filename (string, optional): name of file containing transmission as a function of wavelength
                data. Must be in a format readable by `astropy.table.Table.read()` and use column names `Wavelength`
                and `Transmission`. If the data file does not provide units nm and dimensionless unscaled will be
                assumed.

            sky_mu (Quantity): the sky background surface brightness per arcsecond^2 (in ABmag units) for the band.
        """
        n_args = np.count_nonzero((transmission_filename, chebyshev_params, butterworth_params))
        if n_args != 1:
            raise ValueError("One and only one of `tranmission_filename`, `chebyshev_params` & `butterworth_params`"
                             + "must be specified, got {}!".format(n_args))

        if transmission_filename:
            transmission_data = Table.read(get_pkg_data_filename(os.path.join(data_dir, transmission_filename)))

            if not transmission_data['Wavelength'].unit:
                transmission_data['Wavelength'].unit = u.nm
            self.wavelengths = transmission_data['Wavelength'].quantity.to(u.nm)

            if not transmission_data['Transmission'].unit:
                transmission_data['Transmission'].unit = u.dimensionless_unscaled
            self._transmission = transmission_data['Transmission'].quantity.to(u.dimensionless_unscaled)

            # Create linear interpolator for calculating transmission at arbitrary wavelength
            self._interpolator = interp1d(self.wavelengths, self._transmission, kind='linear', fill_value='extrapolate')

        elif chebyshev_params:
            # Ensure all parameters are present (filling in defaults where necessary) and are the correct types/units
            wave1 = ensure_unit(chebyshev_params['wave1'], u.nm)
            wave2 = ensure_unit(chebyshev_params['wave2'], u.nm)
            order = int(chebyshev_params['order'])
            ripple = chebyshev_params.get('order', 1)
            peak = ensure_unit(chebyshev_params.get('peak', 0.95), u.dimensionless_unscaled)

            # Create a lambda function to calculate transmission at arbitrary wavelength.
            self._interpolator = lambda x: cheby_band(x, wave1, wave2, order, ripple, peak)

        elif butterworth_params:
            # Ensure all parameters are present (filling in defaults where necessary) and are the correct types/units
            wave1 = ensure_unit(butterworth_params['wave1'], u.nm)
            wave2 = ensure_unit(butterworth_params['wave2'], u.nm)
            order = int(butterworth_params['order'])
            peak = ensure_unit(butterworth_params.get('peak', 0.95), u.dimensionless_unscaled)

            # Create a lambda function to calculate transmission at arbitrary wavelength.
            self._interpolator = lambda x: butter_band(x, wave1, wave2, order, peak)

        self.sky_mu = ensure_unit(sky_mu, u.ABmag)

    def transmission(self, waves):
        """Return filter transmission at the given wavelength(s). For filter bandpasses defined by data tables this
        will interpolate/extrapolate as required while for filter bandpasses defined by analytic expressions it will
        be calculated directly.

        Args:
            waves (Quantity): wavelength(s) for which the filter transmission is required

        Returns:
            Quantity: filter transmission at the given wavelength(s)
        """
        waves = ensure_unit(waves, u.nm)
        trans = self._interpolator(waves)

        # Make sure interpolation/extrapolation doesn't take transmission unphysically below zero or above 1
        trans = np.where(trans > 0, trans, 0)
        trans = np.where(trans <= 1, trans, 1)

        # Put units back (interpolator doesn't work with Quantity)
        return ensure_unit(trans, u.dimensionless_unscaled)


def butter_band(w, w1, w2, N, peak=0.95):
    """
    Simple Butterworth bandpass filter function in wavelength space
    To be more realistic this should probably be a Chebyshev Type I function
    instead, and should definitely include cone angle effect but at f/5.34
    (Space Eye focal ratio) the latter at least is pretty insignficant
    """
    # Bandpass implemented as low pass and high pass in series
    g1 = np.sqrt(1 / (1 + (w1/w).to(u.dimensionless_unscaled)**(2*N)))
    g2 = np.sqrt(1 / (1 + (w/w2).to(u.dimensionless_unscaled)**(2*N)))
    return peak * g1 * g2


def cheby_band(w, w1, w2, N, ripple=1, peak=0.95):
    """
    Simple Chebyshev Type I bandpass filter function in wavelength space
    To be more realistic this should definitely include cone angle effect
    but at f/5.34 (Space Eye focal ratio) that is pretty insignficant
    """
    # Bandpass implemented as low pass and high pass in series
    g1 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w1/w).to(u.dimensionless_unscaled).value)**2)
    g2 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w/w2).to(u.dimensionless_unscaled).value)**2)
    return peak * g1 * g2
