import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brentq
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
                 *kwargs):
        """Class representing an optical bandpass filter. The filter bandpass can be defined either by a table of
        transmission versus wavelength data or by one of the included analytic functions: Butterworth function or
        Chebyshev Type I.

        Args:
            transmission_filename (string, optional): name of file containing transmission as a function of wavelength
                data. Must be in a format readable by `astropy.table.Table.read()` and use column names `Wavelength`
                and `Transmission`. If the data file does not provide units nm and dimensionless unscaled will be
                assumed.
            chebyshev_params (dict, optional):
            butterworth_params(dict, optional):
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
            ripple = chebyshev_params.get('ripple', 1)
            peak = ensure_unit(chebyshev_params.get('peak', 0.95), u.dimensionless_unscaled)

            # Create a lambda function to calculate transmission at arbitrary wavelength.
            scale = peak / cheby_band(w=(wave1 + wave2)/2, w1=wave1, w2=wave2, N=order, ripple=ripple, scale=1)
            self._interpolator = lambda x: cheby_band(x, w1=wave1, w2=wave2, N=order, ripple=ripple, scale=scale)

        elif butterworth_params:
            # Ensure all parameters are present (filling in defaults where necessary) and are the correct types/units
            wave1 = ensure_unit(butterworth_params['wave1'], u.nm)
            wave2 = ensure_unit(butterworth_params['wave2'], u.nm)
            order = int(butterworth_params['order'])
            peak = ensure_unit(butterworth_params.get('peak', 0.95), u.dimensionless_unscaled)

            # Create a lambda function to calculate transmission at arbitrary wavelength.
            scale = peak / butter_band(w=(wave1 + wave2) / 2, w1=wave1, w2=wave2, N=order, scale=1)
            self._interpolator = lambda x: butter_band(x, w1=wave1, w2=wave2, N=order, scale=scale)

        # Calculate filter parameters (peak transmission, FWHM, lambda_c)
        # Will use the interpolator for all filter types but intial guesses will depend on whether we have tabular
        # data or not
        if hasattr(self, 'wavelengths'):
            # Tabular filter
            self._peak = self._transmission.max()
            self._lambda_peak = self.wavelengths[self._transmission.argmax()]
            above_half_max = np.arange(len(self._transmission))[self._transmission > 0.5 * self._peak]
            blue_half_max_a = self.wavelengths[above_half_max[0] - 1]
            blue_half_max_b = self.wavelengths[above_half_max[0]]
            red_half_max_a = self.wavelengths[above_half_max[-1]]
            red_half_max_b = self.wavelengths[above_half_max[-1] + 1]

            peak_results = minimize_scalar(lambda x: -self.transmission(x).value,
                                           method='Bounded',
                                           bounds=(blue_half_max_b.value, red_half_max_a.value))
        else:
            # Parameterised filter.
            self._peak = peak
            self._lambda_peak = (wave1 + wave2) / 2
            blue_half_max_a = wave1 - (wave2 - wave1)
            blue_half_max_b = self._lambda_peak
            red_half_max_a = (wave1 + wave2) / 2
            red_half_max_b = wave2 + (wave2 - wave1)

            peak_results = minimize_scalar(lambda x: -self.transmission(x).value,
                                           method='Bounded',
                                           bounds=(wave1.value, wave2.value))

        if not peak_results.success:
            raise RuntimeError("Failed to find peak of filter transmission profile!")

        self._lambda_peak = peak_results.x * u.nm
        self._peak = self.transmission(self._lambda_peak)

        print(self.transmission(blue_half_max_a), self.transmission(blue_half_max_b))

        blue_half_max_results = brentq(lambda x: self.transmission(x).value - self._peak.value  / 2,
                                       blue_half_max_a.value,
                                       blue_half_max_b.value,
                                       full_output=True)

        red_half_max_results = brentq(lambda x: self.transmission(x).value - self._peak.value / 2,
                                      red_half_max_a.value,
                                      red_half_max_b.value,
                                      full_output=True)

        if not (blue_half_max_results[1].converged and red_half_max_results[1].converged):
            raise RuntimeError("Problem finding half maximum points of filter transmission profile!")

        self._lambda_c = (blue_half_max_results[0] + red_half_max_results[0]) * u.nm / 2
        self._FWHM = (red_half_max_results[0] - blue_half_max_results[0]) * u.nm

    @property
    def peak(self):
        """Peak transmission of the filter

        Returns:
            Quantity: peak transmission of the filter in dimensionless unscaled units
        """
        return self._peak

    @property
    def lambda_peak(self):
        """Wavelength at peak transmission of the filter

        Returns:
            Quantity: wavelength at peak transmission
        """
        return self._lambda_peak

    @property
    def FWHM(self):
        """Full-Width at Half Maximum of the filter

        Returns:
            Quantity: FWHM of the filter transmission profile
        """
        return self._FWHM

    @property
    def lambda_c(self):
        """Central wavelength of the filter, defined here as the mid point between the two wavelengths where
        transmission is half peak transmission.

        Returns:
            Quantity: central wavelength
        """
        return self._lambda_c

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


def butter_band(w, w1, w2, N, scale=0.95):
    """
    Simple Butterworth bandpass filter function in wavelength space
    To be more realistic this should probably be a Chebyshev Type I function
    instead, and should definitely include cone angle effect but at f/5.34
    (Space Eye focal ratio) the latter at least is pretty insignficant
    """
    # Bandpass implemented as low pass and high pass in series
    w = ensure_unit(w, u.nm)
    g1 = np.sqrt(1 / (1 + (w1/w).to(u.dimensionless_unscaled)**(2*N)))
    g2 = np.sqrt(1 / (1 + (w/w2).to(u.dimensionless_unscaled)**(2*N)))
    return scale * g1 * g2


def cheby_band(w, w1, w2, N, ripple=1, scale=0.95):
    """
    Simple Chebyshev Type I bandpass filter function in wavelength space
    To be more realistic this should definitely include cone angle effect
    but at f/5.34 (Space Eye focal ratio) that is pretty insignficant
    """
    # Bandpass implemented as low pass and high pass in series
    w = ensure_unit(w, u.nm)
    g1 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w1/w).to(u.dimensionless_unscaled).value)**2)
    g2 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w/w2).to(u.dimensionless_unscaled).value)**2)
    return scale * g1 * g2
