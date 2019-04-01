"""
Optical filters
"""
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brentq
from scipy.special import eval_chebyt

from astropy import units as u
from astropy.table import Table

from gunagala.utils import ensure_unit, get_table_data


class Filter:
    """
    Class representing an optical bandpass filter.

    The filter bandpass can be defined either by a table of transmission
    versus wavelength data or by one of the included analytic functions:
    Butterworth function or Chebyshev Type I.

    Parameters
    ----------
    transmission : astropy.table.Table or str, optional
        Filter transmission as a function of wavelength data, either as an
        astropy.table.Table object or the name of a file that can be read
        by `astropy.table.Table.read()`. The filename can be either the
        path to a user file or the name of one of gunagala's included
        files. The table must use column names `Wavelength` and
        `Transmission`. If the table does not specify units then nm and
        dimensionless unscaled are assumed.
    chebyshev_params : dict, optional
        Dictionary containing the parameters wave1, wave2, order, ripple
        and peak for the Chebyshev Type I parameterised filter model.
    butterworth_params : dict, optional
        Dictionary containing the parameters wave1, wave2, order and peak
        for the Butterworth parameterised filter model.
    apply_aoi : bool, optional
        Whether to model angle of incidence effects due to installation of
        the filter in a converging beam. If the filter is to be installed
        in a pupil or the transmission profile already includes these
        effects this should be set to False. If set to True then calls to
        the `transmission()` method will have to specify the range of
        angles of incidence. Default False
    n_eff : float, optional
        Effective refractive index value for the filter coatings. This is
        used only by the angle of incidence effect model. Typical values
        for real interference filters range from ~1.5 to ~2, and are in
        general polarisation dependent (not modelled here). Default 1.75.
    theta_range : astropy.units.Quantity, optional
        2 element Quantity specifying the range of angles of incidence
        (min, max). If specified this will be used to model the effect of
        a converging beam on the calculated filter parameters (FWHM,
        lambda_c, etc).

    """
    def __init__(self,
                 transmission=None,
                 chebyshev_params=None,
                 butterworth_params=None,
                 apply_aoi=False,
                 n_eff=1.75,
                 theta_range=None,
                *kwargs):

        n_args = np.count_nonzero((transmission, chebyshev_params, butterworth_params))
        if n_args != 1:
            raise ValueError("One and only one of `tranmission`, `chebyshev_params` & `butterworth_params`"
                             + "must be specified, got {}!".format(n_args))

        self.apply_aoi = apply_aoi
        self.n_eff = n_eff
        if theta_range:
            self._theta_range = ensure_unit(theta_range, u.radian)
        else:
            self._theta_range = None

        if transmission:
            self.wavelengths, self._transmission = get_table_data(transmission,
                                                                  column_names=('Wavelength', 'Transmission'),
                                                                  column_units=(u.nm, u.dimensionless_unscaled))

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

            # Store parameters for later reference
            self._params = {'wave1': wave1,
                            'wave2': wave2,
                            'order': order,
                            'ripple': ripple,
                            'scale': scale,
                            'peak': peak}

        elif butterworth_params:
            # Ensure all parameters are present (filling in defaults where necessary) and are the correct types/units
            wave1 = ensure_unit(butterworth_params['wave1'], u.nm)
            wave2 = ensure_unit(butterworth_params['wave2'], u.nm)
            order = int(butterworth_params['order'])
            peak = ensure_unit(butterworth_params.get('peak', 0.95), u.dimensionless_unscaled)

            # Create a lambda function to calculate transmission at arbitrary wavelength.
            scale = peak / butter_band(w=(wave1 + wave2) / 2, w1=wave1, w2=wave2, N=order, scale=1)
            self._interpolator = lambda x: butter_band(x, w1=wave1, w2=wave2, N=order, scale=scale)

            # Store parameters for later reference
            self._params = {'wave1': wave1,
                            'wave2': wave2,
                            'order': order,
                            'scale': scale,
                            'peak': peak}

        self._update_properties()

    @property
    def theta_range(self):
        """
        2 element Quantity specifying the range of angles of incidence
        (min, max).

        Returns
        -------
        theta_range: astropy.units.Quantity
            2 element Quantity specifying the range of angles of incidence
            (min, max).
        """
        return self._theta_range

    @theta_range.setter
    def theta_range(self, theta_range):
        self._theta_range = ensure_unit(theta_range, u.radian)
        self._update_properties()

    @property
    def peak(self):
        """
        Peak transmission of the filter

        Returns
        -------
        peak : astropy.units.Quantity
            Peak transmission of the filter in dimensionless unscaled units
        """
        return self._peak

    @property
    def lambda_peak(self):
        """
        Wavelength at peak transmission of the filter

        Returns
        -------
        lambda_peak : astropy.units.Quantity
            Wavelength at peak transmission
        """
        return self._lambda_peak

    @property
    def FWHM(self):
        """
        Full-Width at Half Maximum of the filter

        Returns
        -------
        FWHM : astropy.units.Quantity
            FWHM of the filter transmission profile
        """
        return self._FWHM

    @property
    def lambda_c(self):
        """
        Central wavelength of the filter, defined here as the mid point
        between the two wavelengths where transmission is half peak
        transmission.

        Returns
        -------
        lambda_c : astropy.units.Quantity
            Central wavelength
        """
        return self._lambda_c

    def transmission(self, waves, theta_range=None):
        """
        Return filter transmission at the given wavelength(s).

        For filter bandpasses defined by data tables this will
        interpolate/extrapolate as required while for filter bandpasses
        defined by analytic expressions it will be calculated directly.

        Parameters
        ----------
        waves : astropy.units.Quantity
            Wavelength(s) for which the filter transmission is required
        theta_range : astropy.units.Quantity, optional
            2 element quantity specifying the range of angles of incidence
            (min, max). If specified this will be used to model the effect
            of a converging beam on the filter bandpass. If not specified
            the default values set when creating the Filter instance will
            be used.

        Returns
        -------
        waves : astropy.units.Quantity
            Filter transmission at the given wavelength(s)
        """
        waves = ensure_unit(waves, u.nm)

        if self.apply_aoi and (theta_range or self.theta_range):
            if theta_range:
                theta_range = ensure_unit(theta_range, u.radian)
            else:
                theta_range = self.theta_range

            # Thetas spanning range with equal spacing in theta^2 (approximate weighting by solid angle)
            thetas = np.linspace(theta_range.min()**2, theta_range.max()**2, num=10)**0.5
            thetas = thetas.reshape((1, len(thetas)))

            # For each input wavelength create shifted effective wavelengths for each angle of incidence.
            len_waves = 1 if waves.isscalar else len(waves)  # len() fails on scalar Quantities
            shifted_waves = waves.reshape((len_waves, 1)) / (1 - (np.sin(thetas / self.n_eff))**2)**0.5

            # Use interpolator to look up transmission for every shifted wavelength
            trans = self._interpolator(shifted_waves)

            # Finally take mean over thetas (effectively an approximate integral over solid angle)
            trans = trans.mean(axis=1)

        else:
            trans = self._interpolator(waves)

        # Make sure interpolation/extrapolation doesn't take transmission unphysically below zero or above 1
        trans = np.where(trans > 0, trans, 0)
        trans = np.where(trans <= 1, trans, 1)

        # Put units back (interpolator doesn't work with Quantity, nor does np.where)
        return ensure_unit(trans, u.dimensionless_unscaled)

    def _update_properties(self):
        # Calculate filter parameters (peak transmission, FWHM, lambda_c)
        # Will use the interpolator for all filter types but optimizer/solver bounds will depend on whether we have
        # tabular data or not
        if hasattr(self, 'wavelengths'):
            # Tabular filter
            self._peak = self._transmission.max()
            self._lambda_peak = self.wavelengths[self._transmission.argmax()]
            above_half_max = np.arange(len(self._transmission))[self._transmission > 0.5 * self._peak]
            if self.apply_aoi and self.theta_range:
                blue_half_max_a = self.wavelengths[above_half_max[0] - 1] * \
                    (1 - (np.sin(self.theta_range.max() / self.n_eff))**2)**0.5
            else:
                blue_half_max_a = self.wavelengths[above_half_max[0] - 1]
            blue_half_max_b = self.wavelengths[above_half_max[0]]

            if self.apply_aoi and self.theta_range:
                red_half_max_a = self.wavelengths[above_half_max[-1]] * \
                    (1 - (np.sin(self.theta_range.max() / self.n_eff))**2)**0.5
            else:
                red_half_max_a = self.wavelengths[above_half_max[-1]]
            red_half_max_b = self.wavelengths[above_half_max[-1] + 1]

        else:
            # Parameterised filter.
            wave1 = self._params['wave1']
            wave2 = self._params['wave2']
            self._lambda_peak = (wave1 + wave2) / 2
            if self.apply_aoi and self.theta_range:
                blue_half_max_a = (wave1 - (wave2 - wave1)) * \
                    (1 - (np.sin(self.theta_range.max() / self.n_eff))**2)**0.5
            else:
                blue_half_max_a = wave1 - (wave2 - wave1)
            blue_half_max_b = self._lambda_peak
            red_half_max_a = (wave1 + wave2) / 2
            red_half_max_b = wave2 + (wave2 - wave1)

            peak_results = minimize_scalar(lambda x: -self.transmission(x).value,
                                           method='Bounded',
                                           bounds=(blue_half_max_a.value, red_half_max_b.value))

            if not peak_results.success:
                raise RuntimeError("Failed to find peak of filter transmission profile!")

            self._lambda_peak = peak_results.x * u.nm
            self._peak = self.transmission(self._lambda_peak)

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


def butter_band(w, w1, w2, N, scale=0.95):
    """
    Simple Butterworth bandpass filter function in wavelength space.

    Parameters
    ----------
    w : astropy.units.Quantity
        Wavelength
    w1 : astropy.units.Quantity
        Wavelength of short wavelength edge of bandpass
    w2 : astropy.units.Quantity
        Wavelength of long wavelength edge of bandpass
    N : int
        Order of the Butterworth function
    scale : float, optional
        Scaling to apply to the transmission of the Butterworth function,
        default 0.95

    Returns
    -------
    transmission : astropy.units.Quantity
        Filter transmission at wavelength `w`.
    """
    # Bandpass implemented as low pass and high pass in series
    w = ensure_unit(w, u.nm)
    g1 = np.sqrt(1 / (1 + (w1/w).to(u.dimensionless_unscaled)**(2*N)))
    g2 = np.sqrt(1 / (1 + (w/w2).to(u.dimensionless_unscaled)**(2*N)))
    return scale * g1 * g2


def cheby_band(w, w1, w2, N, ripple=1, scale=0.95):
    """
    Simple Chebyshev Type I bandpass filter function in wavelength space.

    Parameters
    ----------
    w : astropy.units.Quantity
        Wavelength
    w1 : astropy.units.Quantity
        Wavelength of short wavelength edge of bandpass
    w2 : astropy.units.Quantity
        Wavelength of long wavelength edge of bandpass
    N : int
        Order of the Chebyshev function
    ripple : float, optional
        Scaling to apply to the ripple of the Chebyshev function, default
        1.0
    scale : float, optional
        Scaling to apply to the transmission of the Chebyshev function,
        default 0.95

    Returns
    -------
    transmission : astropy.units.Quantity
        Filter transmission at wavelength `w`.
    """
    # Bandpass implemented as low pass and high pass in series
    w = ensure_unit(w, u.nm)
    g1 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w1/w).to(u.dimensionless_unscaled).value)**2)
    g2 = 1 / np.sqrt(1 + ripple**2 * eval_chebyt(N, (w/w2).to(u.dimensionless_unscaled).value)**2)
    return scale * g1 * g2
