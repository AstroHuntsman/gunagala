import os

import numpy as np
from scipy.special import eval_chebyt

from astropy import units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from .utils import ensure_unit


data_dir = 'data/performance_data'


class Filter:

    def __init__(self, transmission_filename, sky_mu):
        """Class representing an optical bandpass filter

        Args:
            transmission_filename (string): name of file containing transmission as a function of wavelength data. Must
            be in a format readable by `astropy.table.Table.read()` and use column names `Wavelength` and
            `Transmission`. If the data file does not provide units nm and dimensionless unscaled will be assumed.
            sky_mu (Quantity): the sky background surface brightness per arcsecond^2 (in ABmag units) for the band.
        """

        transmission_data = Table.read(get_pkg_data_filename(os.path.join(data_dir, transmission_filename)))

        if not transmission_data['Wavelength'].unit:
            transmission_data['Wavelength'].unit = u.nm
        self.wavelengths = transmission_data['Wavelength'].quantity.to(u.nm)

        if not transmission_data['Transmission'].unit:
            transmission_data['Transmission'].unit = u.dimensionless_unscaled
        self.transmission = transmission_data['Transmission'].quantity.to(u.dimensionless_unscaled)

        self.sky_mu = ensure_unit(sky_mu, u.ABmag)


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
