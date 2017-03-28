import os
import numpy as np

from astropy import units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from .utils import ensure_unit


data_dir = 'data/performance_data'


class Optic:

    def __init__(self, aperture, focal_length, throughput_filename, central_obstruction=0 * u.mm):
        """ Class representing the overall optical system (e.g. a telescope, including any field flattener, focal
        reducer or reimaging optics). The Filter class should be used for optical filters.

        Args:
            aperture (Quantity): diameter of the entrance pupil
            focal_length (Quantity): effective focal length
            throughput_filename (string): name of file containing optical throughput as a function of wavelength data.
                Must be in a format readable by `astropy.table.Table.read()` and use column names `Wavelength` and
                `Throughput`. If the data file does not provide units nm and dimensionless unscaled are assumed.
            central_obstruction (Quantity, optional): diameter of the central obstruction of the entrance pupil, if any.
        """

        self.aperture = ensure_unit(aperture, u.mm)
        self.central_obstruction = ensure_unit(central_obstruction, u.mm)

        self.aperture_area = np.pi * (self.aperture**2 - self.central_obstruction**2).to(u.m**2) / 4

        self.focal_length = ensure_unit(focal_length, u.mm)

        print(os.path.join(data_dir, throughput_filename))
        tau_data = Table.read(get_pkg_data_filename(os.path.join(data_dir, throughput_filename)))

        if not tau_data['Wavelength'].unit:
            tau_data['Wavelength'].unit = u.nm
        self.wavelengths = tau_data['Wavelength'].quantity.to(u.nm)

        if not tau_data['Throughput'].unit:
            tau_data['Throughput'].unit = u.dimensionless_unscaled
        self.throughput = tau_data['Throughput'].quantity.to(u.dimensionless_unscaled)
