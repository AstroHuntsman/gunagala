"""
Optics, e.g. a telescope or lens
"""

import os
import numpy as np

from astropy import units as u
from astropy.table import Table

from gunagala.utils import ensure_unit, get_table_data


class Optic:
    """
    Class representing the overall optical system.

    The optical system includes all optics (e.g. telescope, including any
    field flattener, focal reducer or reimaging optics) with the exception
    of optical filters, which are handled by the `optical_filter.Filter`
    class. At present only circular pupils (aperture) and are supported,
    but central obstructions (also circular) can be modelled.

    Parameters
    ----------
    aperture : astropy.units.Quantity
        Diameter of the optical system entrance pupil (effective aperture
        diameter).
    focal_length : astropy.units.Quantity
        Effective focal length of the optical system as a whole.
    throughput : astropy.table.Table or str
        Optical throughput as a function of wavelength data, either as an
        astropy.table.Table object or the name of a file that can be read
        by `astropy.table.Table.read()`. The filename can be either the
        path to a user file or the name of one of gunagala's included
        files. The table must use column names `Wavelength` and
        `Throughput`. If the table does not specify units then nm and
        dimensionless unscaled are assumed.
    central_obstruction : astropy.units.Quantity, optional
        Diameter of the central obstruction of the entrance pupil, if any.
        If not specified an unobstructed pupil is assumed.

    Attributes
    ----------
    aperture : astropy.units.Quantity
        Same as parameters.
    central_obstruction : astropy.units.Quantity
        Same as parameters.
    aperture_area : astropy.units.Quantity
        Effective collecting are of the optical system aperture, including
        the effects of the central obstruction, if any.
    focal_length : astropy.units.Quantity
        Same as parameters.
    focal_ratio : astropy.units.Quantity
        Effective focal ratio (F/D) of the optical system
    theta_range : astropy.units.Quantity
        Pair of angles corresponding to the minimum and maximum angles of
        incidence in focal plane of the optical system. These can be used
        by `optical_filter.Filter` objects to calculate cone angle effects
        for focal plane filters. These are automatically calculated from
        the central obstruction and entrance pupil diameters and effective
        focal length assuming a telecentric output. If the optical system
        is far telecentricty these values should now be used.
    wavelengths : astropy.units.Quantity
        Sequence of wavelength values from the tabulated throughput data,
        loaded from `throughput_filename`.
    throughput : astropy.units.Quantity
        Sequence of throughput values from the tabulated throughput data,
        loaded from `throughout_filename`.
    """
    def __init__(self, aperture, focal_length, throughput, central_obstruction=0 * u.mm):

        self.aperture = ensure_unit(aperture, u.mm)
        self.central_obstruction = ensure_unit(central_obstruction, u.mm)

        self.aperture_area = np.pi * (self.aperture**2 - self.central_obstruction**2).to(u.m**2) / 4

        self.focal_length = ensure_unit(focal_length, u.mm)

        self.focal_ratio = (self.focal_length / self.aperture).to(u.dimensionless_unscaled)

        # Calculate beam half-cones angles at the focal plane
        if central_obstruction == 0 * u.mm:
            theta_min = 0 * u.radian
        else:
            theta_min = np.arctan((self.central_obstruction / 2) / self.focal_length)

        theta_max = np.arctan((self.aperture / 2) / self.focal_length)

        self.theta_range = u.Quantity((theta_min, theta_max)).to(u.degree)

        self.wavelengths, self.throughput = get_table_data(throughput, data_dir='data/performance_data',
                                                           column_names = ('Wavelength', 'Throughput'),
                                                           column_units = (u.nm, u.dimensionless_unscaled))
