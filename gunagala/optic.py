"""
Optics, e.g. a telescope or lens
"""

import os
import functools
import numpy as np

from astropy import units as u
from astropy.table import Table

from gunagala.utils import ensure_unit, get_table_data, array_sequence_equal


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

        self.wavelengths, self.throughput = get_table_data(throughput,
                                                           column_names=('Wavelength', 'Throughput'),
                                                           column_units=(u.nm, u.dimensionless_unscaled))

def make_throughput(surfaces):
    """
    Constructs a table of throughout vs wavelength from the numbers and types of optical surfaces

    Utility function to provide a simple estimate of optical throughput versus wavelength given
    the numbers and types of optical surfaces. Uses tabulated data included within gunagala for
    a number of common optical surfaces. To get a list of the available surface types call
    list_surfaces(). Mirror coating data from by Thorlabs (http://www.tholabs.com/)

    Parameters
    ----------
    surfaces: list of tuples
        List containing tuples of (surface name, )

    Returns
    -------
    table: astropy.table.Table
        Table with columns Wavelength and Throughput
    """
    wavelengths_list = []
    throughputs_list = []
    for surface_name, surface_number in surfaces:
        assert surface_name in list_surfaces(), \
            "{} is not one the supported surface types!".format(surface_name)
        if _surfaces[surface_name][:8] == 'Thorlabs':
            # Thorlabs mirror coating data. Needs some extra keyword arguments.
            wavelengths, throughputs = get_table_data(_surfaces[surface_name],
                                                      column_names=('Wavelength', 'Reflectance'),
                                                      column_units=(u.um, u.percent),
                                                      names=('Comments 1', 'Comments 2',
                                                             'Wavelength', 'Reflectance P',
                                                             'Reflectance S', 'Reflectance'),
                                                      encoding='utf-8')
        else:
            wavelengths, throughputs = get_table_data(_surfaces[surface_name],
                                                      column_names=('Wavelength', 'Transmission'),
                                                      column_units=(u.nm, u.dimensionless_unscaled))
        wavelengths_list.append(wavelengths)
        throughputs = throughputs.to(u.dimensionless_unscaled)
        throughputs = throughputs**surface_number
        throughputs_list.append(throughputs)

    if len(wavelengths_list) == 1:
        # Only one surface type.
        throughput_table = Table((wavelengths_list[0], throughputs_list[0]),
                                 names=('Wavelength', 'Throughput'))
    elif array_sequence_equal(wavelengths_list):
        # All surfaces have the same wavelength sampling, easy to combine.
        throughput_combined = functools.reduce(lambda x, y: x*y, throughputs_list)
        throughput_table = Table((wavelengths_list[0], throughput_combined),
                                 names=('Wavelength', 'Throughput'))
    else:
        # Some surfaces use different wavelength sampling. Need to resample before combining.
        raise NotImplementedError("Surface types have different wavelength sampling.")

    return throughput_table


def list_surfaces():
    surfaces = list(_surfaces.keys())
    surfaces.sort()
    return surfaces


_surfaces = {'aluminium_12deg_protected': 'Thorlabs_Protected_Aluminum_Coating_12deg.csv',
             'aluminium_45deg_protected': 'Thorlabs_Protected_Aluminum_Coating_45deg.csv',
             'gold_12deg_protected': 'Thorlabs_Protected_Gold_Coating_12deg.csv',
             'gold_45deg_protected': 'Thorlabs_Protected_Gold_Coating_45deg.csv',
             'silver_12deg_protected': 'Thorlabs_Protected_Silver_Coating_12deg.csv',
             'silver_45deg_protected': 'Thorlabs_Protected_Silver_Coating_45deg.csv',
             'gold_45deg_unprotected': 'Thorlabs_Unprotected_Gold_Coating_45deg.csv'}
