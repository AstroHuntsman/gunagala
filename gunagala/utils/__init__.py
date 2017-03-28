# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`
import astropy.units as u


def ensure_unit(arg, unit):
    """Ensures that the argument is using the required unit"""
    if not isinstance(arg, u.Quantity):
        arg = arg * unit
    return arg.to(unit)
