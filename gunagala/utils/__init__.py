# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`
import astropy.units as u


def ensure_unit(arg, unit):
    """
    Ensures that the argument has the requested units, performing
    conversions as necessary.

    Parameters
    ----------
    arg : astropy.units.Quantity or compatible
        Argument to be coerced into the requested units. Can be an
        `astropy.units.Quantity` instance or any numeric type or sequence
        that is compatible with the `Quantity` constructor (e.g.
        a `numpy.array`, `list` of `float`, etc.).
    unit : astropy.units.Unit
        Requested units.

    Returns
    -------
    arg : astropy.units.Quantity
        `arg` as an `astropy.units.Quantity` with units of `unit`.
    """
    if not isinstance(arg, u.Quantity):
        arg = arg * unit
    return arg.to(unit)
