# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`
import os
import astropy.units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename


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
    try:
        arg = arg.to(unit)
    except u.UnitConversionError as err:
        # arg is a Quantity or compatible class, but the units are incompatible.
        raise err
    except:
        # Some other exception means arg isn't a Quantity or compatible. Try converting it.
        arg = arg * unit

    return arg


def get_table_data(data_table, data_dir, column_names, column_units):
    if not isinstance(data_table, Table):
        # data_table isn't a Table, assume it's a filename.
        if not os.path.exists(data_table):
            # Not a (valid) path to a user file, look in package data directories
            try:
                data_table = get_pkg_data_filename(os.path.join(data_dir, data_table),
                                                   package='gunagala')
            except:
                # Not in package data directories either
                raise IOError("Couldn't find data table {}!".format(data_table))
        data_table = Table.read(data_table)

    data = []
    for name, unit in zip(column_names, column_units):
        try:
            column = data_table[name]
        except KeyError:
            raise ValueError("Data table has no column named {}!".format(name))
        if not column.unit:
            column.unit = unit
            data.append(column.quantity)
        else:
            data.append(column.quantity.to(unit))

    return data
