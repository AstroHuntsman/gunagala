# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-module is destined for common non-package specific utility
# functions that will ultimately be merged into `astropy.utils`
import os
import functools
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


def get_table_data(data_table, column_names, column_units,
                   data_dir='data/performance_data', **kwargs):
    """
    Parses a data table to extract specified columns, converted to Quantity with specified units.

    Parameters
    ----------
    data_table: astropy.table.Table or str
        The data table for parsing, either as an astropy.table.Table object or the name of a file
        that can be read by `astropy.table.Table.read()`. The filename can be either the path to a
        user file or the name of one of gunagala's included files.
    column_names: sequence
        Names of the columns to extract from the table
    column_units: sequence
        Desired units for the extracted columns. If data_table specifies units for its columns then
        the extracted columns will be converted to these units. If not then the specified units
        will be added to the corresponding column.

    Additional keyword arguments will be passed to the call to astropy.table.Table.read()
    if reading a Table from a file. See the documentation for Table.read() for details
    of the available parameters.

    Returns
    -------
    data: tuple of astropy.units.Quantity
        Tuple of Quantity objects corresponding to the named columns, with the specified units.
    """
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
        data_table = Table.read(data_table, **kwargs)

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


def array_sequence_equal(array_sequence, reference=None):
    """
    Determine if all array objects in a sequence are equal.

    Parameters
    ----------
    array_sequence: sequence of numpy.array
        Sequence of numpy.array or compatible type (e.g. astropy.unit.Quantity) objects to compare.
        The objects must support element-wise comparision and implement an any() method.
    reference: numpy.array, optional
        If given all arrays in the sequence will be compared with reference, otherwise they will
        be compared with each other.

    Returns
    -------
    equal: bool
        True if all arrays in the sequence are equal (or equal to reference, if given), otherwise
        False.
    """
    n_arrays = len(array_sequence)
    if n_arrays == 0:
        raise ValueError('array_sequence must contain at least one array object!')
    elif n_arrays = 1:
        if reference is None:
            return True
        else:
            return (array_list[0] == reference).all()
    else:
        if reference is None:
            reference = array_sequence[0]
        comparisons = map(lambda x: (x==reference).all(), array_sequence)
        return functools.reduce(lambda x, y: x and y, comparisons)
