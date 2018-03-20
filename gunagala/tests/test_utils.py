import pytest
import numpy as np
import astropy.units as u
from astropy.table import Column

from gunagala import utils


def test_ensure_unit_quantity():
    q = (1.0, 42) * u.imperial.furlong
    result = utils.ensure_unit(q, u.m)
    assert (result == q.to(u.m)).all()


def test_ensure_unit_floats():
    f = (1.0, 42.0)
    result = utils.ensure_unit(f, u.m)
    assert (result == f * u.m).all()


def test_ensure_unit_column():
    c = Column(data=(1.0, 42), name='test', unit=u.imperial.furlong)
    result = utils.ensure_unit(c, u.m)
    assert (result == c.to(u.m)).all()


def test_ensure_unit_bad_unit():
    with pytest.raises(u.UnitConversionError):
        q = (1.0, 42) * u.fortnight
        result = utils.ensure_unit(q, u.m)


def test_ensure_unit_non_numeric():
    with pytest.raises(ValueError):
        s = "fortytwo"
        result = utils.ensure_unit(s, u.m)


def test_array_empty():
    with pytest.raises(ValueError):
        utils.array_sequence_equal([])


def test_array_single():
    assert utils.array_sequence_equal([(1.2, 1.3) * u.m]) is True


def test_array_twice():
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.4) * u.m)) is True


def test_array_different():
    assert utils.array_sequence_equal([(1.2, 1.4) * u.m, (1.2, 1.3) * u.m]) is False


def test_array_length():
    assert utils.array_sequence_equal([(1.2, 1.4) * u.m, (1.2, 1.4, 1.3) * u.m]) is False


def test_array_units():
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.4) * u.imperial.furlong)) is False


def test_array_type():
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, np.array((1.2, 1.4)))) is False


def test_array_reference():
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.4) * u.m),
                                      reference=(1.2, 1.4) * u.m) is True
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.3) * u.m),
                                      reference=(1.2, 1.4) * u.m) is False
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.4) * u.m),
                                      reference=(1.2, 1.3) * u.m) is False
    assert utils.array_sequence_equal(((1.2, 1.4) * u.m, (1.2, 1.4) * u.m),
                                      reference=(1.2, 1.4, 1.3) * u.m) is False
