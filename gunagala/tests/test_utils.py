import pytest
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
