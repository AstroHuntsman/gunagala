import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..sky import Sky, Simple, ZodiacalLight


@pytest.fixture(scope='module')
def simple():
    sky = Simple(surface_brightness = {'g': 21 * u.ABmag, 'r': 20 * u.ABmag})
    return sky


@pytest.fixture(scope='module')
def zl():
    sky = ZodiacalLight()
    return sky


def test_base():
    base = Sky()
    with pytest.raises(NotImplementedError):
        base.surface_brightness()


def test_simple(simple):
    assert isinstance(simple, Simple)


def test_simple_surface_brightness(simple):
    assert simple.surface_brightness('g') == 21 * u.ABmag
    assert simple.surface_brightness('r') == 20 * u.ABmag
    with pytest.raises(ValueError):
        simple.surface_brightness('i')


def test_zl(zl):
    assert isinstance(zl, ZodiacalLight)


def test_zl_surface_brightness(zl):
    waves = (0.4, 0.5, 0.6, 0.7) * u.micron
    sb_func = zl.surface_brightness()
    sb = sb_func(waves)
    assert isinstance(sb, u.Quantity)
    assert sb.unit == u.photon * u.second**-1 * u.m**-2 *  u.um**-1 * u.arcsecond**-2


def test_zl_relative_brightness(zl):
    position_text_1 = "2h 34m 21s -31d 52m 29.1s"
    position_text_2 = "2h 43m 25s -30d 15m 36.2s"
    position = SkyCoord((position_text_1, position_text_2))
    time = Time.now()
    rb = zl.relative_brightness(position_text_1, time)
    assert isinstance(rb, np.ndarray)
    assert rb.shape == ()
    rb = zl.relative_brightness(position, time)
    assert isinstance(rb, np.ndarray)
    assert rb.shape == (2,)
