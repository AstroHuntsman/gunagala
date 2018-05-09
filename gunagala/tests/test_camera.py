import pytest
import astropy.units as u

from gunagala.camera import Camera


@pytest.fixture(scope='module')
def ccd():
    ccd = Camera(bit_depth=16,
                 full_well=25500 * u.electron / u.pixel,
                 gain=0.37 * u.electron / u.adu,
                 bias=1100 * u.adu / u.pixel,
                 readout_time=0.9 * u.second,
                 pixel_size=5.4 * u.micron / u.pixel,
                 resolution=(3326, 2504) * u.pixel,
                 read_noise=9.3 * u.electron / u.pixel,
                 dark_current=0.04 * u.electron / (u.pixel * u.second),
                 minimum_exposure=0.1 * u.second,
                 QE_filename='ML8300M_QE.csv')
    return ccd


def test_camera(ccd):
    assert isinstance(ccd, Camera)


def test_saturation_level(ccd):
    assert ccd.saturation_level == min(25500 * u.electron / u.pixel,
                                       (((2**16 - 1) - 1100) * 0.37 * u.electron / u.pixel))


def test_max_noise(ccd):
    assert ccd.max_noise == (ccd.saturation_level * u.electron / u.pixel + (9.3 * u.electron / u.pixel)**2)**0.5
