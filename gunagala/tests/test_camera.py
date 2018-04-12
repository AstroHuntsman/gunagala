import pytest
from scipy import stats
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
                 QE='ML8300M_QE.csv')
    return ccd


def test_camera(ccd):
    assert isinstance(ccd, Camera)


def test_saturation_level(ccd):
    assert ccd.saturation_level == min(25500 * u.electron / u.pixel,
                                       (((2**16 - 1) - 1100) * 0.37 * u.electron / u.pixel))


def test_max_noise(ccd):
    assert ccd.max_noise == (ccd.saturation_level * u.electron / u.pixel + (9.3 * u.electron / u.pixel)**2)**0.5


def test_dark_frame():
    shape = 0.5
    loc = 0.4
    scale = 0.02
    dist = stats.lognorm(s=shape, loc=loc, scale=scale)
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
                 QE='ML8300M_QE.csv',
                 dark_current_dist=dist,
                 dark_current_seed=42)
    assert isinstance(ccd.dark_frame, u.Quantity)
    assert (ccd.dark_frame.shape * u.pixel == ccd.resolution).all()
    fitted_params = stats.lognorm.fit(ccd.dark_frame.value[0:100,0:100])
    assert fitted_params[0] == pytest.approx(shape, rel=0.02)
    assert fitted_params[1] == pytest.approx(loc, rel=0.02)
    assert fitted_params[2] == pytest.approx(scale, rel=0.02)


def test_dark_seed():
    shape = 0.5
    loc = 0.4
    scale = 0.02
    dist = stats.lognorm(s=shape, loc=loc, scale=scale)
    ccd1 = Camera(bit_depth=16,
                  full_well=25500 * u.electron / u.pixel,
                  gain=0.37 * u.electron / u.adu,
                  bias=1100 * u.adu / u.pixel,
                  readout_time=0.9 * u.second,
                  pixel_size=5.4 * u.micron / u.pixel,
                  resolution=(3326, 2504) * u.pixel,
                  read_noise=9.3 * u.electron / u.pixel,
                  dark_current=0.04 * u.electron / (u.pixel * u.second),
                  minimum_exposure=0.1 * u.second,
                  QE='ML8300M_QE.csv',
                  dark_current_dist=dist,
                  dark_current_seed=42)
    ccd2 = Camera(bit_depth=16,
                  full_well=25500 * u.electron / u.pixel,
                  gain=0.37 * u.electron / u.adu,
                  bias=1100 * u.adu / u.pixel,
                  readout_time=0.9 * u.second,
                  pixel_size=5.4 * u.micron / u.pixel,
                  resolution=(3326, 2504) * u.pixel,
                  read_noise=9.3 * u.electron / u.pixel,
                  dark_current=0.04 * u.electron / (u.pixel * u.second),
                  minimum_exposure=0.1 * u.second,
                  QE='ML8300M_QE.csv',
                  dark_current_dist=dist,
                  dark_current_seed=42)
    assert (ccd1.dark_frame == ccd2.dark_frame).all()


def test_dark_no_seed():
    shape = 0.5
    loc = 0.4
    scale = 0.02
    dist = stats.lognorm(s=shape, loc=loc, scale=scale)
    ccd1 = Camera(bit_depth=16,
                  full_well=25500 * u.electron / u.pixel,
                  gain=0.37 * u.electron / u.adu,
                  bias=1100 * u.adu / u.pixel,
                  readout_time=0.9 * u.second,
                  pixel_size=5.4 * u.micron / u.pixel,
                  resolution=(3326, 2504) * u.pixel,
                  read_noise=9.3 * u.electron / u.pixel,
                  dark_current=0.04 * u.electron / (u.pixel * u.second),
                  minimum_exposure=0.1 * u.second,
                  QE='ML8300M_QE.csv',
                  dark_current_dist=dist)
    ccd2 = Camera(bit_depth=16,
                  full_well=25500 * u.electron / u.pixel,
                  gain=0.37 * u.electron / u.adu,
                  bias=1100 * u.adu / u.pixel,
                  readout_time=0.9 * u.second,
                  pixel_size=5.4 * u.micron / u.pixel,
                  resolution=(3326, 2504) * u.pixel,
                  read_noise=9.3 * u.electron / u.pixel,
                  dark_current=0.04 * u.electron / (u.pixel * u.second),
                  minimum_exposure=0.1 * u.second,
                  QE='ML8300M_QE.csv',
                  dark_current_dist=dist)
    assert (ccd1.dark_frame != ccd2.dark_frame).all()
