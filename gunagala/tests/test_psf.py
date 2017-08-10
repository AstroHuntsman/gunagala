import pytest
import numpy as np
import astropy.units as u

from ..psf import PSF, Moffat_PSF


@pytest.fixture(scope='module')
def psf():
    psf = Moffat_PSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


def test_base():
    with pytest.raises(TypeError):
        # Try to instantiate abstract base class, should fail
        psf_base = PSF(FWHM=1 / 30 * u.arcminute)


def test_moffat(psf):
    assert isinstance(psf, Moffat_PSF)
    assert isinstance(psf, PSF)


def test_FWHM(psf):
    assert psf.FWHM == 2 * u.arcsecond
    psf.FWHM = 4 * u.arcsecond
    assert psf.FWHM == 1 / 15 * u.arcminute
    with pytest.raises(ValueError):
        psf.FWHM = -1 * u.degree
    psf.FWHM = 2 * u.arcsecond


def test_pixel_scale(psf):
    psf.pixel_scale = 2.85 * u.arcsecond / u.pixel
    assert psf.pixel_scale == 2.85 * u.arcsecond / u.pixel


def test_n_pix(psf):
    assert psf.n_pix == 4.25754067000986 * u.pixel


def test_peak(psf):
    assert psf.peak == 0.7134084656751443 / u.pixel


def test_shape(psf):
    assert psf.shape == 4.7
    psf.shape = 2.5
    assert psf.shape == 2.5
    with pytest.raises(ValueError):
        psf.shape = 0.5
    psf.shape = 4.7


def test_pixellated(psf):
    pixellated = psf.pixellated()
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (21, 21)
    pixellated = psf.pixellated(size=7.2)
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (7, 7)
    pixellated = psf.pixellated(offsets=(0.3, -0.7))
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (21, 21)
    with pytest.raises(ValueError):
        psf.pixellated(size=-1.3)
