import pytest
import numpy as np
import astropy.units as u

from gunagala.psf import PSF, MoffatPSF, PixellatedPSF


@pytest.fixture(scope='module')
def psf():
    psf = MoffatPSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


@pytest.fixture(scope='module')
def pix_psf():
    psf_data = np.array([[0.0, 0.0, 0.1, 0.0, 0.0],
                         [0.0, 0.3, 0.7, 0.4, 0.0],
                         [0.1, 0.8, 1.0, 0.6, 0.1],
                         [0.0, 0.2, 0.7, 0.3, 0.0],
                         [0.0, 0.0, 0.1, 0.0, 0.0]])
    psf = PixellatedPSF(psf_data=psf_data,
                        psf_sampling=1 * u.arcsecond / u.pixel,
                        psf_centre=(2, 2),
                        oversampling=10,
                        pixel_scale=(2 / 3) * u.arcsecond / u.pixel)
    return psf


def test_base():
    with pytest.raises(TypeError):
        # Try to instantiate abstract base class, should fail
        psf_base = PSF(FWHM=1 / 30 * u.arcminute)


def test_moffat(psf):
    assert isinstance(psf, MoffatPSF)
    assert isinstance(psf, PSF)


def test_pix(pix_psf):
    assert isinstance(pix_psf, PixellatedPSF)
    assert isinstance(pix_psf, PSF)


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


def test_pixel_scale_pix(pix_psf):
    pix_psf.pixel_scale = (1 / 3) * u.arcsecond / u.pixel
    assert pix_psf.pixel_scale == (1 / 3) * u.arcsecond / u.pixel
    pix_psf.pixel_scale = (2 / 3) * u.arcsecond / u.pixel


def test_n_pix(psf):
    assert psf.n_pix == 4.25754067000986 * u.pixel


def test_n_pix_pix(pix_psf):
    assert pix_psf.n_pix.to(u.pixel).value == pytest.approx(21.01351017)


def test_peak(psf):
    assert psf.peak == 0.7134084656751443 / u.pixel


def test_peak_pix(pix_psf):
    assert pix_psf.peak.to(1 / u.pixel).value == pytest.approx(0.08073066)


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


def test_pixellated(pix_psf):
    pixellated = pix_psf.pixellated()
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (21, 21)
    pixellated = pix_psf.pixellated(size=7.2)
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (7, 7)
    pixellated = pix_psf.pixellated(offsets=(0.3, -0.7))
    assert isinstance(pixellated, np.ndarray)
    assert pixellated.shape == (21, 21)
    with pytest.raises(ValueError):
        pix_psf.pixellated(size=-1.3)
