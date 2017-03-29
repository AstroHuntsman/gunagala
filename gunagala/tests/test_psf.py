import pytest
import astropy.units as u

from ..psf import PSF, Moffat_PSF


@pytest.fixture(scope='module')
def psf():
    psf = Moffat_PSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


def test_psf_base():
    with pytest.raises(TypeError):
        # Try to instantiate abstract base class, should fail
        psf_base = PSF(FWHM=1 / 30 * u.arcminute)


def test_psf_moffat(psf):
    assert isinstance(psf, Moffat_PSF)
    assert isinstance(psf, PSF)
    assert psf.FWHM == 2 * u.arcsecond
    psf.pixel_scale = 2.85 * u.arcsecond / u.pixel
    assert psf.pixel_scale == 2.85 * u.arcsecond / u.pixel
    assert psf.n_pix == 4.25754067000986 * u.pixel
    assert psf.peak == 0.7134084656751443 / u.pixel
