import pytest
import numpy as np
import astropy.units as u

from gunagala.psf import PSF, MoffatPSF, PixellatedPSF


@pytest.fixture(scope='module')
def psf_moffat():
    psf = MoffatPSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


@pytest.fixture(scope='module')
def psf_pixellated():
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


def test_moffat(psf_moffat):
    assert isinstance(psf, MoffatPSF)
    assert isinstance(psf, PSF)


def test_pix(psf_pixellated):
    assert isinstance(pix_psf, PixellatedPSF)
    assert isinstance(pix_psf, PSF)


def test_FWHM(psf_moffat):
    assert psf.FWHM == 2 * u.arcsecond
    psf.FWHM = 4 * u.arcsecond
    assert psf.FWHM == 1 / 15 * u.arcminute
    with pytest.raises(ValueError):
        psf.FWHM = -1 * u.degree
    psf.FWHM = 2 * u.arcsecond


def test_pixel_scale(psf_moffat):
    psf.pixel_scale = 2.85 * u.arcsecond / u.pixel
    assert psf.pixel_scale == 2.85 * u.arcsecond / u.pixel


def test_pixel_scale_pix(psf_pixellated):
    pix_psf.pixel_scale = (1 / 3) * u.arcsecond / u.pixel
    assert pix_psf.pixel_scale == (1 / 3) * u.arcsecond / u.pixel
    pix_psf.pixel_scale = (2 / 3) * u.arcsecond / u.pixel


@pytest.mark.parameterize("psf, expected_n_pix", [
    (psf_moffat,4.25754067000986),
    (psf_pixellated,21.06994544)],
    ids=["moffat", "pixellated"]
    )


def test_n_pix(psf, expected_n_pix):
    assert psf.n_pix / u.pixel == pytest.approx(expected_n_pix)


def test_peak(psf_moffat):
    assert psf.peak == 0.7134084656751443 / u.pixel


def test_peak_pix(psf_pixellated):
    assert pix_psf.peak * u.pixel == pytest.approx(0.08073066)


def test_shape(psf_moffat):
    assert psf.shape == 4.7
    psf.shape = 2.5
    assert psf.shape == 2.5
    with pytest.raises(ValueError):
        psf.shape = 0.5
    psf.shape = 4.7


@pytest.mark.parametrize("psf, expected", [
    (psf_moffat, (21, 21)),
    (psf_pixellated, (21, 21))],
    ids=["moffat", "pixellated"]
    )


def test_pixellated_square(psf, expected):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert psf.pixellated().shape == expected
    assert (psf.pixellated() >= 0 ).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, expected", [
    (psf_moffat, (7,9)),
    (psf_pixellated, (7,9))],
    ids=["moffat", "pixellated"]
    )


def test_pixellated_rectangle(psf, expected):
    assert isinstance(psf.pixellated(size=(7.2, 9.2)), np.ndarray)
    assert psf.pixellated(size=(7.2, 9.2)).shape == expected
    assert (psf.pixellated(size=(7.2, 9.2)) >= 0 ).all()
    assert np.isfinite(psf.pixellated(size=(7.2, 9.2))).all()


@pytest.mark.parametrize("psf, expected", [
    (psf_moffat, (21, 21)),
    (psf_pixellated, (21, 21))],
    ids=["moffat", "pixellated"]
    )


def test_pixellated_offsets(psf, expected):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert psf.pixellated().shape == expected
    assert (psf.pixellated() >= 0 ).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, expected", [
    (psf_moffat, (1.3, -1.3)),
    (psf_pixellated, (-1.3, 1.3))],
    ids=["moffat", "pixellated"]
    )


def test_pixellated(psf, expected):
    with pytest.raises(ValueError):
        psf.pixellated(size=expected)
