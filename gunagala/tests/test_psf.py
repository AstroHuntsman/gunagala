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
    assert isinstance(psf_moffat, MoffatPSF)
    assert isinstance(psf_moffat, PSF)


def test_pix(psf_pixellated):
    assert isinstance(psf_pixellated, PixellatedPSF)
    assert isinstance(psf_pixellated, PSF)


def test_FWHM(psf_moffat):
    assert psf_moffat.FWHM == 2 * u.arcsecond
    psf_moffat.FWHM = 4 * u.arcsecond
    assert psf_moffat.FWHM == 1 / 15 * u.arcminute
    with pytest.raises(ValueError):
        psf_moffat.FWHM = -1 * u.degree
    psf_moffat.FWHM = 2 * u.arcsecond


def test_pixel_scale(psf_moffat):
    psf_moffat.pixel_scale = 2.85 * u.arcsecond / u.pixel
    assert psf_moffat.pixel_scale == 2.85 * u.arcsecond / u.pixel


def test_pixel_scale_pix(psf_pixellated):
    psf_pixellated.pixel_scale = (1 / 3) * u.arcsecond / u.pixel
    assert psf_pixellated.pixel_scale == (1 / 3) * u.arcsecond / u.pixel
    psf_pixellated.pixel_scale = (2 / 3) * u.arcsecond / u.pixel


moffat = psf_moffat()
pixellated = psf_pixellated()


# @pytest.mark.parametrize("psf, expected_n_pix", [
#     (moffat, 4.25754067000986),
#     (pixellated, pytest.approx(21.06994544))],
#     ids=["moffat", "pixellated"]
# )
# def test_n_pix(psf, expected_n_pix):
#     assert psf.n_pix  == expected_n_pix


def test_n_pix(psf_moffat):
    assert psf_moffat.n_pix == 4.25754067000986 * u.pixel


def test_n_pix_pix(psf_pixellated):
    assert psf_pixellated.n_pix / u.pixel == pytest.approx(21.069945447)


def test_peak(psf_moffat):
    assert psf_moffat.peak == 0.7134084656751443 / u.pixel


def test_peak_pix(psf_pixellated):
    assert psf_pixellated.peak * u.pixel == pytest.approx(0.08073066)


def test_shape(psf_moffat):
    assert psf_moffat.shape == 4.7
    psf_moffat.shape = 2.5
    assert psf_moffat.shape == 2.5
    with pytest.raises(ValueError):
        psf_moffat.shape = 0.5
    psf_moffat.shape = 4.7


@pytest.mark.parametrize("psf, image_size", [
    (moffat, (21, 21)),
    (pixellated, (21, 21))],
    ids=["moffat", "pixellated"]
)
def test_pixellated_square(psf, image_size):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert psf.pixellated().shape == image_size
    assert (psf.pixellated() >= 0).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, image_size", [
    (moffat, (7, 9)),
    (pixellated, (7, 9))],
    ids=["moffat", "pixellated"]
)
def test_pixellated_rectangle(psf, image_size):
    assert isinstance(psf.pixellated(size=(7.2, 9.2)), np.ndarray)
    assert psf.pixellated(size=(7.2, 9.2)).shape == image_size
    assert (psf.pixellated(size=(7.2, 9.2)) >= 0).all()
    assert np.isfinite(psf.pixellated(size=(7.2, 9.2))).all()


@pytest.mark.parametrize("psf, image_size", [
    (moffat, (21, 21)),
    (pixellated, (21, 21))],
    ids=["moffat", "pixellated"]
)
def test_pixellated_offsets(psf, image_size):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert psf.pixellated().shape == image_size
    assert (psf.pixellated() >= 0).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, test_size", [
    (moffat, (1.3, -1.3)),
    (pixellated, (-1.3, 1.3))],
    ids=["moffat", "pixellated"]
)
def test_pixellated(psf, test_size):
    with pytest.raises(ValueError):
        psf.pixellated(size=test_size)
