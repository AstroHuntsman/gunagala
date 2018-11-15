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


@pytest.mark.parametrize("psf, type", [
    (psf_moffat(), MoffatPSF),
    (psf_pixellated(), PixellatedPSF)],
    ids=["moffat", "pixellated"]
)
def test_instance(psf, type):
    assert isinstance(psf, type)
    assert isinstance(psf, PSF)


def test_FWHM(psf_moffat):
    assert psf_moffat.FWHM == 2 * u.arcsecond
    psf_moffat.FWHM = 4 * u.arcsecond
    assert psf_moffat.FWHM == 1 / 15 * u.arcminute
    with pytest.raises(ValueError):
        psf_moffat.FWHM = -1 * u.degree
    psf_moffat.FWHM = 2 * u.arcsecond


@pytest.mark.parametrize("psf, t_pixel_scale, pixel_scale", [
    (psf_moffat(), 2.85, 2.85),
    (psf_pixellated(), (1 / 3), (2 / 3))],
    ids=["moffat", "pixellated"]
)
def test_pixel_scale(psf, t_pixel_scale, pixel_scale):
    psf = t_pixel_scale * u.arcsecond / u.pixel
    assert psf == t_pixel_scale * u.arcsecond / u.pixel
    psf = pixel_scale * u.arcsecond / u.pixel


# @pytest.mark.parametrize("psf, expected_n_pix", [
#     (psf_moffat(), 4.25754067000986),
#     (psf_pixellated(), 21.06994544)],
#     ids=["moffat", "pixellated"]
# )
# def test_n_pix(psf, expected_n_pix):
#     print(psf, psf.n_pix)
#     assert psf.n_pix / u.pixel == pytest.approx(expected_n_pix)


# def test_n_pix(psf_moffat):
#     assert psf_moffat.n_pix == 4.25754067000986 * u.pixel


def test_n_pix_pix(psf_pixellated):
    assert psf_pixellated.n_pix / u.pixel == pytest.approx(21.069945447)


# @pytest.mark.parametrize("psf, expected_peak", [
#     (psf_moffat(), 0.7134084656751443),
#     (psf_pixellated(), 0.08073066)],
#     ids=["moffat", "pixellated"]
# )
# def test_peak(psf, expected_peak):
#     print(psf, psf.peak)
#     assert psf.peak * u.pixel == pytest.approx(expected_peak)


# def test_peak(psf_moffat):
#     assert psf_moffat.peak == 0.7134084656751443 / u.pixel


def test_peak_pix(psf_pixellated):
    assert psf_pixellated.peak * u.pixel == pytest.approx(0.08073066)


def test_shape(psf_moffat):
    assert psf_moffat.shape == 4.7
    psf_moffat.shape = 2.5
    assert psf_moffat.shape == 2.5
    with pytest.raises(ValueError):
        psf_moffat.shape = 0.5
    psf_moffat.shape = 4.7


# This te 
@pytest.mark.parametrize("psf, image_size", [
    (psf_moffat(), (21, 21)),
    (psf_pixellated(), (21, 21)),
    (psf_moffat(), (7, 9)),
    (psf_pixellated(), (7, 9))]
)
def test_pixellated_rectangle(psf, image_size):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert isinstance(psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)), np.ndarray)
    assert psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)).shape == image_size
    assert (psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)) >= 0).all()
    assert np.isfinite(psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2))).all()


@pytest.mark.parametrize("psf", [
    (psf_moffat()),
    (psf_pixellated())],
    ids=["moffat", "pixellated"]
)
def test_pixellated_offsets(psf):
    assert isinstance(psf.pixellated(offsets=(0.3, -0.7)), np.ndarray)
    assert psf.pixellated().shape == (21, 21)
    assert (psf.pixellated() >= 0).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, test_size", [
    (psf_moffat(), (1.3, -1.3)),
    (psf_pixellated(), (-1.3, 1.3))],
    ids=["moffat", "pixellated"]
)
def test_pixellated(psf, test_size):
    with pytest.raises(ValueError):
        psf.pixellated(size=test_size)
