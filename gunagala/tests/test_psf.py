import pytest
import numpy as np
import astropy.units as u

from gunagala.psf import PSF, MoffatPSF, PixellatedPSF


@pytest.fixture(scope='module')
def psf_moffat():
    return make_moffat()


def make_moffat():
    psf = MoffatPSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


@pytest.fixture(scope='module')
def psf_pixellated():
    return make_pixellated()


def make_pixellated():
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
        _ = PSF(FWHM=1 / 30 * u.arcminute)


@pytest.mark.parametrize("psf, type", [
    (make_moffat(), MoffatPSF),
    (make_pixellated(), PixellatedPSF)],
    ids=["moffat", "pixellated"]
)
def test_instance(psf, type):
    assert isinstance(psf, type)
    assert isinstance(psf, PSF)


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


def test_n_pix(psf_moffat):
    assert psf_moffat.n_pix == 4.25754067000986 * u.pixel


def test_n_pix_pix(psf_pixellated):
    assert psf_pixellated.n_pix.to(u.pixel).value == pytest.approx(21.0699454)


def test_peak(psf_moffat):
    assert psf_moffat.peak == 0.7134084656751443 / u.pixel


def test_peak_pix(psf_pixellated):
    assert psf_pixellated.peak.to(1 / u.pixel).value == pytest.approx(0.08073066)


def test_shape(psf_moffat):
    assert psf_moffat.shape == 4.7
    psf_moffat.shape = 2.5
    assert psf_moffat.shape == 2.5
    with pytest.raises(ValueError):
        psf_moffat.FWHM = -1 * u.degree
    psf_moffat.FWHM = 2 * u.arcsecond




@pytest.mark.parametrize("psf, t_pixel_scale, pixel_scale", [
    (make_moffat(), 2.85, 2.85),
    (make_pixellated(), (1 / 3), (2 / 3))],
    ids=["moffat", "pixellated"]
)
def test_pixel_scale(psf, t_pixel_scale, pixel_scale):
    psf.pixel_scale = t_pixel_scale * u.arcsecond / u.pixel
    assert psf.pixel_scale == t_pixel_scale * u.arcsecond / u.pixel
    psf.pixel_scale = pixel_scale * u.arcsecond / u.pixel


@pytest.mark.parametrize("psf, expected_n_pix, pixel_scale", [
    (make_moffat(), 4.25754067000986, 2.85),
    (make_pixellated(), 21.06994544, (2 / 3))],
    ids=["moffat", "pixellated"]
)
def test_n_pix(psf, expected_n_pix, pixel_scale):
    psf.pixel_scale = pixel_scale * u.arcsecond / u.pixel
    assert psf.n_pix.to(u.pixel).value == pytest.approx(expected_n_pix)


@pytest.mark.parametrize("psf, expected_peak, pixel_scale", [
    (make_moffat(), 0.7134084656751443, 2.85),
    (make_pixellated(), 0.08073066, (2 / 3))],
    ids=["moffat", "pixellated"]
)
def test_peak(psf, expected_peak, pixel_scale):
    psf.pixel_scale = pixel_scale * u.arcsecond / u.pixel
    assert psf.peak.to(1 / (u.pixel)).value == pytest.approx(expected_peak)





@pytest.mark.parametrize("psf, image_size", [
    (make_moffat(), (21, 21)),
    (make_pixellated(), (21, 21)),
    (make_moffat(), (7, 9)),
    (make_pixellated(), (7, 9))],
    ids=["moffat_square",
         "pixellated_square",
         "moffat_rectangle",
         "pixellated_rectangle"]
)
def test_pixellated_dimension(psf, image_size):
    assert isinstance(psf.pixellated(), np.ndarray)
    assert isinstance(psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)), np.ndarray)
    assert psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)).shape == image_size
    assert (psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2)) >= 0).all()
    assert np.isfinite(psf.pixellated(size=(
        image_size[0] + 0.2, image_size[1] + 0.2))).all()


@pytest.mark.parametrize("psf, offset", [
    (make_moffat(), (0.0, 0.0)),
    (make_pixellated(), (0.0, 0.0)),
    (make_moffat(), (0.3, -0.7)),
    (make_pixellated(), (0.3, -0.7))],
    ids=["moffat_centre_offsets",
         "pixellated_centre_offsets",
         "moffat_noncentre_offsets",
         "pixellated_noncentre_offsets"]
)
def test_offsets(psf, offset):
    assert isinstance(psf.pixellated(offsets=offset), np.ndarray)
    assert psf.pixellated().shape == (21, 21)
    assert (psf.pixellated() >= 0).all()
    assert np.isfinite(psf.pixellated()).all()


@pytest.mark.parametrize("psf, test_size", [
    (make_moffat(), (1.3, -1.3)),
    (make_pixellated(), (-1.3, 1.3))],
    ids=["moffat", "pixellated"]
)
def test_pixellated_invalid_size(psf, test_size):
    with pytest.raises(ValueError):
        psf.pixellated(size=test_size)
