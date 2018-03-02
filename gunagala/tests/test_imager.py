# Tests for the signal-to-noise module
import pytest
import numpy as np
import astropy.units as u
import astropy.constants as c

from gunagala.optic import Optic
from gunagala.optical_filter import Filter
from gunagala.camera import Camera
from gunagala.psf import PSF, Moffat_PSF
from gunagala.sky import Sky, Simple, ZodiacalLight
from gunagala.imager import Imager
from gunagala.imager import create_imagers


@pytest.fixture(scope='module')
def lens():
    lens = Optic(aperture=14 * u.cm,
                 focal_length=0.391 * u.m,
                 throughput='canon_throughput.csv')
    return lens


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


@pytest.fixture(scope='module')
def filters():
    g = Filter(transmission_filename='astrodon_g.csv')
    r = Filter(chebyshev_params={'wave1': 550 * u.nm,
                                 'wave2': 700 * u.nm,
                                 'order': 50,
                                 'ripple': 0.14,
                                 'peak': 0.95})
    return {'g': g, 'r': r}


@pytest.fixture(scope='module', params=('g', 'r'))
def filter_name(request):
    return request.param


@pytest.fixture(scope='module', params=('SSO', 'ZL'))
def sky(request):
    if request.param == 'SSO':
        sky = Simple(surface_brightness={'g': 22.5, 'r': 21.5})
    elif request.param == 'ZL':
        sky = ZodiacalLight()
    else:
        pytest.fail("Unknown sky type '{}'!".format(request.param))

    return sky


@pytest.fixture(scope='module')
def psf():
    psf = Moffat_PSF(FWHM=1 / 30 * u.arcminute, shape=4.7)
    return psf


@pytest.fixture(scope='module')
def imager(lens, ccd, filters, psf, sky):
    imager = Imager(optic=lens, camera=ccd, filters=filters, psf=psf, sky=sky, num_imagers=5, num_per_computer=5)
    return imager


def test_init(imager):
    assert isinstance(imager, Imager)
    assert imager.pixel_scale == (5.4 * u.micron / (391 * u.mm * u.pixel)).to(u.arcsecond / u.pixel,
                                                                              equivalencies=u.dimensionless_angles())
    assert imager.pixel_area == (5.4 * u.micron /
                                 (391 * u.mm * u.pixel)).to(u.arcsecond / u.pixel,
                                                            equivalencies=u.dimensionless_angles())**2 * u.pixel
    assert (imager.field_of_view == (3326, 2504) * u.pixel * imager.pixel_scale).all()


def test_conversions(imager, filter_name):
    mag = 25 * u.ABmag

    # Round trip from AB magnitudes to photo-electrons per second and back
    rate = imager.ABmag_to_rate(mag, filter_name)
    assert imager.rate_to_ABmag(rate, filter_name) == mag

    # Round trip from AB magnitudes to total flux and bacl
    flux = imager.ABmag_to_flux(mag, filter_name)
    assert imager.flux_to_ABmag(flux, filter_name) == mag

@pytest.mark.xfail
def test_conversions_2(imager, filter_name):
    mag = 25 * u.ABmag

    # AB magnitudes to flux, manual conversion to rate, then rate back to mag
    flux = imager.ABmag_to_flux(mag, filter_name)
    photon_energy = c.h * c.c / (imager.pivot_wave[filter_name] * u.photon)
    rate = flux * imager.optic.aperture_area * imager.efficiency[filter_name] / photon_energy
    assert imager.rate_to_ABmag(rate, filter_name) == mag

    # AB magnitudes to rate, manaul conversion to flux, then back to mag
    rate = imager.ABmag_to_rate(mag, filter_name)
    photon_energy = c.h * c.c / (imager.pivot_wave[filter_name] * u.photon)
    flux = rate * photon_energy / (imager.optic.aperture_area * imager.efficiency[filter_name])
    assert imager.flux_to_ABmag(flux, filter_name) == mag

def test_time_calculations(imager, filter_name):
    total_exposure_time = 28 * u.hour
    sub_exp_time = 10 * u.minute
    n_subs = (total_exposure_time / sub_exp_time).to(u.dimensionless_unscaled)
    exp_list = np.ones(int(n_subs)) * sub_exp_time

    # Round trip test, including rounding down.
    total_elapsed_time = imager.total_elapsed_time(exp_list)
    assert imager.total_exposure_time(total_elapsed_time + sub_exp_time / 2, sub_exp_time)

def test_extended_snr(imager, filter_name):
    sb = 25 * u.ABmag
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # Calculate signal to noise ratio given surface brightness and exposure time
    snr = imager.extended_source_snr(surface_brightness=sb,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub,
                                     calc_type='per arcsecond squared',
                                     saturation_check=True)

    # Calculating exposure time given surface brightness and calculated SNR should match original exposure time
    # SNR target reduced a tiny amount to prevent finite numerical precision causing exposure time to get rounded up.
    assert t_exp == imager.extended_source_etc(surface_brightness=sb,
                                               filter_name=filter_name,
                                               snr_target=snr * 0.999999999999,
                                               sub_exp_time=t_sub,
                                               calc_type='per arcsecond squared',
                                               saturation_check=True)

    # Calculating surface brightness given exposure time and SNR should match original surface brightness
    assert sb == imager.extended_source_limit(total_exp_time=t_exp,
                                              filter_name=filter_name,
                                              snr_target=snr,
                                              sub_exp_time=t_sub,
                                              calc_type='per arcsecond squared')

    # Can't use pixel binning with per arcsecond squared signal, noise values
    with pytest.raises(ValueError):
        imager.extended_source_signal_noise(surface_brightness=sb,
                                            filter_name=filter_name,
                                            total_exp_time=t_exp,
                                            sub_exp_time=t_sub,
                                            calc_type='per arcsecond squared',
                                            saturation_check=True,
                                            binning=4)

    # Can't calculate signal to noise ratio per banana, either.
    with pytest.raises(ValueError):
        imager.extended_source_snr(surface_brightness=sb,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=t_sub,
                                   calc_type='per banana',
                                   saturation_check=False,
                                   binning=16)


def test_extended_saturation(imager, filter_name):
    sb = 25 * u.ABmag
    t_exp = 28 * u.hour
    t_sub = 28 * u.hour

    # Very long sub-exposure should saturate and give zero signal to noise
    snr = imager.extended_source_snr(surface_brightness=sb,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub)
    assert snr == 0 * u.dimensionless_unscaled

    # Not if the saturation check is turned off though.
    snr = imager.extended_source_snr(surface_brightness=sb,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub,
                                     saturation_check=False)
    assert snr > 0 * u.dimensionless_unscaled

    # Should work with the sky measurement option, too.
    snr = imager.extended_source_snr(surface_brightness=None,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub)
    assert snr == 0 * u.dimensionless_unscaled

    # ETC should also give zero
    t = imager.extended_source_etc(surface_brightness=sb,
                                   filter_name=filter_name,
                                   sub_exp_time=t_sub,
                                   snr_target=3.0)
    assert t == 0 * u.second

    t = imager.extended_source_etc(surface_brightness=sb,
                                   filter_name=filter_name,
                                   sub_exp_time=t_sub,
                                   snr_target=3.0,
                                   saturation_check=False)
    assert t > 0 * u.second

    t = imager.extended_source_etc(surface_brightness=None,
                                   filter_name=filter_name,
                                   sub_exp_time=t_sub,
                                   snr_target=3.0)
    assert t == 0 * u.second


def test_sky_snr(imager, filter_name):
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # If surface brightness is set to None (or anything False) should get the signal to noise ratio
    # for measurement of the sky brightness.
    snr = imager.extended_source_snr(surface_brightness=None,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub)
    assert snr > 0 * u.dimensionless_unscaled

    # Can also do this with the ETC, and should get consistent answers.
    t = imager.extended_source_etc(surface_brightness=None,
                                   filter_name=filter_name,
                                   sub_exp_time=t_sub,
                                   snr_target=snr * 0.999999999999,)
    assert t == t_exp


def test_extended_binning(imager, filter_name):
    sb = 25 * u.ABmag
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # Per pixel SNR shoudl scale with pixel binning^0.5
    snr_1_pix = imager.extended_source_snr(surface_brightness=sb,
                                           filter_name=filter_name,
                                           total_exp_time=t_exp,
                                           sub_exp_time=t_sub,
                                           calc_type='per pixel',
                                           saturation_check=False)
    snr_4_pix = imager.extended_source_snr(surface_brightness=sb,
                                           filter_name=filter_name,
                                           total_exp_time=t_exp,
                                           sub_exp_time=t_sub,
                                           calc_type='per pixel',
                                           saturation_check=False,
                                           binning=4)
    assert snr_4_pix == 2 * snr_1_pix

    # Binned exposure time given surface brightness and SNR should match original exposure time.
    assert t_exp.to(u.second).value == pytest.approx(imager.extended_source_etc(surface_brightness=sb,
                                                                                filter_name=filter_name,
                                                                                snr_target=snr_4_pix,
                                                                                sub_exp_time=t_sub,
                                                                                saturation_check=False,
                                                                                binning=4).to(u.second).value,
                                                                                rel=0.1)

def test_extended_arrays(imager, filter_name):
    # SNR functions should handle arrays values for any of the main arguments.
    assert len(imager.extended_source_snr(surface_brightness=(20.0, 25.0) * u.ABmag,
                                          filter_name=filter_name,
                                          total_exp_time=28 * u.hour,
                                          sub_exp_time=600 * u.second)) == 2

    assert len(imager.extended_source_snr(surface_brightness=25.0 * u.ABmag,
                                          filter_name=filter_name,
                                          total_exp_time=(10, 20, 30) * u.hour,
                                          sub_exp_time=(200, 400, 600) * u.second)) == 3

    assert len(imager.extended_source_etc(surface_brightness=25 * u.ABmag,
                                          filter_name=filter_name,
                                          snr_target=(3.0, 5.0),
                                          sub_exp_time=600 * u.second)) == 2

    assert len(imager.extended_source_etc(surface_brightness=25 * u.ABmag,
                                          filter_name=filter_name,
                                          snr_target=1.0,
                                          sub_exp_time=(200, 400, 600) * u.second)) == 3

    assert len(imager.extended_source_limit(total_exp_time=(20, 30) * u.hour,
                                            filter_name=filter_name,
                                            snr_target=1.0,
                                            sub_exp_time=600 * u.second)) == 2

    assert len(imager.extended_source_limit(total_exp_time=28 * u.hour,
                                            filter_name=filter_name,
                                            snr_target=(3.0, 5.0),
                                            sub_exp_time=(300, 600) * u.second)) == 2


def test_extended_rates(imager, filter_name):
    # SNR function optionally accept electrons / pixel per second instead of AB mag per arcsecond^2
    rate = 0.1 * u.electron / (u.pixel * u.second)
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # Calculate signal to noise ratio given surface brightness and exposure time
    snr = imager.extended_source_snr(surface_brightness=rate,
                                     filter_name=filter_name,
                                     total_exp_time=t_exp,
                                     sub_exp_time=t_sub,
                                     calc_type='per arcsecond squared',
                                     saturation_check=True)

    # Calculating exposure time given surface brightness and calculated SNR should match original exposure time
    # SNR target reduced a tiny amount to prevent finite numerical precision causing exposure time to get rounded up.
    assert t_exp == imager.extended_source_etc(surface_brightness=rate,
                                               filter_name=filter_name,
                                               snr_target=snr * 0.999999999999,
                                               sub_exp_time=t_sub,
                                               calc_type='per arcsecond squared',
                                               saturation_check=True)

    # Calculating surface brightness given exposure time and SNR should match original surface brightness
    assert imager.rate_to_SB(rate, filter_name) == imager.extended_source_limit(total_exp_time=t_exp,
                                                                                filter_name=filter_name,
                                                                                snr_target=snr,
                                                                                sub_exp_time=t_sub,
                                                                                calc_type='per arcsecond squared')

    # Can't use pixel binning with per arcsecond squared signal, noise values
    with pytest.raises(ValueError):
        imager.extended_source_signal_noise(surface_brightness=rate,
                                            filter_name=filter_name,
                                            total_exp_time=t_exp,
                                            sub_exp_time=t_sub,
                                            calc_type='per arcsecond squared',
                                            saturation_check=True,
                                            binning=4)


def test_point_snr(imager, filter_name):
    b = 25 * u.ABmag
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # Calculate signal to noise ratio given brightness and exposure time
    snr = imager.point_source_snr(brightness=b,
                                  filter_name=filter_name,
                                  total_exp_time=t_exp,
                                  sub_exp_time=t_sub,
                                  saturation_check=True)

    # Calculating exposure time given brightness and calculated SNR should match original exposure time
    # SNR target reduced a tiny amount to prevent finite numerical precision causing exposure time to get rounded up.
    assert t_exp == imager.point_source_etc(brightness=b,
                                            filter_name=filter_name,
                                            snr_target=snr * 0.999999999999,
                                            sub_exp_time=t_sub,
                                            saturation_check=True)

    # Calculating brightness given exposure time and SNR should match original brightness
    assert b.value == pytest.approx(imager.point_source_limit(total_exp_time=t_exp,
                                                              filter_name=filter_name,
                                                              snr_target=snr,
                                                              sub_exp_time=t_sub).value)


def test_point_arrays(imager, filter_name):
    # SNR functions should handle arrays values for any of the main arguments.
    assert len(imager.point_source_snr(brightness=(20.0, 25.0) * u.ABmag,
                                       filter_name=filter_name,
                                       total_exp_time=28 * u.hour,
                                       sub_exp_time=600 * u.second)) == 2

    assert len(imager.point_source_snr(brightness=25.0 * u.ABmag,
                                       filter_name=filter_name,
                                       total_exp_time=(10, 20, 30) * u.hour,
                                       sub_exp_time=(200, 400, 600) * u.second)) == 3

    assert len(imager.point_source_etc(brightness=25 * u.ABmag,
                                       filter_name=filter_name,
                                       snr_target=(3.0, 5.0),
                                       sub_exp_time=600 * u.second)) == 2

    assert len(imager.point_source_etc(brightness=25 * u.ABmag,
                                       filter_name=filter_name,
                                       snr_target=1.0,
                                       sub_exp_time=(200, 400, 600) * u.second)) == 3

    assert len(imager.point_source_limit(total_exp_time=(20, 30) * u.hour,
                                         filter_name=filter_name,
                                         snr_target=1.0,
                                         sub_exp_time=600 * u.second)) == 2

    assert len(imager.point_source_limit(total_exp_time=28 * u.hour,
                                         filter_name=filter_name,
                                         snr_target=(3.0, 5.0),
                                         sub_exp_time=(300, 600) * u.second)) == 2


def test_point_rates(imager, filter_name):
    rate = 0.1 * u.electron / u.second
    t_exp = 28 * u.hour
    t_sub = 600 * u.second

    # Calculate signal to noise ratio given brightness and exposure time
    snr = imager.point_source_snr(brightness=rate,
                                  filter_name=filter_name,
                                  total_exp_time=t_exp,
                                  sub_exp_time=t_sub,
                                  saturation_check=True)

    # Calculating exposure time given brightness and calculated SNR should match original exposure time
    # SNR target reduced a tiny amount to prevent finite numerical precision causing exposure time to get rounded up.
    assert t_exp == imager.point_source_etc(brightness=rate,
                                            filter_name=filter_name,
                                            snr_target=snr * 0.999999999999,
                                            sub_exp_time=t_sub,
                                            saturation_check=True)

    # Calculating brightness given exposure time and SNR should match original brightness.
    # This particular comparison seems to fail due to floating point accuracy, need to allow some tolerance.
    assert imager.rate_to_ABmag(rate, filter_name).value == pytest.approx(imager.point_source_limit(total_exp_time=t_exp,
                                                                                                    filter_name=filter_name,
                                                                                                    snr_target=snr,
                                                                                                    sub_exp_time=t_sub).value,
                                                                          abs=1e-14)


def test_exposure(imager):
    t_elapsed = 2700 * u.second
    t_sub = 600 * u.second
    t_exp = imager.total_exposure_time(t_elapsed, t_sub)
    assert t_exp == 4 * t_sub


def test_elapsed(imager):
    exp_list = (150, 300, 600, 600) * u.second
    t_elapsed = imager.total_elapsed_time(exp_list)
    assert t_elapsed == 1650 * u.second + 4 * imager.num_per_computer * imager.camera.readout_time


def test_extended_sat_mag(imager, filter_name):
    t_exp = 28 * u.hour
    t_sub = 600 * u.second
    sat_mag = imager.extended_source_saturation_mag(sub_exp_time=t_sub, filter_name=filter_name)

    assert imager.extended_source_snr(surface_brightness=sat_mag.value - 0.01,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=t_sub) == 0 * u.dimensionless_unscaled

    assert imager.extended_source_snr(surface_brightness=sat_mag.value + 0.01,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=t_sub) != 0 * u.dimensionless_unscaled

    assert imager.extended_source_snr(surface_brightness=sat_mag.value - 0.01,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=t_sub,
                                      saturation_check=False) != 0 * u.dimensionless_unscaled

    assert imager.extended_source_etc(surface_brightness=sat_mag.value - 0.01,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=t_sub) == 0 * u.second

    assert imager.extended_source_etc(surface_brightness=sat_mag.value + 0.01,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=t_sub) != 0 * u.second

    assert imager.extended_source_etc(surface_brightness=sat_mag.value - 0.01,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=t_sub,
                                      saturation_check=False) != 0 * u.second


def test_extended_sat_exp(imager, filter_name):
    sb = 10 * u.ABmag
    t_exp = 28 * u.hour
    sat_exp = imager.extended_source_saturation_exp(surface_brightness=sb, filter_name=filter_name)

    assert imager.extended_source_snr(surface_brightness=sb,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=sat_exp * 1.01) == 0 * u.dimensionless_unscaled

    assert imager.extended_source_snr(surface_brightness=sb,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=sat_exp * 0.99) != 0 * u.dimensionless_unscaled

    assert imager.extended_source_snr(surface_brightness=sb,
                                      filter_name=filter_name,
                                      total_exp_time=t_exp,
                                      sub_exp_time=sat_exp * 1.01,
                                      saturation_check=False) != 0 * u.dimensionless_unscaled

    assert imager.extended_source_etc(surface_brightness=sb,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=sat_exp * 1.01) == 0 * u.second

    assert imager.extended_source_etc(surface_brightness=sb,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=sat_exp * 0.99) != 0 * u.second

    assert imager.extended_source_etc(surface_brightness=sb,
                                      filter_name=filter_name,
                                      snr_target=3.0,
                                      sub_exp_time=sat_exp * 1.01,
                                      saturation_check=False) != 0 * u.second

    assert imager.extended_source_saturation_mag(sub_exp_time=sat_exp, filter_name=filter_name) == sb


def test_point_sat_mag(imager, filter_name):
    t_exp = 28 * u.hour
    t_sub = 600 * u.second
    sat_mag = imager.point_source_saturation_mag(sub_exp_time=t_sub, filter_name=filter_name)

    assert imager.point_source_snr(brightness=sat_mag.value - 0.01,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=t_sub) == 0 * u.dimensionless_unscaled

    assert imager.point_source_snr(brightness=sat_mag.value + 0.01,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=t_sub) != 0 * u.dimensionless_unscaled

    assert imager.point_source_snr(brightness=sat_mag.value - 0.01,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=t_sub,
                                   saturation_check=False) != 0 * u.dimensionless_unscaled

    assert imager.point_source_etc(brightness=sat_mag.value - 0.01,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=t_sub) == 0 * u.second

    assert imager.point_source_etc(brightness=sat_mag.value + 0.01,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=t_sub) != 0 * u.second

    assert imager.point_source_etc(brightness=sat_mag.value - 0.01,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=t_sub,
                                   saturation_check=False) != 0 * u.second


def test_point_sat_exp(imager, filter_name):
    b = 10 * u.ABmag
    t_exp = 28 * u.hour
    sat_exp = imager.point_source_saturation_exp(brightness=b, filter_name=filter_name)

    assert imager.point_source_snr(brightness=b,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=sat_exp * 1.01) == 0 * u.dimensionless_unscaled

    assert imager.point_source_snr(brightness=b,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=sat_exp * 0.99) != 0 * u.dimensionless_unscaled

    assert imager.point_source_snr(brightness=b,
                                   filter_name=filter_name,
                                   total_exp_time=t_exp,
                                   sub_exp_time=sat_exp * 1.01,
                                   saturation_check=False) != 0 * u.dimensionless_unscaled

    assert imager.point_source_etc(brightness=b,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=sat_exp * 1.01) == 0 * u.second

    assert imager.point_source_etc(brightness=b,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=sat_exp * 0.99) != 0 * u.second

    assert imager.point_source_etc(brightness=b,
                                   filter_name=filter_name,
                                   snr_target=3.0,
                                   sub_exp_time=sat_exp * 1.01,
                                   saturation_check=False) != 0 * u.second

    assert imager.point_source_saturation_mag(sub_exp_time=sat_exp, filter_name=filter_name) == b


def test_sequence(imager, filter_name):
    brightest = 10 * u.ABmag
    ratio = 2.0
    t_max = 600 * u.second
    faintest = 25 * u.ABmag

    t_exps = imager.exp_time_sequence(bright_limit=brightest,
                                      filter_name=filter_name,
                                      exp_time_ratio=ratio,
                                      longest_exp_time=t_max,
                                      faint_limit=faintest)

    # Shortest exposure time should be less than the time to saturation on the brightest targets
    assert t_exps[0] <= imager.point_source_saturation_exp(brightness=brightest, filter_name=filter_name)

    # Ratio between exposures should be close to exp_time_ratio (apart from rounding to nearest 0.01 seconds)
    assert (t_exps[1] / t_exps[0]).value == pytest.approx(2, rel=0.1)

    # Total exposure time should be close to the required exposure time for a non-HDR sequence reaching the same
    # depth.
    non_hdr_t_exp = imager.point_source_etc(brightness=faintest,
                                            filter_name=filter_name,
                                            snr_target=5.0,
                                            sub_exp_time=t_max)
    assert t_exps.sum() > non_hdr_t_exp - 5 * u.second
    assert t_exps.sum().value == pytest.approx(non_hdr_t_exp.value, abs=t_max.value * 2)

    # Can't set both num_long_exp and faint_limit
    with pytest.raises(ValueError):
        imager.exp_time_sequence(bright_limit=brightest,
                                 filter_name=filter_name,
                                 exp_time_ratio=ratio,
                                 longest_exp_time=t_max,
                                 num_long_exp=5,
                                 faint_limit=faintest)
    # Or neither
    with pytest.raises(ValueError):
        imager.exp_time_sequence(bright_limit=brightest,
                                 filter_name=filter_name,
                                 exp_time_ratio=ratio,
                                 longest_exp_time=t_max)

    # If given a really bright target then shortest exposure should just be close to the minimum that the camera
    # is capabable off.
    t_exps_bright = imager.exp_time_sequence(bright_limit=-10 * u.ABmag,
                                             filter_name=filter_name,
                                             exp_time_ratio=ratio,
                                             longest_exp_time=t_max,
                                             faint_limit=faintest)

    assert t_exps_bright[0] >= imager.camera.minimum_exposure
    assert t_exps_bright[0].value == pytest.approx(imager.camera.minimum_exposure.value, rel=ratio)

    # If given a faint minimum magnitude (won't saturate in longest subs) then should return a non-HDR sequence.
    t_exps = imager.exp_time_sequence(bright_limit=20 * u.ABmag,
                                      filter_name=filter_name,
                                      exp_time_ratio=ratio,
                                      longest_exp_time=t_max,
                                      faint_limit=faintest)
    assert (t_exps == t_max).all()

    # Can instead specify number of long exposures directly.
    t_exps_n_long = imager.exp_time_sequence(bright_limit=brightest,
                                             filter_name=filter_name,
                                             exp_time_ratio=ratio,
                                             longest_exp_time=t_max,
                                             num_long_exp=3)
    assert (t_exps_n_long == t_max).sum() == 3

    # Can also directly specify the shortest exposure time. Will get rounded down to next integer exponent of the
    # exposure time ratio.
    t_exps_short = imager.exp_time_sequence(shortest_exp_time=100 * u.second,
                                            filter_name=filter_name,
                                            exp_time_ratio=ratio,
                                            longest_exp_time=t_max,
                                            num_long_exp=3)
    assert t_exps_short[0] == 75 * u.second

    # Can't specicify both bright_limit and shortest_exp_time
    with pytest.raises(ValueError):
        imager.exp_time_sequence(bright_limit=brightest,
                                 filter_name=filter_name,
                                 shortest_exp_time=100 * u.second,
                                 exp_time_ratio=ratio,
                                 longest_exp_time=t_max,
                                 num_long_exp=3)

    # Or neither
    with pytest.raises(ValueError):
        imager.exp_time_sequence(exp_time_ratio=ratio,
                                 filter_name=filter_name,
                                 longest_exp_time=t_max,
                                 num_long_exp=3)

    # Can override default SNR target of 5.0.
    t_exps_low_snr = imager.exp_time_sequence(bright_limit=brightest,
                                              filter_name=filter_name,
                                              exp_time_ratio=ratio,
                                              longest_exp_time=t_max,
                                              faint_limit=faintest,
                                              snr_target=2.5)
    assert t_exps_low_snr.sum().value == pytest.approx(t_exps.sum().value / 4, rel=0.1)


def test_snr_vs_mag(imager, filter_name, tmpdir):
    exp_times = imager.exp_time_sequence(filter_name=filter_name,
                                         shortest_exp_time=100 * u.second,
                                         exp_time_ratio=2,
                                         longest_exp_time=600 * u.second,
                                         num_long_exp=4)

    # Basic usage
    mags, snrs = imager.snr_vs_ABmag(exp_times=exp_times, filter_name=filter_name)

    # With plot
    plot_path = tmpdir.join('snr_vs_mag_test.png')
    mags, snrs = imager.snr_vs_ABmag(exp_times=exp_times, filter_name=filter_name, plot=plot_path.strpath)
    assert plot_path.check()

    # Finer sampling
    mags2, snrs2 = imager.snr_vs_ABmag(exp_times=exp_times, filter_name=filter_name,
                                       magnitude_interval=0.01 * u.ABmag)
    assert len(mags2) == pytest.approx(len(mags) * 2, abs=1)

    # Different signal to noise threshold
    mags3, snrs3 = imager.snr_vs_ABmag(exp_times=exp_times, filter_name=filter_name,
                                       snr_target=10.0)
    # Roughly background limited at faint end, 10 times higher SNR should be about 2.5 mag brighter
    assert mags3[-1].value == pytest.approx(mags[-1].value - 2.5, abs=0.1)


def test_create_imagers():
    imagers = create_imagers()
    assert isinstance(imagers, dict)
    for key, value in imagers.items():
        assert type(key) == str
        assert isinstance(value, Imager)
