"""
Imaging instruments
"""
import math
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import poisson, norm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from astropy import constants as c
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from gunagala.optic import Optic
from gunagala.optical_filter import Filter
from gunagala.camera import Camera
from gunagala.psf import PSF, MoffatPSF
from gunagala.sky import Sky, Simple, ZodiacalLight
from gunagala.config import load_config
from gunagala.utils import ensure_unit

def create_imagers(config=None):
    """
    Parse config and create a corresponding dictionary of Imager objects.

    Parses a configuration and creates a dictionary of Imager objects
    corresponding to the imaging instruments described by that config.
    The config can be passed to the function as a dictionary object,
    otherwise the config will be read from the
    `gunagala/data/performance.yaml` and, if it exists, the
    `performance_local.yaml` file.

    Parameters
    ----------
    config : dict, optional
        a dictionary containing the performance data configuration. If not
        specified `load_config()` will be used to attempt to load a
        `performance.yaml` and/or `performance_local.yaml` file and use
        the resulting config.

    Returns
    -------
    imagers: dict
        dictionary of `Imager` objects.
    """

    if config is None:
        config = load_config('performance')

    # Caches for instantiated objects
    optics = dict()
    cameras = dict()
    filters = dict()
    psfs = dict()
    skys = dict()
    imagers = dict()

    # Setup imagers
    for name, imager_info in config['imagers'].items():
        optic_name = imager_info['optic']
        try:
            # Try to get from cache
            optic = optics[optic_name]
        except KeyError:
            # Create optic from this imager
            optic_info = config['optics'][optic_name]
            optic = Optic(**optic_info)

            # Put in cache
            optics[optic_name] = optic
            camera_name = imager_info['camera']
        try:
            # Try to get from cache
            camera = cameras[camera_name]
        except KeyError:
            # Create camera for this imager
            camera_info = config['cameras'][camera_name]
            if type(camera_info['resolution']) == str:
                camera_info['resolution'] = [int(a) for a in camera_info['resolution'].split(',')]
            camera = Camera(**camera_info)

            # Put in cache
            cameras[camera_name] = camera

        bands = {}
        band_names = imager_info['filters']
        for band_name in band_names:
            try:
                # Try to get from cache
                bands[band_name] = filters[band_name]
            except KeyError:
                # Create Filter for this imager
                filter_info = config['filters'][band_name]
                bands[band_name] = Filter(**filter_info)

                # Put in cache
                filters[band_name] = bands[band_name]

        psf_name = imager_info['psf']
        # Don't cache these as their attributes get modified by the Imager they're associated with so
        # each Imager should get a new instance.
        psf_info = config['psfs'][psf_name]
        assert issubclass(globals()[psf_info['model']], PSF)
        psf = globals()[psf_info['model']](**psf_info)

        sky_name = imager_info['sky']
        try:
            # Try to get one from the cache
            sky = skys[sky_name]
        except KeyError:
            # Create sky for this imagers
            sky_info = config['skys'][sky_name]
            assert issubclass(globals()[sky_info['model']], Sky)
            sky = globals()[sky_info['model']](**sky_info)

        imagers[name] = Imager(optic,
                               camera,
                               bands,
                               psf,
                               sky,
                               imager_info.get('num_imagers', 1),
                               imager_info.get('num_per_computer', 1))
    return imagers


class Imager:
    """
    Class representing an astronomical imaging instrument.

    Class representing a complete astronomical imaging system, including
    optics, optical filters and camera.

    Also includes point spread function and sky background models.
    Optionally it can be used to represent an array of identical,
    co-aligned imager using the `num_imagers` parameter to specify the
    number of copies.

    Parameters
    ----------
    optic : gunagala.optic.Optic
        Optical system model.
    camera : gunagala.camera.Camera
        Camera (image sensor) model.
    filters : dict of gunagala.filter.Filter
        Dictionary of optical filter models.
    psf : gunagala.psf.PSF
        Point spread function model.
    sky : gunagala.sky.Sky
        Sky background model.
    num_imagers : int, optional
        the number of identical, co-aligned imagers represented by this
        `Imager`. The default is 1.
    num_per_computer : int, optional
        number of cameras connected to each computer. Used in situations
        where multiple cameras must be readout sequentially so the
        effective readout time is equal to the readout time of a single
        camera multiplied by the `num_per_computer`. The default is 1.

    Attributes
    ----------
    optic : gunagala.optic.Optic
        Same as parameters.
    camera : gunagala.camera.Camera
        Same as parameters.
    filters : dict
        Same as parameters.
    psf : gunagala.psf.PSF
        Same as parameters.
    sky : gunagala.sky.Sky
        Same as parameters.
    num_imagers : int
        Same as parameters.
    num_per_computer : int
        Same as parameters.
    filter_names : list of str
        List of filter names from `filters`.
    pixel_scale : astropy.units.Quantity
        Pixel scale in arcseconds/pixel units.
    pixel_area : astropy.units.Quantity
        Pixel area in arseconds^2/pixel units.
    field_of_view : astropy.units.Quantity
        Field of view (horizontal, vertical) in degrees.
    wcs : astropy.wcs.WCS
        Template world coordinate system (WCS) for sky coordinate/pixel
        coordinate mapping.
    wavelengths : astropy.units.Quantity
        List of wavelengths used for wavelength dependent
        attributes/calculations.
    efficiencies : dict of astropy.units.Quantity
        End to end efficiency as a function of wavelegth for each filter
        bandpass.
    efficiency : dict of astropy.units.Quantity
        Mean end to end efficiencies for each filter bandpass.
    mean_wave : dict of astropy.units.Quantity
        Mean wavelength for each filter bandpass.
    pivot_wave : dict of astropy.units.Quantity
        Pivot wavelength for each filter bandpass.
    bandwidth : dict of astropy.units.Quantity
        Bandwidths for each filter bandpass (STScI definition).
    sky_rate : dict of astropy.units.Quantity
        Detected electrons/s/pixel due to the sky background for each
        filter bandpass.
    """
    def __init__(self, optic, camera, filters, psf, sky, num_imagers=1, num_per_computer=1):

        if not isinstance(optic, Optic):
            raise ValueError("'{}' is not an instance of the Optic class!".format(optic))
        if not isinstance(camera, Camera):
            raise ValueError("'{}' is not an instance of the Camera class!".format(camera))
        for band in filters.values():
            if not isinstance(band, Filter):
                raise ValueError("'{}' is not an instance of the Filter class!".format(band))
        if not isinstance(psf, PSF):
            raise ValueError("'{}' is not an instance of the PSF class!".format(psf))
        if not isinstance(sky, Sky):
            raise ValueError("'{}' is not an instance of the Sky class!".format(sky))

        self.optic = optic
        self.camera = camera
        self.filter_names = filters.keys()
        self.filters = filters
        self.psf = psf
        self.sky = sky
        self.num_imagers = int(num_imagers)
        self.num_per_computer = int(num_per_computer)

        # Calculate pixel scale, area
        self.pixel_scale = (self.camera.pixel_size / self.optic.focal_length)
        self.pixel_scale = self.pixel_scale.to(u.arcsecond / u.pixel,
                                               equivalencies=u.equivalencies.dimensionless_angles())
        self.pixel_area = self.pixel_scale**2 * u.pixel  # arcsecond^2 / pixel
        self.psf.pixel_scale = self.pixel_scale

        # Calculate field of view.
        self.field_of_view = (self.camera.resolution * self.pixel_scale)
        self.field_of_view = self.field_of_view.to(u.degree, equivalencies=u.dimensionless_angles())

        # Pass focal plane beam half-cone angles to Filters so that angle of incidence effects on filter transmission
        # profile will be modelled where appropriate.
        for filter in self.filters.values():
            filter.theta_range = self.optic.theta_range

        # Construct a simple template WCS to store the focal plane configuration parameters
        self.wcs = WCS(naxis=2)
        self.wcs._naxis1 = self.camera.resolution[0].value
        self.wcs._naxis2 = self.camera.resolution[1].value
        self.wcs.wcs.crpix = [(self.camera.resolution[0].value + 1)/2,
                              (self.camera.resolution[1].value + 1)/2]
        self.wcs.wcs.cdelt = [self.pixel_scale.to(u.degree / u.pixel).value,
                              self.pixel_scale.to(u.degree / u.pixel).value]
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        # Calculate end to end efficiencies, etc.
        self._efficiencies()

        # Calculate sky count rates for later use
        self.sky_rate = {}
        for filter_name in self.filter_names:
            # Get surface brightness from sky model
            sb = sky.surface_brightness(filter_name=filter_name)
            if callable(sb):
                # Got a callable back, this should give us surface brightness as a function of wavelength
                surface_brightness = sb(self.wavelengths)

                # Work out what *sort* of surface brightness we got and do something appropriate
                try:
                    surface_brightness = surface_brightness.to(u.photon / (u.second * u.m**2 * u.arcsecond**2 * u.nm))
                except u.UnitConversionError:
                    raise ValueError("I don't know what to do with this!")
                else:
                    # Got photon spectral flux density. Integrate with product of efficiency, aperture area, pixel area
                    sky_rate = np.trapz(surface_brightness *
                                        self.efficiencies[filter_name] *
                                        self.optic.aperture_area *
                                        self.pixel_area, x=self.wavelengths)
                    self.sky_rate[filter_name] = sky_rate.to(u.electron / (u.second * u.pixel))
            else:
                # Not a callable, should be a Simple sky model which just returns AB magnitudes per square arcsecond
                self.sky_rate[filter_name] = self.SB_to_rate(sb, filter_name)

    def extended_source_signal_noise(self, surface_brightness, filter_name, total_exp_time, sub_exp_time,
                                     calc_type='per pixel', saturation_check=True, binning=1):
        """
        Calculates the signal and noise for an extended source with given
        surface brightness.

        Calculates the signal and noise for an extended source with given
        surface brightness. Alternatively can calculate the signal and
        noise for measurements of the sky background itself by setting the
        source surface brightness to None.

        Parameters
        ----------
        surface_brightness : astropy.units.Quantity or callable or None
            Surface brightness per arcsecond^2 of the source in ABmag
            units, or an equivalent count rate in photo-electrons per
            second per pixel, or a callable object that return surface
            brightness in spectral flux density units as a function of
            wavelength. Set to None or False to calculate the signal and
            noise for the sky background.
        filter_name : str
            Name of the optical filter to use
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        sub_exp_time : astropy.units.Quantity
            Length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal & noise per pixel or signal &
            noise per arcsecond^2. Default is 'per pixel'
        saturation_check : bool, optional
            If `True` will set both signal and noise to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.
        binning : int, optional
            Pixel binning factor. Cannot be used with calculation type
            'per arcsecond squared', will raise `ValueError` if you try.

        Returns
        -------
        signal : astropy.units.Quantity
            Total signal, units determined by calculation type.
        noise: astropy.units.Quantity
            Total noise, units determined by calculaton type.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        if calc_type not in ('per pixel', 'per arcsecond squared'):
            raise ValueError("Invalid calculation type '{}'!".format(calc_type))

        if calc_type == 'per arcsecond squared' and binning != 1:
            raise ValueError("Cannot specify pixel binning with calculation type 'per arcsecond squared'!")

        if surface_brightness:
            # Given a source brightness
            if callable(surface_brightness):
                # Surface brightness is a callable, should return surface brightness as a function of wavelength
                sfd = surface_brightness(self.wavelengths)
                # Check that the returned Quantity has the right dimensionality, if so calculate rate.
                try:
                    sfd = sfd.to(u.photon * u.second**-1 * u.m**-2 * u.arcsecond**-2 * u.nm**-1)
                except u.UnitConversionError:
                    raise ValueError("I don't know what to do with this!")
                else:
                    rate = np.trapz(sfd *
                                    self.efficiencies[filter_name] *
                                    self.optic.aperture_area *
                                    self.pixel_area, x=self.wavelengths)
                    rate = rate.to(u.electron * u.second**-1 * u.pixel**-1)
            else:
                # Not callable, should be either band averaged surface brightness in AB mag units or a detected
                # count rate.
                if not isinstance(surface_brightness, u.Quantity):
                    surface_brightness = surface_brightness * u.ABmag
                try:
                    # If surface brightness is a count rate this should work
                    rate = surface_brightness.to(u.electron / (u.pixel * u.second))
                except u.core.UnitConversionError:
                    # Direct conversion failed so assume we have surface brightness in ABmag, call conversion function
                    rate = self.SB_to_rate(surface_brightness, filter_name)
        else:
            # Measuring the sky background itself.
            rate = self.sky_rate[filter_name]

        total_exp_time = ensure_unit(total_exp_time, u.second)
        sub_exp_time = ensure_unit(sub_exp_time, u.second)

        # Round total exposure time to an integer number of sub exposures. One or both of total_exp_time or
        # sub_exp_time may be Quantity arrays, need np.ceil
        number_subs = np.ceil(total_exp_time / sub_exp_time)
        total_exp_time = number_subs * sub_exp_time

        # Noise sources (per pixel for single imager)
        signal = (rate * total_exp_time).to(u.electron / u.pixel)
        # If calculating the signal & noise for the sky itself need to avoid double counting it here
        sky_counts = self.sky_rate[filter_name] * total_exp_time if surface_brightness else 0 * u.electron / u.pixel
        dark_counts = self.camera.dark_current * total_exp_time
        total_read_noise = number_subs**0.5 * self.camera.read_noise

        noise = ((signal + sky_counts + dark_counts) * (u.electron / u.pixel) + total_read_noise**2)**0.5
        noise = noise.to(u.electron / u.pixel)

        # Saturation check
        if saturation_check:
            if surface_brightness:
                saturated = self._is_saturated(rate, sub_exp_time, filter_name)
            else:
                # Sky counts already included in _is_saturated, need to avoid counting them twice
                saturated = self._is_saturated(0 * u.electron / (u.pixel * u.second), sub_exp_time, filter_name)
            # np.where strips units, need to manually put them back.
            signal = np.where(saturated, 0, signal) * u.electron / u.pixel
            noise = np.where(saturated, 0, noise) * u.electron / u.pixel

        # Totals per (binned) pixel for all imagers.
        signal = signal * self.num_imagers * binning
        noise = noise * (self.num_imagers * binning)**0.5

        # Optionally convert to totals per arcsecond squared.
        if calc_type == 'per arcsecond squared':
            signal = signal / self.pixel_area  # e/arcseconds^2
            noise = noise / (self.pixel_scale * u.arcsecond)  # e/arcseconds^2

        return signal, noise

    def extended_source_snr(self, surface_brightness, filter_name, total_exp_time, sub_exp_time,
                            calc_type='per pixel', saturation_check=True, binning=1):
        """
        Calculates the signal to noise ratio for an extended source with
        given surface brightness.

        Calculates the signal to noise ratio for an extended source with
        given surface brightness. Alternatively can calculate the signal
        to noise for measurements of the sky background itself by setting
        the source surface brightness to None.

        Parameters
        ----------
        surface_brightness : astropy.units.Quantity or callable or None
            Surface brightness per arcsecond^2 of the source in ABmag
            units, or an equivalent count rate in photo-electrons per
            second per pixel, or a callable object that return surface
            brightness in spectral flux density units as a function of
            wavelength. Set to None or False to calculate the signal to
            noise ratio for the sky background.
        filter_name : str
            Name of the optical filter to use
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        sub_exp_time : astropy.units.Quantity
            Length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal to noise ratio per pixel or
            signal to noise ratio per arcsecond^2. Default is 'per pixel'
        saturation_check : bool, optional
            If `True` will set the signal to noise ratio to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.
        binning : int, optional
            Pixel binning factor. Cannot be used with calculation type
            'per arcsecond squared', will raise `ValueError` if you try.

        Returns
        -------
            snr : astropy.units.Quantity
                signal to noise ratio, dimensionless unscaled units
        """
        signal, noise = self.extended_source_signal_noise(surface_brightness, filter_name, total_exp_time,
                                                          sub_exp_time, calc_type, saturation_check, binning)

        # np.where() strips units, need to manually put them back
        snr = np.where(noise != 0.0, signal / noise, 0.0) * u.dimensionless_unscaled

        return snr

    def extended_source_etc(self, surface_brightness, filter_name, snr_target, sub_exp_time, calc_type='per pixel',
                            saturation_check=True, binning=1):
        """
        Calculates the total exposure time required to reach a given
        signal to noise ratio for a given extended source surface
        brightness.

        Calculates the total exposure time required to reach a given
        signal to noise ratio for a given extended source surface
        brightness. Alternatively can calculate the required time for
        measurements of the sky background itself by setting the source
        surface brightness to None.

        Parameters
        ----------
        surface_brightness : astropy.units.Quantity or None
            Surface brightness per arcsecond^2 of the source in ABmag
            units, or an equivalent count rate in photo-electrons per
            second per pixel. Set to None or False to calculate the
            required exposure time for measurements of the sky background.
        filter_name : str
            Name of the optical filter to use
        snr_target : astropy.units.Quantity
            The desired signal to noise ratio, dimensionless unscaled units
        sub_exp_time : astropy.units.Quantity
            length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal to noise ratio per pixel or
            signal to noise ratio per arcsecond^2. Default is 'per pixel'
        saturation_check : bool, optional
            If `True` will set the exposure time to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.
        binning : int, optional
            Pixel binning factor. Cannot be used with calculation type
            'per arcsecond squared', will raise `ValueError` if you try.

        Returns
        -------
        total_exp_time : astropy.units.Quantity
            Total exposure time required to reach a signal to noise ratio
            of at least `snr_target`, rounded up to an integer multiple of
            `sub_exp_time`.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        if calc_type not in ('per pixel', 'per arcsecond squared'):
            raise ValueError("Invalid calculation type '{}'!".format(calc_type))

        if calc_type == 'per arcsecond squared' and binning != 1:
            raise ValueError("Cannot specify pixel binning with calculation type 'per arcsecond squared'!")

        # Convert target SNR per array combined, binned pixel to SNR per unbinned pixel
        snr_target = ensure_unit(snr_target, u.dimensionless_unscaled)
        snr_target = snr_target / (self.num_imagers * binning)**0.5

        if calc_type == 'per arcsecond squared':
            # If snr_target was given as per arcseconds squared need to mutliply by square root of
            # pixel area to convert it to a per pixel value.
            snr_target = snr_target * self.pixel_scale / (u.arcsecond / u.pixel)

        if surface_brightness:
            # Given a source brightness
            if not isinstance(surface_brightness, u.Quantity):
                surface_brightness = surface_brightness * u.ABmag
            try:
                # If surface brightness is a count rate this should work
                rate = surface_brightness.to(u.electron / (u.pixel * u.second))
            except u.core.UnitConversionError:
                # Direct conversion failed so assume we have surface brightness in ABmag, call conversion function
                rate = self.SB_to_rate(surface_brightness, filter_name)
        else:
            # Measuring the sky background itself.
            rate = self.sky_rate[filter_name]

        sub_exp_time = ensure_unit(sub_exp_time, u.second)

        # If required total exposure time is much greater than the length of a sub-exposure then
        # all noise sources (including read noise) are proportional to t^0.5 and we can use a
        # simplified expression to estimate total exposure time.
        if surface_brightness:
            noise_squared_rate = ((rate +
                                   self.sky_rate[filter_name] +
                                   self.camera.dark_current) * (u.electron / u.pixel) +
                                   self.camera.read_noise**2 / sub_exp_time)
        else:
            # Avoiding counting sky noise twice when the target is the sky background itself
            noise_squared_rate = ((rate +
                                   self.camera.dark_current) * (u.electron / u.pixel) +
                                   self.camera.read_noise**2 / sub_exp_time)

        noise_squared_rate = noise_squared_rate.to(u.electron**2 / (u.pixel**2 * u.second))
        total_exp_time = (snr_target**2 * noise_squared_rate / rate**2).to(u.second)

        # Now just round up to the next integer number of sub-exposures, being careful because the total_exp_time
        # and/or sub_exp_time could be Quantity arrays instead of scalars. The simplified expression above is exact
        # for integer numbers of sub exposures and signal to noise ratio monotonically increases with exposure time
        # so the final signal to noise be above the target value.
        number_subs = np.ceil(total_exp_time / sub_exp_time)

        if saturation_check:
            if surface_brightness:
                saturated = self._is_saturated(rate, sub_exp_time, filter_name)
            else:
                # Sky counts already included in _is_saturated, need to avoid counting them twice
                saturated = self._is_saturated(0 * u.electron / (u.pixel * u.second), sub_exp_time, filter_name)

            number_subs = np.where(saturated, 0, number_subs)

        return number_subs * sub_exp_time

    def extended_source_limit(self, total_exp_time, filter_name, snr_target, sub_exp_time, calc_type='per pixel',
                              binning=1, enable_read_noise=True, enable_sky_noise=True, enable_dark_noise=True):
        """
        Calculates the limiting extended source surface brightness for a
        given minimum signal to noise ratio and total exposure time.

        Parameters
        ----------
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        filter_name : str
            Name of the optical filter to use
        snr_target : astropy.units.Quantity
            The desired signal to noise ratio, dimensionless unscaled units
        sub_exp_time : astropy.units.Quantity
            length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal to noise ratio per pixel or
            signal to noise ratio per arcsecond^2. Default is 'per pixel'
        binning : int, optional
            Pixel binning factor. Cannot be used with calculation type
            'per arcsecond squared', will raise `ValueError` if you try.
        enable_read_noise : bool, optional
            If `False` calculates limit as if read noise were zero, default
            `True`
        enable_sky_noise : bool, optional
            If `False` calculates limit as if sky background were zero,
            default `True`
        enable_dark_noise : bool, optional
            If False calculates limits as if dark current were zero,
            default `True`

        Returns
        -------
        surface_brightness : astropy.units.Quantity
            Limiting source surface brightness per arcsecond squared, in AB mag units.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        if calc_type not in ('per pixel', 'per arcsecond squared'):
            raise ValueError("Invalid calculation type '{}'!".format(calc_type))

        if calc_type == 'per arcsecond squared' and binning != 1:
            raise ValueError("Cannot specify pixel binning with calculation type 'per arcsecond squared'!")

        # Convert target SNR per array combined, binned pixel to SNR per unbinned pixel
        snr_target = ensure_unit(snr_target, u.dimensionless_unscaled)
        snr_target = snr_target / (self.num_imagers * binning)**0.5

        if calc_type == 'per arcsecond squared':
            # If snr_target was given as per arcseconds squared need to mutliply by square root of
            # pixel area to convert it to a per pixel value.
            snr_target = snr_target * self.pixel_scale / (u.arcsecond / u.pixel)

        total_exp_time = ensure_unit(total_exp_time, u.second)
        sub_exp_time = ensure_unit(sub_exp_time, u.second)

        # Round total exposure time to an integer number of sub exposures. One or both of total_exp_time or
        # sub_exp_time may be Quantity arrays, need np.ceil
        number_subs = np.ceil(total_exp_time / sub_exp_time)
        total_exp_time = number_subs * sub_exp_time

        # Noise sources
        sky_counts = self.sky_rate[filter_name] * total_exp_time if enable_sky_noise else 0.0 * u.electron / u.pixel
        dark_counts = self.camera.dark_current * total_exp_time if enable_dark_noise else 0.0 * u.electron / u.pixel
        total_read_noise = number_subs**0.5 * \
            self.camera.read_noise if enable_read_noise else 0.0 * u.electron / u.pixel

        noise_squared = ((sky_counts + dark_counts) * (u.electron / u.pixel) + total_read_noise**2)
        noise_squared.to(u.electron**2 / u.pixel**2)

        # Calculate science count rate for target signal to noise ratio
        a = total_exp_time**2
        b = -snr_target**2 * total_exp_time * u.electron / u.pixel  # Units come from converting signal counts to noise
        c = -snr_target**2 * noise_squared

        rate = (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)
        rate = rate.to(u.electron / (u.pixel * u.second))

        return self.rate_to_SB(rate, filter_name)

    def ABmag_to_rate(self, mag, filter_name):
        """
        Converts AB magnitudes to photo-electrons per second in the image
        sensor

        Parameters
        ----------
        mag : astropy.units.Quantity
            Source brightness in AB magnitudes
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        rate : astropy.units.Quantity
            Corresponding photo-electrons per second
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        mag = ensure_unit(mag, u.ABmag)

        # First convert to incoming spectral flux density per unit frequency
        f_nu = mag.to(u.W / (u.m**2 * u.Hz),
                      equivalencies=u.equivalencies.spectral_density(self.pivot_wave[filter_name]))
        # Then convert to photo-electron rate using the 'sensitivity integral' for the instrument
        rate = f_nu * self.optic.aperture_area * self._iminus1[filter_name] * u.photon / c.h

        return rate.to(u.electron / u.second)

    def rate_to_ABmag(self, rate, filter_name):
        """
        Converts photo-electrons per second in the image sensor to AB
        magnitudes

        Parameters
        ----------
        rate : astropy.units.Quantity
            Photo-electrons per second
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        mag : astropy.units.Quantity
            Corresponding source brightness in AB magnitudes
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        rate = ensure_unit(rate, u.electron / u.second)

        # First convert to incoming spectral flux density using the 'sensitivity integral' for the instrument
        f_nu = rate * c.h / (self.optic.aperture_area * self._iminus1[filter_name] * u.photon)
        # Then convert to AB magnitudes
        return f_nu.to(u.ABmag, equivalencies=u.equivalencies.spectral_density(self.pivot_wave[filter_name]))

    def SB_to_rate(self, mag, filter_name):
        """
        Converts surface brightness AB magnitudes (per arcsecond squared)
        to photo-electrons per pixel per second.

        Parameters
        ----------
        mag : astropy.units.Quantity
            Source surface brightness in AB magnitudes
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        rate : astropy.units.Quantity
            Corresponding photo-electrons per pixel per second

        Notes
        -----
        At the time of writing `astropy.units` did not support the
        commonly used (but dimensionally nonsensical) expression of
        surface brightness in 'magnitudes per arcsecond squared'.
        Consequently the `mag` surface brightness parameter should
        have a units of `astropy.unit.ABmag`, the 'per arcsecond
        squared' is implied.
        """
        # Use ABmag_to_rate() to convert to electrons per second, then multiply by pixel area
        SB_rate = self.ABmag_to_rate(mag, filter_name) * self.pixel_area / (u.arcsecond**2)
        return SB_rate.to(u.electron / (u.second * u.pixel))

    def rate_to_SB(self, SB_rate, filter_name):
        """
        Converts photo-electrons per pixel per second to surface brightness
        AB magnitudes (per arcsecond squared)

        Parameters
        ----------
        SB_rate : astropy.units.Quantity
            Photo-electrons per pixel per second
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        mag : astropy.units.Quantity
            Corresponding source surface brightness in AB magnitudes

        Notes
        -----
        At the time of writing `astropy.units` did not support the
        commonly used (but dimensionally nonsensical) expression of
        surface brightness in 'magnitudes per arcsecond squared'.
        Consequently the `mag` surface brightness return value has units
        of `astropy.unit.ABmag`, the 'per arcsecond squared' is implied.
        """
        SB_rate = ensure_unit(SB_rate, u.electron / (u.second * u.pixel))
        # Divide by pixel area to convert to electrons per second per arcsecond^2
        rate = SB_rate * u.arcsecond**2 / self.pixel_area
        # Use rate_to_ABmag() to convert to AB magnitudes
        return self.rate_to_ABmag(rate, filter_name)

    def ABmag_to_flux(self, mag, filter_name):
        """
        Converts brightness of the target to total flux, integrated over
        the filter band.

        Parameters
        ----------
        mag : astropy.units.Quantity
            Brightness of the target in AB magnitudes
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        flux : astropy.units.Quantity
            Corresponding total flux in Watts per square metre

        Notes
        -----
        The conversion between band averaged magnitudes and total flux
        depends somewhat on the spectrum of the source. For this
        calculation we assume :math:`F_\\nu` is constant.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        mag = ensure_unit(mag, u.ABmag)

        # First convert to spectral flux density per unit wavelength
        f_nu = mag.to(u.W / (u.m**2 * u.Hz),
                      equivalencies=u.equivalencies.spectral_density(self.pivot_wave[filter_name]))
        # Then use pre-calculated integral to convert to total flux in the band (assumed constant F_nu)
        flux = f_nu * c.c * self._iminus2[filter_name] * u.photon / u.electron

        return flux.to(u.W / u.m**2)

    def flux_to_ABmag(self, flux, filter_name):
        """
        Converts total flux of the target, integrated over the filter
        band, to magnitudes.

        Parameters
        ----------
        flux : astropy.units.Quantity
            Total flux in Watts per square metre
        filter_name : str
            Name of the optical filter to use

        Returns
        -------
        mag : astropy.units.Quantity
            Corresponding brightness of the target in AB magnitudes

        Notes
        -----
        The conversion between band averaged magnitudes and total flux
        depends somewhat on the spectrum of the source. For this
        calculation we assume :math:`F_\\nu` is constant.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        flux = ensure_unit(flux, u.W / u.m**2)

        # First convert from total flux to spectral flux densitity per
        # unit wavelength, using the pre-caluclated integral.
        f_nu = flux  * u.electron / (self._iminus2[filter_name] * c.c * u.photon)

        # Then convert spectral flux density to magnitudes
        return f_nu.to(u.ABmag,
                       equivalencies=u.equivalencies.spectral_density(self.pivot_wave[filter_name]))

    def total_exposure_time(self, total_elapsed_time, sub_exp_time):
        """
        Calculates total exposure time given a total elapsed time and
        sub-exposure time.

        The calculation includes readout time overheads (but no others, at
        present) and rounds down to an integer number of sub-exposures.

        Parameters
        ----------
        total_elapsed_time : astropy.units.Quantity
            Total elapsed time
        sub_exp_time : astropy.units.Quantity
            Exposure time of individual sub-exposures

        Returns
        -------
        total_exposure_time : astropy.units.Quantity
            Maximum total exposure time possible in an elapsed time of no more than `total_elapsed_time`
        """
        total_elapsed_time = ensure_unit(total_elapsed_time, u.second)
        sub_exp_time = ensure_unit(sub_exp_time, u.second)

        num_of_subs = np.floor(total_elapsed_time / (sub_exp_time + self.camera.readout_time * self.num_per_computer))
        total_exposure_time = num_of_subs * sub_exp_time
        return total_exposure_time

    def total_elapsed_time(self, exp_list):
        """
        Calculates the total elapsed time required for a given a list of
        sub-exposure times.

        The calculation add readout time overheads (but no others, at
        present) and sums the elapsed time from all sub-exposures.

        Parameters
        ----------
        exp_list : astropy.units.Quantity
            List of sub-exposure times

        Returns
        -------
        elapsed_time : astropy.units.Quantity
            Total elapsed time required to execute the list of sub exposures
        """
        exp_list = ensure_unit(exp_list, u.second)

        elapsed_time = exp_list.sum() + len(exp_list) * self.num_per_computer * self.camera.readout_time
        return elapsed_time

    def point_source_signal_noise(self, brightness, filter_name, total_exp_time, sub_exp_time, saturation_check=True):
        """
        Calculates the signal and noise for a point source of a given
        brightness, assuming PSF fitting photometry

        The returned signal and noise values are the weighted sum over the
        pixels in the source image, where the weights are the normalised
        pixel values of the PSF model being fit to the data.

        Parameters
        ----------
        brightness : astropy.units.Quantity
            Brightness of the source in ABmag units, or an equivalent
            count rate in photo-electrons per second.
        filter_name : str
            Name of the optical filter to use
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        sub_exp_time : astropy.units.Quantity
            Length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal & noise per pixel or signal &
            noise per arcsecond^2. Default is 'per pixel'
        saturation_check : bool, optional
            If `True` will set both signal and noise to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.

        Returns
        -------
        signal : astropy.units.Quantity
            Effective total signal in units of electrons
        noise: astropy.units.Quantity
            Effective total noise in units of electrons

        Notes
        ----------
        The PSF fitting signal to noise calculations follow the example
        of http://www.stsci.edu/itt/review/ihb_cy14.WFPC2/ch6_exposuretime6.html

        The values will depend on the position of the centre of the PSF
        relative to the pixel grid, this calculation assumes the worst
        case of PSF centres on a pixel corner. Conversely it
        optimistically assumes that the PSF model exactly matches the PSF
        of the data.
        """
        if not isinstance(brightness, u.Quantity):
            brightness = brightness * u.ABmag

        try:
            # If brightness is a count rate this should work
            rate = brightness.to(u.electron / u.second)
        except u.core.UnitConversionError:
            # Direct conversion failed so assume we have brightness in ABmag, call conversion function
            rate = self.ABmag_to_rate(brightness, filter_name)

        # For PSF fitting photometry the signal to noise calculation is equivalent to dividing the flux equally
        # amongst n_pix pixels, where n_pix is the sum of the squares of the pixel values of the PSF.  The psf
        # object pre-calculates n_pix for the worst case where the PSF is centred on the corner of a pixel.

        # Now calculate effective signal and noise, using binning to calculate the totals.
        signal, noise = self.extended_source_signal_noise(rate / self.psf.n_pix, filter_name,
                                                          total_exp_time, sub_exp_time,
                                                          saturation_check=False, binning=self.psf.n_pix / u.pixel)
        signal = signal * u.pixel
        noise = noise * u.pixel

        # Saturation check. For point sources need to know maximum fraction of total electrons that will end up
        # in a single pixel, this is available as psf.peak. Can use this to calculate maximum electrons per pixel
        # in a single sub exposure, and check against saturation_level.
        if saturation_check:
            saturated = self._is_saturated(rate * self.psf.peak, sub_exp_time, filter_name)
            # np.where strips units, need to manually put them back.
            signal = np.where(saturated, 0.0, signal) * u.electron
            noise = np.where(saturated, 0.0, noise) * u.electron

        return signal, noise

    def point_source_snr(self, brightness, filter_name, total_exp_time, sub_exp_time, saturation_check=True):
        """
        Calculates the signal to noise ratio for a point source of a given
        brightness, assuming PSF fitting photometry

        The returned signal to noise ratio refers to the weighted sum over
        the pixels in the source image, where the weights are the
        normalised pixel values of the PSF model being fit to the data.

        Parameters
        ----------
        brightness : astropy.units.Quantity
            Brightness of the source in ABmag units, or an equivalent
            count rate in photo-electrons per second.
        filter_name : str
            Name of the optical filter to use
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        sub_exp_time : astropy.units.Quantity
            Length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal & noise per pixel or signal &
            noise per arcsecond^2. Default is 'per pixel'
        saturation_check : bool, optional
            If `True` will set both signal and noise to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.

        Returns
        -------
        snr : astropy.units.Quantity
            signal to noise ratio dimensionless unscaled units
        """
        signal, noise = self.point_source_signal_noise(brightness, filter_name,
                                                       total_exp_time, sub_exp_time, saturation_check)

        # np.where() strips units, need to manually put them back.
        snr = np.where(noise != 0.0, signal / noise, 0.0) * u.dimensionless_unscaled

        return snr

    def point_source_etc(self, brightness, filter_name, snr_target, sub_exp_time, saturation_check=True):
        """ Calculates the total exposure time required to reach a given signal to noise ratio for a given point
        source brightness.

        Parameters
        ----------
        brightness : astropy.units.Quantity
            Brightness of the source in ABmag units, or an equivalent
            count rate in photo-electrons per second.
        filter_name : str
            Name of the optical filter to use
        snr_target : astropy.units.Quantity
            The desired signal to noise ratio, dimensionless unscaled units
        sub_exp_time : astropy.units.Quantity
            length of individual sub-exposures
        saturation_check : bool, optional
            If `True` will set the exposure time to zero where the
            electrons per pixel in a single sub-exposure exceed the
            saturation level. Default is `True`.

        Returns
        -------
        total_exp_time : astropy.units.Quantity
            Total exposure time required to reach a signal to noise ratio
            of at least `snr_target`, rounded up to an integer multiple of
            `sub_exp_time`.
        """
        if not isinstance(brightness, u.Quantity):
            brightness = brightness * u.ABmag

        try:
            # If brightness is a count rate this should work
            rate = brightness.to(u.electron / u.second)
        except u.core.UnitConversionError:
            # Direct conversion failed so assume we have brightness in ABmag, call conversion function
            rate = self.ABmag_to_rate(brightness, filter_name)

        total_exp_time = self.extended_source_etc(rate / self.psf.n_pix, filter_name, snr_target, sub_exp_time,
                                                  saturation_check=False, binning=self.psf.n_pix / u.pixel)

        # Saturation check. For point sources need to know maximum fraction of total electrons that will end up
        # in a single pixel, this is available as psf.peak. Can use this to calculate maximum electrons per pixel
        # in a single sub exposure, and check against saturation_level.
        if saturation_check:
            saturated = self._is_saturated(rate * self.psf.peak, sub_exp_time, filter_name)
            # np.where() strips units, need to manually put them back
            total_exp_time = np.where(saturated, 0.0, total_exp_time) * u.second

        return total_exp_time

    def point_source_limit(self, total_exp_time, filter_name, snr_target, sub_exp_time,
                           enable_read_noise=True, enable_sky_noise=True, enable_dark_noise=True):
        """Calculates the limiting point source surface brightness for a given minimum signal to noise ratio and
        total exposure time.

        Parameters
        ----------
        total_exp_time : astropy.units.Quantity
            Total length of all sub-exposures. If necessary will be
            rounded up to an integer multiple of `sub_exp_time`
        filter_name : str
            Name of the optical filter to use
        snr_target : astropy.units.Quantity
            The desired signal to noise ratio, dimensionless unscaled units
        sub_exp_time : astropy.units.Quantity
            length of individual sub-exposures
        calc_type : {'per pixel', 'per arcsecond squared'}
            Calculation type, either signal to noise ratio per pixel or
            signal to noise ratio per arcsecond^2. Default is 'per pixel'
        enable_read_noise : bool, optional
            If `False` calculates limit as if read noise were zero, default
            `True`
        enable_sky_noise : bool, optional
            If `False` calculates limit as if sky background were zero,
            default `True`
        enable_dark_noise : bool, optional
            If False calculates limits as if dark current were zero,
            default `True`

        Returns
        -------
        brightness : astropy.units.Quantity
            Limiting point source brightness, in AB mag units.
        """
        # For PSF fitting photometry the signal to noise calculation is equivalent to dividing the flux equally
        # amongst n_pix pixels, where n_pix is the sum of the squares of the pixel values of the PSF.  The psf
        # object pre-calculates n_pix for the worst case where the PSF is centred on the corner of a pixel.

        # Calculate the equivalent limiting surface brighness, in AB magnitude per arcsecond^2
        equivalent_SB = self.extended_source_limit(total_exp_time, filter_name, snr_target, sub_exp_time,
                                                   binning=self.psf.n_pix / u.pixel,
                                                   enable_read_noise=enable_read_noise,
                                                   enable_sky_noise=enable_sky_noise,
                                                   enable_dark_noise=enable_dark_noise)

        # Multiply the limit by the area (in arcsecond^2) of n_pix pixels to convert back to point source magnitude
        # astropy.units.ABmag doesn't really support arithmetic at the moment, have to strip units.
        return (equivalent_SB.value - 2.5 * np.log10(self.psf.n_pix * self.pixel_area / u.arcsecond**2).value) * u.ABmag

    def extended_source_saturation_mag(self, sub_exp_time, filter_name, n_sigma=3.0):
        """
        Calculates the surface brightness of the brightest extended source
        that would definitely not saturate the image sensor in a given (sub)
        exposure time.

        Parameters
        ----------
        sub_exp_time : astropy.units.Quantity
            Length of the (sub) exposure.
        filter_name : str
            Name of the optical filter to use.
        n_sigma : float, optional
            Safety margin to leave between the maximum expected electrons
            per pixel and the nominal saturation level, in multiples of
            the noise, default 3.0

        Returns
        -------
        surface_brightness : astropy.units.Quantity
            Surface brightness per arcsecond^2 of the brightest extended
            source that will definitely not saturate, in AB magnitudes.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        sub_exp_time = ensure_unit(sub_exp_time, u.second)
        max_rate = (self.camera.saturation_level - n_sigma * self.camera.max_noise) / sub_exp_time
        max_source_rate = max_rate - self.sky_rate[filter_name] - self.camera.dark_current

        return self.rate_to_SB(max_source_rate, filter_name)

    def point_source_saturation_mag(self, sub_exp_time, filter_name, n_sigma=3.0):
        """
        Calculates the magnitude of the brightest point source that would
        definitely not saturate the image sensor in a given (sub) exposure
        time.

        Parameters
        ----------
        sub_exp_time : astropy.units.Quantity
            Length of the (sub) exposure.
        filter_name : str
            Name of the optical filter to use.
        n_sigma : float, optional
            Safety margin to leave between the maximum expected electrons
            per pixel and the nominalsaturation level, in multiples of
            the noise, default 3.0

        Returns
        -------=
        brighness : astropy.units.Quantity
            AB magnitude of the brightest point source that will definitely not saturate.
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        sub_exp_time = ensure_unit(sub_exp_time, u.second)
        max_rate = (self.camera.saturation_level - n_sigma * self.camera.max_noise) / sub_exp_time
        max_source_rate = max_rate - self.sky_rate[filter_name] - self.camera.dark_current

        return self.rate_to_ABmag(max_source_rate / self.psf.peak, filter_name)

    def extended_source_saturation_exp(self, surface_brightness, filter_name, n_sigma=3.0):
        """
        Calculates the maximum (sub) exposure time that will definitely
        avoid saturation for an extended source of given surface
        brightness.

        Parameters
        ----------
        surface_brightness : astropy.units.Quantity
            Surface brightness per arcsecond^2 of the source in ABmag
            units, or an equivalent count rate in photo-electrons per
            second per pixel
        filter_name : str
            Name of the optical filter to use
        n_sigma : float, optional
            Safety margin to leave between the maximum expected electrons
            per pixel and the nominalsaturation level, in multiples of
            the noise, default 3.0

        Returns
        -------
        sub_exp_time : astropy.units.Quantity
            Maximum length of (sub) exposure that will definitely avoid saturation
        """
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        if not isinstance(surface_brightness, u.Quantity):
            brightness = brightness * u.ABmag

        try:
            # If surface brightness is a count rate this should work
            rate = surface_brightness.to(u.electron / (u.pixel * u.second))
        except u.core.UnitConversionError:
            # Direct conversion failed so assume we have surface brightness in ABmag, call conversion function
            rate = self.SB_to_rate(surface_brightness, filter_name)

        total_rate = rate + self.sky_rate[filter_name] + self.camera.dark_current

        max_electrons_per_pixel = self.camera.saturation_level - n_sigma * self.camera.max_noise

        return max_electrons_per_pixel / total_rate

    def point_source_saturation_exp(self, brightness, filter_name, n_sigma=3.0):
        """
        Calculates the maximum (sub) exposure time that will definitely
        avoid saturation for point source of given brightness

        Parameters
        ----------
        brightness : astropy.units.Quantity
            Brightness of the source in ABmag units, or an equivalent
            count rate in photo-electrons per second.
        filter_name : str
            Name of the optical filter to use
        n_sigma : float, optional
            Safety margin to leave between the maximum expected electrons
            per pixel and the nominalsaturation level, in multiples of
            the noise, default 3.0

        Returns
        -------
        sub_exp_time : astropy.units.Quantity
            Maximum length of (sub) exposure that will definitely avoid saturation
        """
        if not isinstance(brightness, u.Quantity):
            brightness = brightness * u.ABmag

        try:
            # If brightness is a count rate this should work
            rate = brightness.to(u.electron / u.second)
        except u.core.UnitConversionError:
            # Direct conversion failed so assume we have brightness in ABmag, call conversion function
            rate = self.ABmag_to_rate(brightness, filter_name)

        # Convert to maximum surface brightness rate by multiplying by maximum flux fraction per pixel
        return self.extended_source_saturation_exp(rate * self.psf.peak, filter_name)

    def exp_time_sequence(self,
                          filter_name,
                          bright_limit=None,
                          shortest_exp_time=None,
                          longest_exp_time=None,
                          faint_limit=None,
                          num_long_exp=None,
                          exp_time_ratio=2.0,
                          snr_target=5.0):
        """
        Calculates a sequence of sub exposures to use to span a given
        range of either point source brightness or exposure time.

        If required the sequence will begin with an 'HDR block' of
        progressly increasing exposure time, followed by 1 or more
        exposures of equal length with the number of long exposures either
        specified directly or calculated from the faintest point source
        that the sequence is intended to detect.

        Parameters
        ----------
        filter_name : str
            Name of the optical filter to use.
        bright_limit : astropy.units.Quantity, optional
            Brightness in ABmag of the brightest point sources that we
            want to avoid saturating on, will be used to calculate a
            suitable shortest exposure time. Optional, but one and only
            one of `bright_limit` and `shortest_exp_time` must be
            specified.
        shortest_exp_time : astropy.units.Quantity, optional
            Shortest sub exposure time to include in the sequence.
            Optional, but one and only one of `bright_limit` and
            `shortest_exp_time` must be specified.
        longest_exp_time : astropy.units.Quantity
            Longest sub exposure time to include in the sequence.
        faint_limit : astropy.units.Quantity, optional
            Brightness in ABmag of the faintest point sources that we want
            to be able to detect in the combined data from the sequence.
            Optional, but one and only one of `faint_limit` and
            `num_long_exp` must be specified.
        num_long_exp : int, optional
            Number of repeats of the longest sub exposure to include in
            the sequence. Optional, but one and only one of `faint_limit`
            and `num_long_exp` must be specified.
        exp_time_ratio : float, optional
            Ratio between successive sub exposure times in the HDR block,
            default 2.0
        snr_target : float, optional
            Signal to noise ratio threshold for detection at
            `faint_limit`, default 5.0

        Returns
        -------
        exp_times : astropy.units.Quantity
            Sequence of sub exposure times
        """
        # First verify all the inputs
        if filter_name not in self.filter_names:
            raise ValueError("This Imager has no filter '{}'!".format(filter_name))

        if bool(bright_limit) == bool(shortest_exp_time):
            raise ValueError("One and only one of bright_limit and shortest_exp_time must be specified!")

        if bool(faint_limit) == bool(num_long_exp):
            raise ValueError("one and only one of faint_limit and num_long_exp must be specified!")

        longest_exp_time = ensure_unit(longest_exp_time, u.second)
        if longest_exp_time < self.camera.minimum_exposure:
            raise ValueError("Longest exposure time shorter than minimum exposure time of the camera!")

        if bright_limit:
            # First calculate exposure time that will just saturate on the brightest sources.
            shortest_exp_time = self.point_source_saturation_exp(bright_limit, filter_name)
        else:
            shortest_exp_time = ensure_unit(shortest_exp_time, u.second)

        # If the brightest sources won't saturate even for the longest requested exposure time then HDR mode isn't
        # necessary and we can just use the normal ETC to create a boring exposure time list.
        if shortest_exp_time >= longest_exp_time:
            if faint_limit:
                total_exp_time = self.point_source_etc(brightness=faint_limit,
                                                       filter_name=filter_name,
                                                       sub_exp_time=longest_exp_time,
                                                       snr_target=snr_target)
                num_long_exp = int(total_exp_time / longest_exp_time)

            exp_times = num_long_exp * [longest_exp_time]
            exp_times = u.Quantity(exp_times)
            return exp_times

        # Round down the shortest exposure time so that it is a exp_time_ratio^integer multiple of the longest
        # exposure time
        num_exp_times = int(math.ceil(math.log(longest_exp_time / shortest_exp_time, exp_time_ratio)))
        shortest_exp_time = (longest_exp_time / (exp_time_ratio ** num_exp_times))

        # Ensuring the shortest exposure time is not lower than the minimum exposure time of the cameras
        if shortest_exp_time < self.camera.minimum_exposure:
            shortest_exp_time *= exp_time_ratio**math.ceil(math.log(self.camera.minimum_exposure / shortest_exp_time,
                                                                    exp_time_ratio))
            num_exp_times = int(math.log(longest_exp_time / shortest_exp_time, exp_time_ratio))

        # Creating a list of exposure times from the shortest exposure time to the one directly below the
        # longest exposure time
        exp_times = [shortest_exp_time.to(u.second) * exp_time_ratio**i for i in range(num_exp_times)]

        if faint_limit:
            num_long_exp = 0
            # Signals and noises from each of the sub exposures in the HDR sequence
            signals, noises = self.point_source_signal_noise(brightness=faint_limit,
                                                             filter_name=filter_name,
                                                             sub_exp_time=u.Quantity(exp_times),
                                                             total_exp_time=u.Quantity(exp_times))
            # Running totals
            net_signal = signals.sum()
            net_noise_squared = (noises**2).sum()

            # Check is signal to noise target reach, add individual long exposures until it is.
            while net_signal / net_noise_squared**0.5 < snr_target:
                num_long_exp += 1
                signal, noise = self.point_source_signal_noise(brightness=faint_limit,
                                                               filter_name=filter_name,
                                                               sub_exp_time=longest_exp_time,
                                                               total_exp_time=longest_exp_time)
                net_signal += signal
                net_noise_squared += noise**2

        exp_times = exp_times + num_long_exp * [longest_exp_time]
        exp_times = u.Quantity(exp_times)
        exp_times = np.around(exp_times, decimals=2)

        return exp_times

    def snr_vs_ABmag(self, exp_times, filter_name, magnitude_interval=0.02 * u.ABmag, snr_target=1.0, plot=None):
        """
        Calculates PSF fitting signal to noise ratio as a function of
        point source brightness for the combined data resulting from a
        given sequence of sub exposures.

        Optionally generates a plot of the results. Automatically choses
        limits for the magnitude range based on the saturation limit of
        the shortest exposure and the sensitivity limit of the combined
        data.

        Parameters
        ----------
        exp_times : astropy.units.Quantity
            Sequence of sub exposure times.
        filter_name : str
            Name of the optical filter to use.
        magnitude_interval : astropy.units.Quantity, optional
            Step between consecutive brightness values, default 0.02 mag
        snr_target : float, optional
            signal to noise threshold used to set faint limit of magnitude
            range, default 1.0
        plot : str, optional
            Filename for the plot of SNR vs magnitude. If not given no
            plots will be generated.

        Returns
        -------
        magnitudes : astropy.units.Quantity
            Sequence of point source brightnesses in AB magnitudes
        snrs : astropy.units.Quantity
            signal to noise ratios for point sources of the brightnesses
            in `magnitudes`
        """
        magnitude_interval = ensure_unit(magnitude_interval, u.ABmag)

        longest_exp_time = exp_times.max()

        if (exp_times == longest_exp_time).all():
            hdr = False
            # All exposures the same length, use direct calculation.

            # Magnitudes ranging from the sub exposure saturation limit to a SNR of 1 in the combined data.
            magnitudes = np.arange(self.point_source_saturation_mag(longest_exp_time.value, filter_name),
                                   self.point_source_limit(total_exp_time=exp_times.sum(),
                                                           filter_name=filter_name,
                                                           sub_exp_time=longest_exp_time,
                                                           snr_target=snr_target).value,
                                   magnitude_interval.value) * u.ABmag
            # Calculate SNR directly.
            snrs = self.point_source_snr(brightness=magnitudes,
                                         filter_name=filter_name,
                                         total_exp_time=exp_times.sum(),
                                         sub_expt_time=longest_exp_time)

        else:
            hdr = True
            # Have a range of exposure times.
            # Magnitudes ranging from saturation limit of the shortest sub exposure to the SNR of 1 limit for a non-HDR
            # sequence of the same total exposure time.
            magnitudes = np.arange(self.point_source_saturation_mag(exp_times.min(), filter_name).value,
                                   self.point_source_limit(total_exp_time=exp_times.sum(),
                                                           filter_name=filter_name,
                                                           sub_exp_time=longest_exp_time,
                                                           snr_target=snr_target).value,
                                   magnitude_interval.value) * u.ABmag

            # Split into HDR block & long exposure repeats
            hdr_exposures = np.where(exp_times != longest_exp_time)

            # Quantity array for running totals of signal and noise squared at each magnitude
            total_signals = np.zeros(magnitudes.shape) * u.electron
            total_noises_squared = np.zeros(magnitudes.shape) * u.electron**2

            # Signal to noise for each individual exposure in the HDR block
            for exp_time in exp_times[hdr_exposures]:
                signals, noises = self.point_source_signal_noise(brightness=magnitudes,
                                                                 filter_name=filter_name,
                                                                 total_exp_time=exp_time,
                                                                 sub_exp_time=exp_time)
                total_signals += signals
                total_noises_squared += noises**2

            # Direct calculation for the repeated exposures
            num_long_exps = (exp_times == longest_exp_time).sum()
            signals, noises = self.point_source_signal_noise(brightness=magnitudes,
                                                             filter_name=filter_name,
                                                             total_exp_time=num_long_exps * longest_exp_time,
                                                             sub_exp_time=longest_exp_time)
            total_signals += signals
            total_noises_squared += noises**2

            snrs = total_signals / (total_noises_squared)**0.5

        if plot:
            if hdr:
                non_hdr_snrs = self.point_source_snr(brightness=magnitudes,
                                                     filter_name=filter_name,
                                                     total_exp_time=exp_times.sum(),
                                                     sub_exp_time=longest_exp_time)
            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(magnitudes, snrs, 'b-', label='HDR mode')
            if hdr:
                ax1.plot(magnitudes, non_hdr_snrs, 'c:', label='Non-HDR mode')
                ax1.legend(loc='upper right', fancybox=True, framealpha=0.3)
            ax1.set_xlabel('Point source brightness / AB magnitude')
            ax1.set_ylabel('Signal to noise ratio')
            ax1.set_title('Point source PSF fitting signal to noise ratio for combined data')

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.semilogy(magnitudes, snrs, 'b-', label='HDR mode')
            if hdr:
                ax2.semilogy(magnitudes, non_hdr_snrs, 'c:', label='Non-HDR mode')
                ax2.legend(loc='upper right', fancybox=True, framealpha=0.3)
            ax2.set_xlabel('Point source brightness / AB magnitude')
            ax2.set_ylabel('Signal to noise ratio')
            ax2.set_title('Point source PSF fitting signal to noise ratio for combined data')

            fig.savefig(plot)
            plt.close(fig)

        return magnitudes.to(u.ABmag), snrs.to(u.dimensionless_unscaled)

    def get_pixel_coords(self, centre):
        """
        Utility function to return a SkyCoord array containing the on sky position
        of the centre of all the pixels in the image, given a SkyCoord for the
        field centre
        """
        # Ensure centre is a SkyCoord (this allows entering centre as a string)
        if not isinstance(centre, SkyCoord):
            centre = SkyCoord(centre)

        # Set field centre coordinates in internal WCS
        self.wcs.wcs.crval = [centre.icrs.ra.value, centre.icrs.dec.value]

        # Arrays of pixel coordinates
        XY = np.meshgrid(np.arange(self.wcs._naxis1), np.arange(self.wcs._naxis2))

        # Convert to arrays of RA, dec (ICRS, decimal degrees)
        RAdec = self.wcs.all_pix2world(XY[0], XY[1], 0)

        return SkyCoord(RAdec[0], RAdec[1], unit='deg')

    def make_noiseless_image(self, centre, time, f):
        """
        Function to create a noiseless simulated image for a given image centre and observation time.
        """
        electrons = np.zeros((self.wcs._naxis2, self.wcs._naxis1)) * u.electron / u.second

        # Calculate observed zodiacal light background.
        # Get relative zodical light brightness for each pixel
        # Note, side effect of this is setting centre of self.wcs
        pixel_coords = self.get_pixel_coords(centre)
        zl_rel = zl.relative_brightness(pixel_coords, time)

        # TODO: calculate area of each pixel, for now use nominal pixel scale^2
        # Finally multiply to get an observed zodical light image
        zl_obs = self.zl_obs_ep * zl_rel * self.pixel_scale**2

        electrons += zl_obs

        noiseless = ccdproc.CCDData(electrons, wcs=self.wcs)

        return noiseless

    def make_image_real(self, noiseless, exp_time, subtract_dark = False):
        """
        Given a noiseless simulated image in electrons per pixel add dark current,
        Poisson noise and read noise, and convert to ADU using the predefined gain.
        """
        # Scale photoelectron rates by exposure time
        data = noiseless.data * noiseless.unit * exp_time
        # Add dark current
        data += self.dark_frame * exp_time
        # Force to electron units
        data = data.to(u.electron)
        # Apply Poisson noise. This is not unit-aware, need to restore them manually
        data = (poisson.rvs(data/u.electron)).astype('float64') * u.electron
        # Apply read noise. Again need to restore units manually
        data += norm.rvs(scale=self.read_noise/u.electron, size=data.shape) * u.electron
        # Optionally subtract a Perfect Dark
        if subtract_dark:
            data -= (self.dark_frame * exp_time).to(u.electron)
        # Convert to ADU
        data /= self.gain
        # Force to adu (just here to catch unit problems)
        data = data.to(u.adu)
        # 'Analogue to digital conversion'
        data = np.where(data < 2**16 * u.adu, data, (2**16 - 1) * u.adu)
        data = data.astype('uint16')
        # Data type conversion strips units so need to put them back manually
        image = ccdproc.CCDData(data, wcs=noiseless.wcs, unit=u.adu)
        image.header['EXPTIME'] = exp_time
        image.header['DARKSUB'] = subtract_dark

        return image

    def _is_saturated(self, rate, sub_exp_time, filter_name, n_sigma=3.0):
        # Total electrons per pixel from source, sky and dark current
        electrons_per_pixel = (rate + self.sky_rate[filter_name] + self.camera.dark_current) * sub_exp_time
        # Consider saturated if electrons per pixel is closer than n sigmas of noise to the saturation level
        return electrons_per_pixel > self.camera.saturation_level - n_sigma * self.camera.max_noise

    def _efficiencies(self):
        # Fine wavelength grid spaning maximum range of instrument response
        waves = np.arange(self.camera.wavelengths.value.min(), self.camera.wavelengths.value.max(), 0.05) * u.nm
        self.wavelengths = waves

        # Sensitivity integrals for each filter bandpass
        self._iminus1 = {}
        self._iminus2 = {}
        # End to end efficiency as a function of wavelegth for each filter bandpass
        self.efficiencies = {}
        # Mean end to end efficiencies for each filter bandpass
        self.efficiency = {}
        # Mean wavelength for each filter bandpass
        self.mean_wave = {}
        # Pivot wavelengths for each filter bandpass
        self.pivot_wave = {}
        # Bandwidths for each filter bandpass (STScI definition)
        self.bandwidth = {}

        # Interpolators for throughput and QE. Will move these into the Optics and Camera classes later.
        tau = interp1d(self.optic.wavelengths, self.optic.throughput, kind='linear', fill_value='extrapolate')
        qe = interp1d(self.camera.wavelengths, self.camera.QE, kind='linear', fill_value='extrapolate')

        for name, band in self.filters.items():
            # End-to-end efficiency. Need to put units back after interpolation
            effs = tau(waves) * band.transmission(waves) * qe(waves) * u.electron / u.photon

            # Band averaged efficiency, effective wavelengths, bandwidth (STSci definition), flux_integral
            i0 = np.trapz(effs, x=waves)
            i1 = np.trapz(effs * waves, x=waves)
            self._iminus1[name] = np.trapz(effs / waves, x=waves)  # This one is useful later
            self._iminus2[name] = np.trapz(effs / waves**2, x=waves)

            self.efficiencies[name] = effs
            self.efficiency[name] = i0 / (waves[-1] - waves[0])
            self.mean_wave[name] = i1 / i0
            self.pivot_wave[name] = (i1 / self._iminus1[name])**0.5
            self.bandwidth[name] = i0 / effs.max()

    def _gamma0(self):
        """
        Calculates 'gamma0', the number of photons/second/pixel at the top of atmosphere
        that corresponds to 0 AB mag/arcsec^2 for a given band, aperture & pixel scale.
        """
        # Spectral flux density corresponding to 0 ABmag, pseudo-SI units
        sfd_0 = (0 * u.ABmag).to(u.W / (u.m**2 * u.um),
                                 equivalencies=u.equivalencies.spectral_density(self.pivot_wave))
        # Change to surface brightness (0 ABmag/arcsec^2)
        sfd_sb_0 = sfd_0 / u.arcsecond**2
        # Average photon energy
        energy = c.h * c.c / (self.pivot_wave * u.photon)
        # Divide by photon energy & multiply by aperture area, pixel area and bandwidth to get photons/s/pixel
        photon_flux = (sfd_sb_0 / energy) * self.optic.aperture_area * self.pixel_area * self.bandwidth

        self.gamma0 = photon_flux.to(u.photon / (u.s * u.pixel))
