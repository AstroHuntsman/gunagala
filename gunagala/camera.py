"""
Cameras (stricly the image sensor subsystem, not including optics, optical filters, etc)
"""
import os
import numpy as np

from astropy import units as u
from astropy.table import Table

from gunagala.utils import ensure_unit, get_table_data


class Camera:
    """
    Class representing a camera.

    Here 'camera' refers to the image sensor, associated electronics,
    shutter, etc., but does not include any of the optical components
    of the system.

    Parameters
    ----------
    bit_depth : int
        Bits per pixel used by the camera analogue to digital
        converters.
    full_well : astropy.units.Quantity
        Number of photo-electrons each pixel can receive before
        saturating.
    gain : astropy.units.Quantity
        Number of photo-electrons corresponding to one ADU in the
        digital data.
    bias : astropy.units.Quantity
        Bias level of image sensor, in ADU / pixel units. Used when
        determining saturation level.
    readout_time : astropy.units.Quantity
        Time required to read the data from the image sensor.
    pixel_size : astropy.units.Quantity
        Pixel pitch. Square pixels are assumed.
    Resolution : astropy.units.Quantity
        Two element Quantity containing the number of pixels across
        the image sensor in both vertical & horizontal directions.
        (y, x)
    read_noise astropy.units.Quantity
        Intrinsic noise of image sensor and readout electronics, in
        electrons/pixel units.
    dark_current : astropy.units.Quantity
        Rate of accumlation of dark signal, in electrons/second/pixel
        units.
    QE : astropy.table.Table or str
        Quantum efficiency as a function of wavelength data, either as an
        astropy.table.Table object or the name of a file that can be read
        by `astropy.table.Table.read()`. The filename can be either the
        path to a user file or the name of one of gunagala's included
        files. The table must use column names `QE` andv`Throughput`. If
        the table does not specify units then nm and electron / photon
        are assumed.
    minimum_exposure : astropy.units.Quantity
        Length of the shortest exposure that the camera is able to
        take.
    dark_current_dist : scipy.stats.rv_continuous, optional
        A 'frozen' continuous random variable object that describes the distribution
        of dark currents for the pixels in the image sensor. Used to create a `dark frame`
        of uncorrelated dark current values. If not given no dark frame is created.
    dark_current_seed: int, optional
        Seed used to initialise the random number generator before creating the dark
        frame. Set to a fixed value if you need to repeatedly generate the same dark frame.

    Attributes
    ----------
    bit_depth : int
        Same as parameters
    full_well : astropy.units.Quantity
        Same as parameters
    gain : astropy.units.Quantity
        Same as parameters
    bias : astropy.units.Quantity
        Same as parameters
    readout_time : astropy.units.Quantity
        Same as parameters
    pixel_size : astropy.units.Quantity
        Same as parameters
    resolution : astropy.units.Quantity
        Same as parameters
    read_noise : astropy.units.Quantity
        Same as parameters
    dark_current : astropy.units.Quantity
        Same as parameters
    minimum_exposure : astropy.units.Quantity
        Same as parameters
    saturation_level : astropy.units.Quantity
        Lowest of `full_well` and `2**bit_depth - 1 - bias`
    max_noise : astropy.units.Quantity
        Poisson + readout noise corresponding to `saturation_level`
    wavelenghts : astropy.units.Quantity
        Sequence of wavelengths from the QE data
    QE : astropy.units.Quantity
        Sequence of quantum efficiency values from the QE data.
    dark_frame: astropy.units.Quantity or None
        Array containing the dark current values for the pixels of the image sensor
    """
    def __init__(self,
                 bit_depth,
                 full_well,
                 gain,
                 bias,
                 readout_time,
                 pixel_size,
                 resolution,
                 read_noise,
                 dark_current,
                 QE,
                 minimum_exposure,
                 dark_current_dist=None,
                 dark_current_seed=None):

        self.bit_depth = int(bit_depth)
        self.full_well = ensure_unit(full_well, u.electron / u.pixel)
        self.gain = ensure_unit(gain, u.electron / u.adu)
        self.bias = ensure_unit(bias, u.adu / u.pixel)
        self.readout_time = ensure_unit(readout_time, u.second)
        self.pixel_size = ensure_unit(pixel_size, u.micron / u.pixel)
        self.resolution = ensure_unit(resolution, u.pixel)
        self.read_noise = ensure_unit(read_noise, u.electron / u.pixel)
        self.dark_current = ensure_unit(dark_current, u.electron / (u.second * u.pixel))
        self.minimum_exposure = ensure_unit(minimum_exposure, u.second)

        # Calculate a saturation level corresponding to the lower of the 'analogue' (full well)
        # and 'digital' (ADC) limit, in electrons.
        self.saturation_level = min(self.full_well,
                                    ((2**self.bit_depth - 1) * u.adu / u.pixel - self.bias) * self.gain)

        # Calculate the noise at the saturation level
        self.max_noise = (self.saturation_level * u.electron / u.pixel + self.read_noise**2)**0.5

        self.wavelengths, self.QE = get_table_data(QE,
                                                   column_names=('Wavelength', 'QE'),
                                                   column_units=(u.nm, u.electron / u.photon))

        # Generate dark frame
        if dark_current_dist is not None:
            try:
                dark_current_dist.rvs
            except AttributeError:
                raise ValueError("dark_current_dist ({}) has no rvs() method!".format(dark_current_dist))
            self.dark_frame = self._make_dark_frame(dark_current_dist, dark_current_seed)
        else:
            self.dark_frame = None

    def _make_dark_frame(self, distribution, seed=None):
        """
        Function to create a dark current 'image' in electrons per second per pixel.

        Creates an array of random, uncorrelated dark current values drawn from the
        statistical distribution defined by the `distribution` parameter and returns
        it as an astropy.units.Quantity.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous
            A 'frozen' continuous random variable object that describes the distribution
            of dark current values for the pixels in the image sensor.
        seed: int
            Seed used to initialise the random number generator before creating the dark
            frame. Set to a fixed value if you need to repeatedly generate the same dark
            frame.
        """
        if seed is not None:
            # Initialise RNG
            np.random.seed(seed)

        dark_frame = distribution.rvs(size=self.resolution.value.astype(int))
        dark_frame = dark_frame * u.electron / (u.second * u.pixel)\

        if seed is not None:
            # Re-initialise RNG with random seed
            np.random.seed()

        return dark_frame
