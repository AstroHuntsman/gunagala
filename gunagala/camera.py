"""
Cameras (stricly the image sensor subsystem, not including optics, optical filters, etc)
"""
import os

from astropy import units as u
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from gunagala.utils import ensure_unit


data_dir = 'data/performance_data'


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
        the image sensor in both horizontal & vertical directions.
    read_noise astropy.units.Quantity
        Intrinsic noise of image sensor and readout electronics, in
        electrons/pixel units.
    dark_current : astropy.units.Quantity
        Rate of accumlation of dark signal, in electrons/second/pixel
        units.
    QE_filename : str
        Name of a file containing quantum efficieny as a function of
        wavelength data. Must be in a format readable by
        `astropy.table.Table.read()` and use column names `Wavelength`
        and `QE`. If the data file does not provide units nm and
        dimensionless unscaled will be assumed.
    minimum_exposure : astropy.units.Quantity
        Length of the shortest exposure that the camera is able to
        take.

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
    """
    def __init__(self, bit_depth, full_well, gain, bias, readout_time, pixel_size, resolution, read_noise,
                 dark_current, QE_filename, minimum_exposure):

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

        # Calculate a saturation level corresponding to the lower of the 'analogue' (full well) and 'digital'
        # (ADC) limit, in electrons.
        self.saturation_level = min(self.full_well, ((2**self.bit_depth - 1) * u.adu / u.pixel - self.bias) * self.gain)

        # Calculate the noise at the saturation level
        self.max_noise = (self.saturation_level * u.electron / u.pixel + self.read_noise**2)**0.5

        QE_data = Table.read(get_pkg_data_filename(os.path.join(data_dir, QE_filename)))

        if not QE_data['Wavelength'].unit:
            QE_data['Wavelength'].unit = u.nm
        self.wavelengths = QE_data['Wavelength'].quantity.to(u.nm)

        if not QE_data['QE'].unit:
            QE_data['QE'].unit = u.electron / u.photon
        self.QE = QE_data['QE'].quantity.to(u.electron / u.photon)
