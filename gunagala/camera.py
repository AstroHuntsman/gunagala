import os

from astropy import units as u
from astropy.table import Table

from .imager import ensure_unit


class Camera:

    def __init__(self, bit_depth, full_well, gain, bias, readout_time, pixel_size, resolution, read_noise,
                 dark_current, QE_filename, minimum_exposure):
        """Class representing a camera, which in this case means the image sensor, associated electronics, shutter,
        etc., but does not include any of the optical components of the system.

        Args:
            bit_depth (int): bits per pixel used by the camera analogue to digital converters
            full_well (Quantity): number of photo-electrons each pixel can receive before saturating
            gain (Quantity): number of photo-electrons corresponding to one ADU in the digital data
            bias (Quantity): bias level of image sensor, in ADU / pixel units. Used when determining saturation level.
            readout_time (Quantity): time required to read the data from the image sensor
            pixel_size (Quantity): pixel pitch
            resolution (Quantity): number of pixels across the image sensor in both horizontal & vertical directions
            read_noise (Quantity): intrinsic noise of image sensor and readout electronics, in electrons/pixel units
            dark_current (Quantity): rate of accumlation of dark signal, in electrons/second/pixel units
            QE_filename (string): name of a file containing quantum efficieny as a function of wavelength data. Must
                be in a format readable by `astropy.table.Table.read()` and use column names `Wavelength` and `QE`.
                If the data file does not provide units nm and dimensionless unscaled will be assumed.
            minimum_exposure (Quantity): length of the shortest exposure that the camera is able to take.
        """

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

        QE_data = Table.read(os.path.join('.data/performance_data', QE_filename))

        if not QE_data['Wavelength'].unit:
            QE_data['Wavelength'].unit = u.nm
        self.wavelengths = QE_data['Wavelength'].quantity.to(u.nm)

        if not QE_data['QE'].unit:
            QE_data['QE'].unit = u.electron / u.photon
        self.QE = QE_data['QE'].quantity.to(u.electron / u.photon)
