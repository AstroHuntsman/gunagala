import pytest
import astropy.units as u
from astropy.table import Table

from gunagala.optic import Optic


@pytest.fixture(scope='module')
def lens():
    lens = Optic(aperture=14 * u.cm,
                 focal_length=0.391 * u.m,
                 throughput='canon_throughput.csv')
    return lens


@pytest.fixture(scope='module')
def telescope():
    telescope = Optic(aperture=279 * u.mm,
                      focal_length=620 * u.mm,
                      central_obstruction=129 * u.mm,
                      throughput='rasa_tau.csv')
    return telescope


def test_lens(lens):
    assert isinstance(lens, Optic)
    assert lens.aperture == 140 * u.mm
    assert lens.focal_length == 39.1 * u.cm
    assert lens.central_obstruction == 0 * u.mm


def test_telescope(telescope):
    assert isinstance(telescope, Optic)
    assert telescope.aperture == 0.279 * u.m
    assert telescope.focal_length == 62 * u.cm
    assert telescope.central_obstruction == 129 * u.mm


def test_table_throughput():
    ws = [800, 1800] * u.nm
    tps = [0.6, 0.6] * u.dimensionless_unscaled
    throughput = Table(data = [ws, tps], names=['Wavelength', 'Throughput'])
    telescope = Optic(aperture=279 * u.mm,
                      focal_length=620 * u.mm,
                      central_obstruction=129 * u.mm,
                      throughput=throughput)
    assert isinstance(telescope, Optic)
    assert (telescope.wavelengths == ws).all()
    assert (telescope.throughput == tps).all()


def test_file_throughput(tmpdir):
    ws = [800, 1800] * u.nm
    tps = [0.6, 0.6] * u.dimensionless_unscaled
    throughput = Table(data = [ws, tps], names=['Wavelength', 'Throughput'])
    throughput_path = str(tmpdir.join('test_throughput.csv'))
    throughput.write(throughput_path)
    telescope = Optic(aperture=279 * u.mm,
                      focal_length=620 * u.mm,
                      central_obstruction=129 * u.mm,
                      throughput=throughput_path)
    assert isinstance(telescope, Optic)
    assert (telescope.wavelengths == ws).all()
    assert (telescope.throughput == tps).all()
