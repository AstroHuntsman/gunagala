import pytest
import astropy.units as u
from astropy.table import Table

from gunagala.optic import Optic, list_surfaces, make_throughput


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


def test_bad_throughput():
    with pytest.raises(IOError):
        telescope = Optic(aperture=279 * u.mm,
                          focal_length=620 * u.mm,
                          central_obstruction=129 * u.mm,
                          throughput='notavalidthroughputfile')


def test_bad_columns():
    with pytest.raises(ValueError):
        ws = [800, 1800] * u.nm
        tps = [0.6, 0.6] * u.dimensionless_unscaled
        throughput = Table(data = [ws, tps], names=['Banana', 'Cucumber'])
        telescope = Optic(aperture=279 * u.mm,
                          focal_length=620 * u.mm,
                          central_obstruction=129 * u.mm,
                          throughput=throughput)


def test_bad_units():
    with pytest.raises(u.UnitConversionError):
        ws = [800, 1800] * u.imperial.furlong
        tps = [0.6, 0.6] * u.fortnight
        # Furlong is actually a valid unit here, but fortnight isn't
        throughput = Table(data = [ws, tps], names=['Wavelength', 'Throughput'])
        telescope = Optic(aperture=279 * u.mm,
                          focal_length=620 * u.mm,
                          central_obstruction=129 * u.mm,
                          throughput=throughput)


def test_list_surfaces():
    surfaces = list_surfaces()
    assert len(surfaces) > 0
    assert isinstance(surfaces[0], str)


def test_throughput_one_one():
    surfaces = list_surfaces()
    throughput = make_throughput([('aluminium_12deg_protected', 1)])
    assert isinstance(throughput, Table)


def test_throughput_two_one():
    surfaces = list_surfaces()
    throughput = make_throughput([('aluminium_12deg_protected', 1),
                                  ('gold_12deg_protected', 1)])
    assert isinstance(throughput, Table)


def test_throughput_one_two():
    surfaces = list_surfaces()
    throughput = make_throughput([('silver_12deg_protected', 2)])
    assert isinstance(throughput, Table)


def test_throughput_bad_surface():
    with pytest.raises(AssertionError):
        throughput = make_throughput([('volume', 1)])


@pytest.mark.xfail(raises=NotImplementedError)
def test_throughput_different_wavelengths():
    surfaces = list_surfaces()
    throughput = make_throughput([('aluminium_12deg_protected', 1),
                                  ('aluminium_45deg_protected', 1)])
    assert isinstance(throughput, Table)
