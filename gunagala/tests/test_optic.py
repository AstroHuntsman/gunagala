import pytest
import astropy.units as u

from ..optic import Optic


@pytest.fixture(scope='module')
def lens():
    lens = Optic(aperture=14 * u.cm,
                 focal_length=0.391 * u.m,
                 throughput_filename='canon_throughput.csv')
    return lens


@pytest.fixture(scope='module')
def telescope():
    telescope = Optic(aperture=279 * u.mm,
                      focal_length=620 * u.mm,
                      central_obstruction=129 * u.mm,
                      throughput_filename='rasa_tau.csv')
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
