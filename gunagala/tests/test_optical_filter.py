import pytest
import astropy.units as u

from ..optical_filter import Filter


@pytest.fixture(scope='module', params=('table', 'chebyshev', 'butterworth'))
def optical_filter(request):
    if request.param == 'table':
        bandpass = Filter(transmission_filename='astrodon_g.csv',
                          sky_mu=22.5 * u.ABmag)
    elif request.param == 'chebyshev':
        bandpass = Filter(chebyshev_params={'wave1': 0.700 * u.micron,
                                            'wave2': 855.5,
                                            'order': 50,
                                            'ripple': 0.14,
                                            'peak': 0.95},
                          sky_mu=20.5 * u.ABmag)
    elif request.param == 'butterworth':
        bandpass = Filter(butterworth_params={'wave1': 708 * u.nm,
                                              'wave2': 855.5,
                                              'order': 3,
                                              'peak': 0.95},
                          sky_mu=20.5 * u.ABmag)
    else:
        pytest.fail("Unknown filter type {}!".format(request.param))

    return bandpass


def test_filter_table(optical_filter):
    assert isinstance(optical_filter, Filter)
    waves = (0.3, 0.4, 0,5, 0,6, 0,7, 0.8, 0.9, 1.0, 1.1) * u.um
    trans = optical_filter.transmission(waves)
    assert isinstance(trans, u.Quantity)
    assert len(trans) == len(waves)
    assert trans.unit == u.dimensionless_unscaled
    assert (trans <= 1).all()
    assert (trans >= 0).all()


def test_filter_bad():
    with pytest.raises(ValueError):
        Filter(transmission_filename='astrodon_g.csv',
               chebyshev_params={'wave1': 0.700 * u.micron,
                                            'wave2': 855.5,
                                            'order': 50,
                                            'ripple': 0.14,
                                            'peak': 0.95},
               sky_mu=21.5 * u.ABmag)
