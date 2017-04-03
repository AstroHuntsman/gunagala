import pytest
import numpy as np
import astropy.units as u

from ..optical_filter import Filter


@pytest.fixture(scope='module', params=('table', 'chebyshev', 'butterworth'))
def optical_filter(request):
    if request.param == 'table':
        bandpass = Filter(transmission_filename='astrodon_g.csv')
    elif request.param == 'chebyshev':
        bandpass = Filter(chebyshev_params={'wave1': 0.700 * u.micron,
                                            'wave2': 855.5,
                                            'order': 50,
                                            'ripple': 0.14,
                                            'peak': 0.95})
    elif request.param == 'butterworth':
        bandpass = Filter(butterworth_params={'wave1': 700 * u.nm,
                                              'wave2': 855.5,
                                              'order': 250,
                                              'peak': 0.95})
    else:
        pytest.fail("Unknown filter type {}!".format(request.param))

    return bandpass


def test_filter(optical_filter):
    assert isinstance(optical_filter, Filter)
    waves = np.arange(0.3, 1.1, 0.01) * u.um
    trans = optical_filter.transmission(waves)
    assert isinstance(trans, u.Quantity)
    assert len(trans) == len(waves)
    assert trans.unit == u.dimensionless_unscaled
    assert (trans <= 1).all()
    assert (trans >= 0).all()


def test_filter_parameters(optical_filter):
    waves = np.arange(0.3, 1.1, 0.01) * u.um
    trans = optical_filter.transmission(waves)
    assert optical_filter.peak == pytest.approx(0.95, abs=0.05)
    assert optical_filter.lambda_peak.to(u.nm).value == pytest.approx(waves[trans.argmax()].to(u.nm).value,
                                                                      abs=10)
    if hasattr(optical_filter, 'wavelengths'):
        assert isinstance(optical_filter.FWHM, u.Quantity)
        assert isinstance(optical_filter.lambda_c, u.Quantity)
    else:
        assert optical_filter.FWHM.to(u.nm).value == pytest.approx(155.5, rel=0.1)
        assert optical_filter.lambda_c.to(u.nm).value == pytest.approx((700 + 855.5) / 2, rel=0.1)


def test_filter_bad():
    with pytest.raises(ValueError):
        Filter(transmission_filename='astrodon_g.csv',
               chebyshev_params={'wave1': 0.700 * u.micron,
                                 'wave2': 855.5,
                                 'order': 50,
                                 'ripple': 0.14,
                                 'peak': 0.95})
