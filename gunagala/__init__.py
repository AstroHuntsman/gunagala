# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is a Python package from modelling the performance of astronomical instruments, including SNR/sensitivity limit/ETC calculations and simulating data.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .signal_to_noise import *
    from .astroimsim import *
