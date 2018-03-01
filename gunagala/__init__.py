# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Performance modelling for astronomical instruments

This is a Python package for modelling the performance of astronomical instruments, including
SNR/ETC/sensitivity limit calculations and generation of simulated data.
"""

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.

    #from .example_mod import *
    pass
