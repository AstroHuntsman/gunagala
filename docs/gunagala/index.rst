****************
Gunagala
****************

Gungala is a Python package for modelling the performance of astronomical instruments, including SNR/ETC/sensitivity
limit calculations and generation of simulated data.

This package is not intended for rigorous, end-to-end simulations of telescope and instrument systems. Instead gunagala
implements parameterised models of instrument components in order to enable rapid, efficient evaluation of instrument
performance. Anticipated uses include exposure time calculators, selection of commercial off the shelf components and
exploration of the design parameter space and for custom components.

Gunagala includes a library of performance parameters for a number of existing commercial off the shelf instrument
components (e.g. CCD cameras, optical filters, telescopes and camera lenses) and the user can easily add new/custom
components either through YAML based config files or programmatically in Python.

Gunagala is the word for sky in Kamilaroi/Gamilaraay, the language of the Traditional Owners of the land on which Siding
Spring Observatory stands
(ref: `www.dnathan.com <http://www.dnathan.com/language/gamilaraay/dictionary/GAM_G.HTM#gunagala>`_).

Reference/API
=============

.. automodapi:: gunagala
