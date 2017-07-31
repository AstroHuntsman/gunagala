****************
Gunagala
****************

Introduction
============

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

Installation
============

Installing with pip
-------------------

To install using the Python package manager ``pip`` use the following command::

  pip install git+https://github.com/AstroHuntsman/gunagala.git

Alternatively to install in 'editable mode' use::

  pip install -r git+https://github.com/AstroHuntsman/gunagala.git

Depending on the configuration of your system you may want to use pip's ``--user`` or ``--root`` options to change
the install location. See the pip documentation for details.

Pip will automatically install the Python packages required by Gunagala (``numpy``, ``scipy``, ``astropy``, ``PyYAML``,
``matplotlib`` and their dependencies) if they are not already installed.

Installing from source
----------------------

The project source is in a GitHub repository at https://github.com/AstroHuntsman/gunagala. 

Running the test suite
----------------------

After installing Gunagala it is recommended that you run the suite of units tests. This can be done at the command line
using::

  $ python setup.py test

or from within a Python interpreter with::

  >>> import gunagala
  >>> gunagala.test()

Examples
========

The Gunagala package includes several examples in the form of `Jupyter <https://jupyter.org>`_ notebooks. These can be
found in the `gungala/examples` directory after installing Gunagala, or they can be viewed directly in the GitHub
repository by going to https://github.com/AstroHuntsman/gunagala/tree/master/examples and clicking on the ``.ipynb``
files.

Reference/API
=============

.. automodapi:: gunagala

.. automodapi:: gunagala.imager
  :allowed-package-names: gunagala.imager

.. automodapi:: gunagala.optic
  :allowed-package-names: gunagala.optic

.. automodapi:: gunagala.optical_filter
  :allowed-package-names: gunagala.optical_filter
