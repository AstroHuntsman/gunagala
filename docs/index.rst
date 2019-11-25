****************
Gunagala
****************

Introduction
============

Gungala is a Python package for modelling the performance of astronomical instruments, including
SNR/ETC/sensitivity limit calculations and generation of simulated data.

This package is not intended for rigorous, end-to-end simulations of telescope and instrument
systems. Instead gunagala implements parameterised models of instrument components in order to
enable rapid, efficient evaluation of instrument performance. Anticipated uses include exposure time
calculators, selection of commercial off the shelf components and exploration of the design
parameter space for custom components.

Gunagala includes a library of performance parameters for a number of existing commercial off the
shelf instrument components (e.g. CCD cameras, optical filters, telescopes and camera lenses) and
the user can easily add new/custom components either through YAML based config files or
programmatically in Python.

Gunagala is named as a gesture of respect to the traditional custodians of the land on which Siding
Spring Observatory sits, the Kamilaroi people of northern New South Wales. Gunagala is 'sky' in
the Kamilaroi/Gamilaraay language
(ref: `www.dnathan.com <http://www.dnathan.com/language/gamilaraay/dictionary/>`_). Aboriginal
Australians have studied the night skies above Australia for at least 50000 years. To learn more
about Aboriginal astronomy please visit http://www.aboriginalastronomy.com.au/.


Installation
============

Installing with pip
-------------------
.. highlight:: console

To install using the Python package manager `pip` use the following command::

  $ pip install git+https://github.com/AstroHuntsman/gunagala.git

Alternatively to install in 'editable mode' use::

  $ pip install -e git+https://github.com/AstroHuntsman/gunagala.git

Depending on the configuration of your system you may want to use `pip`'s ``--user`` or ``--root``
options to change the install location. See the pip documentation for details.

`Pip` will automatically install the Python packages required by Gunagala (`numpy`, `scipy`,
`astropy`, `PyYAML`, `matplotlib` and their dependencies) if they are not already installed. If you
want to install specific versions of the required packages from other sources do this before
installing Gunagala.

Installing from source
----------------------

The project source is in a GitHub repository at https://github.com/AstroHuntsman/gunagala. To
install using git on the command line::

  $ cd ~/Build
  $ git clone https://github.com/AstroHuntsman/gunagala.git
  $ cd gunagala
  $ python setup.py install

Alternatively if you expect to make changes to the Gunagala code install with the ``develop``
command instead::

  $ python setup.py develop

Setuptools will automatically install the Python packages required by Gunagala (`numpy`, `scipy`,
`astropy`, `PyYAML`, `matplotlib` and their dependencies) if they are not already installed. If you
want to install specific versions of the required packages from other sources do this before
installing Gunagala.

Running the test suite
----------------------

After installing Gunagala it is recommended that you run the suite of units tests. This can be done
at the command line using::

  $ python setup.py test

.. highlight:: python3

or from within a Python interpreter with::

  >> import gunagala
  >> gunagala.test()

Examples
========

The Gunagala package includes several examples in the form of `Jupyter <https://jupyter.org>`_
notebooks. These can be found in the `gungala/examples` directory after installing Gunagala, or they
can be viewed directly in the GitHub repository by going to
https://github.com/AstroHuntsman/gunagala/tree/master/examples and clicking on the ``.ipynb`` files.

Contributing
============

Please submit bug reports or feature requests in the form of GitHub Issues at
https://github.com/AstroHuntsman/gunagala/issues. For code contributions please fork and clone the
repository, create a feature branch and submit a Pull Request.  We recommend the
`astropy Developer Documentation <http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_
for a description of suitable workflows.

Changelog
=========

.. include:: ../CHANGES.rst

Reference/API
=============

.. automodapi:: gunagala
  :no-main-docstr:

.. automodapi:: gunagala.imager
  :skip: ensure_unit
  :skip: load_config
  :skip: Filter
  :skip: Optic
  :skip: Camera
  :skip: PSF
  :skip: Sky
  :skip: Simple
  :skip: MoffatPSF
  :skip: ZodiacalLight
  :skip: WCS
  :skip: SkyCoord
  :skip: CCDData
  :skip: interp1d
  :skip: InvalidTransformError
  :no-inheritance-diagram:

.. automodapi:: gunagala.optic
  :skip: Table
  :skip: ensure_unit
  :skip: get_table_data
  :skip: array_sequence_equal
  :no-inheritance-diagram:

.. automodapi:: gunagala.optical_filter
  :skip: Table
  :skip: interp1d
  :skip: brentq
  :skip: ensure_unit
  :skip: get_table_data
  :skip: minimize_scalar
  :skip: eval_chebyt
  :no-inheritance-diagram:

.. automodapi:: gunagala.camera
  :skip: Table
  :skip: ensure_unit
  :no-inheritance-diagram:

.. automodapi:: gunagala.psf
  :skip: discretize_model
  :skip: Fittable2DModel
  :skip: Moffat2D

.. automodapi:: gunagala.sky
  :skip: interp1d
  :skip: RectSphereBivariateSpline
  :skip: SmoothBivariateSpline
  :skip: SkyCoord
  :skip: GeocentricTrueEcliptic
  :skip: get_sun
  :skip: Angle
  :skip: Time
  :skip: ensure_unit

.. automodapi:: gunagala.utils
  :skip: get_pkg_data_filename
