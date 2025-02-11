Gunagala
===================================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

This is a Python package for modelling the performance of astronomical instruments, including
SNR/ETC/sensitivity limit calculations and generation of simulated data. The documentation is at
`gunagala.readthedocs.io <https://gunagala.readthedocs.io/>`_

Gunagala is named as a gesture of respect to the traditional custodians of the land on which Siding
Spring Observatory sits, the Kamilaroi people of northern New South Wales. Gunagala is 'sky' in
the Kamilaroi/Gamilaraay language
(ref: `www.dnathan.com <http://www.dnathan.com/language/gamilaraay/dictionary/>`_ ). Aboriginal
Australians have studied the night skies above Australia for at least 50000 years. To learn more
about Aboriginal astronomy please visit http://www.aboriginalastronomy.com.au/.

.. image:: https://github.com/AstroHuntsman/gunagala/actions/workflows/pythontest.yml/badge.svg
    :target: https://github.com/AstroHuntsman/gunagala/actions
    :alt: Python package tests

.. image:: https://coveralls.io/repos/github/AstroHuntsman/gunagala/badge.svg?branch=develop
    :target: https://coveralls.io/github/AstroHuntsman/gunagala?branch=develop
    :alt: Coverage status

.. .. image:: https://readthedocs.org/projects/gunagala/badge/?version=develop
..    :target: http://gunagala.readthedocs.io/en/develop/?badge=develop
..   :alt: Documentation Status


Development
-----------

To install gunagala in development mode, clone the repository and run:

.. code-block:: bash

    git clone https://github.com/AstroHuntsman/gunagala.git
    cd gunagala
    pip install -e .

To run the tests on various python versions, install tox (`pip install tox`) and run:

.. code-block:: bash

    tox

Or run tests in your current python environement:

.. code-block:: bash

    cd gunagala
    pip install -e .[test]
    pytest

To build the documentation, install the package in development mode and run: 

.. code-block:: bash

    pip install -e .[docs]
    cd docs
    make html

Or just run:

.. code-block:: bash

    tox -e build_docs -- -aE

License
-------

This project is Copyright (c) Anthony Horton and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! gunagala is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
gunagala based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
