Installation
============

.. highlight:: bash

Penguins can be installed via pip::

   pip install penguins

**Penguins requires Python 3.7 or later.** Other dependencies are:

* ``numpy`` v1.17.0+
* ``matplotlib``
* ``seaborn``

.. note:: 
   The only actual feature that requires 3.7 is ``from __future__ import annotations`` (`PEP 563 <https://www.python.org/dev/peps/pep-0563/>`_), but I hope you'll forgive me as I wanted to use this as an opportunity to learn about the type annotation options in recent Python versions.
   In principle, this is not really necessary.
   I could use strings in the type hints instead of the actual classes.
   If you need penguins on an older version of Python, raise an issue and we can work on it.
