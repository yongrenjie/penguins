Installation
============

.. highlight:: bash

Penguins hasn't been pushed to pypi yet, so the only way to install it is via my GitHub repo::

   git clone https://github.com/yongrenjie/penguins
   cd penguins
   pip install .

**Penguins requires Python 3.7 or later.** Other dependencies are fairly standard packages for data analysis:

* ``numpy`` v1.17.0+
* ``matplotlib``
* ``pandas``
* ``seaborn``

(The only actual feature that requires 3.7 is ``from __future__ import annotations`` (`PEP 563 <https://www.python.org/dev/peps/pep-0563/>`_), but I hope you'll forgive me as I wanted to use this as an opportunity to learn about the static typing options in recent Python versions.)
