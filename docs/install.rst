Installation
============

.. highlight:: bash

Penguins hasn't been pushed to pypi yet, so the only way to install it is via my GitHub repo::

   git clone https://github.com/yongrenjie/penguins
   cd penguins
   pip install .

Penguins' requirements are:

* Python 3.6+
* ``numpy`` v1.17.0+
* ``matplotlib`` (any version)

Technically there are no features that require recent versions of Python or numpy, but I used `f-strings <https://www.python.org/dev/peps/pep-0498/>`_ in the code, and I also feed :class:`pathlib.Path` objects to :func:`np.fromfile() <numpy.fromfile>` which relies on a relatively recent version of ``numpy``.
