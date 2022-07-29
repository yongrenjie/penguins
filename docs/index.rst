.. |penguin| replace:: 🐧

Welcome to penguins!
====================

|penguin| *Penguins: an Easy and NullPointerException-free Gateway to Unpacking and Illustrating NMR Spectra* |penguin|

**Penguins** is a Python 3 package intended for generating publication-quality plots of NMR spectra in a programmatic, reproducible fashion.

The current penguins version is |version|. penguins requires Python 3.7 or above, and can be installed via ``pip``::

   pip install penguins

As of the current version, penguins only supports Bruker data, and can only work with processed data; there are no functions for working with raw data.


About this documentation
------------------------

The documentation is divided into five major sections (largely adhering to the principles outlined by `Divio <https://documentation.divio.com/introduction/>`_):

1. **Tutorials**: intended as a brief introduction to penguins' core features.

2. *(In progress)* **How-to guides**: ways of accomplishing specific tasks which are likely to occur frequently.

3. *(In progress)* **Gallery**: case studies from real-life plots with step-by-step walkthroughs.

4. **Explanations**: a more in-depth look at how penguins works behind the scenes.

5. **Reference**: a full description of the penguins API.


Contact
-------

If you have (any) question, comments, or would like to report a bug, please `create an issue on GitHub <https://github.com/yongrenjie/penguins/issues>`_.


Contents
========

.. toctree::
   :maxdepth: 2
   
   install
   tutorials/index
   howto/index
   gallery/index
   explanations/index
   reference/index

* :ref:`genindex`
* :ref:`search`
