.. penguins documentation master file, created by
   sphinx-quickstart on Wed Jun 10 13:21:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |penguin| replace:: 🐧

Welcome to penguins!
====================

|penguin| *Penguins: an Easy and NullPointerException-free Gateway to Unpacking and Illustrating NMR Spectra* |penguin|

::

   import penguins as pg
   import matplotlib.pyplot as plt

   example_spec = pg.read("data/exam2d_HC/", 3, 1)
   pg.plot2d(example_spec,
             contour_levels=(1e6, None, None),
             contour_colors=("blue", "red"))
   plt.xlim(7.4, 0.6)
   plt.ylim(133.8, 6.8)
   plt.show()

.. image:: images/exam2d_hc.png
   :width: 400
   :align: center
   :alt: HSQC spectrum of cyclosporin (TopSpin example data)
   :class: no-scaled-link

Navigation
==========

.. toctree::
   :hidden:
   
   install
   quickstart
   topspin
   reference

* :doc:`reference`
* :ref:`genindex`
* :ref:`search`
