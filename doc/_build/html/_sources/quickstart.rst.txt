Quickstart
==========


Importing a spectrum
--------------------

Importing spectra is done with :func:`~penguins.read`::

   import penguins as pg
   hsqc = pg.read("/opt/topspin4.0.8/examdata/exam2d_HC", 3, 1)

:func:`~penguins.read` takes three parameters: the path to the *spectrum folder*, the expno, and the procno. (If you are not familiar with TopSpin's directory layout yet, see :doc:`topspin`.)

This returns one of several possible Dataset objects, depending on the dimensionality of the spectrum selected. Note that there is no support for 3D or higher spectra.


Reading parameters
------------------

Spectral parameters (both acquisition and processing) can be accessed via their TopSpin names using dictionary-like syntax::

   hsqc["ns"]            # 16
   hsqc["td"]            # np.array(256, 1024)
   hsqc["si"]            # np.array(1024, 1024)
   hsqc["nuc1"]          # ('13C', '1H')

For 2D spectra, parameters which have values in both the indirect and direct dimensions are stored as either a tuple or a :class:`numpy.ndarray`, depending on whether the underlying values can be coerced to a float or not. Some parameters (such as those above) only make sense as ints, and those are stored as ints, not floats.

The first element is always the value for the indirect (*f*:subscript:`1`) dimension, and the second element the value for the direct (*f*:subscript:`2`) dimension.


Plotting spectra
----------------

Plotting is done in three stages:

1. *Stage* the spectrum (or spectra) to be plotted.

   Options that are specific to each individual spectrum, such as bounds on the plotting window, colours, etc. are specified at this stage.

2. *Construct* the plot.

   Options that affect the entire plot, such as horizontal/vertical offset between spectra, plot limits, etc. are specified at this stage.

3. *Display* the plot.

   You can either display the window using :func:`~penguins.pgplot.show()`, or save a figure using :func:`~penguins.pgplot.savefig`. Both act as wrappers around the corresponding ``matplotlib`` functions.

An example::

   hsqc.stage(bounds=((57, 7), (4, 0.5)),     # STAGING
              colors=("blue", "red"),
              levels=(1e6, None, None)
              )
   pg.plot()                                  # CONSTRUCT
   pg.show()                                  # DISPLAY

will give the following inset of the alkyl region of that HSQC:

.. image:: images/quickstart_plot2d.png
   :align: center

An example of a 1D plot is as follows::

   prot = pg.read("/opt/topspin4.0.8/examdata/exam1d_1H", 1, 1)
   prot.stage(bounds=(6.2, 4.8),  # use (None, 5) for everything ≥ 5 ppm
              color="darkviolet",
              label=r"Cyclosporin $^1$H")
   pg.plot()
   pg.show()

.. image:: images/quickstart_plot1d.png
   :align: center

For a more complete explanation of the options, please see :doc:`plot`.
