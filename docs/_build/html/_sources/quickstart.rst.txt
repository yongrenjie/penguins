Quickstart
==========


Importing a spectrum
--------------------

Importing spectra is done with :func:`~penguins.read`. :func:`~penguins.read` takes three parameters: the path to the *spectrum folder*, the expno, and the procno. This returns one of several possible ``Dataset`` objects, depending on the dimensionality of the spectrum selected::

   >>> import penguins as pg
   >>> hsqc_ds = pg.read("data/rot1", 3, 1)
   >>> hsqc_ds
   Dataset2D('/Users/yongrenjie/penguins/tests/data/rot1/3/pdata/1')

(If you are not familiar with TopSpin's directory layout yet, see :doc:`topspin`.) Note that there is no support for 3D or higher spectra.


Reading parameters
------------------

Spectral parameters (both acquisition and processing) can be accessed via their TopSpin names using dictionary-like syntax::

   >>> hsqc_ds["ns"]
   16
   >>> hsqc_ds["td"]   # see also "Non-Uniform Sampling"
   array([ 128, 2048])
   >>> hsqc_ds["si"]
   array([1024, 2048])
   >>> hsqc_ds["nuc1"]
   ('13C', '1H')

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

   You can either display the window using :func:`~penguins.show()`, or save a figure using :func:`~penguins.savefig`. Both act as wrappers around the corresponding ``matplotlib`` functions.

An example::

   # Staging
   hsqc_ds.stage(f1_bounds=(81, 11),           # (upper, lower) in ppm
                 f2_bounds=(4.2, 1),
                 colors=("blue", "red"),       # (positive, negative)
                 levels=(5e4, None, None)      # (base level, increment, number of levels)
                 )
   # Construct
   pg.plot()
   # Display
   pg.show()

will give the following inset of the HSQC:

.. image:: images/quickstart_plot2d.png
   :align: center

An example of a 1D plot is as follows::

   # Staging
   prot = pg.read("data/rot1", 1, 1)
   prot.stage(bounds=(7, None),          # no right bound: stretches to right edge of spectrum
              color="darkviolet",
              label=r"$\mathrm{^{1}H}$ spectrum")   # using some LaTeX syntax
   # Construct
   pg.plot()
   # Display
   pg.show()

.. image:: images/quickstart_plot1d.png
   :align: center

For a more complete explanation of the options, please see :doc:`plot1d`.
