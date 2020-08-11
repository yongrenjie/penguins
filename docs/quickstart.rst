Quickstart
==========


Importing a spectrum
--------------------

Importing spectra is done with :func:`penguins.read`. :func:`~penguins.read` takes three parameters: the path to the *spectrum folder*, the expno, and the procno. This returns one of several possible Dataset objects, depending on the dimensionality of the spectrum selected::

   >>> import penguins as pg
   >>> pg.read("tests/data/rot1", 2, 1)
   Dataset1D('/Users/yongrenjie/penguins/tests/data/rot1/2/pdata/1')

(If you are not familiar with TopSpin's directory layout yet, see :doc:`topspin`.)
Note that penguins is only capable of reading Bruker data, and there is no support for 3D or higher spectra.


Accessing parameters
--------------------

Spectral parameters (both acquisition and processing parameters) can be accessed via their TopSpin names using dictionary-like syntax::

   >>> c13 = pg.read("tests/data/rot1", 2, 1)
   >>> c13["pulprog"]
   'zgpg30'
   >>> c13["td"]
   65536
   >>> hsqc = pg.read("tests/data/rot1", 3, 1)
   >>> hsqc["si"]
   array([1024, 2048])
   >>> hsqc["nuc1"]
   ('13C', '1H')

For 2D spectra, parameters which have values in both the indirect and direct dimensions are stored as either a tuple or a |ndarray|, depending on whether the underlying values are numeric.
The first element is always the value for the indirect dimension (f1), and the second element the value for the direct dimension (f2).

Penguins does not aspire to be a full processing package; it mainly focuses on plotting.
However, there are some functions for processing datasets, including integration and generation of projections/slices.
See :doc:`datasets` for more information.


Plotting spectra
----------------

Plotting is done in three stages. A brief summary is as follows:

1. *Stage* the spectrum (or spectra) to be plotted.

   Options that are specific to each individual spectrum, such as the range of chemical shifts to be plotted, legend labels, colours, etc. are specified at this stage. This uses the ``stage()`` methods on 1D and 2D Dataset objects.

2. *Construct* the plot.

   Options that affect the layout of the entire plot, such as horizontal/vertical offset between spectra, axis labels, etc. are specified at this stage. This uses the :func:`penguins.mkplot()` function.

3. *Display* the plot.

   The two main choices here are to display the window using :func:`penguins.show()`, or save a figure using :func:`penguins.savefig`. Both are simple wrappers around the corresponding ``matplotlib`` functions.

An example of a simple 1D plot is as follows.

.. plot::

   prot = pg.read("tests/data/rot1", 1, 1)
   # Staging
   prot.stage(bounds="2..7",   # plot between 2 and 7 ppm
              label=r"My proton spectrum")
   # Construct
   pg.mkplot()
   # Display
   pg.show()

The most important keyword argument for staging spectra is *bounds*. This can be specified either as a string ``lower..upper`` or a tuple of floats ``(lower, upper)``:

+---------------------+-------------+------------------+
| Region of interest  | String form | Tuple form       |
+=====================+=============+==================+
| Entire spectrum     | ``""``      | ``(None, None)`` |
+---------------------+-------------+------------------+
| Below 5 ppm only    | ``"..5"``   | ``(None, 5)``    |
+---------------------+-------------+------------------+
| Above 5 ppm only    | ``"5.."``   | ``(5, None)``    |
+---------------------+-------------+------------------+
| Between 4 and 5 ppm | ``"4..5"``  | ``(4, 5)``       |
+---------------------+-------------+------------------+

An example for a 2D spectrum is as follows:

.. plot::

   hsqc = pg.read("tests/data/pt2", 4, 1)
   # Staging
   hsqc.stage(f1_bounds="11..140",
              f2_bounds=(0.5, 8.5),
              levels=(2.5e4, None, None))
   # Construct plot.
   pg.mkplot(autolabel="nucl")
   # Display
   pg.show()

There are a few details worth mentioning here, since these are likely to be frequently used.

 - During staging, the *f1_bounds* and *f2_bounds* parameters are specified using exactly the same formats as before.

 - Contour levels are specified using the *levels* parameter as a tuple of ``(baselev, increment, nlev)``. In total, ``nlev`` positive contours will be drawn at heights of ``baselev``, ``baselev * increment``, ``baselev * (increment ** 2)``, ..., and ``nlev`` negative contours will be drawn at the corresponding negative values.

 - In the second stage of plot construction, the *autolabel* parameter generates nice axes labels that show the nucleus being measured.

From here, you may want to consider reading :doc:`paradigm` to get to know penguins' overall approach to plotting.
