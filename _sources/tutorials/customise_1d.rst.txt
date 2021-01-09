Tutorial 3: Customising 1D plots
================================

Here, we will learn how to change various aspects of the basic 1D plot.
We will continue using the ``data_1d`` object that we imported in :doc:`/tutorials/read`.

Recall that plotting consists of three parts: *staging*, *construction*, and *display*.
In this tutorial, we will only cover options that pertain to **staging**: that is, options that only affect individual spectra.
In the :doc:`next tutorial </tutorials/multiple_1d>`, when we discuss how to plot multiple spectra on the same axes, we will discuss the various options for construction.

-------------

Bounds
------

Our original plot in :doc:`/tutorials/plot` covers a spectral width of 16 ppm, but only around 7 ppm of the spectrum contains signals.
As a refresher, these are the steps which we took to make it:

.. plot::
   :format: doctest
   :context: reset

    >>> import penguins as pg
    >>> data_1d = pg.read(".", 1, 1)    # read in data
    >>> 
    >>> data_1d.stage()                 # stage
    >>> pg.mkplot()                     # construct
    >>> pg.show()                       # display

It would probably be a good idea to only show the region between, say, 0.3 ppm and 7 ppm.
To do this, we pass the *bounds* parameter to ``stage()``.
If you have the previous figure still open, close it first, and re-stage the dataset again.
This will use the same dataset to make a new plot:

.. plot::
   :format: doctest
   :context: close-figs

    >>> data_1d.stage(bounds="0.3..7")
    >>> pg.mkplot()
    >>> pg.show()

In this example, we specified the bounds as a string ``{low}..{high}``.
The bounds can alternatively be specified as a tuple of numbers::

    >>> data_1d.stage(bounds=(0.3, 7))    # same as above

If you do not want to specify a lower (or upper) bound, then omit it from the string, or pass ``None`` as one of the tuple components.
The following handy table describes all the possibilities (the last case is equivalent to simply not passing the *bounds* parameter at all).

+----------------------+-------------+------------------+
| Region to be plotted | String form | Tuple form       |
+======================+=============+==================+
| Between 4 and 5 ppm  | ``"4..5"``  | ``(4, 5)``       |
+----------------------+-------------+------------------+
| Below 5 ppm only     | ``"..5"``   | ``(None, 5)``    |
+----------------------+-------------+------------------+
| Above 5 ppm only     | ``"5.."``   | ``(5, None)``    |
+----------------------+-------------+------------------+
| Entire spectrum      | ``".."``    | ``(None, None)`` |
+----------------------+-------------+------------------+

|v|

----------------

Colours and other aesthetics
----------------------------

The other major spectrum-specific options are aesthetic choices, such as colours, linewidths, and so on.
The way ``stage()`` handles this is by taking keyword arguments and passing them along to ``matplotlib``'s |plot| function (which penguins ultimately uses to draw the actual spectrum).

That means that *any* configuration option that you can pass to |plot| can *also* be passed to ``stage()``.
The full list of options is described in the ``matplotlib`` documentation, but some of the most commonly used ones are listed here.

 - ``color``: any of the formats listed in :std:doc:`tutorials/colors/colors`. If you don't specify a colour, penguins will draw from the ``deep`` palette in the ``seaborn`` package.
 - ``linewidth``: a float. The default value is ``1``.
 - ``linestyle``: set to (for example) ``--`` for a dashed line.

As a horrible example, let's try using *all* of these options together:

.. plot::
   :format: doctest
   :context: close-figs

    >>> data_1d.stage(bounds="0.3..7", color="red", linewidth=0.5, linestyle="--")
    >>> pg.mkplot()
    >>> pg.show()

