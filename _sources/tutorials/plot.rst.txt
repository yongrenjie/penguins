Tutorial 2: Plotting data
=========================

In this tutorial (as well as all the following ones), we will reuse the ``data_1d`` and ``data_2d`` objects that we imported in the previous tutorial.

------------------------

The three stages of plotting
----------------------------

Plotting spectra using penguins is accomplished through three distinct steps:

1. **Staging**. In this step, a spectrum (i.e. the dataset returned by `read()`) is associated with an |Axes| which it will be plotted on.

   *Individual spectrum-level options* (e.g. the plot bounds and line colour) are specified here.

2. **Construction**. Here, all the spectra that were previously staged on a particular |Axes| are plotted.

   *Plot-level options* (e.g. horizontal / vertical separation between spectra, and axis labels) are specified here.

3. **Display**. The plot just needs to be displayed, and that's what we do here.

We will now illustrate each of these steps using a basic example.

.. note::
   If this is your first experience with matplotlib, it's okay: these first few tutorials are designed to not require any explicit understanding of matplotlib.

---------------------------

A minimal 1D example
--------------------

**Staging** in general simply refers to calling the ``stage()`` method on the Dataset objects:

.. plot::
   :format: doctest
   :context: reset
   :include-source: no
   :nofigs:
    
   >>> import penguins as pg
   >>> data_1d = pg.read(".", 1, 1)
   >>> data_2d = pg.read(".", 2, 1)

.. plot::
   :format: doctest
   :context:
   :nofigs:

   >>> data_1d    # we imported this in tutorial 1
   Dataset1D('/Users/yongrenjie/penguins/tests/nmrdata/1/pdata/1')
   >>> data_1d.stage()

This doesn't produce any visible output (yet); what it *has* done is it has "registered" the spectrum above, so that it will be plotted in the future.

**Construction** uses the `mkplot()` function:

.. plot::
   :format: doctest
   :context:
   :nofigs:

   >>> pg.mkplot()
   (<Figure size 600x400 with 1 Axes>, <AxesSubplot:xlabel='$\\rm ^{1}H$ (ppm)'>)

When we call `mkplot()`, we can see from the output that a |Figure| and an |Axes| (technically an ``AxesSubplot``) have been set up for us.
The *x*-axis label of the plot has automatically been set to a string which renders as $\mathrm{^1H~(ppm)}$.

Finally, **display** the plot using `show()`:

.. plot::
   :format: doctest
   :context:

   >>> pg.show()

Once you type this, the 1D plot above should be displayed in a new window.
You can manipulate this plot, zoom in/out, or save it as a picture using the various buttons available.

As you may expect, penguins provides a number of options that can be passed in order to customise these plots.
These will be described shortly in `customise_1d`, but before that, a quick 2D example.

---------------------------

A minimal 2D example
--------------------

2D spectra are plotted in exactly the same way:

.. plot::
   :format: doctest
   :context: close-figs

   >>> data_2d   # we imported this earlier, too
   Dataset2D('/Users/yongrenjie/penguins/tests/nmrdata/2/pdata/1')
   >>> data_2d.stage()
   >>> pg.mkplot()
   (<Figure size 500x500 with 1 Axes>, <AxesSubplot:xlabel='$\\rm ^{1}H$ (ppm)', ylabel='$\\rm ^{1}H$ (ppm)'>)
   >>> pg.show()

Notice that the |Axes| is set to a square shape: penguins does this automatically for 2D spectra.
However, there are some aspects of this plot which are clearly undesirable.
In `customise_2d` we will see how to improve it.

---------------------------

Axis labels and titles
----------------------

Before moving on, let's take a moment to look at one of the most basic ways of customising a plot.
The *x*-axis label, *y*-axis label, and title can be specified as arguments to the ``mkplot()`` function during the construction stage.
An example will likely be the clearest way of illustrating this.
The following plot is exactly the same as the 2D plot above, just with several new arguments passed to ``mkplot()``.

.. plot::
   :format: doctest
   :context: close-figs

   >>> data_2d.stage()
   >>> pg.mkplot(xlabel="My xlabel", ylabel="My ylabel", title="My title")
   >>> pg.show()

