Tutorial 4: Plotting multiple 1Ds
=================================

It is quite common to want to plot multiple 1D spectra on the same plot: in fact, so common that we will dedicate an entire introductory tutorial to it.

Because we will be doing quite a number of repetitive calls here, I suggest moving these commands into a Python script (or a Jupyter notebook), instead of typing them into the interactive REPL.
All of the following code examples are scripts which can be run from inside the ``nmrdata`` folder.
You can put the script anywhere you like, but will have to adjust the path to the ``nmrdata`` folder appropriately, so that penguins can find the data.

If you are using the sample ``nmrdata`` spectra, **start by placing these lines at the top of your script (or notebook cell):**

.. plot::
   :context: reset
   :nofigs:

   import penguins as pg
   em = pg.read(".", 1, 1)  # this is the same as `data_1d` in previous tutorials
   gm = pg.read(".", 1, 2)  # this is new

These two datasets have the same raw data, but are processed with different window functions: ``em`` uses an exponential window function, whereas ``gm`` uses a Gaussian window function.
You are, of course, free to substitute this with any of your own data.

----------------------

Staging multiple spectra
------------------------

This is easy: just stage them with any of the options described in :doc:`customise_1d`.
For the purposes of this tutorial, let's focus on the multiplet at 4.63 ppm.
The coupling is barely resolved in the ``em`` spectrum, but can be seen more clearly in the ``gm`` spectrum.
Add the following lines to your script, then run it:

.. plot::
   :context:

    em.stage(bounds="4.62..4.65", linestyle="--")   # stage the first one
    gm.stage(bounds="4.62..4.65")                   # stage the second one
    pg.mkplot()                                     # construct as usual
    pg.show()                                       # display as usual

-----------------------

Adding a legend
---------------

To add a plot legend, do one of the following:

1. Pass the legend labels as the ``label`` keyword argument when staging. penguins will automatically create a legend if any of the plots are staged with the ``label`` keyword.

2. Manually call ``matplotlib``'s :func:`~matplotlib.pyplot.legend` function.

If you do the latter, we assume you are at least a little familiar with ``matplotlib``, and won't discuss it further.
Here's an example of the former. Modify the two calls to ``stage()`` to include a ``label`` argument:

.. plot::
   :context: close-figs

    em.stage(bounds="4.62..4.65", linestyle="--", label="Exponential")
    gm.stage(bounds="4.62..4.65", label="Gaussian")
    pg.mkplot()
    pg.show()

Because the label is just any old Python string, you can combine this with the attribute access discussed in :doc:`/tutorials/read`.
The f-string syntax in Python 3.7+ is especially useful for this: for example, ``label=f"lb={em['lb']}"`` would create a legend label which contains the value of the ``LB`` parameter for that dataset.

-------------------------

Horizontal and vertical offsets
-------------------------------

TopSpin's multiple display mode has a few settings which allow you to stack spectra or to shift them by a constant amount, which is really useful for plotting.
However, these are not *spectrum-specific* options: they are options which affect the *entire plot*.
That's why these options will be passed to ``mkplot()``, rather than ``stage()``.

Horizontal offsets, controlled by the ``hoffset`` parameter, are specified in units of ppm.
The first spectrum will not be shifted, and subsequent spectra will be shifted rightwards (i.e. to lower chemical shift):

.. plot::
   :context: close-figs

    em.stage(bounds="4.62..4.65", label="Exponential")
    gm.stage(bounds="4.62..4.65", label="Gaussian")
    pg.mkplot(hoffset=0.03)
    pg.show()

Vertical offsets ``voffset`` are specified as a fraction of the height of the tallest spectrum.
Successive spectra are moved upwards.
Note that ``1`` is not the best value: often ``1.1`` or so is better as that adds a bit of empty space between plots.

.. plot::
   :context: close-figs

    em.stage(bounds="4.62..4.65", label="Exponential")
    gm.stage(bounds="4.62..4.65", label="Gaussian")
    pg.mkplot(voffset=1)
    pg.show()

Of course, you can specify *both* ``hoffset`` and ``voffset`` to get spectra that are diagonally offset from one another.
This is commonly used, for example, when plotting a series of related spectra.

----------------------

voffset and stacked
-------------------

We're going to do something weird for a while.
Stage ``em`` one time and ``gm`` three times (this effectively creates a few copies of ``gm`` to be plotted), then call `mkplot()` with ``voffset=1.1``:

.. plot::
   :context: close-figs

    em.stage(); gm.stage(); gm.stage(); gm.stage()
    pg.mkplot(voffset=1.1)
    pg.show()

You will notice that the three ``gm`` spectra are separated by quite a bit of white space.
This is the expected behaviour: the vertical separation between each plot is defined to be 1.1 times the height of the tallest spectrum (which in this case is ``em``).

In many cases this is OK, but in some situations this may not be desirable.
However, in some situations this may not be desirable.
What we *can* do is to separate each spectrum by its own height, thus ensuring that there isn't any excessive whitespace.
penguins lets you do this quickly using the ``stacked`` argument:

.. plot::
   :context: close-figs

    em.stage(); gm.stage(); gm.stage(); gm.stage()
    pg.mkplot(stacked=True)
    pg.show()

Which you use will depend on the scenario as well as your personal taste.

