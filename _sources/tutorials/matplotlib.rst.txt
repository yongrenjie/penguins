Tutorial 7: Using matplotlib
============================

penguins is a *wrapper* around matplotlib, which is a library that provides plotting capabilities in Python.
What this means is that all the code in penguins ultimately calls a series of matplotlib functions.
The penguins functions are provided as "higher-level" constructs, which allow you to directly work with and plot NMR data, without worrying about "low-level" implementation details.

However, you will notice that on its own, penguins' plotting capabilities are slightly limited.
For example, it doesn't provide a function which lets you write arbitrary text on a graph.

The reason for this is because penguins (generally) does not try to reinvent the wheel.
If matplotlib already has a way to do something, then penguins will not re-implement it; instead, you should simply call the matplotlib function itself.

.. note::

   This tutorial assumes almost no prior knowledge of matplotlib.
   However, if you are not familiar with matplotlib, it is a good idea to check out some of the :std:doc:`introductory Matplotlib tutorials <tutorials/index>`
   matplotlib's documentation itself -- although very extensive -- can appear slightly intimidating to a beginner.
   There are also a great number of other tutorials online which may be more approachable.

------------------------

Accessing Figures and Axes
--------------------------

Two of the key objects required for matplotlib are the |Figure| and |Axes| objects.
The |Figure| refers to the *entire* plot area, whereas |Axes| refers only to one set of, well, axes.
For more information, see the official matplotlib documentation on :std:doc:`gallery/showcase/anatomy`, or any other online resource.

The easiest way to access these is via the `mkplot()` function.
Up until now, we have not bothered to collect anything as the return value from `mkplot()`.
In fact, it returns precisely the |Figure| and |Axes| objects.

One of the easiest things we can do with the |Axes| object is to change the *x*-axis label.
(You could do that more easily by passing the *xlabel* argument to `mkplot()`, but we'll ignore that for demonstration purposes.)

.. plot::

    import penguins as pg

    data_1d = pg.read(".", 1, 1)
    data_1d.stage()
    fig, ax = pg.mkplot()  # faster: pg.mkplot(xlabel="My x label")

    ax.set_xlabel("My x label")
    pg.show()

Alternatively, if you use `subplots()`, you'll get a |Figure| and a list of |Axes|.
This was mentioned previously in :doc:`subplots`.
It allows you to control each Axes individually.
Here we'll add a title to each of the two Axes, and then we'll add a *suptitle* for the entire Figure.
Because the Figure encompasses both Axes, the suptitle is sort of an overarching title for the entire plot.

.. plot::

    import penguins as pg

    data_1d = pg.read(".", 1, 1)
    data_2d = pg.read(".", 2, 1)

    fig, axs = pg.subplots2d(1, 2)

    data_1d.stage(axs[0])   # 1D data on the left
    pg.mkplot(axs[0])
    axs[0].set_title("1D data")

    data_2d.stage(axs[1])   # 2D data on the right
    pg.mkplot(axs[1])
    axs[1].set_title("2D data")

    fig.suptitle("Some NMR data with different dimensions")
    pg.show()

You might notice that the Figure suptitle overlaps with the Axes titles.
This can be fixed with a call to `cleanup_figure()` at the very end. 
.. plot::

    import penguins as pg

    data_1d = pg.read(".", 1, 1)
    data_2d = pg.read(".", 2, 1)

    fig, axs = pg.subplots2d(1, 2)

    data_1d.stage(axs[0])   # 1D data on the left
    pg.mkplot(axs[0])
    axs[0].set_title("1D data")

    data_2d.stage(axs[1])   # 2D data on the right
    pg.mkplot(axs[1])
    axs[1].set_title("2D data")

    fig.suptitle("Some NMR data with different dimensions")
    pg.cleanup_figure()
    pg.show()
