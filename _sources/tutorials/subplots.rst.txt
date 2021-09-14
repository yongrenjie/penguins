Tutorial 6: Subplots
====================

Many of the most useful plots involve having multiple plots in one figure, for example to present a series of spectra, or to compare multiple different spectra.
In this tutorial, we will cover the basic usage of *subplots*.
We will assume you have already read in the ``data_1d`` and ``data_2d`` objects (either using the test data provided, or with your own data).

.. plot::
   :include-source: false
   :nofigs:
   :context:

   import penguins as pg
   data_1d = pg.read(".", 1, 1)
   data_2d = pg.read(".", 2, 1)

--------------------------------

Getting to know subplots2d()
----------------------------

The first part of creating a plot with multiple subplots is to first specify the layout of the subplots.
This can be done using the `subplots2d` function:

.. plot::
   :nofigs:
   :context:

   fig, axs = pg.subplots2d(2, 2)    # creates a 2-by-2 grid of subplots

`subplots2d()` returns two matplotlib objects.
The first, ``fig``, is a handle for the entire |Figure| (which encompasses all four subplots).
The second, ``axs``, is a 2D numpy |ndarray| whose elements are the individual |Axes| which we can plot on.

Each subplot has one |Axes|, and we can access these by selecting the correct element of the 2D |ndarray|.
For example, to access the top-left subplot, we need to manipulate only the top-left |Axes|, which can be accessed via ``axs[0][0]``.
Likewise, the top-right |Axes| can be accessed via ``axs[0][1]``.

The `subplots2d()` function automatically sets the figure size such that each subplot has (approximately) a 4 inch by 4 inch area.
This is a good starting size for 2D spectra, hence the name of the function.
If you want to specify the figure size (i.e. the size of the *entire* figure, not the size of a single subplot!) then you may do so by passing a tuple of ``(width, height)`` as the ``figsize`` keyword argument of `subplots2d()`, viz.

::

   fig, axs = pg.subplots2d(2, 2, figsize=(10, 4))

.. note::

    If you are already familiar with matplotlib, you will know that matplotlib itself has a |subplots| function.
    You can use that if you prefer: penguins' `subplots2d()` is merely a wrapper around this with a default ``figsize``.
    penguins also provides a direct wrapper, :func:`~penguins.subplots()`, that *doesn't* set a default ``figsize``.

----------------------------------

The ax keyword argument
-----------------------

So far, we have carried out staging and construction on a single Axes.
But wait! We didn't actually *specify* that we wanted this.
It turns out that if you don't specify it, penguins takes care of automatically setting up a new Axes for us to plot on, which is why the previous plots worked perfectly well.

Now, though, we need a bit more control over the process.
There are four different Axes which we need to plot separately on, and if we don't explicitly specify which Axes we want to plot on, penguins has no way of figuring this out on its own.
We therefore need to pass the ``ax`` parameter to both the staging method ``stage()``, as well as the construction function ``mkplot()``.

.. plot::
   :context:

   data_2d.stage(ax=axs[0][0], levels=3e5)   # top-left
   pg.mkplot(ax=axs[0][0])
   data_2d.stage(ax=axs[0][1], levels=3e5)   # top-right
   pg.mkplot(ax=axs[0][1])
   data_2d.stage(ax=axs[1][0], levels=3e5)   # bottom-left
   pg.mkplot(ax=axs[1][0])
   data_2d.stage(ax=axs[1][1], levels=3e5)   # bottom-right
   pg.mkplot(ax=axs[1][1])

---------------------------------

For loops and zip()
-------------------

Typing this out by hand is pretty inefficient.
It is far better to use a for loop which accomplishes the same thing.

Recall that ``axs`` was a 2D array of Axes instances.
This is analogous to a list of lists, and if we directly iterated over ``axs``, such as ``for ax in axs``, ``ax`` wouldn't be the individual Axes themselves but rather lists of Axes.
Instead, we can iterate over ``axs.flat``, which behaves *like* a 1D array (technically, it's an `iterator <numpy.ndarray.flat>`):

.. plot::
   :context: close-figs

   fig, axs = pg.subplots2d(2, 2)
   for ax in axs.flat:
       # Inside this loop, `ax` refers to an individual Axes.
       # It is also the name of the keyword parameter.
       data_2d.stage(ax=ax, levels=3e5)
       pg.mkplot(ax=ax)


Obviously, it's not particularly useful to plot the same thing four times.
However, you can easily customise each plot by staging a different dataset, or by passing various options to ``stage()`` as well as ``mkplot()``, as was already described in previous tutorials.
For this purpose, the builtin Python function :py:func:`zip()` is incredibly useful.
Let's see, for example, how we can plot the same spectrum 4 times with different contour levels.

.. plot::
   :context: close-figs

   fig, axs = pg.subplots2d(2, 2)

   contour_levels = [1e4, 3e4, 1e5, 3e5]
   titles = ["Lots of noise", "Some noise",
             "Just a bit of noise", "No noise"]

   for ax, level, title in zip(axs.flat, contour_levels, titles):
       data_2d.stage(ax=ax, levels=level)
       pg.mkplot(ax=ax, title=title)

Using ``zip()`` allows us to write a for loop which consumes multiple lists in parallel.
That is to say, each time we advance through the for loop, we advance through all of the lists provided to ``zip()`` simultaneously, such that the first contour level and the first title are associated with the first Axes, and so on.

**This is one of the most common "patterns" used for subplots in penguins.**
As another example, you can zip together a list of datasets with ``axs.flat`` to plot a series of different spectra.


--------

Labelling axes
--------------

When preparing graphics for publication, a common requirement is that each subplot must be labelled with a letter (for example).
The `label_axes` function takes care of this quite simply.
It takes a list of |Axes|, and a format string ``fstr`` in which the curly braces are replaced with the relevant character.
Further options for customisation can be found in the reference documentation.

.. plot::
   :context: close-figs

   fig, axs = pg.subplots2d(2, 2)

   contour_levels = [1e4, 3e4, 1e5, 3e5]
   titles = ["Lots of noise", "Some noise",
             "Just a bit of noise", "No noise"]

   for ax, level, title in zip(axs.flat, contour_levels, titles):
       data_2d.stage(ax=ax, levels=level)
       pg.mkplot(ax=ax, title=title)

   pg.label_axes(axs, fstr="({})", fontsize=12, fontweight="semibold")


