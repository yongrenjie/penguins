Subplots
========

Many of the most useful plots involve having multiple plots on one figure, so it arguably deserves a page of its own.

Penguins also provides a wrapper around :func:`plt.subplots() <matplotlib.pyplot.subplots>`. The first thing to do, therefore, is to call :func:`~penguins.subplots()`::

   # Create subplots
   _, axs = pg.subplots(2, 2) 

At this point, I should mention that it is not a *pure* wrapper.
If you do not specify the ``figsize`` parameter, then :func:`~penguins.subplots()` will automatically choose ``(4 * ncols, 4 * nrows)`` as the size.
In other words, every subplot will be 4 inches by 4 inches.
Of course, you can manually specify ``figsize=(width, height)`` if you don't like this.

Now, :func:`~penguins.subplots()` returns a tuple containing the :class:`~matplotlib.figure.Figure` instance and a :class:`np.ndarray <numpy.ndarray>` of :class:`~matplotlib.axes.Axes` instances.
In order to plot the correct spectrum on the correct ``Axes``, we need to pass the appropriate ``Axes`` instance to :func:`~penguins.mkplot()`.
A common technique is to iterate over ``axs.flat``.
In this case, what we can do is to set up the list of datasets to be plotted, their ``baselev`` values, and their titles, then use :py:func:`zip` to iterate over all the lists one at a time::

   # Set up the lists.
   # 15N HMQC; 13C HSQC; COSY; NOESY
   spectra = [pg.read("data/noah", i, 1) for i in range(1, 5)]
   levels = [7e3, 2.3e4, 8.5e5, 8.9e4]
   titles = [r"$^{15}$N HMQC", r"$^{13}$C HSQC", "COSY", "NOESY"]
   clr = ("blue", "red")  # we use the same colours for all spectra
   # Iterate over the lists.
   for spec, ax, lvl, title, char in zip(spectra, axs.flat, levels, titles, "abcd"):
       # Staging proceeds as normal
       spec.stage(levels=lvl, colors=clr)
       # When constructing the plot, you need to pass the correct axis instance
       pg.mkplot(axis=ax,
                 title=title,
                 autolabel="nucl")
       # Add a label in the top left corner of each spectrum.
       ax.text(x=0.02, y=0.97, s=f"({char})", transform=ax.transAxes,
               fontweight="semibold", verticalalignment="top")
    # Display as usual (outside the loop)
    pg.show()
 
.. image:: images/cookbook_subplots.svg

Incidentally, we needed *three* sets of curly braces inside the ``xlabel`` and ``ylabel`` strings. One is for the f-string variable substitution; the other two get collapsed into one set of *literal* curly braces. The literal curly braces are needed for the LaTeX parser to superscript the entire mass number (or else we would end up with ``$^15$N``: :superscript:`1`\ 5N).


More examples will be added soon.


.. note:: If you want to do anything with :func:`~penguins.get_properties()`, then you need to do it inside the loop, as each call to :func:`~penguins.mkplot()` will reset the :class:`~penguins.pgplot.PlotProperties` instance.
