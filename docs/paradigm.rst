Penguins' Plot Paradigm
=======================

It should come as no surprise that penguins is essentially a wrapper around ``matplotlib``.
For those who have used ``matplotlib`` before, the way in which penguins constructs plots likely seems familiar; however, it also undoubtedly does not feel *quite* the same.  To other users, it may be a little bit reminiscent of ``git``'s three-stage process (add, commit, push): in fact, the term "stage" was indeed stolen from ``git``.

In any case, this document attempts to explain a little bit of how penguins works behind the scenes, utilising ``matplotlib`` in each plotting stage.
You probably don't *need* to read this before the :doc:`plot1d` and :doc:`plot2d` pages, but we do suggest reading this *before* moving on to the :doc:`cookbook`.

.. note::
   Penguins tries its best to prevent you, the user, from needing to import ``matplotlib`` explicitly. Therefore, at each stage, it tries to provide enough tools for users to do most sensible things. Consequently, none of the examples in the documentation require a ``matplotlib`` import.

   Note that this doesn't extend to *methods* on ``matplotlib`` classes such as :class:`~matplotlib.axes.Axes`: those are fair game, since you don't need to actually import ``matplotlib`` to use them.
   
   If you find that you *do* need to import ``matplotlib``, that's of course fine; but please do get in touch so that we can see whether it is worth implementing a penguins interface.


Staging and the PHA
-------------------

Calling the :meth:`~penguins.dataset.Dataset1D.stage()` methods on Dataset objects has the main job of constructing a :class:`~penguins.pgplot.PlotObject1D` or :class:`~penguins.pgplot.PlotObject2D` object, depending on the dimensionality of the spectrum.
These objects essentially encapsulate *all* the information that :func:`plt.plot() <matplotlib.pyplot.plot>` (1D) and :func:`plt.contour() <matplotlib.pyplot.contour>` need to plot the graphs. For example, for 2D spectra, the full contour levels are calculated from the parameters given when :class:`~penguins.pgplot.PlotObject2D` is instantiated.

Once constructed, the PlotObjects are added to what is known as the **plot holding area**, or PHA. The main bit of the PHA is the ``plot_queue`` attribute, which is the list of PlotObjects. The PHA's other job is to keep track of which default colours have been generated so far.

As of now, the PHA will only accept newly staged spectra if they have the same dimensionality as the existing spectra in the PHA, although we may relax that. You can access the PHA using the :func:`~penguins.get_pha()` function, but I cannot yet think of a use case for that, except for this code snippet which demonstrates what's going on::

   >>> ds1 = pg.read("data/pt2", 1, 1)   # 1H
   >>> ds2 = pg.read("data/pt2", 2, 1)   # 13C
   >>> # This creates two PlotObject1D objects and adds them to the plot queue.
   >>> ds1.stage(); ds2.stage()
   >>> pg.get_pha()
   <penguins.pgplot.PlotHoldingArea object at 0x1150aab50>
   >>> # The plot queue is a list of the two objects we created earlier.
   >>> pg.get_pha().plot_queue
   [<penguins.pgplot.PlotObject1D object at 0x10f7a1d10>, <penguins.pgplot.PlotObject1D object at 0x115731550>]
   >>> # This is a 2D spectrum, so we can't stage it!
   >>> ds3 = pg.read("data/pt2", 3, 1)   # COSY
   >>> ds3.stage()
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "/Users/yongrenjie/penguins/penguins/dataset.py", line 418, in stage
       pgplot.stage2d(self, *args, **kwargs)
     File "/Users/yongrenjie/penguins/penguins/pgplot.py", line 360, in stage2d
       raise TypeError("Plot queue already contains 1D spectra.")
   TypeError: Plot queue already contains 1D spectra.

Note that at this point we have not yet actually called any ``matplotlib`` functions.
The PHA is a notion that purely belongs to penguins.
It may seem slightly confusing that plot options have to be specified here, given that ``matplotlib``'s functions are not being called yet.
The reason is because penguins tries to abstract away the (relatively) low-level ``matplotlib`` interface, where you would have to calculate the *x*- and *y*-data yourself, then call :func:`plt.plot() <matplotlib.pyplot.plot>` on every spectrum with a different set of options.
It seemed far more logical to tie the *dataset-specific* options to a Dataset method.


Constructing a plot
-------------------

Each instance of the PHA is associated with one plot. :func:`~penguins.mkplot()` performs several jobs when it is called:

1. If the ``ax`` parameter is not provided, then chooses the currently active :class:`~matplotlib.axes.Axes` to plot spectra on.

2. Iterates over the plot queue and plots all the spectra in them on the given :class:`~matcontourlib.axes.Axes`. For 1D spectra this is done with :meth:`ax.plot() <matcontourlib.axes.Axes.plot>`, and for 2D spectra :meth:`ax.contour() <matcontourlib.axes.Axes.contour>`.

3. Stores some properties from the plots in the PHA, such as colours and vertical heights of stacked spectra. These can be accessed via :func:`~penguins.get_properties()`, and are wiped on the next call to :func:`~penguins.mkplot()`.

4. Empties the PHA plot queue and restarts the colour cycle.

This has the further implication that *every spectrum in the same PHA will be plotted on the same set of Axes*.
This does not matter much if you only have one set of ``Axes``, but if you want to do something like subplots, then you need to follow the correct order of operations so that the right spectra are on the right ``Axes``.
As a trivial example, consider what happens if you stage a spectrum *after* calling :func:`~penguins.mkplot()`::

   >>> ds1.stage()    # adds 1H to the PHA
   >>> pg.mkplot()    # empties the PHA, also calls ax.plot() on the 1H data
   >>> ds2.stage()    # adds 13C to the PHA, but it's never plotted
   >>> pg.show()      # will only have the 1H!

If you want to do anything with the :class:`~matplotlib.figure.Figure` and :class:`~matplotlib.axes.Axes` objects, such as setting the ``Axes`` position, **the best time to do it is after calling** :func:`~penguins.mkplot()`.
:func:`~penguins.mkplot()` returns ``(fig, ax)`` for you to carry out any other methods you may want to.

(Side note: if you are wondering about :func:`~penguins.mkinset()`, it basically creates the inset axes, passes it as a parameter to :func:`~penguins.mkplot()`, then draws the box and lines connecting the inset to the main spectrum.)


Displaying the plot
-------------------

At this stage all the necessary ``matplotlib`` functions have been called, so all we need to do is to show the plot using :func:`plt.show() <matplotlib.pyplot.show>`.
After reading about the previous two stages, you will be glad to know that penguins does not try to overcomplicate this.
The only suggestion we make is to use :func:`~penguins.show()` instead: it saves you from having to import ``matplotlib``, keeping in line with our ethos. 😄

