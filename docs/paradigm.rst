Penguins' Plot Paradigm
=======================

It should come as no surprise that penguins is essentially a wrapper around matplotlib.
The aim of this page is to (try to) explain a little bit of how penguins works behind the scenes, utilising matplotlib in each plotting stage.

One of the general principles that I have tried to adhere to is to minimise, or even completely obviate, the need for the user to explicitly import matplotlib.
Consequently, none of the examples in the documentation require a matplotlib import.

However, at the same time, I have also tried to retain as much compatibility with matplotlib as possible.
That means that should you want to customise the plot using the raw power of matplotlib, you should be able to do it, as far as possible.
Indeed, in many of the more advanced examples, we will call methods on |Axes| to customise the plots further.
(Note that using |Axes| methods doesn't require you, the user, to import matplotlib, so that's fair game!)


Staging and the PHA
-------------------

Calling the :meth:`1D <penguins.dataset._1D_PlotMixin.stage>` and :meth:`2D <penguins.dataset._2D_PlotMixin.stage>` methods on Dataset objects has the main job of constructing :class:`~penguins.pgplot.PlotObject1D` or :class:`~penguins.pgplot.PlotObject2D` objects respectively.
These objects essentially encapsulate *all* the information that |plot| (1D) and |contour| (2D) need to plot the graphs.
For example, for 2D spectra, all the contour levels are calculated from the parameters given inside the initialisation code of `PlotObject2D`.

Once constructed, the PlotObjects are added to what is known as the **plot holding area**, or PHA.
The PHA is an instance of `PlotHoldingArea`; each |Axes| has its own PHA, which can be accessed with ``ax.pha`` (although there is no real reason why one would ever need to access it).
At this point, we have not yet actually called any matplotlib functions; we have merely populated the PHA with a set of PlotObjects.

Since the PlotObjects contain *all* the information that needs to be passed to |plot| or |contour|, this means that any dataset-specific options must be passed along to ``stage()``.


Constructing a plot
-------------------

`mkplot` performs several jobs when it is called:

1. If the ``ax`` parameter is not provided, then chooses the currently active |Axes| to plot spectra on.

2. Iterates over that |Axes|'s plot holding area and plots all the spectra in them on the given |Axes|. For 1D spectra this is done with |plot|, and for 2D spectra |contour|. (Options such as *hoffset* affect every spectrum, which is why it is passed to `mkplot()` and not ``stage()``.)

3. Fixes the plot axes such that they resemble a typical NMR plot (e.g. inverting axes limits, etc.), then sets up the axes labels and title (if appropriate).

3. Stores some plot properties, such as colours and the vertical heights of stacked spectra, in an instance of `PlotProperties`. This object is also tied to the particular |Axes|, and can be accessed with ``ax.prop``.

4. Resets that |Axes|'s PHA to a brand-new instance of `PlotHoldingArea`.

If you want to do anything with the |Figure| and |Axes| objects, **the best time to do it is after calling** `mkplot`. It conveniently returns ``(fig, ax)`` for you to carry out any other methods you may want to (if you didn't already have a handle to the |Axes|).


Displaying the plot
-------------------

At this stage all the necessary matplotlib functions have been called, so all we need to do is to show the plot using |show|.
Penguins does not try to further complicate this step.
The only suggestion we make is to use `penguins.show()` instead: it saves you from having to import matplotlib. :)


Addendum
--------

Depending on your familiarity with matplotlib, you may find that the three-stage model is somewhat similar to, or rather dissimilar from, matplotlib itself.
It is rather different from typical matplotlib usage, where you don't need to "construct" a plot; you just call ``plt.plot()`` or something similar, then go straight to the "display" stage.
However, behind the scenes each call to ``plt.plot()`` actually creates an :class:`~matplotlib.artist.Artist` object which is not *actually* drawn on the figure until you call ``fig.draw()`` or something similar.
This is automatically done behind-the-scenes by matplotlib, so the "construction" stage is invisible to the ordinary user.

A different parallel can be drawn with the version control software git.
The most basic git workflow involves a three-step process of adding, committing, then pushing.
In a way, penguins' staging is similar to running ``git add``, and indeed the word "stage" was stolen from git terminology.
