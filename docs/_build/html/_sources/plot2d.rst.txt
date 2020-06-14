.. |br| raw:: html

    <br />

2D Plotting in Detail
=====================

Many concepts of the interface here are exactly analogous to that for 1D spectra, so will not be detailed here again.


Step 1: Staging spectra
-----------------------

The :meth:`Dataset2D.stage <penguins.dataset.Dataset2d.stage>` method also delegates to :func:`penguins.pgplot.stage2d()`::

   # These are entirely equivalent
   ds2.stage(*args, **kwargs)    # ds2: a 2D dataset
   pg.pgplot.stage2d(ds, *args, **kwargs)

..
   * ** * ** This comment stops vim from highlighting everything as italicised.

Multiple spectra can be plotted by staging each of them individually.

.. currentmodule:: penguins.pgplot

.. function:: stage2d(dataset, f1_bounds=None, f2_bounds=None, levels=None, colors=None, label=None, plot_options=None)

   Uses the information provided to create a :class:`penguins.pgplot.PlotObject2D` object, which is then appended to the plot holding area.

   :param dataset: A 2D dataset object.
   :param f1_bounds: *(optional)* A tuple of floats ``(upper, lower)`` specifying the section of the indirect axis to plot. Both should be chemical shifts. If either ``upper`` or ``lower`` are none, then the upper (lower) bound is the maximum (or minimum) chemical shift. If not provided, defaults to the entire spectrum.

      |br|
      It is a little unfortunate that an *upper* bound for *f*:subscript:`1` is *lower* in terms of the spectrum layout, but that is an unfortunate quirk of NMR spectroscopy. Swapping the order would make it inconsistent with the *f*:subscript:`2` notation, which is not a particularly appealing option, but if there are strong opinions then please get in touch.

   :param f2_bounds: *(optional)* A tuple of floats ``(upper, lower)`` specifying the section of the direct axis to plot. Both should be chemical shifts. If either ``upper`` or ``lower`` are none, then the upper (lower) bound is the maximum (or minimum) chemical shift. If not provided, defaults to the entire spectrum.

      |br|
      Note that, as before, the ``bounds`` parameters do not merely affect the *plot limits*. They restrict the portion of the spectrum which is actually plotted (and ``matplotlib`` chooses sensible plot limits to reflect that).
   :param levels: *(optional)* A tuple ``(baselev, increment, nlev)`` specifying the levels at which to draw contours. ``baselev`` and ``increment`` are floats; ``nlev`` is an int. This is directly analogous  to the contour levels in TopSpin's ``edlev`` dialog box. Contours are drawn at ``±(baselev * (increment ** N))`` where ``N`` ranges from ``0`` to ``nlev - 1``.

      |br|
      The default ``baselev`` is chosen individually for each spectrum, according to TopSpin's algorithm of ``(35 * stdev(data))`` (see *"Processing Parameters and Commands"* documentation). ``increment`` by default is 1.5 and ``nlev`` is 10. Any of the three parameters can be passed as ``None`` in order to use the defaults.

      |br|
      ``baselev`` is, by far, the most likely parameter to be customised. Therefore, instead of passing a tuple of ``(baselev, None, None)``, you can also pass a single float ``baselev``.

   :param colors: *(optional)* A tuple of valid ``matplotlib`` colors ``(positive, negative)``. The colours will be used for positive and negative contours respectively. See :std:doc:`matplotlib:tutorials/colors/colors` for more information. The default colours are drawn from Seaborn's "deep" palette (see Seaborn's :std:doc:`seaborn:tutorial/color_palettes`).

      |br|
      Note that as of now there is no way to disable contours of a particular sign. The easiest way around that is to make the unwanted contour have the same colour as the figure background.

   :param str label: *(optional)* A string to display in the legend of the plot. See below for an example of how this looks.

   :param plot_options: *(optional)* Key-value options which are passed on directly to :func:`plt.contour() <matplotlib.pyplot.contour>`. Note that the ``colors`` parameter will override the corresponding key in ``plot_options``, if present.
   :type plot_options: dict

   :returns: None.


As a slightly contrived example, here we stage the same HMBC dataset four times with different values of ``baselev``::

   d = pg.read("data/pt2", 5, 1)   # HMBC
   # Split spectrum into four portions
   bottom_f1, top_f1 = (None, 100), (100, None)
   left_f2, right_f2 = (None, 4.5), (4.5, None)
   # To make this less boring you could use a double listcomp or
   # itertools.product(), but here we'll do it the repetitive way
   # Recall levels=1e2 is the same as levels=(1e2, None, None).
   d.stage(f1_bounds=bottom_f1, f2_bounds=left_f2,  levels=1e2)
   d.stage(f1_bounds=top_f1,    f2_bounds=left_f2,  levels=1e3)
   d.stage(f1_bounds=bottom_f1, f2_bounds=right_f2, levels=1e4)
   d.stage(f1_bounds=top_f1,    f2_bounds=right_f2, levels=1e5)
   # Construct and display
   pg.plot(); pg.show()

.. image:: images/plot2d_baselev.png
   :align: center

Notice that ``baselev`` in the bottom-left and top-left sectors are certainly too low; we are plotting mostly noise. The top-right sector with ``baselev=1e5`` doesn't pick up any noise, but the peaks are getting a little bit difficult to see. The best value of ``baselev`` is probably close to ``1e4`` as seen in the bottom-right. It turns out that TopSpin's algorithm suggests a value of approximately ``1.9e4``.

The next section contains more information about how to choose a base level.


Choosing base levels
--------------------

Choosing a base level for a 2D plot can be tricky. Even though the TopSpin algorithm is usually a decent initial guess, there is often a slightly better value. Instead of using trial-and-error to find this, there are a couple of ways to do this efficiently.

The first is to open the spectrum in TopSpin. Scroll up or down to decrease or increase the base level to a comfortable point, then type in ``edlev`` at the TopSpin command line and note down the base level. This number can be directly passed to ``stage2d()``. You do need to be careful about which number you choose. Sometimes these are the same, but sometimes they aren't:

.. image:: images/topspin_edlev.png
   :align: center
   :width: 70%
   :class: no-scaled-link


Alternatively, penguins provides a :meth:`~penguins.dataset.Dataset2D.find_baselev()` method on :class:`~penguins.dataset.Dataset2D` classes which opens an interactive plot window where you can adjust ``baselev`` using a slider::

   hmbc = pg.read("data/pt2", 5, 1)
   hmbc.find_baselev()     # opens the following window

.. image:: images/find_baselev.png
   :align: center

The slider is logarithmic and the value displayed on the right is the base-10 logarithm of the real ``baselev``. The initial ``baselev`` is given by TopSpin's algorithm (``10 ** 4.27 ≈ 1.9e4``, as before). After you find a comfortable value, click "OK"; penguins will print the final value of ``baselev`` to standard output. Note that this method does *not* work in Jupyter notebooks.

Under the hood, the :meth:`find_baselev()` method calls :func:`~penguins.pgplot._make_contour_slider` with the dataset object as the first parameter.

.. function:: _make_contour_slider(dataset, increment=None, nlev=4)

   Creates an interactive plot window with a slider controlling ``baselev``.

   :param dataset: 2D dataset object.
   :param float increment: *(optional)* Desired increment. Defaults to ``(1.5 ** 10) ** (1 / nlev)``. This value is chosen so that the ``nlev`` contours can cover a similar dynamic range as with ``increment=1.5`` and ``nlev=10``.
   :param int nlev: *(optional)* Desired number of contour levels. Note that every time the slider value is changed, ``matplotlib`` has to redraw all the contours from scratch. Therefore, it is advisable to keep this number fairly small (the default of 4 seems to work reasonably well). Anecdotally, the quality of the plot does not suffer very much.



Step 2: Constructing the plot
-----------------------------

Plot construction is done using :func:`~penguins.plot()`. If the holding area consists of 2D spectra, then it delegates to :func:`~penguins.pgplot._plot2d()`.

.. currentmodule:: penguins

.. function:: plot(figsize=None, figstyle="default", offset=(0, 0), title=None, xlabel=r"$f_2$ (ppm)", ylabel=r"$f_1$ (ppm)", legend_loc="best", close=True, empty_pha=True)
   :noindex:

   Calls :func:`plt.plot() <matplotlib.pyplot.plot>` on each spectrum in the holding area. Also calls several ``matplotlib`` functions in order to make the plot more aesthetically pleasing. Finally, empties the plot holding area if ``empty_pha`` is set to True.
   
   All keyword arguments below are optional:

   :param figsize: Tuple of floats specifying ``(width, height)`` of plot in inches.

   :param str figstyle: Specifies the overall plot style.

      * ``"default"`` enables minor ticks, makes the axes slightly thicker than usual, and calls :func:`plt.tight_layout() <matplotlib.pyplot.tight_layout>`.
      * ``"mpl_natural"`` does not change any settings from the original.

      There are no other styles right now, but this list may be expanded in future.

   :param offset: Tuple of floats ``(f1_offset, f2_offset)`` specifying offset between adjacent spectra. If ``f1_offset`` (or ``f2_offset``) is positive, then later spectra are shifted upwards (or rightwards). Also, the default of this argument is a tuple ``(0, 0)``, but Sphinx doesn't display the parentheses in the function signature correctly.

   :param str title: Plot title.

   :param str xlabel: Label for *x*-axis.

   :param str ylabel: Label for *y*-axis.

   :param legend_loc: Legend location. Passed directly as the ``loc`` parameter to :func:`plt.legend <matplotlib.pyplot.legend>`, where a full description can be found. Defaults to ``"best"``, which means that ``matplotlib`` will attempt to choose a position that doesn't collide with the existing plots.

   :param bool close: Close all previously used figures before constructing a new plot. This shouldn't be changed by the end user.

   :param bool empty_pha: Empty the holding area after constructing a plot. This also causes the ``seaborn`` colour generator to restart. Setting this to False can be useful if one wants to construct several figures by adding one spectrum at at time, since you don't have to stage the first dataset every time after calling :func:`~penguins.show` or :func:`~penguins.savefig`.

   :returns: Tuple of (:class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`) objects corresponding to the plot.

As before, :func:`~penguins.plot()` returns ``(fig, ax)`` so that various ``matplotlib`` methods can be used.

Here's an example where we plot the same HSQC spectrum four times but give the plot a nonzero offset to make it seem as if the peaks are shifting::

   d = pg.read("data/rot1", 3, 1)   # HSQC
   # Make up some data 
   temps = [240, 250, 260, 270]  # in K
   blues = [f"#00{cc}ff" for cc in ["00", "55", "a6", "ea"]]
   reds = [f"#ff{cc}00" for cc in ["00", "55", "a6", "ea"]]
   # Stage each of them with different colours and label
   for temp, blue, red in zip(temps, blues, reds):
      d.stage(colors=(blue, red),
              f1_bounds=(80, 11),
              f2_bounds=(4.2, 0.6),
              levels=2.8e5,
              label=f"{temp} K"
              )
   # Separate each plot a little bit
   pg.plot(offset=(0.2, 0.02), legend_loc="upper left")
   pg.show()

.. image:: images/plot2d_offset.png

Of course, if your peaks are *truly* shifting, then you should load each separately (e.g. by adding an ``expno`` list as another argument to ``zip()``) and you will not need to put in a fake ``offset``. And you can also get the temperature via the ``TE`` parameter (i.e. ``d.stage(..., label=str(d["te"]))``).

PS: If anybody wants to contribute a series of real HSQC spectra for this plot please get in touch.


Step 3: Displaying the plot
---------------------------

This step is the same as in the 1D case. For more information, please see the :ref:`corresponding section <plot1d_display>` in the 1D chapter.
