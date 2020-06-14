1D Plotting in Detail
=====================

As described in :doc:`quickstart`, routine plotting of spectra can be done fairly easily. This page aims to describe the various options available when plotting spectra.

The options here are applicable to all 1D datasets (:class:`~penguins.dataset.Dataset1D`, :class:`~penguins.dataset.Dataset1DProj`, and :class:`~penguins.dataset.Dataset1DProjVirtual`).

Step 1: Staging spectra
-----------------------

Firstly, it should be noted that the :meth:`stage()` methods on 1D dataset classes simply call :func:`penguins.pgplot.stage1d()` on the Dataset object itself, with all the arguments passed along. Therefore, if ``ds`` is a 1D dataset object, the following invocations are entirely equivalent (although the former is easier to type)::

   ds.stage(*args, **kwargs)
   pg.pgplot.stage1d(ds, *args, **kwargs)

Multiple spectra can be plotted by staging each of them individually.

.. currentmodule:: penguins.pgplot

.. function:: stage1d(dataset, scale=None, bounds=None, label=None, color=None, plot_options=None)

   Penguins maintains a *"holding area"*, which is a global list of plots which have been registered but not yet plotted. The :func:`~penguins.pgplot.stage1d()` function uses the information provided to create a :class:`penguins.pgplot.PlotObject1D` object, which is then appended to that list.

   :param dataset: A 1D dataset object.
   :param scale: *(optional)* Indicates factor to scale spectrum intensity by.
   :type scale: float
   :param bounds: *(optional)* A tuple of floats ``(upper, lower)`` specifying the section of the spectrum to plot. Both should be chemical shifts. If either ``upper`` or ``lower`` are none, then the upper (lower) bound is the leftmost (or rightmost) edge of the spectrum. If not provided, defaults to the entire spectrum. Note that this does not merely affect the *plot limits*. It restricts the portion of the spectrum which is actually plotted (and ``matplotlib`` chooses sensible plot limits to reflect that).
   :param label: *(optional)* Text to be displayed in the plot legend. Some LaTeX-like syntax is possible using raw strings: see :std:doc:`tutorials/text/mathtext`.
   :type label: str
   :param color: *(optional)* A valid ``matplotlib`` color. See :std:doc:`matplotlib:tutorials/colors/colors` for more information. The default colour palette used is Seaborn's "deep" (see Seaborn's :std:doc:`seaborn:tutorial/color_palettes`).
   :type color: str
   :param plot_options: *(optional)* Key-value options which are passed on directly to :func:`plt.plot() <matplotlib.pyplot.plot>`. Note that the ``color`` and ``label`` parameters will override the corresponding keys in ``plot_options``, if present.
   :type plot_options: dict

   :returns: None.

::

   ds1 = pg.read("data/pt2", 1, 1)
   
   # This label demonstrates some of the LaTeX capabilities.
   # The colour for this one defaults to the first item in Seaborn/deep.
   ds1.stage(bounds=(8.5, 7.5),
             label=r"$\mathrm{C_{20}H_{28}N_2O_4S}$",
             plot_options={"linestyle": '--'})

   # You can stage the same dataset multiple times with different options.
   ds1.stage(scale=0.2,           # 1/5 the height of first spectrum
             bounds=(8.5, 8),
             label="Yes, that is the actual formula",
             color="hotpink")

   pg.plot(); pg.show()

.. image:: images/plot1d_stage.png
   :align: center


Step 2: Constructing the plot
-----------------------------

Plot construction is done using :func:`~penguins.plot()`.

.. currentmodule:: penguins

.. function:: plot(figsize=None, figstyle="default", stacked=False, voffset=0, hoffset=0, title=None, xlabel="Chemical shift (ppm)", ylabel="Intensity(au)", close=True, empty_pha=True)

   Calls :func:`plt.plot() <matplotlib.pyplot.plot>` on each spectrum in the holding area. Also calls several ``matplotlib`` functions in order to make the plot more aesthetically pleasing. Finally, empties the plot holding area if ``empty_pha`` is set to True.
   
   All keyword arguments below are optional:

   :param tuple(float,float) figsize: (width, height) of plot in inches.

   :param str figstyle: Specifies the overall plot style.

      * ``"default"`` removes all spines except the bottom one, enables minor ticks, and makes the axis slightly thicker.
      * ``"mpl_natural"`` does not change any settings from the original.

      There are no other styles right now, but this list may be expanded in future. In any case, this can later be customised further (see below).

   :param bool stacked: True to make spectra tightly stacked (i.e. not superimposed). Overrides any value given in the ``voffset`` parameter.

   :param float voffset: Fraction of maximum height to vertically offset spectra by. The height of a spectrum refers to the total width it spans in the *y*-axis, and the maximum height refers to the largest such height of all spectra in the holding area. This is useful for offsetting spectra by a *constant amount*. Note that stacked spectra have a *variable* vertical offset between each spectrum, because each spectrum will have a different height. An example of the difference is shown below.

   :param float hoffset: Amount of horizontal offset between adjacent spectra in ppm. If this is positive, then later spectra are shifted to the right.

   :param str title: Plot title.

   :param str xlabel: Label for *x*-axis.

   :param str ylabel: Label for *y*-axis. This would never be used unless you use ``figstyle=mpl_natural``, or manually reenable the *y*-axis display.

   :param bool close: Close all previously used figures before constructing a new plot. This shouldn't be changed by the end user.

   :param bool empty_pha: Empty the holding area after constructing a plot. This also causes the ``seaborn`` colour generator to restart. Setting this to False can be useful if one wants to construct several figures by adding one spectrum at at time, since you don't have to stage the first dataset every time after calling :func:`~penguins.show` or :func:`~penguins.savefig`.

   :returns: Tuple of (:py:class:`~matplotlib.figure.Figure`, :py:class:`~matplotlib.axes.Axes`) objects corresponding to the plot.

:func:`~penguins.plot()` conveniently returns ``(fig, ax)``, so that you do not need to call :func:`plt.gcf() <matplotlib.pyplot.gcf>` or :func:`plt.gca() <matplotlib.pyplot.gca>`. Therefore you can carry out any other operations you wish to after this. For example, to add extra text to the plot::

   _, ax = pg.plot()    # _ denotes a throwaway variable
   ax.text(0, 0.5, "Aromatic region", transform=ax.transAxes)

(see :meth:`matplotlib.axes.Axes.text` for an explanation of the parameters). For even more customisation you can of course import ``matplotlib`` itself and utilise the full library of functions there. 

Here is an example of a stacked plot versus one with ``voffset=1.1``. The extra 0.1 is there to ensure that there is some padding between adjacent spectra (when using ``stacked``, penguins also makes sure to add ``0.1 * maxheight`` padding)::

   ds2 = pg.read("data/pt2", 2, 1)          # 13C spectrum
   ds2.stage(bounds=None, color="black")    # Full spectrum
   ds2.stage(bounds=(150, 100))             # Three subspectra
   ds2.stage(bounds=(100, 50))
   ds2.stage(bounds=(50, 0))
   pg.plot(stacked=True, title="stacked")   # Either this...
   pg.plot(voffset=1.1, title="voffset")    # ...or this, but not both!
   pg.show()

.. image:: images/plot1d_stacked.png
   :width: 49%
   :class: no-scaled-link

.. image:: images/plot1d_voffset.png
   :width: 49%
   :class: no-scaled-link

In the ``stacked`` plot, the second spectrum (from the bottom; in blue) occupies a relatively small vertical space, because its intensity is relatively low (does not contain the large CDCl\ :subscript:`3` peak at 77 ppm).
However, in the ``voffset`` plot, it occupies the same vertical space as the other spectra, because the vertical offset between spectra is a constant.
The question of which is *better* is left as a personal decision for you to make. Indeed, you may find that different situations call for different settings.


.. _plot1d_display:

Step 3: Displaying the plot
---------------------------

This step is exactly analogous to using :func:`plt.show() <matplotlib.pyplot.show>`. You can also save an image using :func:`plt.savefig() <matplotlib.pyplot.savefig>`. For convenience, penguins provides functions of the same name which are simply wrappers around the corresponding ``matplotlib`` functions::

   # These are entirely equivalent to plt.show() and plt.savefig()
   # Display the plot:
   pg.show()
   # or save an image:
   pg.savefig("/Users/yongrenjie/Desktop/nice_plot.png", dpi=500)

.. function:: show(*args, **kwargs)

   Calls :func:`plt.show(*args, **kwargs) <matplotlib.pyplot.show>`.
..
   * ** * ** This comment stops vim from highlighting everything as italicised.

.. function:: savefig(*args, **kwargs)

   Calls :func:`plt.savefig(*args, **kwargs) <matplotlib.pyplot.savefig>`.
..
   * ** * ** This comment stops vim from highlighting everything as italicised.

