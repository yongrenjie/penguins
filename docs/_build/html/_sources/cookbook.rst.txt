Plotting Cookbook
=================


Adding text to a spectrum
-------------------------

Adding text is best done using :meth:`ax.text() <matplotlib.axes.Axes.text>`. For example::

   ds = pg.read("data/pt2", 1, 1)
   ds.stage(bounds="7..8.5")
   _, ax = pg.mkplot()
   ax.text(x=7.27, y=0.8, s=r"$\mathrm{CHCl_3}$",
           color="red",
           transform=ax.get_xaxis_transform())
   pg.show()

.. image:: images/cookbook_text.png
   :align: center

The ``transform`` parameter above ensures that the *x*-coordinate is specified in terms of *data* coordinates (i.e. the actual chemical shift on the *x*-axis), whereas the *y*-coordinate is specified in terms of *axis* coordinates (i.e. bottom is 0 and top is 1). For 1D plots, the *y*-axis is the spectral intensity and spans several orders of magnitude, so without the transformation it is difficult to guess how high to place the text.

It turns out that for :meth:`ax.text() <matplotlib.axes.Axes.text>`, the default coordinate system is *data* coordinates along both axes (note that this is somewhat unusual for ``matplotlib``; it's far more normal for axis coordinates to be the default). So for 2D plots, you can simply leave out the ``transform`` parameter altogether and use ``ax.text(x=f2_ppm, y=f1_ppm, s="text")``.


Insets
------

To be done.



Subplots
--------

To be done.


A complete example
------------------

As a more complicated example, let's try to plot five 1D NOE spectra with different mixing times. This is what happens when we naively stage all of them::

   noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
   for noe in noes:
       mixing_time = int(noe["d8"] * 1000)  # d8 itself is in seconds
       noe.stage(label=f"{mixing_time} ms",
                 bounds="0..6")
   pg.mkplot(voffset=0.01, hoffset=0.05)
   pg.show()

.. image:: images/cookbook_noesy1.png
   :align: center

There are a couple of ways of stopping the intense on-resonance peak from dominating the spectrum. One way is to use the ``dfilter`` parameter of :meth:`~penguins.dataset.Dataset1D.stage()`. ``dfilter`` must be a function that takes the spectrum intensity at each point (a float) and returns ``True`` or ``False`` depending on whether we want the point or not. Here we use a ``lambda`` but you can define a proper function if you want. Also, if you prefer having the on-resonance peak negative, you can either reprocess in TopSpin or use ``scale=-1``, as below::

   noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
   # Calculate the height of the intense peak
   maxheight = np.amax(noes[0].proc_data())
   for noe in noes:
       mixing_time = int(noe["d8"] * 1000)
       noe.stage(label=f"{mixing_time} ms",
                 bounds="0..6",
                 dfilter=(lambda i: i < 0.02 * maxheight),
                 scale=-1)
   # Note that the max heights of the staged spectra have changed,
   # so voffset needs to be adjusted as well.
   pg.mkplot(voffset=0.4, hoffset=0.05)
   pg.show()
                
.. image:: images/cookbook_noesy2.png
   :align: center

As an alternative to that, you could just manually set the plot limits. When you display a graph, you can hover over the graph and ``matplotlib`` will tell you the coordinates of your current cursor position. Jot some good values down and pass them to :meth:`ax.set_xlim <matplotlib.axes.Axes.set_xlim>` and :meth:`ax.set_ylim <matplotlib.axes.Axes.set_ylim>`::

    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    for noe in noes:
        mixing_time = int(noe["d8"] * 1000)
        noe.stage(label=f"{mixing_time} ms",
                  bounds="0..6",
                  scale=-1)
    _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
    ax.set_xlim(6.2, -0.3)   # must be (larger, smaller) so that it's inverted
    ax.set_ylim(-2.1e4, 1.4e5)
    pg.show()

.. image:: images/cookbook_noesy3.png
   :align: center

Let's assume that we like this second option. The following discussion will apply to both, anyway, so you can tailor it to your liking.

If we wanted to display the mixing time next to each individual spectrum and not in the legend, then we just need to call :meth:`ax.text() <matplotlib.axes.Axes.text>` on an appropriate coordinate.
The *x*-coordinate is easy to choose, but the *y*-coordinate is not so easy, until we find :func:`~penguins.get_properties()`.
This returns a :class:`~penguins.pgplot.PlotProperties` class, which has an attribute ``voffsets`` listing the vertical offset of each spectrum in *data* coordinates::

   noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
   for noe in noes:
       noe.stage(bounds="0..6", scale=-1)
   _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
   ax.set_xlim(6.2, -0.3)
   ax.set_ylim(-2.1e4, 1.4e5)
   # Get the vertical offset of each spectrum, in data coordinates
   voffsets = pg.pgplot.get_properties().voffsets
   # Now add each bit of text at the appropriate voffset
   for voffset, noe in zip(voffsets, noes):
       mixing_time_label = f"{int(noe['d8'] * 1000)} ms"
       ax.text(x=0.6, y=voffset,
               s=mixing_time_label)
   pg.show()

.. image:: images/cookbook_noesy4.png
   :align: center

Not bad, but the text needs to be lifted a little.
Now, :class:`~penguins.pgplot.PlotProperties` doesn't try to be overly clever with the values it stores, since it doesn't know what you want to use them for; it trusts that you will use them wisely.
In this case, all we need to do is to add some extra height (this bit pretty much *has* to be trial-and-error, since we don't want to hard-code a value).

We could also horizontally displace the text a little bit, just like the spectra, by subtracting ``(n * 0.05)`` from each successive *x*-coordinate. This would match the ``hoffset=0.5`` parameter passed to :func:`~penguins.mkplot()`. And finally, we can reuse the colours of the original plot via ``PlotProperties.colors``::

   noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
   for noe in noes:
       noe.stage(bounds="0..6", scale=-1)
   _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
   ax.set_xlim(6.2, -0.3)   # must be (larger, smaller)
   ax.set_ylim(-2.1e4, 1.4e5)
   # Get the properties of each spectrum
   heights = pg.pgplot.get_properties().heights
   colors = pg.pgplot.get_properties().colors
   for n, (color, height, noe) in enumerate(zip(colors, heights, noes)):
       mixing_time_label = f"{int(noe['d8'] * 1000)} ms"
       ax.text(x=(0.6 - n * 0.05), y=height+2e3,
               s=mixing_time_label,
               color=color)
   pg.show()

.. image:: images/cookbook_noesy5.png
   :align: center

