Adding arbitrary text
---------------------

This uses the `Axes.text()<matplotlib.axes.Axes.text>` function.
Note that in the script below, writing ``x=0.5, y=0.5`` refers to so-called *data* coordinates, i.e. 0.5 ppm in both dimensions.
This is most straightforward for labelling 2D data.

If you want *axes* coordinates, i.e. using ``x=0.5, y=0.5`` to refer to the middle of the spectrum, then you need to pass an extra argument ``transform=ax.transAxes`` to `text()<matplotlib.axes.Axes.text>`, where `ax` is the Axes instance you are plotting on.

Coordinate transforms are covered very thoroughly in the :std:doc:`Matplotlib transforms tutorial<tutorials/advanced/transforms_tutorial>`.

Other keyword arguments (to control e.g. colour, font weight, or alignment) can be found in the Matplotlib documentation for `text()<matplotlib.axes.Axes.text>`.

.. plot:: howto/add_text.py

