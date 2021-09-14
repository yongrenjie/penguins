Adding arbitrary text
---------------------

This uses the `ax.text()<matplotlib.axes.Axes.text>` function.
Note that writing ``x=0.5, y=0.5`` refers to so-called *data* coordinates, i.e. 0.5 ppm in both dimensions.
This is most straightforward for labelling 2D data.
If you want *axes* coordinates, i.e. using ``x=0.5, y=0.5`` to refer to the middle of the spectrum, then you need to pass an extra argument ``transform=ax.transAxes`` to `text()`.

.. plot:: add_text.py


