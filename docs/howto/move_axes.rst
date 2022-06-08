Changing the axes layouts
=========================

``matplotlib``, by default, creates plots with the *y*-axis on the left side of the plot.
In plots of 2D spectra, it is often desirable to move the *y*-axis to the right.

It is possible to do this with a series of ``matplotlib`` functions, but ``penguins`` provides a built-in function for this purpose, `ymove()`.
You can simply call this function on its own and it will move the axes labels to the very sensible top-right position.
Note that this function must be called *after* `mkplot()`.

.. plot:: howto/ymove_default.py

More specifically, `ymove()` allows you to choose between three available styles.
All of them move the *y*-axis to the right, but differ in where they place the *y*-axis label.

- *"topright"*: Moves the label to the top-right and places it in a horizontal orientation. (This is the default shown above)
- *"midright"*: Moves the label to the middle of the axis and places it sideways next to the tick labels.
- *"topspin"*: Rotates tick labels as well as the axis label. This mimics the normal display in TopSpin.

These are more easily illustrated with a diagram rather than with text.

.. plot:: howto/ymove.py

There is also an analogous function `xmove()`, but it is not very fully developed at this point in time, and in general should not really be needed unless you have a particular "house style" to follow.
