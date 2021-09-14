Tutorial 5: Customising 2D plots
================================

In this tutorial, we will go back to looking at *staging* options, but for 2D spectra instead of 1D spectra.

-----------------------

Contour levels and colours
--------------------------

Let's revisit the original 2D plot that we did back in :doc:`/tutorials/plot`.
As before, it's probably a good idea to move this into a script (or notebook).

.. plot::
   :context: reset

    import penguins as pg
    data_2d = pg.read(".", 2, 1)
    data_2d.stage()
    pg.mkplot()
    pg.show()

The most obvious problem is that the contour levels chosen are too *low*: that is, we are plotting quite a bit of noise along with the peaks.
penguins tries to choose a reasonable contour level (using the same algorithm as TopSpin), but you will find that for nearly every spectrum you plot you will have to specify the contour levels manually.

This can be done using the ``levels`` parameter.
Passing a float here will lead to that float being used as the *base level*, i.e. the lowest level for which contours are drawn.
The smaller this number, the more noise you will pick up.
Finding a good number usually requires some experimentation.

.. plot::
   :context: close-figs

    data_2d.stage(levels=3e5)
    pg.mkplot()
    pg.show()

Contours are always shown for both positive and negative levels.
If you want to disable negative contours, the best way is currently to set the *negative contour colour* to be the same as the background.
Contour colours are a 2-tuple of strings (positive colour and negative colour), passed as the ``colors`` parameter (note the American English spelling!).

.. plot::
   :context: close-figs

    # contour levels reduced so that the negative ones are more visible
    data_2d.stage(levels=1e5, colors=("yellowgreen", "purple"))
    pg.mkplot()
    pg.show()

-----------------------

Bounds
------

If you are already familiar with :doc:`specifying bounds for 1D spectra </tutorials/customise_1d>`, then 2D ones will be easy.
Instead of a single ``bounds`` parameter, you can now pass ``f1_bounds`` and/or ``f2_bounds``.
The syntax is exactly the same, so either a string or a tuple can be passed:

.. plot::
   :context: close-figs

    data_2d.stage(levels=3e5, f1_bounds="0.3..7", f2_bounds=(0.3, 7))
    pg.mkplot()
    pg.show()

