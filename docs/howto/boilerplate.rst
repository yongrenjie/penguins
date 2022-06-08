Reducing boilerplate code
=========================

A frequent task is to plot multiple spectra on multiple axes in a subplot.

Of course, the most direct way is simply to use ``for`` loops as well as `enumerate()` and `zip()`.
This is already far better than individually plotting on each Axes by.
However, there are also many ways in which ``penguins`` tries to reduce the amount of boilerplate code required for this:

 - You can `read()` in multiple datasets at a go by passing a list or other iterable of expnos::

    # Reads in one dataset
    pg.read(path, 1)
    
    # "Slow" way of reading in a list of four datasets
    [pg.read(path, expno) for expno in range(1, 5)]

    # "Fast" way, equivalent to the above
    pg.read(path, range(1, 5))

 - Staging still has to be done individually, unfortunately.

 - The `mkplots()` function essentially performs `mkplot()` on a series of axes::

    # Construct one plot with a given title
    pg.mkplot(ax, title=title)

    # "Slow" way of constructing a series of plots
    for ax, title in zip(axs, titles):
        pg.mkplot(ax, title=title)

    # "Fast" way, equivalent to the above
    pg.mkplots(axs, titles=titles)


