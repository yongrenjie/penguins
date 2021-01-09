Tutorial 7: Using matplotlib
============================

penguins is a *wrapper* around matplotlib, which is a library that provides plotting capabilities in Python.
What this means is that all the code in penguins ultimately calls a series of matplotlib functions.
The penguins functions are provided as "higher-level" constructs, which allow you to directly work with and plot NMR data, without worrying about "low-level" implementation details.

However, you will notice that on its own, penguins' plotting capabilities are slightly limited.
For example, it doesn't provide a function which lets you write arbitrary text on a graph.

The reason for this is because penguins (generally) does not try to reinvent the wheel.
If matplotlib already has a way to do something, then penguins will not re-implement it; instead, you should simply call the matplotlib function itself.
For writing arbitrary text, for example, one should call the :meth:`ax.text()<matplotlib.axes.Axes.text>` method on the |Axes| of interest.
penguins, in turn, gives you access to the |Axes| object via :func:`~penguins.subplots()` or `mkplot()`.


In general, one can interleave matplotlib and penguins functions without many issues.
In this regard, penguins behaves very similarly to other plotting packages such as `seaborn<http://seaborn.pydata.org/>`_.

.. note::

   If you are not familiar with matplotlib, it is a good idea to check out some of the `introductory matplotlib tutorials <https://matplotlib.org/tutorials/index.html>`_.
   matplotlib's documentation itself -- although very extensive -- can appear slightly intimidating to a beginner.
   There are also a great number of other tutorials online which may be more approachable.
