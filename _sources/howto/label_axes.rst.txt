Labelling axes
==============

``penguins`` provides a `label_axes()` function, which expects a list (or other iterable) of Axes objects.

For a typical "publication-style" labelling, the syntax can appear a bit unusual at first::

    fig, axs = ...
    
    ...  # do your plotting here

    pg.label_axes(axs, fstr="({})", fontweight="semibold")

The ``fstr`` parameter is a string which is then *formatted* with the label you are using, meaning that the curly braces ``{}`` are replaced with the label text. 
By default, ``penguins`` uses alphabetical labels, i.e. abcdef... although numeric labels and Roman numerals are also available.
So, the curly braces *inside* the parentheses are replaced with this label and so what ``penguins`` gives you is labels which read ``(a)``, ``(b)``, and so on.

Other parameters such as ``fontweight`` are just passed directly to `ax.text()<matplotlib.axes.Axes.text>`.

.. plot:: howto/label_example1.py

If you don't pass the ``fstr`` parameter, then by default only the label itself is printed (i.e. the letters) without anything around it.
This style also sees some use in publications.

.. plot:: howto/label_example2.py

The documentation for `label_axes()` provides a few more options which you can use to customise the labelling (e.g. if you want to start from a different letter/number).
