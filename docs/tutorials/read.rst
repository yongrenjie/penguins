Tutorial 1: Reading data
========================

In TopSpin, each NMR experiment has its own "expno", and can be found in a folder numbered by its expno.
If you downloaded the penguins test data, you will find that it contains two such numbered datasets, which correspond to different "expnos" in TopSpin.

Of course, feel free to use your own data: your top-level data directory should also contain one or more numbered expno folders, and each of these datasets must have already been processed in TopSpin.

-------

Importing spectra: pg.read()
----------------------------

Spectra can be imported using the `read()` function.
`read()` takes three parameters: the path to the *spectrum folder*, the *expno*, and the *procno*.

We are going to assume here that you are using the penguins test data, but if you aren't, you just need to adapt these three arguments accordingly.
First, open a terminal and ``cd`` to the folder which contains the numbered expno datasets.
Then, launch the interactive Python REPL, and type::

    >>> import penguins as pg
    >>> data_1d = pg.read(".", 1, 1)
    >>> data_1d
    Dataset1D('/Users/yongrenjie/penguins/tests/nmrdata/1/pdata/1')

The ``"."`` argument refers to the current directory that you are in: in general, this can be any folder that contains Bruker NMR data.
The ``1`` and ``1`` that follow mean that we are importing the dataset with expno 1 and procno 1.
If you call the `read()` function as

::

    pg.read(PATH, EXPNO, PROCNO)

then penguins will expect to find processed NMR data at the folder ``<PATH>/<EXPNO>/pdata/<PROCNO>``.

Let's now try it with some 2D data::

    >>> data_2d = pg.read(".", 2, 1)
    >>> data_2d
    Dataset2D('/Users/yongrenjie/penguins/tests/nmrdata/2/pdata/1')

Notice how the returned object is a ``Dataset2D`` object now, as opposed to a ``Dataset1D`` object previously: penguins is capable of automatically detecting the dimensionality of the data you import.


----------------------


Accessing spectrum parameters
-----------------------------

Once imported using `read()`, spectrum parameters can be accessed using dictionary-like syntax.
All parameters are referred to using their standard Bruker names, just as you would type them in the TopSpin command line::

    >>> data_1d["p1"]
    9.22
    >>> data_1d["pulprog"]
    'zg60'

So, we can tell that this is a proton spectrum acquired using a 90° pulse width of 9.22 µs (the units that are used are always consistent with those in TopSpin).

For the 2D case, many parameters differ between the two spectral dimensions.
In these cases, they are returned either as a numpy `ndarray <numpy.ndarray>` or as a tuple.
The first always refers to the indirect-dimension ($f_1$) value, and the second to the direct-dimension ($f_2$) value::

    >>> data_2d["p1"]    # no difference between f1 and f2, so just a plain float
    10.83
    >>> data_2d["td"]
    array([ 256, 1024])
    >>> data_2d["nuc1"]
    ('1H', '1H')

Accessing the value of TD in $f_1$ is simply then ``data_2d["td"][0]``.
