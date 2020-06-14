Full reference
==============


.. currentmodule:: penguins

.. function:: read(path, expno, procno)

   Creates a :class:`~penguins.dataset.Dataset` object from the spectrum name folder, expno, and procno. Automatically detects the dimensionality of the spectrum being read in.

   :param path: Path to the spectrum name folder.
   :type path: str or pathlib.Path
   :param expno: Expno of desired spectrum.
   :type expno: int
   :param procno: Procno of desired spectrum.
   :type procno: int
   :return: Object belonging to the appropriate subclass of :class:`~penguins.dataset.Dataset` corresponding to the spectrum.
   :rtype: :class:`~penguins.dataset.Dataset1D`, :class:`~penguins.dataset.Dataset1DProj`, or :class:`~penguins.dataset.Dataset2D`

|

.. function:: read_abs(path)

   Same as :func:`read`, but takes only one parameter, which is the procno folder.
   
   :param path: Direct path to the procno folder.
   :type path: str or Path
   :return: Object belonging to the appropriate subclass of :class:`~penguins.dataset.Dataset` corresponding to the spectrum.
   :rtype: :class:`~penguins.dataset.Dataset1D`, :class:`~penguins.dataset.Dataset1DProj`, or :class:`~penguins.dataset.Dataset2D`

|

.. currentmodule:: penguins.dataset

.. class:: Dataset

   Parent class of :class:`Dataset1D`, :class:`Dataset1DProj`, and :class:`Dataset2D`. The general interface is described here, but the methods are not actually implemented on this superclass. Refer to the specific subclasses for implementation details.

   You can instantiate subclasses of :class:`Dataset` directly, but there should never be a *need* to. Use :func:`~penguins.read` or :func:`~penguins.read_abs` for that instead.

   .. attribute:: pars

      Case-insensitive dictionary (:class:`_parDict`) which holds values of acquisition and processing parameters. The ``__getitem__`` and ``__setitem__`` special methods of :class:`Dataset` are redirected to the corresponding special methods of the underlying :class:`_parDict`, which means that parameters can be accessed easily using ``dataset["param"]``.

   .. method:: ppm_scale()

      Generates a linearly spaced array of SI points from the highest chemical shift to the lowest. This can then be used as the frequency axis when plotting.

      :rtype: :class:`numpy:numpy.ndarray`

   .. method:: hz_scale()

      Generates a linearly spaced array of SI points from the highest frequency to the lowest (in Hz).

      :rtype: :class:`numpy:numpy.ndarray`

   .. method:: proc_data()

      Returns the pure real part of the processed data.

      :rtype: :class:`numpy:numpy.ndarray`

   .. method:: raw_data()

      Returns the raw data (`fid` or `ser`).

      :rtype: :class:`numpy:numpy.ndarray`

|

.. class:: Dataset1D

   Class representing a 1D spectrum.
   
|

.. class:: Dataset1DProj

   Class representing a 1D projection of a 2D spectrum.

|

.. class:: Dataset2D

   Class representing a 2D spectrum.

|

.. class:: _parDict

   Modified dictionary which holds acquisition and processing parameters. The keys are case-insensitive and are internally always converted to lowercase.

   Unlike in other packages (e.g. ``nmrglue``), parameters are not immediately loaded when the :class:`Dataset` class is instantiated. Only a few key parameters (e.g. ``TD``, ``SI``, ``O1``, ``SW``, ``SFO1``) are initially set. This is meant to avoid cluttering the dictionary with irrelevant parameters.

   Parameters are loaded (by parsing the ``procs``/``acqus``/``proc2s``/``acqu2s`` files) simply by looking them up (via ``__getitem__``). Therefore, the dictionary behaves *as if* it is fully populated, in the sense that calling ``pardict["par"]`` will always return the corresponding value (if it exists), but the value is only read *on demand*.

   The lookup function attempts to be clever and convert the parameters to floats if possible; otherwise, the parameters are stored as strings. There are currently several exceptions to this rule, such as ``TD`` and ``SI``, which are stored as ints. However, this list is not complete, and if there is a parameter that should be an int but isn't, it would be great if you could report it.

   For 2D spectra, string parameters are stored as a tuple of ``(f1_value, f2_value)``. Float parameters are stored as a :class:`numpy:numpy.ndarray` to facilitate elementwise manipulations (e.g. calculating ``O1P`` in both dimensions at one go).
