Datasets
--------

.. currentmodule:: penguins.dataset

.. note::
   Since you're here, you probably deserve to know how the source code is organised.
   Dataset classes are implemented using mixins.
   It's not pretty, but I think it's the best way, and I spent a long time thinking about this.
   I've tried to follow the principles outlined in Chapter 12 of Luciano Ramalho's *Fluent Python*, and I think I've succeeded.

   Anyway, what happens is that we have a `_Dataset` superclass which contains all the top-level behaviour that is shared between all datasets.
   As of the time of writing, this behaviour is only the dictionary-style parameter lookup.
   After that, we have a series of mixins for 1D raw data, 2D raw data, 1D processed data, 2D processed data, 1D plotting, and 2D plotting.
   (All of these mixins have very recognisable names in the source code.)
   The `Dataset1D` class simply inherits the methods that it needs to.
   In this case it would inherit from `_Dataset`, `_1D_RawDataMixin`, `_1D_ProcDataMixin`, and `_1D_PlotMixin`.
   Likewise, the `Dataset2D` class inherits all the 2D methods.

   This seems to be an utterly stupid use of mixins until you realise that `Dataset1DProj` inherits from `_2D_RawDataMixin`, `_1D_ProcDataMixin`, and `_1D_PlotMixin`.
   In other words, it has behaviour that sometimes mimics the 2D data, and sometimes mimics 1D data.
   So I chose to use multiple inheritance as a way of not repeating code.
   I've provided "aggregate classes" which have useful groups of methods, and the user should never actually have to deal with any of the issues associated with multiple inheritance.

.. autoclass:: _Dataset

|v|

.. autoclass:: _parDict
   :members: ..

|v|

1D mixins
*********

.. autoclass:: _1D_RawDataMixin
   :members: ..

   .. automethod:: raw_data
.. The :members: line stops autodoc from documenting any methods. We then documented the methods manually so that they would be documented fully instead of as a one-liner.

|v|

.. autoclass:: _1D_ProcDataMixin
   :members: ..

   .. automethod:: proc_data
   .. automethod:: integrate
   .. automethod:: bounds_to_slice
   .. automethod:: nuclei_to_str

|v|

.. autoclass:: _1D_PlotMixin
   :members: ..

   .. automethod:: stage

|v|

2D mixins
*********

.. autoclass:: _2D_RawDataMixin
   :members: ..

|v|

.. autoclass:: _2D_ProcDataMixin
   :members: ..
   
   .. automethod:: proc_data
   .. automethod:: integrate
   .. automethod:: bounds_to_slice
   .. automethod:: nuclei_to_str

|v|

.. autoclass:: _2D_PlotMixin
   :members: ..

   .. automethod:: stage
   .. automethod:: find_baselev

|v|

Actual Dataset classes
**********************

To be done.

.. autoclass:: Dataset1D
   :members: ..

|v|

.. autoclass:: Dataset1DProj
   :members: ..

|v|

.. autoclass:: Dataset1DProjVirtual
   :members: ..

|v|

.. autoclass:: Dataset2D
   :members: ..
