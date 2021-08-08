Reading NUS data
================

Non-uniform sampling is an acquisition method where an incomplete set of *t*:subscript:`1` increments are acquired. The reconstruction of the full spectrum can be accomplished by several methods, and if signal-to-noise is not limiting, can be used to greatly speed up acquisition of multidimensional spectra.

From penguins' point of view, once they have been processed, there is almost no difference between NUS spectra and uniformly sampled spectra. However, you may need to watch out for the ``TD`` parameter::

   >>> # you can get this dataset from
   >>> # https://github.com/yongrenjie/penguins-testdata
   >>> nus_hsqc = pg.read("penguins-testdata", 9)
   >>> nus_hsqc["td"]
   array([ 128, 2048])
   >>> nus_hsqc["nustd"]
   array([512,   0])
   >>> nus_hsqc["nusamount"]
   25.0

What this means is that this spectrum has been acquired with 25% sampling: only 128 points were acquired in the indirect dimension, but the reconstructed spectrum has 512 points. Therefore, if you plan to use ``TD`` for any reason whatsoever, it may be worth inserting a line such as::

   >>> nus_hsqc["td"][0] = max(nus_hsqc["td"][0], nus_hsqc["nustd"][0])
   >>> nus_hsqc["td"]
   array([ 512, 2048])

Thankfully, this has zero impact on penguins' plotting routines since these depend on ``SI`` and not ``TD``.
