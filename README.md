[![Documentation Status](https://readthedocs.org/projects/penguins/badge/?version=latest)](https://penguins.readthedocs.io/en/latest)
[![Travis CI Build Status](https://travis-ci.com/yongrenjie/penguins.svg?branch=master)](https://travis-ci.com/github/yongrenjie/penguins)
![Code Coverage](tests/coverage.svg)

## Penguins: an Easy, NPE-free Gateway to Unpacking and Illustrating NMR Spectra

```
pip install penguins
```

`penguins` is a Python 3 package intended mainly for generating publication-quality plots of NMR spectra in a programmatic, reproducible fashion.
Here's a small example of the output (a similar plot to Figure 2 in [*Angew. Chem. Int. Ed.* **2017**, *56* (39), 11779–11783](https://doi.org/10.1002/anie.201705506)):

<div align="center"><img src="docs/images/cookbook_subplots.svg" height="550"></div>

Documentation can be found at https://penguins.readthedocs.io, although it's not complete yet and arguably not very well-organised (I'm working on it!).

Note that `penguins` is still in development, so the interface should not be assumed to be stable.

[Regarding the 'NPE' in the name: I made this package after getting fed up of getting `java.lang.NullPointerException` in TopSpin's Plot tab. It's especially bad on OS X Catalina, but even on Windows it's rather buggy. Also, penguins are cute. 🐧]
