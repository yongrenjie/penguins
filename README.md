[![Travis CI Build Status](https://travis-ci.com/yongrenjie/penguins.svg?branch=master)](https://travis-ci.com/github/yongrenjie/penguins)
![Code Coverage](https://raw.githubusercontent.com/yongrenjie/penguins/master/tests/coverage.svg)
[![PyPI version](https://badge.fury.io/py/penguins.svg)](https://badge.fury.io/py/penguins)

## Penguins: an Easy, NPE-free Gateway to Unpacking and Illustrating NMR Spectra

```
pip install penguins
```

`penguins` is a Python 3 package intended mainly for generating publication-quality plots of NMR spectra in a programmatic, reproducible fashion.

Documentation can be found at https://yongrenjie.github.io/penguins.
Everything is documented in one form or another, although I'm still working on tutorial-type pages which are easier to read than the docstrings.

Note that `penguins` is still in development, so the interface should not be assumed to be (too) stable. For now, I am working directly on the master branch, and rebasing/force pushing may occur without warning.

---------

As an example of the output, here's Figure 2 from [*Angew. Chem. Int. Ed.* **2017**, *56* (39), 11779–11783](https://doi.org/10.1002/anie.201705506):

<div align="center"><img src="https://raw.githubusercontent.com/yongrenjie/penguins/master/docs/images/angew_example.png" height="500"></div>

And here's a similar plot that can be done in under 20 lines of PEP8-compliant code using penguins (it's the same compound, but not the same data):

<div align="center"><img src="https://raw.githubusercontent.com/yongrenjie/penguins/master/docs/images/readme_example.png" height="550"></div>

```python
import penguins as pg
hmqc, hsqc, cosy, noesy = (pg.read("tests/data/noah", i, 1)
                           for i in range(1, 5))

fig, axs = pg.subplots(2, 2)
hmqc.stage(axs[0, 0], levels=7e3, f1_bounds="110..130", f2_bounds="7..9.5")
hsqc.stage(axs[0, 1], levels=4e4, f1_bounds="12..65", f2_bounds="0.5..5")
cosy.stage(axs[1, 0], levels=8e5)
noesy.stage(axs[1, 1], levels=9e4)

titles = [r"$^{15}$N HMQC", r"$^{13}$C HSQC", "COSY", "NOESY"]
for ax, title, char in zip(axs.flat, titles, "abcd"):
    pg.mkplot(ax, title=title, autolabel="nucl")
    ax.text(x=0.02, y=0.98, s=f"({char})", transform=ax.transAxes,
            fontweight="semibold", verticalalignment="top")
    pg.move_ylabel(ax, pos="topright")

pg.cleanup_axes()
pg.show()
```

----------

[Regarding the 'NPE' in the name: I made this package after getting fed up of getting `java.lang.NullPointerException` in TopSpin's Plot tab. It's especially bad on OS X Catalina, but even on Windows it's rather buggy. Also, penguins are cute. 🐧]
