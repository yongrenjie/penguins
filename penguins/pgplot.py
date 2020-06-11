from __future__ import annotations   # PEP 563

from itertools import zip_longest
from collections import abc
from typing import Union, Sequence, MutableMapping, Optional, Tuple, Any

import matplotlib.pyplot as plt  # type: ignore

from . import dataset as ds
from .type_aliases import *


# -- 1D PLOTTING ----------------------------------------------

default_1d_plotoptions = {"linewidth": 1}


def plot1d(spec: ds.TDataset1D,
           plot_options: Optional[MutableMapping] = None,
           label: Optional[str] = None,
           bounds: Optional[TBounds1D] = None,
           color: Optional[str] = None,
           ) -> Tuple[Any, Any]:
    """
    Plots one 1D spectrum. Gets plot1ds() to do the hard work.
    This is just a helper function to avoid having to write a singleton list
    every time you want to plot one spectrum only.
    """
    po = [plot_options] if plot_options is not None else None
    l = [label] if label is not None else None
    b = [bounds] if bounds is not None else None
    c = [color] if color is not None else None
    return plot1ds([spec], plot_options=po, labels=l, bounds=b, colors=c)


def plot1ds(specs: Sequence[ds.TDataset1D],
            ppm_offset: float = 0,
            plot_options: Optional[Sequence[MutableMapping]] = None,
            labels: Optional[Sequence[Optional[str]]] = None,
            bounds: Optional[Sequence[TBounds1D]] = None,
            colors: Optional[Sequence[Optional[str]]] = None,
            ) -> Tuple[Any, Any]:
    """
    Plots a series of 1D spectra on the same axes.
    """
    # Set default options. Using zip_longest(..., fillvalue=None), [] behaves
    # just as a list of Nones, i.e. the default values will be used.
    plot_options = plot_options or []
    labels = labels or []
    bounds = bounds or []
    colors = colors or []

    # Check that they're all 1D
    # Annoyingly, this is a runtime check, so we can't use the Union[...] definition here
    is1d = lambda x: x.__class__ in [ds.Dataset1D,
                                     ds.Dataset1DProj,
                                     ds.Dataset1DProjVirtual]
    if not all(is1d(s) for s in specs):
        raise TypeError(f"Incompatible types {[s.__class__.__name__ for s in specs]}")

    # Do the plotting
    for n, (s, o, l, b, c) in enumerate(zip_longest(specs, plot_options,
                                                    labels, bounds, colors,
                                                    fillvalue=None)):
        if o is not None:
            o = dict(default_1d_plotoptions, **o)
        else:
            o = default_1d_plotoptions
        # Specifying color will override any "color" key in plot_options.
        if c is not None:
            o.update(color=c)

        plt.plot(s.ppm_scale(bounds=b) - (n * ppm_offset),
                 s.proc_data(bounds=b),
                 label=l,
                 **o)
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 4), useMathText=True)
    plt.xlabel("Chemical shift (ppm)")
    plt.ylabel("Intensity (au)")
    xleft, xright = plt.xlim()
    if xleft < xright:
        ax.invert_xaxis()
    if any(labels):
        plt.legend()
    plt.tight_layout()
    return (plt.gca(), plt.gcf())


# -- 2D PLOTTING ----------------------------------------------

default_2d_plotoptions = {"linewidths": 0.7}
default_2d_levs = {"levmin": 2e4,
                   "levn": 10,
                   "levinc": 1.5
                   }
default_2d_colors = {"pos": "blue",
                     "neg": "mediumseagreen"
                     }

def plot2d(spec: ds.Dataset2D,
           plot_options: Optional[MutableMapping] = None,
           contour_levels: Optional[TLevels] = None,
           contour_colors: Optional[TColors] = None,
           bounds: Optional[TBounds2D] = None,
           ) -> Tuple[Any, Any]:
    """
    Plots one 2D spectrum. Gets plot2ds() to do the hard work.
    """
    po = [plot_options] if plot_options is not None else None
    clr = [contour_colors] if contour_colors is not None else None
    b = [bounds] if bounds is not None else None
    return plot2ds([spec], plot_options=po,
                   contour_levels=contour_levels, contour_colors=clr,
                   bounds=b)


def plot2ds(specs: Sequence[ds.Dataset2D],
            ppm_offset: Tuple[float, float] = (0,0),
            plot_options: Optional[Sequence[MutableMapping]] = None,
            contour_levels: Optional[TLevels] = None,
            contour_levels_indiv: Optional[Sequence[TLevels]] = None,
            contour_colors: Optional[Sequence[TColors]] = None,
            bounds: Optional[Sequence[TBounds2D]] = None,
            ) -> Tuple[Any, Any]:
    """
    Plots a series of 2D spectra on the same axes.
    """
    # Set default options. Using zip_longest(..., fillvalue=None), [] behaves
    # just as a list of Nones, i.e. the default values will be used.
    plot_options = plot_options or []
    contour_colors = contour_colors or []
    bounds = bounds or []

    # Contour levels are a bit more subtle. First we need to check if the
    # contour_levels parameter is passed. If it is, then we override
    # contour_levels_indiv.
    chosen_contour_levels: Sequence[TLevels]
    if contour_levels is not None:
        chosen_contour_levels = [contour_levels] * len(specs)
    # Otherwise we use contour_levels_indiv.
    elif contour_levels_indiv is not None:
        chosen_contour_levels = contour_levels_indiv
    # Otherwise it's just empty and left to the defaults.
    else:
        chosen_contour_levels = []

    # Check that they're all 2D
    if not all(s.__class__ == ds.Dataset2D for s in specs):
        raise TypeError(f"Incompatible types {[s.__class__.__name__ for s in specs]}")

    # Plot 2D spectra
    for n, (s, o, lv, clr, b) in enumerate(zip_longest(specs, plot_options,
                                                       chosen_contour_levels,
                                                       contour_colors,
                                                       bounds,
                                                       fillvalue=None)):
        if o is not None:
            o = dict(default_2d_plotoptions, **o)
        else:
            o = default_2d_plotoptions
        b = b or ((None, None), (None, None))  # allow None as a synonym
        f1, f2 = s.ppm_scale(axis=0, bounds=b[0]), s.ppm_scale(axis=1, bounds=b[1])

        # Calculate contour levels
        # TODO: how to find a sensible value for levmin?
        levmin, levn, levinc = lv if lv is not None else (None, None, None)
        # apply defaults if lv contains any Nones
        levmin = levmin or default_2d_levs["levmin"]
        levn   = levn   or default_2d_levs["levn"]
        levinc = levinc or default_2d_levs["levinc"]
        clevels = [-levmin * (levinc ** (levn - i)) for i in range(levn)] + \
            [levmin * (levinc ** i) for i in range(levn)]

        # Contour colours
        cpos, cneg = clr if clr is not None else (None, None)
        # apply defaults
        cpos = cpos or default_2d_colors["pos"]
        cneg = cneg or default_2d_colors["neg"]
        ccolors = ([cneg] * levn) + ([cpos] * levn)

        # plot the data
        plt.contour(f2 - (n * ppm_offset[0]),   # x-axis
                    f1 - (n * ppm_offset[1]),   # y-axis
                    s.proc_data(f1_bounds=b[0], f2_bounds=b[1]),
                    clevels, colors=ccolors, **o)
    ax = plt.gca()
    xleft, xright = plt.xlim()
    yleft, yright = plt.ylim()
    if xleft < xright:
        ax.invert_xaxis()
    if yleft < yright:
        ax.invert_yaxis()
    plt.xlabel(r"$f_2$ (ppm)")
    plt.ylabel(r"$f_1$ (ppm)")
    plt.tight_layout()
    return (plt.gca(), plt.gcf())


# -- Matplotlib wrappers ------------------------------------

def show(*args, **kwargs):
    plt.show(*args, **kwargs)

def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)
