from __future__ import annotations   # PEP 563

from itertools import zip_longest
from collections import abc
from typing import (Union, Iterable, MutableMapping,
                    Optional, Tuple, Any, Deque)

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import AutoMinorLocator  # type: ignore

from . import dataset as ds
from .type_aliases import *


# -- HELPER OBJECTS -------------------------------------------

SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
           "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
            "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
          "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
)

class PlotHoldingArea():
    def __init__(self):
        self.plot_queue: List = []
        # prime the colour generators
        self.colors_1d = self.color_generator_1d()
        self.colors_2d = self.color_generator_2d()

    def color_generator_1d(self):
        while True:
            yield from SEABORN_PALETTES["deep"]

    def color_generator_2d(self):
        while True:
            yield from [("blue", "red"),
                        ("seagreen", "hotpink")]

# Not a good idea to use this directly!
_globalPHA = PlotHoldingArea()

# Go through these methods! It doesn't necessarily make it *safer*, because
# you can still access _globalPHA, but it's easier to keep track of.
def get_pha():
    return _globalPHA

def reset_pha():
    global _globalPHA
    _globalPHA = PlotHoldingArea()


def plot(**kwargs):
    """
    Delegates to _plot1d() or _plot2d() as necessary.
    """
    PHA = get_pha()
    if len(PHA.plot_queue) == 0:
        raise ValueError("No spectra have been staged yet.")
    else:
        if isinstance(PHA.plot_queue[0], PlotObject1D):
            fig, ax = _plot1d(PHA, **kwargs)
        elif isinstance(PHA.plot_queue[0], PlotObject2D):
            fig, ax = _plot2d(PHA, **kwargs)
        else:
            raise TypeError("Plot holding area has invalid entries.")
    # Reset the PHA to being empty
    reset_pha()
    return (fig, ax)


# -- 1D PLOTTING ----------------------------------------------

class PlotObject1D():
    """
    Object that includes a Dataset as well as plotting options, e.g.
    labels, bounds, colours, & generic options passed to matplotlib.
    """
    default_1d_plotoptions = {"linewidth": 1}
    def __init__(self,
                 dataset: ds.TDataset1D,
                 scale: OF = 1,
                 bounds: Optional[TBounds1D] = None,
                 label: OS = None,
                 color: OS = None,
                 plot_options: Optional[MutableMapping] = None):
        self.dataset = dataset
        self.scale = scale
        self._init_bounds(bounds)
        self._init_options(plot_options, color, label)
        self.ppm_scale = self.dataset.ppm_scale(bounds=self.bounds)
        self.proc_data = self.dataset.proc_data(bounds=self.bounds) * self.scale

    def _init_bounds(self, bounds):
        self.bounds = bounds or (None, None)

    def _init_options(self, plot_options, color, label):
        """
        Note that the color parameter will override any color key/value pair
        passed in plot_options.
        """
        options = dict(self.default_1d_plotoptions) # make a copy
        if plot_options is not None:
            options.update(plot_options)
        if color is not None:
            options.update(color=color)
        if label is not None:
            options.update(label=label)
        # If a colour hasn't been chosen by now, get one from the colour
        # generator in PHA.
        if "color" not in options:
            next_color = next(get_pha().colors_1d)
            options.update(color=next_color)
        self.options = options


def stage1d(dataset: ds.TDataset1D,
            scale: OF = 1,
            bounds: Optional[TBounds1D] = None,
            label: OS = None,
            color: OS = None,
            plot_options: Optional[MutableMapping] = None,
            ) -> None:
    """
    Stages a 1D spectrum to be plotted, i.e. sticks it into PHA.plot_queue.
    """
    # Create a PlotObject1D first.
    plot_obj = PlotObject1D(dataset=dataset, scale=scale,
                            bounds=bounds, label=label,
                            color=color, plot_options=plot_options)

    # Check that the plot queue doesn't have 2D spectra.
    # We can just check against the first element. By induction, it is
    # equivalent to checking against every element.
    PHA = get_pha()
    if len(PHA.plot_queue) != 0 and isinstance(PHA.plot_queue[0], PlotObject2D):
        raise TypeError("Plot queue already contains 2D spectra.")
    else:
        PHA.plot_queue.append(plot_obj)


def _plot1d(holding_area: PlotHoldingArea,
            figstyle: str = "default",
            stacked: bool = False,
            voffset: float = 0,
            hoffset: float = 0,
            title: OS = None,
            xlabel: str = "Chemical shift (ppm)",
            ylabel: str = "Intensity (au)",
            ) -> Tuple[Any, Any]:
    """
    Plots all the 1D objects in the PHA.
    """
    make_legend = False
    # Find the maximum height if stacked is True, or if voffset is nonzero.
    if stacked or voffset != 0:
        heights = [np.amax(pobj.proc_data) - np.amin(pobj.proc_data)
                   for pobj in holding_area.plot_queue]
        max_height = max(heights)
    else:
        max_height = 0

    # Iterate over plot objects. Isn't it easy?
    for n, pobj in enumerate(holding_area.plot_queue):
        # Decide whether to make the legend
        if "label" in pobj.options and pobj.options["label"] is not None:
            make_legend = True
        # Calculate vertical offset
        if stacked:
            # Raise each spectrum by the heights of previous spectra,
            # plus a little padding per spectrum.
            vert_offset = sum(heights[0:n]) + (n * 0.1 * max_height)
        else:
            # This covers the case where voffset is 0 as well.
            vert_offset = n * voffset * max_height
        # Plot it!
        plt.plot(pobj.ppm_scale - (n * hoffset),
                 pobj.proc_data + (vert_offset),
                 **pobj.options)

    # Plot formatting
    ax = plt.gca()
    # Only if y-axis is enabled.
    # plt.ylabel(ylabel)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 4), useMathText=True)
    plt.xlabel(xlabel)
    ax.invert_xaxis()
    if make_legend:
        plt.legend()
    # Apply other styles.
    if figstyle == "default":
        # Remove the other spines
        for s in ["top", "left", "right"]:
            ax.spines[s].set_visible(False)
        ax.yaxis.set_visible(False)
        # Make the bottom one thicker
        ax.spines["bottom"].set_linewidth(1.3)
        # Enable minor ticks and make the ticks more visible
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)
    elif figstyle == "mpl_natural":
        pass
    if title:
        plt.title(title)
    plt.tight_layout()
    return (plt.gcf(), plt.gca())


# -- 2D PLOTTING ----------------------------------------------

class Contours:
    """2D plot contours."""
    def __init__(self,
                 levels: Optional[TLevels] = None,
                 colors: Optional[TColors] = None):
        levels = levels or (None, None, None)
        self.read_levels(*levels)
        colors = colors or (None, None)
        self.read_colors(*colors)

    def read_levels(self, base: OF = None,
                    increment: OF = None,
                    number: Optional[int] = None):
        self.base = base or 2e4
        self.increment = increment or 1.5
        self.number = number or 10

    def read_colors(self,
                    color_positive: OS = None,
                    color_negative: OS = None):
        if color_positive is None or color_negative is None:
            # Means we need to get a color set from the PHA generator.
            next_positive, next_negative = next(get_pha().colors_2d)
            self.color_positive = color_positive or next_positive
            self.color_negative = color_negative or next_negative
        else:
            self.color_positive = color_positive
            self.color_negative = color_negative

    def generate_contour_levels(self):
        neg = [-self.base * (self.increment ** (self.number - i)) for i in range(self.number)]
        pos = [self.base * (self.increment ** i) for i in range(self.number)]
        return neg + pos

    def generate_contour_colors(self):
        neg = [self.color_negative] * self.number
        pos = [self.color_positive] * self.number
        return neg + pos


class PlotObject2D():
    """
    Object that includes a Dataset as well as plotting options, e.g.
    contours, bounds, colours, & generic options passed to matplotlib.
    """
    default_2d_plotoptions = {"linewidths": 0.7}
    def __init__(self,
                 dataset: ds.Dataset2D,
                 bounds: Optional[TBounds2D] = None,
                 levels: Optional[TLevels] = None,
                 colors: Optional[TColors] = None,
                 plot_options: Optional[MutableMapping] = None):
        self.dataset = dataset
        self._init_bounds(bounds)
        self.contours = Contours(levels, colors)
        self.clevels = self.contours.generate_contour_levels()
        self.ccolors = self.contours.generate_contour_colors()
        self._init_options(plot_options)
        # self.options will include the colors key.
        self.f1_scale = self.dataset.ppm_scale(axis=0, bounds=self.bounds[0])
        self.f2_scale = self.dataset.ppm_scale(axis=1, bounds=self.bounds[1])
        self.proc_data = self.dataset.proc_data(f1_bounds=self.bounds[0],
                                                f2_bounds=self.bounds[1])

    def _init_bounds(self, bounds):
        if bounds is None:
            bounds = ((None, None), (None, None))
        else:
            if bounds[0] is None:
                bounds[0] = (None, None)
            if bounds[1] is None:
                bounds[1] = (None, None)
        self.bounds = bounds

    def _init_options(self, plot_options):
        options = dict(self.default_2d_plotoptions)  # make a copy
        if plot_options is not None:
            options.update(plot_options)
        # As before, the colors parameter will override the colors key provided
        # in plot_options.
        options.update(colors=self.ccolors)
        self.options = options


def stage2d(dataset: ds.Dataset2D,
            bounds: Optional[TBounds2D] = None,
            levels: Optional[TLevels] = None,
            colors: Optional[TColors] = None,
            plot_options: Optional[MutableMapping] = None,
            ) -> None:
    """
    Stages a 2D spectrum.
    """
    plot_obj = PlotObject2D(dataset=dataset, bounds=bounds,
                            levels=levels, colors=colors,
                            plot_options=plot_options)

    # Check that the plot queue doesn't have 1D spectra.
    # We can just check against the first element. By induction, it is
    # equivalent to checking against every element.
    PHA = get_pha()
    if len(PHA.plot_queue) != 0 and isinstance(PHA.plot_queue[0], PlotObject1D):
        raise TypeError("Plot queue already contains 1D spectra.")
    else:
        PHA.plot_queue.append(plot_obj)


def _plot2d(holding_area: PlotHoldingArea,
            offset: Tuple[float, float] = (0, 0),
            ) -> Tuple[Any, Any]:
    """
    Plots all the 2D objects in the PHA.
    """
    # Iterate over plot objects. Isn't it easy?
    for n, pobj in enumerate(holding_area.plot_queue):
        plt.contour(pobj.f2_scale - (n * offset[1]),   # x-axis
                    pobj.f1_scale - (n * offset[0]),   # y-axis
                    pobj.proc_data,
                    levels=pobj.clevels,
                    **pobj.options)
    # Axis formatting
    ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.xlabel(r"$f_2$ (ppm)")
    plt.ylabel(r"$f_1$ (ppm)")
    plt.tight_layout()
    return (plt.gca(), plt.gcf())


# -- Matplotlib wrappers ------------------------------------

def show(*args, **kwargs):
    PHA = get_pha()
    if len(PHA.plot_queue) != 0:
        print("Warning: plot staging area is not empty. "
              "Did you mean to call pg.plot()?")
    plt.show(*args, **kwargs)


def savefig(*args, **kwargs):
    PHA = get_pha()
    if len(PHA.plot_queue) != 0:
        print("Warning: plot staging area is not empty. "
              "Did you mean to call pg.plot()?")
    plt.savefig(*args, **kwargs)
