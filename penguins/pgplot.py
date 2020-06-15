from __future__ import annotations   # PEP 563

from itertools import zip_longest, cycle
from collections import abc
from typing import (Union, Iterable, MutableMapping,
                    Optional, Tuple, Any, Deque)
from numbers import Real

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import AutoMinorLocator  # type: ignore
from matplotlib.legend_handler import HandlerBase  # type: ignore

from . import dataset as ds
from . import main
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
        yield from cycle(SEABORN_PALETTES["deep"])

    def color_generator_2d(self):
        i = 0
        while True:
            yield (SEABORN_PALETTES["deep"][i % 10],
                   SEABORN_PALETTES["deep"][(i + 5) % 10])
            i += 1

class PlotProperties():
    """
    A collection of properties of spectra that have already been constructed
    with mkplot().
    """
    def __init__(self):
        self.heights = []
        self.colors = []
        self.options = []
        self.colors_positive = []
        self.colors_negative = []

# Not a good idea to use these directly!
_globalPHA = PlotHoldingArea()
_globalPP = PlotProperties()

# Go through these methods! It doesn't necessarily make it *safer*, because
# you can still access _globalPHA, but it's easier to keep track of.
def get_pha():
    return _globalPHA

def _reset_pha():
    global _globalPHA
    _globalPHA = PlotHoldingArea()

def get_properties():
    return _globalPP

def _reset_properties():
    global _globalPP
    _globalPP = PlotProperties()

# -- 1D PLOTTING ----------------------------------------------

class PlotObject1D():
    """
    Object that includes a Dataset as well as plotting options, e.g.
    labels, bounds, colours, & generic options passed to matplotlib.
    """
    default_1d_plotoptions = {"linewidth": 1}
    def __init__(self,
                 dataset: ds.TDataset1D,
                 scale: float = 1,
                 bounds: TBounds = "",
                 dfilter: Optional[Callable[[float], bool]] = None,
                 label: OS = None,
                 color: OS = None,
                 plot_options: Optional[MutableMapping] = None):
        self.dataset = dataset
        self.scale = scale
        self.bounds = bounds
        self._init_options(plot_options, color, label)
        self.ppm_scale = self.dataset.ppm_scale(bounds=self.bounds)
        # Handle processed data
        proc_data = self.dataset.proc_data(bounds=self.bounds)
        if dfilter is not None:
            # This is a bit ugly, but if dfilter involves any ands or ors,
            # then proc_data = proc_data[dfilter(proc_data)] will raise
            # a ValueError: Truth value is ambiguous.
            logical_array = np.array([dfilter(i) for i in proc_data])
            proc_data = np.where(logical_array, proc_data, np.nan)
        self.proc_data = proc_data * self.scale

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
            scale: float = 1,
            bounds: TBounds = "",
            dfilter: Optional[Callable[[float], Bool]] = None,
            label: OS = None,
            color: OS = None,
            plot_options: Optional[MutableMapping] = None,
            ) -> None:
    """
    Stages a 1D spectrum to be plotted, i.e. sticks it into PHA.plot_queue.
    """
    # Create a PlotObject1D first.
    plot_obj = PlotObject1D(dataset=dataset, scale=scale,
                            bounds=bounds, dfilter=dfilter,
                            label=label,
                            color=color, plot_options=plot_options)

    # Check that the plot queue doesn't have 2D spectra.
    # We can just check against the first element. By induction, it is
    # equivalent to checking against every element.
    PHA = get_pha()
    if len(PHA.plot_queue) != 0 and isinstance(PHA.plot_queue[0], PlotObject2D):
        raise TypeError("Plot queue already contains 2D spectra.")
    else:
        PHA.plot_queue.append(plot_obj)


def _mkplot1d(holding_area: PlotHoldingArea,
              figstyle: str = "default",
              stacked: bool = False,
              voffset: float = 0,
              hoffset: float = 0,
              title: OS = None,
              xlabel: str = "Chemical shift (ppm)",
              ylabel: str = "Intensity (au)",
              legend_loc: Any = "best",
              ) -> Tuple[Any, Any]:
    """
    Plots all the 1D objects in the PHA.
    """
    make_legend = False
    # Find the maximum height
    heights = [np.nanmax(pobj.proc_data) - np.nanmin(pobj.proc_data)
               for pobj in holding_area.plot_queue]
    max_height = max(heights)

    # Iterate over plot objects
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
        # Add heights and colors to plotproperties.
        pp = get_properties()
        pp.heights.append(vert_offset)
        pp.colors.append(pobj.options["color"])
        pp.options.append(pobj.options)

    # Plot formatting
    ax = plt.gca()
    # Only if y-axis is enabled.
    # plt.ylabel(ylabel)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 4), useMathText=True)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    ax.invert_xaxis()
    if make_legend:
        plt.legend(loc=legend_loc)
    # Apply other styles.
    if figstyle not in ["default", "mpl_natural"]:
        print(f"No figure style corresponding to {figstyle}. Using default.")
        figstyle = "default"
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
        plt.tight_layout()
    elif figstyle == "mpl_natural":
        pass
    return (plt.gcf(), plt.gca())


# -- 2D PLOTTING ----------------------------------------------

class Contours:
    """2D plot contours."""
    def __init__(self,
                 dataset: ds.Dataset2D,
                 levels: TLevels = (None, None, None),
                 colors: TColors = (None, None),
                 ):
        self.dataset = dataset
        if isinstance(levels, float):
            levels = (levels, None, None)
        self.make_levels(*levels)
        self.make_colors(*colors)

    def make_levels(self, base: OF = None,
                    increment: OF = None,
                    number: Optional[int] = None):
        self.base = base or self.dataset._tsbaselev
        self.increment = increment or 1.5
        self.number = number or 10

    def make_colors(self,
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
                 f1_bounds: TBounds = "",
                 f2_bounds: TBounds = "",
                 levels: TLevels = (None, None, None),
                 colors: TColors = (None, None),
                 label: OS = None,
                 plot_options: Optional[MutableMapping] = None):
        self.dataset = dataset
        self.f1_bounds = f1_bounds
        self.f2_bounds = f2_bounds
        self.contours = Contours(self.dataset, levels, colors)
        # can access cpos and cneg with self.contours.color_[positive|negative]
        self.clevels = self.contours.generate_contour_levels()
        self.ccolors = self.contours.generate_contour_colors()
        self._init_options(plot_options)
        self.label = label
        # self.options will include the colors key.
        self.f1_scale = self.dataset.ppm_scale(axis=0, bounds=self.f1_bounds)
        self.f2_scale = self.dataset.ppm_scale(axis=1, bounds=self.f2_bounds)
        self.proc_data = self.dataset.proc_data(f1_bounds=self.f1_bounds,
                                                f2_bounds=self.f2_bounds)

    def _init_options(self, plot_options):
        options = dict(self.default_2d_plotoptions)  # make a copy
        if plot_options is not None:
            options.update(plot_options)
        # As before, the colors parameter will override the colors key provided
        # in plot_options.
        options.update(colors=self.ccolors)
        self.options = options


class ContourLegendHandler(HandlerBase):
    # code lifted mostly from https://stackoverflow.com/a/41765095/
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        line_cpos = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                               color=orig_handle[0])
        line_cneg = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                               color=orig_handle[1])
        return [line_cpos, line_cneg]


def stage2d(dataset: ds.Dataset2D,
            f1_bounds: TBounds = "",
            f2_bounds: TBounds = "",
            levels: TLevels = (None, None, None),
            colors: TColors = (None, None),
            label: OS = None,
            plot_options: Optional[MutableMapping] = None,
            ) -> None:
    """
    Stages a 2D spectrum.
    """
    plot_obj = PlotObject2D(dataset=dataset,
                            f1_bounds=f1_bounds, f2_bounds=f2_bounds,
                            levels=levels, colors=colors,
                            label=label,
                            plot_options=plot_options)

    # Check that the plot queue doesn't have 1D spectra.
    # We can just check against the first element. By induction, it is
    # equivalent to checking against every element.
    PHA = get_pha()
    if len(PHA.plot_queue) != 0 and isinstance(PHA.plot_queue[0], PlotObject1D):
        raise TypeError("Plot queue already contains 1D spectra.")
    else:
        PHA.plot_queue.append(plot_obj)


def _mkplot2d(holding_area: PlotHoldingArea,
              figstyle: str = "default",
              offset: Tuple[float, float] = (0, 0),
              title: OS = None,
              xlabel: str = r"$f_2$ (ppm)",
              ylabel: str = r"$f_1$ (ppm)",
              legend_loc: Any = "best",
              ) -> Tuple[Any, Any]:
    """
    Plots all the 2D objects in the PHA.
    """
    make_legend = False
    legend_colors, legend_labels = [], []
    # Iterate over plot objects
    for n, pobj in enumerate(holding_area.plot_queue):
        plt.contour(pobj.f2_scale - (n * offset[1]),   # x-axis
                    pobj.f1_scale - (n * offset[0]),   # y-axis
                    pobj.proc_data,
                    levels=pobj.clevels,
                    **pobj.options)
        # Construct lists for plt.legend
        if pobj.label is not None:
            make_legend = True
            legend_colors.append((pobj.contours.color_positive,
                                  pobj.contours.color_negative)
                                 )
            legend_labels.append(pobj.label)
    # Axis formatting
    ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Apply other styles.
    if figstyle not in ["default", "mpl_natural"]:
        print(f"No figure style corresponding to {figstyle}. Using default.")
        figstyle = "default"
    if figstyle == "default":
        # Make spines thicker
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)
        # Enable minor ticks and make the ticks more visible
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)
        plt.tight_layout()
    elif figstyle == "mpl_natural":
        pass

    # Make legend. This part is not easy...
    # See https://stackoverflow.com/questions/41752309/ for an example
    if make_legend:
        plt.legend(legend_colors, legend_labels,
                   handler_map={tuple: ContourLegendHandler()},
                   loc=legend_loc)
    return (plt.gcf(), plt.gca())


def _make_contour_slider(dataset: ds.Dataset2D,
                         increment: float = None,
                         nlev: int = 4,
                         ) -> None:
    # Choose contour levels. We reduce nlev to 4 by default so that the
    # plotting is faster -- otherwise it's super laggy. We try to cover the
    # same dynamic range as 1.5 ** 10 by default, unless the user specified
    # an increment.
    initial_baselev = dataset._tsbaselev
    increment = increment or (1.5 ** 10) ** (1 / nlev)
    initial_clev = (initial_baselev, increment, nlev)
    dataset.stage(levels=initial_clev)
    # Maximum level should be the highest intensity of the spectrum.
    max_baselev = np.max(np.abs(dataset.rr))

    # Plot the spectrum on the top portion of the figure.
    fig, ax = plt.subplots()
    _, plot_axes = main.plot(empty_pha=False,
                             figstyle="mpl_natural",
                             )
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Generate a slider.
    from matplotlib.widgets import Slider, Button     # type: ignore
    baselev_axes = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lavender")
    baselev_slider = Slider(baselev_axes, "",
                            2, np.log10(max_baselev),
                            valinit=np.log10(initial_baselev), color="purple")
    # Add some text
    plt.text(0.5, -1.2, r"log$_{10}$(base contour level)",
             horizontalalignment="center",
             transform=baselev_axes.transAxes)

    # Define the behaviour when redrawn
    def redraw(plot_axes: Any, val: float) -> None:
        # Update the internal Contours object
        pobj = get_pha().plot_queue[0]
        pobj.contours.base = 10 ** val
        # Regenerate the contours
        pobj.clevels = pobj.contours.generate_contour_levels()
        # Replot
        plot_axes.cla()
        plt.sca(plot_axes)
        main.plot(close=False, empty_pha=False, figstyle="mpl_natural")

    # Register the redraw function. The argument to on_changed must be a
    # function taking val as its only parameter.
    baselev_slider.on_changed(lambda val: redraw(plot_axes, val))

    # Generate the "OK" button
    okay_axes = plt.axes([0.84, 0.025, 0.1, 0.04])
    okay_button = Button(okay_axes, "OK", color="plum", hovercolor='0.95')
    okay = 0
    # Add the close behaviour
    def set_okay(button):
        nonlocal okay
        okay = 1
        plt.close()
    okay_button.on_clicked(set_okay)

    # Generate the "Reset" button
    reset_axes = plt.axes([0.72, 0.025, 0.1, 0.04])
    reset_button = Button(reset_axes, "Reset", color="plum", hovercolor='0.95')
    # Add the close behaviour
    def reset(button):
        baselev_slider.reset()
    reset_button.on_clicked(reset)

    # Show it to the user.
    main.show()

    # Post-OK actions
    if okay:
        # Print the final value
        fval = 10 ** baselev_slider.val
        fval_short = f"{fval:.2e}".replace("e+0", "e")
        print(f"\n[[{dataset!r}]]\nThe final base level was: {fval} (or {fval_short})\n")
    # Clear out the PHA
    _reset_pha()
    plt.close("all")

