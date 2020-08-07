from __future__ import annotations   # PEP 563

from itertools import zip_longest, cycle
from collections import abc
from typing import (Union, Iterable, Dict, Optional, Any,
                    Tuple, List, Deque, Callable, Sequence, Iterator)
from numbers import Real

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import AutoMinorLocator  # type: ignore
from matplotlib.legend_handler import HandlerBase  # type: ignore
import seaborn as sns  # type: ignore

from . import dataset as ds
from . import main
from .type_aliases import *


# -- HELPER OBJECTS -------------------------------------------

# 1D colour palette to use (from seaborn). By default "deep".
_current_palette = "deep"
# This are the colours from seaborn-bright, but rearranged into nice tuples.
# Honestly three pairs of colours should suffice. If you're plotting more 2D
# spectra than that on the same graph, you probably need to rethink your plot,
# or at least choose your colours manually to illustrate whatever point you're
# trying to make.
_bright_2d = [("#023EFF", "#E8000B"), # blue, red
              ("#1AC938", "#FF7C00"), # green, orange
              ("#8B2BE2", "#F14CC1"), # purple, pink
              ]


def set_palette(palette: str
                ) -> None:
    global _current_palette
    # Change seaborn palette, in case user wants to draw other plots.
    sns.set_palette(palette)
    # Change penguins 1D palette.
    _current_palette = palette


class PlotHoldingArea():
    def __init__(self) -> None:
        self.plot_queue: List = []
        # prime the colour generators
        self.colors_1d = self.color_generator_1d()
        self.colors_2d = self.color_generator_2d()

    def color_generator_1d(self) -> Iterator[str]:
        yield from cycle(sns.color_palette(_current_palette))

    def color_generator_2d(self) -> Iterator[Tuple[str, str]]:
        yield from cycle(_bright_2d)

class PlotProperties():
    """
    A collection of properties of spectra that have already been constructed
    with mkplot().
    """
    def __init__(self) -> None:
        self.hoffsets: List[float] = []
        self.voffsets: List[float] = []
        self.colors: List[str] = []
        self.options: List[Dict[str, Any]] = []
        self.colors_positive: List[str] = []
        self.colors_negative: List[str] = []

# Not a good idea to use these directly!
_globalPHA: PlotHoldingArea = PlotHoldingArea()
_globalPP: PlotProperties = PlotProperties()

# Go through these methods! It doesn't necessarily make it *safer*, because
# you can still access _globalPHA, but it's easier to keep track of.
def get_pha() -> PlotHoldingArea:
    return _globalPHA

def _reset_pha() -> None:
    global _globalPHA
    _globalPHA = PlotHoldingArea()

def get_properties() -> PlotProperties:
    return _globalPP

def _reset_properties() -> None:
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
                 plot_options: Optional[Dict] = None
                 ) -> None:
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

    def _init_options(self,
                      plot_options: Optional[Dict],
                      color: OS,
                      label: OS):
        """
        Note that the color parameter will override any color key/value pair
        passed in plot_options.
        """
        # Make a copy of the default options.
        options: Dict[str, Any] = dict(self.default_1d_plotoptions)
        # Update it using the function arguments
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
        # Finally, set the instance attribute
        self.options = options


def stage1d(dataset: ds.TDataset1D,
            scale: float = 1,
            bounds: TBounds = "",
            dfilter: Optional[Callable[[float], bool]] = None,
            label: OS = None,
            color: OS = None,
            plot_options: Optional[Dict] = None,
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
              ax: Any = None,
              style: str = "1d",
              stacked: bool = False,
              voffset: Union[Sequence, float] = 0,
              hoffset: Union[Sequence, float] = 0,
              title: OS = None,
              autolabel: OS = None,
              xlabel: OS = None,
              ylabel: OS = None,
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

    # Get Axes object
    if ax is None:
        ax = plt.gca()

    # Iterate over plot objects
    for n, pobj in enumerate(holding_area.plot_queue):
        # Calculate the hoffset and voffset for this spectrum. If offset is a
        # sequence then use offset[n], otherwise if it's a float use n * offset
        # Also, mypy really doesn't like try/except.
        try:
            this_hoffset = hoffset[n]   # type: ignore
        except TypeError:  # not a sequence
            this_hoffset = hoffset * n   # type: ignore
        try:
            this_voffset = voffset[n] * max_height   # type: ignore
        except TypeError:  # not a sequence
            if stacked:
                # Raise each spectrum by the heights of previous spectra, plus
                # a little padding per spectrum.
                this_voffset = sum(heights[0:n]) + (n * 0.1 * max_height)
            else:
                # This covers the case where voffset is 0 as well.
                this_voffset = n * voffset * max_height  # type: ignore
        # Decide whether to make the legend
        if "label" in pobj.options and pobj.options["label"] is not None:
            make_legend = True
        # Plot it!
        ax.plot(pobj.ppm_scale - this_hoffset,
                pobj.proc_data + this_voffset,
                **pobj.options)
        # Add heights and colors to plotproperties.
        pp = get_properties()
        pp.hoffsets.append(this_hoffset)
        pp.voffsets.append(this_voffset)
        pp.colors.append(pobj.options["color"])
        pp.options.append(pobj.options)

    # Figure out the x- and y-labels.
    # First, we generate the strings accoring to autolabel.
    f_ylabel = "Intensity (au)"
    if autolabel is not None:
        if autolabel == "nucl":
            f_xlabel = holding_area.plot_queue[0].dataset.nuclei_to_str()
            f_xlabel += " (ppm)"
        else:
            raise ValueError(f"Invalid value '{autolabel}' given for "
                             "parameter autolabel.")
    else:
        f_xlabel = "Chemical shift (ppm)"
    # Then we override them based on the values of mkplot()'s kwargs.
    f_xlabel = xlabel if xlabel is not None else f_xlabel
    f_ylabel = ylabel if ylabel is not None else f_ylabel

    # Plot formatting
    # Only if y-axis is enabled.
    # plt.ylabel(f_ylabel)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 4), useMathText=True)
    if title:
        ax.set_title(title)
    ax.set_xlabel(f_xlabel)
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    if make_legend:
        ax.legend(loc=legend_loc)
    # Apply axis styles.
    style_axes(ax, style)
    return plt.gcf(), ax


def style_axes(ax: Any,
               style: str,
               ) -> None:
    """
    Styles the Axes instance according to the given style. Useful for ensuring
    that all subplots have a homogenous look.
    """
    if style == "1d":
        # Default 1D style. Doesn't draw a bounding box, only has the bottom
        # spine, which is made thicker. Adds extra x-axis ticks.
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
    elif style == "1d_box":
        # 1D but with thick bounding box. Extra x-axis ticks added. The y-axis
        # is still disabled.
        ax.yaxis.set_visible(False)
        # Make spines thicker
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)
        # Enable minor ticks and make the ticks more visible
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)
        plt.tight_layout()
    elif style == "2d":
        # Default 2D style. Makes bounding box thicker and adds x- and y-axis
        # ticks.
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)
        # Enable minor ticks and make the ticks more visible
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)
        plt.tight_layout()
    elif style == "plot":
        # To be used for other plots, e.g. seaborn / matplotlib visualisations.
        # Draws a thicker bounding box around the Axes, but otherwise doesn't
        # try to change axis ticks etc.
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)
        plt.tight_layout()
    elif style == "natural":
        # Literally, do nothing. Not even tight_layout().
        pass
    else:
        raise ValueError(f"Invalid style '{style}' requested.")


# -- 2D PLOTTING ----------------------------------------------

class Contours:
    """2D plot contours."""
    def __init__(self,
                 dataset: ds.Dataset2D,
                 levels: TLevels = (None, None, None),
                 colors: TColors = (None, None),
                 ) -> None:
        self.dataset = dataset
        if isinstance(levels, float):
            levels = (levels, None, None)
        self.make_levels(*levels)
        self.make_colors(*colors)

    def make_levels(self, base: OF = None,
                    increment: OF = None,
                    number: Optional[int] = None
                    ) -> None:
        self.base = base or self.dataset._tsbaselev
        self.increment = increment or 1.5
        self.number = number or 10

    def make_colors(self,
                    color_positive: OS = None,
                    color_negative: OS = None
                    ) -> None:
        if color_positive is None or color_negative is None:
            # Means we need to get a color set from the PHA generator.
            next_positive, next_negative = next(get_pha().colors_2d)
            self.color_positive = color_positive or next_positive
            self.color_negative = color_negative or next_negative
        else:
            self.color_positive = color_positive
            self.color_negative = color_negative

    def generate_contour_levels(self) -> List[float]:
        neg = [-self.base * (self.increment ** (self.number - i)) for i in range(self.number)]
        pos = [self.base * (self.increment ** i) for i in range(self.number)]
        return neg + pos

    def generate_contour_colors(self) -> List[str]:
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
                 dfilter: Optional[Callable[[float], bool]] = None,
                 label: OS = None,
                 plot_options: Optional[Dict] = None
                 ) -> None:
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
        # Handle processed data
        proc_data = self.dataset.proc_data(f1_bounds=self.f1_bounds,
                                           f2_bounds=self.f2_bounds)
        if dfilter is not None:
            logical_array = np.array([dfilter(i) for i in proc_data.flat])
            logical_array = np.reshape(logical_array, np.shape(proc_data))
            proc_data = np.where(logical_array, proc_data, np.nan)
        self.proc_data = proc_data

    def _init_options(self,
                      plot_options: Optional[Dict],
                      ) -> None:
        # Make a copy of the default options
        options: Dict[str, Any] = dict(self.default_2d_plotoptions)
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
            dfilter: Optional[Callable[[float], bool]] = None,
            label: OS = None,
            plot_options: Optional[Dict] = None,
            ) -> None:
    """
    Stages a 2D spectrum.
    """
    plot_obj = PlotObject2D(dataset=dataset,
                            f1_bounds=f1_bounds, f2_bounds=f2_bounds,
                            levels=levels, colors=colors,
                            dfilter=dfilter, label=label,
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
              ax: Any = None,
              style: str = "2d",
              offset: Tuple[float, float] = (0, 0),
              title: OS = None,
              autolabel: OS = None,
              xlabel: OS = None,
              ylabel: OS = None,
              legend_loc: Any = "best",
              ) -> Tuple[Any, Any]:
    """
    Plots all the 2D objects in the PHA.
    """
    make_legend = False
    legend_colors, legend_labels = [], []
    if ax is None:
        ax = plt.gca()
    # Iterate over plot objects
    for n, pobj in enumerate(holding_area.plot_queue):
        ax.contour(pobj.f2_scale - (n * offset[1]),   # x-axis
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

    # Figure out the x- and y-labels.
    # First, we generate the strings accoring to autolabel.
    if autolabel is not None:
        if autolabel == "nucl":
            f_xlabel = holding_area.plot_queue[0].dataset.nuclei_to_str()[1]
            f_xlabel += " (ppm)"
            f_ylabel = holding_area.plot_queue[0].dataset.nuclei_to_str()[0]
            f_ylabel += " (ppm)"
        else:
            raise ValueError(f"Invalid value '{autolabel}' given for "
                             "parameter autolabel.")
    else:
        f_xlabel = r"$f_2$ (ppm)"
        f_ylabel = r"$f_1$ (ppm)"
    # Then we override them based on the values of mkplot()'s kwargs.
    f_xlabel = xlabel if xlabel is not None else f_xlabel
    f_ylabel = ylabel if ylabel is not None else f_ylabel

    # Plot formatting
    # Only if y-axis is enabled.
    # plt.ylabel(f_ylabel)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 4), useMathText=True)
    # Axis formatting
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f_xlabel)
    ax.set_ylabel(f_ylabel)
    # Apply other styles.
    style_axes(ax, style)

    # Make legend. This part is not easy...
    # See https://stackoverflow.com/questions/41752309/ for an example
    if make_legend:
        plt.legend(legend_colors, legend_labels,
                   handler_map={tuple: ContourLegendHandler()},
                   loc=legend_loc)
    return (plt.gcf(), ax)


def _make_contour_slider(dataset: ds.Dataset2D,
                         increment: float = None,
                         nlev: int = 4,
                         ) -> float:
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
    _, plot_axes = main.mkplot(empty_pha=False,
                               style="natural",
                               )
    orig_xlim = plot_axes.get_xlim()
    orig_ylim = plot_axes.get_ylim()
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
    def redraw(plot_axes: Any,
               val: float
               ) -> None:
        # Update the internal Contours object
        pobj = get_pha().plot_queue[0]
        pobj.contours.base = 10 ** val
        # Regenerate the contours
        pobj.clevels = pobj.contours.generate_contour_levels()
        # Replot
        xlim = plot_axes.get_xlim()
        ylim = plot_axes.get_ylim()
        plot_axes.cla()
        plt.sca(plot_axes)
        main.mkplot(close=False, empty_pha=False, style="natural")
        plot_axes.set_xlim(xlim)
        plot_axes.set_ylim(ylim)

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

    # Generate the "Reset zoom" button
    reset_axes = plt.axes([0.66, 0.025, 0.16, 0.04])
    reset_button = Button(reset_axes, "Reset zoom", color="plum", hovercolor='0.95')
    # Add the reset behaviour
    def reset(button):
        plot_axes.set_xlim(orig_xlim)
        plot_axes.set_ylim(orig_ylim)
    reset_button.on_clicked(reset)

    # Show it to the user.
    main.show()

    # Post-OK actions
    if okay:
        # Print the dataset
        print(f"\n[[{dataset!r}]]")
        # Print the initial value
        ival = initial_baselev
        ival_short = f"{ival:.2e}".replace("e+0", "e")
        print(f"The initial base level was: {ival:.2f} (or {ival_short})")
        # Print the final value
        fval = 10 ** baselev_slider.val
        fval_short = f"{fval:.2e}".replace("e+0", "e")
        print(f"The final base level was:   {fval:.2f} (or {fval_short})\n")
    # Clear out the PHA
    _reset_pha()
    plt.close("all")

    return fval if okay else None
