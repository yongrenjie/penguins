from __future__ import annotations   # PEP 563

import warnings
from itertools import zip_longest, cycle
from collections import abc
from typing import (Union, Iterable, Dict, Optional, Any,
                    Tuple, List, Deque, Callable, Sequence, Iterator)

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.legend_handler import HandlerBase  # type: ignore
import seaborn as sns  # type: ignore

from . import dataset as ds
from .exportdeco import export
from .plotutils import *
from .type_aliases import *


# -- Helper objects -------------------------------------------

# 1D color palette to use (from seaborn). By default "deep".
_current_palette: Union[str, List[str]] = "deep"
# Enable it for seaborn as well.
sns.set_palette(_current_palette)
# This are the colors from seaborn-bright, but rearranged into nice tuples.
# Honestly three pairs of colors should suffice. If you're plotting more 2D
# spectra than that on the same graph, you probably need to rethink your plot,
# or at least choose your colors manually to illustrate whatever point you're
# trying to make.
_bright_2d = [("#023EFF", "#E8000B"), # blue, red
              ("#1AC938", "#FF7C00"), # green, orange
              ("#8B2BE2", "#F14CC1"), # purple, pink
              ]


# -- Top-level plotting interface -----------------------------

@export
def mkplot(ax: Any = None,
           empty_pha: bool = True,
           **kwargs
           ) -> Tuple[Any, Any]:
    """Construct a plot from one or more staged spectra.

    Parameters
    ----------
    ax : Axes, optional
        |Axes| instance to plot the spectra on. If not provided, creates new
        |Figure| and |Axes| instances.
    kwargs : dict, optional
        Keyword arguments that are passed on to `_mkplot1d()` or `_mkplot2d()`
        respectively, depending on the dimensionality of the spectrum being
        plotted. In turn, these are passed on to |plot| or |contour|.

    Returns
    -------
    fig : Figure
        |Figure| instance for the active plot.
    ax : Axes
        |Axes| instance for the active plot.

    Other Parameters
    ----------------
    empty_pha : bool, optional
        Whether to empty the `PlotHoldingArea` of the |Axes| after constructing
        the plots. The average user should have no use for this; it only exists
        to make `find_baselev()` work.

    Notes
    -----
    This function itself does not do very much. It mainly performs setup
    and teardown actions, whereas the actual plotting itself is delegated
    to `_mkplot1d()` and `_mkplot2d()`. (Those functions should not be used
    directly.)

    See Also
    --------
    penguins.pgplot._mkplot1d : Keyword arguments for 1D plots are described
                                here.
    penguins.pgplot._mkplot2d : Keyword arguments for 2D plots are described
                                here.
    """
    try:
        # Make sure that there is an active figure...
        if not plt.get_fignums():
            raise ValueError("No active figure found.")
        # Get currently active Axes if it wasn't specified.
        # Note that gca() creates one if there isn't already one...
        if ax is None:
            ax = plt.gca()

        # Check if the PHA exists and isn't empty.
        if not hasattr(ax, "pha") or len(ax.pha.plot_objs) == 0:
            warnings.warn("No plots have been staged on this Axes yet.")
            return None, None
        else:
            # Reset (or create) plot properties
            ax.prop = PlotProperties()
            if isinstance(ax.pha.plot_objs[0], PlotObject1D):
                fig, ax = _mkplot1d(ax=ax, **kwargs)
            elif isinstance(ax.pha.plot_objs[0], PlotObject2D):
                fig, ax = _mkplot2d(ax=ax, **kwargs)
            else:
                raise TypeError("Plot holding area has invalid elements.")
    finally:
        # Reset the PHA to being empty
        if ax is not None and empty_pha:
            ax.pha = PlotHoldingArea()
    return fig, ax


@export
def mkplots(axs: Any = None,
            titles: Sequence[OS] = None,
            **kwargs
            ) -> Tuple[Any, Any]:
    """
    Convenience function which essentially calls mkplot(ax, title=title) for
    ax, title in zip(axs.flat, titles).

    Parameters
    ----------
    axs : list, tuple, or ndarray of Axes (optional)
        If not passed, will iterate over all Axes in the currently active
        figure. >1D arrays (e.g. those returned by `subplots()`) are allowed.

    titles : list or tuple of str (optional)
        A series of subplot titles. Use None or an empty string to not have a
        a title.

    **kwargs : dict
        Other keyword arguments which are passed on to `mkplot()` (and
        consequently the other functions that it calls).

    Returns
    -------
    fig : Figure
        |Figure| instance for the active plot.
    axs : ndarray of Axes
        The same ndarray that was provided.
    """
    # Make sure that there is an active figure...
    if not plt.get_fignums():
        raise ValueError("No active figure found.")
    # Get all active Axes if none were given
    fig = plt.gcf()
    if axs is None:
        axs_it = fig.get_axes()
    else:
        if isinstance(axs, np.ndarray):
            axs_it = axs.flat
        else:
            axs_it = axs  # if not iterable, will raise TypeError later
    # If no titles were given, use an empty list
    if titles is None:
        titles = []
    # Call mkplot() on each Axes.
    for ax, title in zip_longest(axs_it, titles):
        mkplot(ax, title=title, **kwargs)
    return fig, axs


@export
def mkinset(ax: Any,
            pos: Tuple[float, float],
            size: Tuple[float, float],
            transform: Any = None,
            show_zoom: bool = True,
            parent_corners: Tuple[str, str] = ("sw", "se"),
            inset_corners: Tuple[str, str] = ("sw", "se"),
            plot_options: Optional[dict] = None,
            inset_options: Optional[dict] = None,
            ) -> Any:
    """Constructs an inset on a given Axes instance and plots any staged
    spectra on the inset Axes.

    Parameters
    ----------
    ax : Axes
        |Axes| instance to construct the inset inside.
    pos : (float, float)
        Position of lower-left corner of inset axes given as (x, y).
    size : (float, float)
        Size of inset axes, given as (width, height).
    transform : Transform
        |Transform| to use for specifying the coordinates of
        *pos* and *size*. By default, axes coordinates are used for both.
    show_zoom : bool, optional
        Whether to draw lines between the parent and inset axes (to indicate
        the section of the spectrum zoomed into).
    parent_corners : (str, str), optional
        Corners of the parent axes to draw the zoom lines from. Each element of
        the tuple can be chosen from {"southwest", "southeast", "northeast",
        "northwest", "sw", "se", "ne", "nw"}.
    inset_corners : (str, str), optional
        Corners of the inset axes to draw the zoom lines to.
    plot_options : dict, optional
        Dictionary of options passed to `mkplot()`, which is in turn passed to
        either |plot| or |contour|.
    inset_options : dict, optional
        Dictionary of options passed to |mark_inset|.

    Returns
    -------
    inset_ax : Axes
        The |Axes| instance corresponding to the inset.
    """
    # Find the axes to draw on
    ax = ax or plt.gca()
    # Generate inset axes
    inset_ax = ax.inset_axes([*pos, *size], transform=transform)
    plot_options = plot_options or {}
    mkplot(ax=inset_ax, xlabel="", ylabel="", **plot_options)

    # Convert the strings to numbers
    def convert_corner(ax: Any, cornerstr: str, is_inset: bool):
        """
        For non-inverted axes, ne = 1, nw = 2, sw = 3, se = 4
        If x-axis is inverted, ne = 2, nw = 1, sw = 4, se = 3
        If y-axis is inverted, ne = 4, nw = 3, sw = 2, se = 1
        If both are inverted,  ne = 3, nw = 4, sw = 1, se = 2
        However, inset axes ALWAYS behave as if they are not inverted,
        regardless of whether they were inverted or not.
        If there's a more elegant way of writing this, please let me know.
        """
        xinv, yinv = ax.xaxis_inverted(), ax.yaxis_inverted()
        if cornerstr in ["ne", "northeast"]:
            return 1 if (is_inset or (not xinv and not yinv)) else \
                2 if (xinv and not yinv) else \
                4 if (not xinv and yinv) else 3
        elif cornerstr in ["nw", "northwest"]:
            return 2 if (is_inset or (not xinv and not yinv)) else \
                1 if (xinv and not yinv) else \
                3 if (not xinv and yinv) else 4
        elif cornerstr in ["sw", "southwest"]:
            return 3 if (is_inset or (not xinv and not yinv)) else \
                4 if (xinv and not yinv) else \
                2 if (not xinv and yinv) else 1
        elif cornerstr in ["se", "southeast"]:
            return 4 if (is_inset or (not xinv and not yinv)) else \
                3 if (xinv and not yinv) else \
                1 if (not xinv and yinv) else 2
        else:
            raise ValueError("Invalid corner provided to mkinset().")

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # type: ignore

    # The loc1 and loc2 are throwaway values, they'll be replaced later.
    default_inset_options = {"loc1": 1, "loc2": 2,
                             "edgecolor": "silver",
                             }
    # Construct options
    options = dict(default_inset_options)
    if inset_options is not None:
        options.update(inset_options)
    # Make the inset, if requested
    if show_zoom:
        _, line1, line2 = mark_inset(ax, inset_ax, **options)
        # Change the corners from which the lines are drawn.
        line1.loc1 = convert_corner(inset_ax, inset_corners[0], True)
        line1.loc2 = convert_corner(ax, parent_corners[0], False)
        line2.loc1 = convert_corner(inset_ax, inset_corners[1], True)
        line2.loc2 = convert_corner(ax, parent_corners[1], False)
    return inset_ax


# -- Plot holding area and plot properties --------------------

class PlotHoldingArea():
    """Plot holding area which holds staged spectra (as `PlotObject1D` or
    `PlotObject2D` objects) before the plot is constructed using `mkplot()`.

    Each |Axes| has its own PlotHoldingArea instance, which can be accessed
    with ``ax.pha``.

    Attributes
    ----------
    plot_objs : list
        List of `PlotObject1D` or `PlotObject2D` items which have been staged.

    colors_1d : generator object
        Yields colors one at a time from the currently active palette.

    colors_2d : generator object
        Yields selected tuples (positive, negative) of colors, taken from the
        "bright" palette.
    """

    def __init__(self) -> None:
        self.plot_objs: List = []
        # prime the color generators
        self.colors_1d = self._color_generator_1d()
        self.colors_2d = self._color_generator_2d()

    def _color_generator_1d(self) -> Iterator[str]:
        """Yields colors one at a time from the current palette."""
        yield from cycle(sns.color_palette(_current_palette))

    def _color_generator_2d(self) -> Iterator[Tuple[str, str]]:
        yield from cycle(_bright_2d)


class PlotProperties():
    """Stores properties of 1D spectra that have already been plotted using
    `mkplot()`. Only artists are stored for 2D spectra.

    Each |Axes| has its own PlotProperties instance, which can be accessed
    with ``ax.prop`` (but *only* after `mkplot()` has been called on that
    particular Axes).

    Attributes
    ----------
    hoffsets : list of float
        Horizontal offset of each plotted spectrum in ppm.
    voffsets : list of float
        Vertical offset of each plotted spectrum in data coordinates (because
        the *y*-axis of a 1D spectrum is the intensity, this is typically on
        the order of 1e5).
    colors : list of str
        List of colors used for each spectrum.
    options : list of dict
        List of options passed to |plot| for each individual spectrum.
    artists : list of Artists
        List of :class:`~matplotlib.artist.Artist` objects that have been
        plotted on the current |Axes|. For 1D spectra, these are Line2D
        objects; for 2D spectra, they are QuadContourSet objects.
    """
    def __init__(self) -> None:
        self.hoffsets: List[float] = []
        self.voffsets: List[float] = []
        self.colors: List[str] = []
        self.options: List[Dict[str, Any]] = []
        self.artists: List[Any] = []


# -- 1D low-level plotting ------------------------------------

# This is the same as matplotlib's default figure size, but we set it here
# anyway just in case the user has otherwise changed it using .matplotlibrc or
# mpl.rcParams.
default_1d_figsize = (6, 4)

def _stage1d(dataset: ds.TDataset1D,
             ax: Any = None,
             scale: float = 1,
             bounds: TBounds = "",
             dfilter: Optional[Callable[[float], bool]] = None,
             label: OS = None,
             color: OS = None,
             **kwargs: Any,
             ) -> None:
    """Stages a 1D spectrum.

    This constructs a `PlotObject1D` object from the dataset as well as any
    options passed via the keyword arguments, and adds it to the plot queue
    of the |Axes|.

    Note that the preferred way of staging spectra is to call the
    :meth:`~penguins.dataset._1D_PlotMixin.stage` method on the dataset object.

    Parameters
    ----------
    dataset : Dataset1D, Dataset1DProj, or Dataset1DProjVirtual
        Dataset to be staged.
    ax : Axes, optional
        |Axes| instance to tie the plot to. Defaults to the currently active
        Axes.
    scale : float, optional
        Value to scale the spectrum intensity by.
    bounds : str or (float, float), optional
        Region of the spectrum to plot. If given as a string, should be in the
        form ``lower..upper``, where ``lower`` and ``upper`` are both chemical
        shifts. If given as a tuple, should be in the form ``(lower, upper)``.
        Either ``lower`` or ``upper`` can be omitted (in the string) or passed
        as ``None`` (in the tuple) to avoid giving a lower or upper bound. If
        not given, the entire spectrum is plotted.
    dfilter : function :: float -> bool, optional
        Function which takes the intensity at each point as its only parameter
        and returns a boolean. If it returns True, the point is plotted, and
        if it returns False, the point is not plotted.
    label : str, optional
        Label to be used in the plot legend.
    color : str, optional
        Color to use for plotting.
    **kwargs
        Any other keyword arguments are passed as-is to |plot|.

    Returns
    -------
    None

    See Also
    --------
    penguins.dataset._1D_PlotMixin.stage : Equivalent method on `Dataset1D`,
                                           `Dataset1DProj`, and
                                           `Dataset1DProjVirtual` objects, and
                                           the preferred way to stage spectra.
    """
    # If an Axes wasn't specified, get the currently active one.
    if ax is None:
        # Before calling gca(), we check whether a figure exists yet. If it
        # doesn't, then we make a figure ourselves with a default figsize.
        if not plt.get_fignums():  # == empty list if no figures yet.
            plt.figure(figsize=default_1d_figsize)
        ax = plt.gca()
    # Create the plot holding area if it doesn't exist yet
    if not hasattr(ax, "pha"):
        try:
            ax.pha = PlotHoldingArea()
        except AttributeError as e:
            # Check if it's actually an ndarray, as one of the most common
            # mistakes is to pass an ndarray of Axes by iterating over axs
            # instead of axs.flat.
            if isinstance(ax, np.ndarray):
                raise AttributeError("'numpy.ndarray' object has no attribute"
                                     " 'pha'. Did you mean to iterate over"
                                     " axs.flat instead of axs?") from None
            else:
                raise e

    # Check that it doesn't already have 2D spectra. We can just check against
    # the first element. By induction, it is equivalent to checking against
    # every element.
    if len(ax.pha.plot_objs) != 0 and isinstance(ax.pha.plot_objs[0],
                                                 PlotObject2D):
        raise TypeError("Plot queue already contains 1D spectra.")
    # If we reached here, then it's all good and we should make the
    # PlotObject1D then append it to the PHA.
    else:
        plot_obj = PlotObject1D(dataset=dataset,
                                ax=ax,
                                scale=scale,
                                bounds=bounds,
                                dfilter=dfilter,
                                label=label,
                                color=color,
                                plot_options=kwargs)
        ax.pha.plot_objs.append(plot_obj)


class PlotObject1D():
    """Object that includes a 1D dataset as well as other keyword arguments
    passed to `_stage1d()`.

    Any processing done on the spectrum, e.g. scaling, bounds selection, or
    dfilter, is done when this class is initialised.
    """

    default_1d_plotoptions = {"linewidth": 1}
    def __init__(self,
                 dataset: ds.TDataset1D,
                 ax: Any,
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
        self._init_options(ax, plot_options, color, label)
        self.ppm_scale = self.dataset.ppm_scale(bounds=self.bounds)
        self.hz_scale = self.dataset.hz_scale(bounds=self.bounds)
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
                      ax: Any,
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
        # If a color hasn't been chosen by now, get one from the color
        # generator in PHA.
        if "color" not in options:
            next_color = next(ax.pha.colors_1d)
            options.update(color=next_color)
        # Finally, set the instance attribute
        self.options = options


def _mkplot1d(ax: Any = None,
              style: str = "1d",
              tight_layout: bool = True,
              stacked: bool = False,
              voffset: Union[Sequence, float] = 0,
              hoffset: Union[Sequence, float] = 0,
              title: OS = None,
              autolabel: str = "nucl",
              xlabel: OS = None,
              legend_loc: Any = "best",
              units: str = "ppm",
              ) -> Tuple[Any, Any]:
    """Calls |plot| on all the spectra in the plot queue. All hoffset and
    voffset calculations are performed here.

    Note that this function should never be called directly.

    Parameters
    ----------
    ax : Axes, optional
        |Axes| instance to plot the spectra on. If not provided, creates a new
        |Figure| and |Axes|.
    style : str, optional
        Plot style to use. By default this is ``1d``. For the list of plot
        styles, see `style_axes()`.
    tight_layout : bool, optional (default True)
        Whether to call plt.tight_layout() after constructing the plot.
    stacked : bool, optional
        True to make spectra tightly stacked vertically (i.e. not
        superimposed). If True, overrides any value passed in *voffset*.
    voffset : float or list of float, optional
        If given as a float, indicates the amount of vertical offset between
        spectra, in units of the maximum height. The height of a spectrum
        refers to the total width it spans in the *y*-axis, and the maximum
        height refers to the largest such height of all spectra in the plot
        queue.
        Using a float for ``voffset`` is useful for offsetting spectra by a
        constant amount. Note that stacked spectra will have a variable
        vertical offset between each spectrum, because each spectrum will have
        a different height.
        If given as a list, each staged spectrum is offset by the corresponding
        amount (again, this is given in units of maximum height).
    hoffset : float or list of float, optional
        If given as a float, indicates the horizontal offset between adjacent
        spectra. If this is positive, then successive spectra are shifted
        towards the right (the first spectrum is not shifted).  If given as a
        list, each staged spectrum is offset by the corresponding amount (in
        ppm).
    title : str, optional
        Plot title.
    autolabel : str, optional (default: "nucl")
        Automatic label to use for the *x*-axis. The only option available now
        is ``nucl`` (the default), which generates a LaTeX representation of
        the nucleus of the first staged spectrum (e.g. for a proton spectrum,
        using this would automatically generate the *x*-axis label
        ``r"$\\rm ^{1}H$ (ppm)"``).
    xlabel : str, optional
        *x*-Axis label. Overrides the autolabel parameter if given.
    legend_loc : str or (float, float), optional
        Location to place the legend. This is passed as the *loc* parameter to
        |legend|; see the documentation there for the available options.
    units : str from {"ppm", "Hz"}, default "ppm"
        Units to use for plotting. This also determines the units for the
        *hoffset* parameter: if *units* is ``ppm`` then hoffset is interpreted
        as a chemical shift offset, and vice versa.

    Returns
    -------
    fig : Figure
        The currently active |Figure| instance.
    ax : Axes
        The currently active |Axes| instance, which the spectra were just
        plotted on.

    See Also
    --------
    penguins.mkplot : The appropriate interface for plot construction.
    """
    # Flag that determines whether ax.legend is called. This will be set to
    # True if we find any PlotObject1D with a non-empty label.
    make_legend = False
    # Find the maximum height
    heights = [np.nanmax(pobj.proc_data) - np.nanmin(pobj.proc_data)
               for pobj in ax.pha.plot_objs]     # type: ignore
    max_height = max(heights)

    # Get Axes object
    if ax is None:
        ax = plt.gca()
    # Create a PlotProperties instance tied to this Axes
    ax.prop = PlotProperties()

    # Iterate over plot objects
    for n, pobj in enumerate(ax.pha.plot_objs):
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
        if units == "ppm":
            xaxis = pobj.ppm_scale - this_hoffset
        elif units == "Hz":
            xaxis = pobj.hz_scale - this_hoffset
        else:
            raise ValueError('units must be either "ppm" or "Hz".')
        l2d = ax.plot(xaxis,
                      pobj.proc_data + this_voffset,
                      **pobj.options)
        # Add heights and colors to plotproperties.
        ax.prop.hoffsets.append(this_hoffset)
        ax.prop.voffsets.append(this_voffset)
        ax.prop.colors.append(pobj.options["color"])
        ax.prop.options.append(pobj.options)
        ax.prop.artists.append(*l2d)  # plot() returns a list of Line2D

    # Figure out the x- and y-labels. xlabel will override everything if it is
    # manually specified, otherwise, use the value of autolabel.
    if xlabel is not None:
        pass  # use the given xlabel
    else:
        if autolabel == "nucl":
            xlabel = ax.pha.plot_objs[0].dataset.nuclei_to_str()
            xlabel += " (ppm)"
        else:
            raise ValueError(f"Invalid value '{autolabel}' given for "
                             "parameter autolabel.")

    # Format the plot.
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    if make_legend:
        ax.legend(loc=legend_loc)
    # Apply axis styles.
    style_axes(ax, style=style, tight_layout=tight_layout)
    return ax.figure, ax


# -- 2D low-level plotting ------------------------------------

_default_2d_figsize = (5, 5)

class Contours:
    """2D plot contours."""
    def __init__(self,
                 dataset: ds.Dataset2D,
                 ax: Any,
                 levels: TLevels = (None, None, None),
                 colors: TColors = (None, None),
                 ) -> None:
        self.dataset = dataset
        if isinstance(levels, (float, int)):
            levels = (levels, None, None)
        self.make_levels(*levels)
        self.make_colors(ax, *colors)

    def make_levels(self, base: OF = None,
                    increment: OF = None,
                    number: Optional[int] = None
                    ) -> None:
        self.base = base or self.dataset.ts_baselev
        self.increment = increment or 1.5
        self.number = number or 10

    def make_colors(self,
                    ax: Any,
                    color_positive: OS = None,
                    color_negative: OS = None
                    ) -> None:
        if color_positive is None or color_negative is None:
            # Means we need to get a color set from the PHA generator.
            next_positive, next_negative = next(ax.pha.colors_2d)
            self.color_positive = color_positive or next_positive
            self.color_negative = color_negative or next_negative
        else:
            self.color_positive = color_positive
            self.color_negative = color_negative

    def generate_contour_levels(self) -> List[float]:
        neg = [-self.base * (self.increment ** (i))
               for i in range(self.number - 1, -1, -1)]
        pos = [self.base * (self.increment ** i)
               for i in range(self.number)]
        return neg + pos

    def generate_contour_colors(self) -> List[str]:
        neg = [self.color_negative] * self.number
        pos = [self.color_positive] * self.number
        return neg + pos


class PlotObject2D():
    """Object that includes a 2D dataset as well as other keyword arguments
    passed to `_stage2d()`.

    Any processing done on the spectrum, e.g. bounds selection, or dfilter, is
    done when this class is initialised. The contour levels and colors are also
    generated by this class.
    """
    default_2d_plotoptions = {"linewidths": 0.7}
    def __init__(self,
                 dataset: ds.Dataset2D,
                 ax: Any,
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
        self.contours = Contours(self.dataset, ax, levels, colors)
        # can access cpos and cneg with self.contours.color_[positive|negative]
        self.clevels = self.contours.generate_contour_levels()
        self.ccolors = self.contours.generate_contour_colors()
        self._init_options(plot_options)
        self.label = label
        # self.options will include the colors key.
        self.f1_ppm_scale = self.dataset.ppm_scale(axis=0, bounds=self.f1_bounds)
        self.f2_ppm_scale = self.dataset.ppm_scale(axis=1, bounds=self.f2_bounds)
        self.f1_hz_scale = self.dataset.hz_scale(axis=0)
        self.f2_hz_scale = self.dataset.hz_scale(axis=1)
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


def _stage2d(dataset: ds.Dataset2D,
             ax: Any = None,
             f1_bounds: TBounds = "",
             f2_bounds: TBounds = "",
             levels: TLevels = (None, None, None),
             colors: TColors = (None, None),
             dfilter: Optional[Callable[[float], bool]] = None,
             label: OS = None,
             **kwargs: Any
             ) -> None:
    """Stages a 2D spectrum.

    This constructs a `PlotObject2D` object from the dataset as well as any
    options passed via the keyword arguments, and adds it to the plot queue
    of the |Axes|.

    Note that the preferred way of staging spectra is to call the
    :meth:`~penguins.dataset._2D_PlotMixin.stage` method on the dataset object.

    Parameters
    ----------
    dataset : Dataset1D, Dataset1DProj, or Dataset1DProjVirtual
        Dataset to be staged.
    ax : Axes, optional
        |Axes| instance to tie the plot to. Defaults to the currently active
        Axes.
    scale : float, optional
        Value to scale the spectrum intensity by.
    f1_bounds : str or (float, float), optional
        Region of the indirect chemical shifts to plot. If given as a string,
        should be in the form ``lower..upper``, where ``lower`` and ``upper``
        are both chemical shifts. If given as a tuple, should be in the form
        ``(lower, upper)``.  Either ``lower`` or ``upper`` can be omitted (in
        the string) or passed as ``None`` (in the tuple) to avoid giving a
        lower or upper bound. If not given, the entire range of indirect
        chemical shifts is plotted.
    f2_bounds : str or (float, float), optional
        Same as *f1_bounds*, but for the direct dimension.
    levels : float or (float, float, float), optional
        A tuple *(baselev, increment, nlev)* specifying the levels at which to
        draw contours. These represent, respectively, the lowest contour level,
        the multiplicative increment between adjacent contours, and the number
        of contour levels to draw. Contours are always drawn for both positive
        and negative levels.

        If any of these are None, then the default values are used: for
        *baselev* this is TopSpin's default contour level, for *increment* this
        is 1.5, and for *nlev* this is 10.  Alternatively, can be provided as a
        single float *baselev* as shorthand for *(baselev, None, None)*.
    dfilter : function :: float -> bool, optional
        Function which takes the intensity at each point as its only parameter
        and returns a boolean. If it returns True, the point is plotted, and if
        it returns False, the point is not plotted.
    label : str, optional
        Label to be used in the plot legend.
    color : (str, str), optional
        Colors to use for positive and negative contours respectively.
    **kwargs
        Any other keyword arguments are passed as-is to |contour|. Note that
        these will be overridden by other keyword arguments passed to stage().

    Returns
    -------
    None

    See Also
    --------
    _2D_PlotMixin.stage : Equivalent method on Dataset2D objects, and the
                          preferred way of staging spectra.
    """
    # If an Axes wasn't specified, get the currently active one.
    if ax is None:
        # Before calling gca(), we check whether a figure exists yet. If it
        # doesn't, then we make a figure ourselves with a default figsize.
        if not plt.get_fignums():  # == empty list if no figures yet.
            plt.figure(figsize=_default_2d_figsize)
        ax = plt.gca()
    # Create the plot holding area if it doesn't exist yet
    if not hasattr(ax, "pha"):
        try:
            ax.pha = PlotHoldingArea()
        except AttributeError as e:
            # Check if it's actually an ndarray, as one of the most common
            # mistakes is to pass an ndarray of Axes by iterating over axs
            # instead of axs.flat.
            if isinstance(ax, np.ndarray):
                raise AttributeError("'numpy.ndarray' object has no attribute"
                                     " 'pha'. Did you mean to iterate over"
                                     " axs.flat instead of axs?") from None
            else:
                raise e

    # Check that it doesn't already have 1D spectra. We can just check against
    # the first element. By induction, it is equivalent to checking against
    # every element.
    if len(ax.pha.plot_objs) != 0 and isinstance(ax.pha.plot_objs[0],
                                                 PlotObject1D):
        raise TypeError("Plot queue already contains 2D spectra.")
    # If we reached here, then it's all good and we should make the
    # PlotObject2D then append it to the PHA.
    else:
        plot_obj = PlotObject2D(dataset=dataset, ax=ax,
                                f1_bounds=f1_bounds, f2_bounds=f2_bounds,
                                levels=levels, colors=colors,
                                dfilter=dfilter, label=label,
                                plot_options=kwargs)
        ax.pha.plot_objs.append(plot_obj)


def _mkplot2d(ax: Any = None,
              style: str = "2d",
              tight_layout: bool = True,
              offset: Tuple[float, float] = (0, 0),
              title: OS = None,
              autolabel: str = "nucl",
              xlabel: OS = None,
              ylabel: OS = None,
              legend_loc: Any = "best",
              f1_units: str = "ppm",
              f2_units: str = "ppm",
              ) -> Tuple[Any, Any]:
    """Calls |contour| on all the spectra in the plot queue. All offset
    calculations are performed here.

    Note that this function should never be called directly.

    Parameters
    ----------
    ax : Axes, optional
        |Axes| instance to plot the spectra on. If not provided, creates a new
        |Figure| and |Axes|.
    style : str, optional
        Plot style to use. By default this is ``2d``. For the list of plot
        styles, see `style_axes()`.
    tight_layout : bool, optional (default True)
        Whether to call plt.tight_layout() after constructing the plot.
    offset : (float, float), optional
        Amount to offset successive spectra by in units of ppm, provided as
        *(f1_offset, f2_offset)*.
    title : str, optional
        Plot title.
    autolabel : str, optional (default: "nucl")
        Automatic label to use for the *x*-axis. The ``nucl`` option generates
        a LaTeX representation of the nuclei of the first spectrum (e.g. for a
        C–H HSQC, using this would automatically generate the *x*- and *y*-axis
        labels ``r"$\\rm ^{1}H$ (ppm)"`` and ``r"$\\rm ^{13}C$ (ppm)"``
        respectively).  The ``f1f2`` option  generates generic ``f1 (ppm)`` and
        ``f2 (ppm)`` strings. There are no other options for now.
    xlabel : str, optional
        *x*-Axis label. If either *xlabel* or *ylabel* are set, they will
        override the *autolabel* parameter (if only one is set then the other
        axis label will be empty!).
    ylabel : str, optional
        *y*-Axis label.
    legend_loc : str or (float, float), optional
        Location to place the legend. This is passed as the *loc* parameter to
        |legend|; see the documentation there for the available options.
    f1_units : str from {"ppm", "Hz"}, default "ppm"
        Units to use for the f1 dimension. This also determines the units for
        the first value of the *offset* parameter: if *units* is ``ppm`` then
        offset[0] is interpreted as a chemical shift offset, and vice versa.
    f2_units : str from {"ppm", "Hz"}, default "ppm"
        Units to use for the f2 dimension. Likewise, this determines the units
        for the second value of the *offset* parameter.

    Returns
    -------
    fig : Figure
        The currently active |Figure| instance.
    ax : Axes
        The currently active |Axes| instance, which the spectra were just
        plotted on.

    See Also
    --------
    penguins.mkplot : The appropriate interface for plot construction.
    """
    make_legend = False
    legend_colors, legend_labels = [], []
    if ax is None:
        ax = plt.gca()
    # Create a PlotProperties instance for this Axes.
    ax.prop = PlotProperties()

    # Iterate over plot objects
    for n, pobj in enumerate(ax.pha.plot_objs):
        # Figure out which x- and y-axes to use (ppm or Hz)
        if f1_units == "ppm":
            yaxis = pobj.f1_ppm_scale - (n * offset[0])
        elif f1_units == "Hz":
            yaxis = pobj.f1_hz_scale - (n * offset[0])
        else:
            raise ValueError('f1_units must be either "ppm" or "Hz".')
        if f2_units == "ppm":
            xaxis = pobj.f2_ppm_scale - (n * offset[1])
        elif f2_units == "Hz":
            xaxis = pobj.f2_hz_scale - (n * offset[1])
        else:
            raise ValueError('f2_units must be either "ppm" or "Hz".')
        # Plot it.
        cs = ax.contour(xaxis, yaxis,
                        pobj.proc_data,
                        levels=pobj.clevels,
                        **pobj.options)
        # Construct lists for plt.legend
        if pobj.label is not None:
            make_legend = True
            legend_colors.append((pobj.contours.color_positive,
                                  pobj.contours.color_negative))
            legend_labels.append(pobj.label)
        # Add the QuadContourSet object to ax.prop.artists
        ax.prop.artists.append(cs)

    # Figure out the x- and y-labels. If xlabel or ylabel are manually
    # specified, they should override autolabel. Otherwise, look at the value
    # of autolabel.
    if xlabel is not None or ylabel is not None:
        pass
    else:
        if autolabel == "nucl":
            # This is the default case, because the default parameters are
            # xlabel=None, ylabel=None, autolabel="nucl".
            xlabel = ax.pha.plot_objs[0].dataset.nuclei_to_str()[1]
            xlabel += f" ({f2_units})"
            ylabel = ax.pha.plot_objs[0].dataset.nuclei_to_str()[0]
            ylabel += f" ({f1_units})"
        elif autolabel == "f1f2":
            xlabel = f"$f_2$ ({f2_units})"
            ylabel = f"$f_1$ ({f1_units})"
        else:
            raise ValueError(f"Invalid value '{autolabel}' given for "
                             "parameter autolabel.")

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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Apply other styles.
    style_axes(ax, style=style, tight_layout=tight_layout)

    # Make legend. This part is not easy...
    # See https://stackoverflow.com/questions/41752309/ for an example
    if make_legend:
        plt.legend(legend_colors, legend_labels,
                   handler_map={tuple: ContourLegendHandler()},
                   loc=legend_loc)
    return ax.figure, ax


def _find_baselev(dataset: ds.Dataset2D,
                  increment: float = None,
                  nlev: int = 4,
                  ) -> float:
    """Create an interactive slider window to see the effect of changing
    *baselev* on the displayed spectrum.

    Note that this should not be called directly; use the
    :meth:`~penguins._2D_PlotMixin.find_baselev` method on `Dataset2D` objects
    instead.

    Parameters
    ----------
    dataset : Dataset2D
        Dataset to stage.
    increment : float, optional
        Multiplicative increment between adjacent contour levels. This is
        typically not worth changing.
    nlev : int, optional
        Number of contour levels to use. This is not typically worth changing.
        If you really want to, don't go too high, as this causes the plot to be
        very laggy. (It has to call |contour| every time the slider is moved.)

    Returns
    -------
    chosen_baselev : float
        Final value of the baselev before the 'OK' button was pressed.

    See Also
    --------
    penguins.dataset._2D_PlotMixin.find_baselev : Equivalent method on
                                                  `Dataset2D` objects. Usage
                                                  of this is preferred.
    """
    # Choose contour levels. We reduce nlev to 4 by default so that the
    # plotting is faster -- otherwise it's super laggy. We try to cover the
    # same dynamic range as 1.5 ** 10 by default, unless the user specified
    # an increment.
    initial_baselev = dataset.ts_baselev
    increment = increment or (1.5 ** 10) ** (1 / nlev)
    initial_clev = (initial_baselev, increment, nlev)
    # Maximum level of the slider should be the greatest intensity of the
    # spectrum. There's no point going above that.
    max_baselev = np.max(np.abs(dataset.rr))
    # Minimum level is ~100.
    min_baselev = 100

    # Plot the spectrum on the top portion of the figure.
    fig, plot_axes = plt.subplots()
    dataset.stage(plot_axes, levels=initial_clev)
    mkplot(plot_axes, empty_pha=False)
    orig_xlim = plot_axes.get_xlim()
    orig_ylim = plot_axes.get_ylim()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Generate a logarithmic slider.
    from matplotlib.widgets import Slider, Button     # type: ignore
    baselev_axes = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor="lavender")
    baselev_slider = Slider(baselev_axes,
                            "",
                            np.log10(min_baselev),
                            np.log10(max_baselev),
                            valinit=np.log10(initial_baselev),
                            color="purple")
    # Add some text
    baselev_axes.text(0.5, -1.2, r"log$_{10}$(base contour level)",
                      horizontalalignment="center",
                      transform=baselev_axes.transAxes)

    # Define the behaviour when redrawn
    def redraw(plot_axes: Any,
               val: float
               ) -> None:
        # Update the internal Contours object
        pobj = plot_axes.pha.plot_objs[0]
        pobj.contours.base = 10 ** val
        # Regenerate the contours
        pobj.clevels = pobj.contours.generate_contour_levels()
        # Replot
        xlim = plot_axes.get_xlim()
        ylim = plot_axes.get_ylim()
        plot_axes.cla()
        plt.sca(plot_axes)
        mkplot(plot_axes, empty_pha=False, style="natural")
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
    show()

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
    plt.close("all")

    return fval if okay else None


# -- Matplotlib and Seaborn wrappers --------------------------

@export
def subplots(*args, **kwargs) -> None:
    """Direct wrapper around |subplots|.
    """
    return plt.subplots(*args, **kwargs)


@export
def subplots2d(nrows: int = 1,
               ncols: int = 1,
               height_ratios: Optional[List[float]] = None,
               width_ratios: Optional[List[float]] = None,
               **kwargs
               ) -> Tuple[Any, Any]:
    """Wrapper around matplotlib's |subplots| function, which (in addition to
    performing everything |subplots| does) also sets the figure size to
    ``(4 * ncols)`` by ``(4 * nrows)`` by default. This means that every
    subplot will have an area of 4 inches by 4 inches, which is a good size for
    2D spectra.

    Also accepts the gridspec keyword arguments ``height_ratios`` and
    ``width_ratios``. The effect of this is that each subplot will have a
    height equal to the height_ratio times 4, i.e. if one specifies
    ``height_ratios = [0.5, 1]`` then the top row will be 2 inches tall and the
    bottom row 4 inches tall.

    Parameters
    ----------
    nrows : int, optional
        Number of rows.
    ncols : int, optional
        Number of columns.
    height_ratios : list of float, optional
        Ratios of Axes heights.
    width_ratios : list of float, optional
        Ratios of Axes widths.
    kwargs : dict, optional
        Other keyword arguments passed to |subplots|. Note that if either
        *height_ratios* or *width_ratios* are passed, they will override both
        the *figsize* as well as *gridspec_kw* keyword arguments.

    Returns
    -------
    fig : Figure
        |Figure| instance corresponding to the current plot.
    axs : Axes, or ndarray of Axes
        If only one subplot was requested, this is the |Axes| instance.
        Otherwise this is an |ndarray| of |Axes|, one for each subplot. See
        the documentation of |subplots| for further explanation.
    """
    # Empty dictionary, we will fill this as we go along.
    gridspec_kw = {}
    # Determine figure width and figure height. First check if either
    # width_ratios or height_ratios were given: if so, use that to calculate
    # the height.
    if width_ratios is not None or height_ratios is not None:
        # If width_ratios or height_ratios were passed as keyword arguments, we
        # should ignore the gridspec_kw keyword argument.
        kwargs.pop("gridspec_kw", None)
        # Error out if either of the lists don't match the number of
        # columns/rows.
        if width_ratios is not None and len(width_ratios) != ncols:
            raise ValueError("width_ratios must have the same number of"
                             " elements as there are columns.")
        if height_ratios is not None and len(height_ratios) != nrows:
            raise ValueError("height_ratios must have the same number of"
                             " elements as there are rows.")
        # Calculate the width/height.
        if width_ratios is not None:
            fig_width = np.sum(width_ratios) * 4
            gridspec_kw["width_ratios"] = width_ratios
        else:
            fig_width = ncols * 4
        if height_ratios is not None:
            fig_height = np.sum(height_ratios) * 4
            gridspec_kw["height_ratios"] = height_ratios
        else:
            fig_height = nrows * 4
        figsize = (fig_width, fig_height)
        # Remove figsize from kwargs if it's there.
        kwargs.pop("figsize", None)
    # Otherwise, if figsize is there, just use it.
    elif "figsize" in kwargs:
        figsize = kwargs.pop("figsize")
    # Otherwise, use nrows and ncols * 4.
    else:
        figsize = (ncols * 4, nrows * 4)
    return plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=figsize,
                        gridspec_kw=gridspec_kw,
                        **kwargs)


@export
def figure(*args, **kwargs) -> Any:
    """Wrapper around matplotlib's |figure| function.

    If *figsize* is not passed as a keyword argument, then it is chosen to be
    (4, 4) by default.

    Parameters
    ----------
    kwargs : dict, optional
        Other keyword arguments passed to |subplots|.

    Returns
    -------
    fig : Figure
        Newly created |Figure| instance.
    """
    if "figsize" not in kwargs:
        kwargs["figsize"] = (4, 4)
    return plt.figure(**kwargs)


@export
def set_palette(palette: Union[str, List[str]],
                ) -> None:
    """Sets the currently active color palette. The default palette is
    seaborn's ``deep``.

    The palette is used both for staging successive 1D spectra, as well as for
    any plots done with seaborn. For 2D spectra, colors from seaborn's
    ``bright`` palette have been manually chosen. If you want to override these,
    you should directly pass the *colors* parameter to the stage() method.

    Parameters
    ----------
    palette : str or list of str
        Color palette to use. See :std:doc:`tutorial/color_palettes` for a
        full description of the possible options.

    Returns
    -------
    None
    """
    global _current_palette
    # Change seaborn palette, in case user wants to draw other plots.
    sns.set_palette(palette)
    # Change penguins 1D palette.
    _current_palette = palette


@export
def color_palette(palette: Optional[Union[str, List[str]]] = None,
                  ) -> List[str]:
    """Returns a list of colors corresponding to a color palette. If *palette*
    is not provided, returns the colors in the current color palette.

    This is essentially a wrapper around :func:`sns.color_palette
    <seaborn.color_palette>`, but it only offers one argument, and it can't be
    used as a context manager. Use `set_palette` if you want to change the
    active palette.

    Parameters
    ----------
    palette : str or list of str, optional
        The palette to look up. Defaults to the currently active color palette.

    Returns
    -------
    colors : list of str
        The colors in the current color palette.
    """
    if palette is None:
        palette = _current_palette
    return sns.color_palette(palette)


@export
def tight_layout(*args, **kwargs) -> None:
    """Direct wrapper around |tight_layout|.
    """
    return plt.tight_layout(*args, **kwargs)


@export
def show(*args, **kwargs) -> None:
    """Direct wrapper around |show|.
    """
    return plt.show(*args, **kwargs)


@export
def savefig(*args, **kwargs) -> None:
    """Direct wrapper around |savefig|.
    """
    return plt.savefig(*args, **kwargs)


@export
def pause(*args, **kwargs) -> None:
    """Direct wrapper around |pause|.
    """
    return plt.pause(*args, **kwargs)
