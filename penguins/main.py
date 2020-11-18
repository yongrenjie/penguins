from __future__ import annotations

import warnings
from pathlib import Path
from itertools import zip_longest
from typing import Union, Optional, Tuple, Sequence, Any

import numpy as np   # type: ignore
import matplotlib.pyplot as plt    # type: ignore
from matplotlib.ticker import AutoMinorLocator  # type: ignore

from .type_aliases import *
from . import dataset as ds
from . import pgplot
from .pgplot import (set_palette, color_palette)


# -- READING --------------------------------------------

def read(path: Union[str, Path],
         expno: int,
         procno: int = 1) -> ds.TDatasetnD:
    """Create a Dataset object from a spectrum folder, expno, and procno.

    The subclass of Dataset returned is determined by what files are available
    in the spectrum folder. It can be either `Dataset1D`, `Dataset1DProj`, or
    `Dataset2D`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the spectrum name folder.
    expno : int
        Expno of experiment of interest.
    procno : int (optional)
        Procno of processed data. Defaults to 1.

    Returns
    -------
    Dataset
        A `Dataset1D`, `Dataset1DProj`, or `Dataset2D` object depending on the
        detected spectrum dimensionality.
    """
    p = Path(path) / str(expno) / "pdata" / str(procno)
    return read_abs(p)


def read_abs(path: Union[str, Path]
             ) -> ds.TDatasetnD:
    """Create a Dataset object directly from a procno folder.

    There is likely no reason to ever use this because `read()` is far easier
    to use, especially if you are plotting multiple spectra with different
    expnos from the same folder.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the procno folder.

    Returns
    -------
    Dataset
        A `Dataset1D`, `Dataset1DProj`, or `Dataset2D` object depending on the
        detected spectrum dimensionality.

    See Also
    --------
    read : The preferred interface for importing datasets.
    """
    p = Path(path)
    # Figure out which type of spectrum it is.
    if not (p / "procs").exists() or not (p.parents[1] / "acqus").exists():
        raise ValueError(f"Invalid path to spectrum {p}:"
                         " procs or acqus not found")
    if (p.parents[1] / "ser").exists() and (p / "2rr").exists():
        return ds.Dataset2D(p)
    elif (p / "1r").exists() and (p / "used_from").exists():
        try:
            return ds.Dataset1DProj(p)
        # For processed pure shift data, Dataset1DProj will raise a ValueError
        # as it cannot find the projection dimension in the 'used_from' file.
        # The structure of acqus and procs files are more similar to a standard
        # 1D spectrum, so we can simply use Dataset1D instead.
        except ValueError:
            return ds.Dataset1D(p)
    elif (p.parents[1] / "fid").exists() and (p / "1r").exists():
        return ds.Dataset1D(p)
    else:
        raise ValueError(f"Invalid path to spectrum {p}: data files not found")


# -- PLOTTING ------------------------------------------

def subplots(nrows: int = 1,
             ncols: int = 1,
             **kwargs
             ) -> Tuple[Any, Any]:
    """Wrapper around matplotlib's |subplots| function.

    If *figsize* is not passed as a keyword argument, then this by default
    sets the figure size to be ``(4 * ncols)`` by ``(4 * nrows)``. This means
    that every subplot will have an area of 4 inches by 4 inches.

    Parameters
    ----------
    nrows : int, optional
        Number of rows.
    ncols : int, optional
        Number of columns.
    kwargs : dict, optional
        Other keyword arguments passed to |subplots|.

    Returns
    -------
    fig : Figure
        |Figure| instance corresponding to the current plot.
    axs : Axes, or ndarray of Axes
        If only one subplot was requested, this is the |Axes| instance.
        Otherwise this is an |ndarray| of |Axes|, one for each subplot. See
        the documentation of |subplots| for further explanation.
    """
    # This implementation captures nrows and ncols so that we can set figsize
    # automatically. We don't care about the rest of the arguments, so those
    # can just be passed on directly.
    if "figsize" not in kwargs:
        kwargs["figsize"] = (ncols * 4, nrows * 4)
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


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
            ax.prop = pgplot.PlotProperties()
            if isinstance(ax.pha.plot_objs[0], pgplot.PlotObject1D):
                fig, ax = pgplot._mkplot1d(ax=ax, **kwargs)
            elif isinstance(ax.pha.plot_objs[0], pgplot.PlotObject2D):
                fig, ax = pgplot._mkplot2d(ax=ax, **kwargs)
            else:
                raise TypeError("Plot holding area has invalid elements.")
    finally:
        # Reset the PHA to being empty
        if ax is not None and empty_pha:
            ax.pha = pgplot.PlotHoldingArea()
    return fig, ax


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


def tight_layout(*args, **kwargs) -> None:
    """Direct wrapper around |tight_layout|.
    """
    return plt.tight_layout(*args, **kwargs)


def show(*args, **kwargs) -> None:
    """Direct wrapper around |show|.
    """
    return plt.show(*args, **kwargs)


def savefig(*args, **kwargs) -> None:
    """Direct wrapper around |savefig|.
    """
    return plt.savefig(*args, **kwargs)


def pause(*args, **kwargs) -> None:
    """Direct wrapper around |pause|.
    """
    return plt.pause(*args, **kwargs)



# -- PLOTTING UTILITIES --------------------------------

# Arguably these should be in another file? But whatever it is, I'm pretty
# sure that we want these in the main penguins namespace.


def style_axes(ax: Any,
               style: str,
               ) -> None:
    """Styles the given |Axes| instance according to the requested style.

    This is useful for making sure that all subplots in a series have a uniform
    appearance.

    All the styles except ``natural`` call |tight_layout| after they are done.

    Parameters
    ----------
    ax : Axes
        |Axes| instance to style.
    style : str
        Style to be applied. The available options are ``1d``, ``1d_box``,
        ``2d``, ``plot``, and ``natural``.

    Returns
    -------
    None
    """
    def disable_y_axis(ax):
        ax.yaxis.set_visible(False)

    def remove_top_left_right_spines(ax):
        for s in ["top", "left", "right"]:
            ax.spines[s].set_visible(False)

    def set_xaxis_ticks(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)

    def set_xyaxis_ticks(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)

    def thicken_spines(ax):
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_visible(True)
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)

    if style == "1d":
        disable_y_axis(ax)
        set_xaxis_ticks(ax)
        thicken_spines(ax)
        remove_top_left_right_spines(ax)
        tight_layout()
    elif style == "1d_box":
        disable_y_axis(ax)
        set_xaxis_ticks(ax)
        thicken_spines(ax)
        tight_layout()
    elif style == "2d":
        thicken_spines(ax)
        set_xyaxis_ticks(ax)
        tight_layout()
    elif style == "plot":
        thicken_spines(ax)
        tight_layout()
    elif style == "natural":
        pass
    else:
        warnings.warn(f"Invalid style '{style}' requested.")


def cleanup_axes() -> None:
    # Need to draw the figure to get the renderer.
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Iterate over axes and check which ticks overlap with axes label.
    for ax in fig.axes:
        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=r)
        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer=r)
        # Just check every bbox.
        for xtick in ax.xaxis.get_major_ticks():
            xtick_bbox1 = xtick.label1.get_window_extent(renderer=r)
            xtick_bbox2 = xtick.label2.get_window_extent(renderer=r)
            if xtick_bbox1.overlaps(xlabel_bbox):
                xtick.label1.set_visible(False)
            if xtick_bbox2.overlaps(xlabel_bbox):
                xtick.label2.set_visible(False)
        for ytick in ax.yaxis.get_major_ticks():
            ytick_bbox1 = ytick.label1.get_window_extent(renderer=r)
            ytick_bbox2 = ytick.label2.get_window_extent(renderer=r)
            if ytick_bbox1.overlaps(ylabel_bbox):
                ytick.label1.set_visible(False)
            if ytick_bbox2.overlaps(ylabel_bbox):
                ytick.label2.set_visible(False)


def cleanup_figure(padding: float = 0.02
                   ) -> None:
    # Resize subplots so that their titles don't clash with the figure legend.
    tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Get minimum bbox y-extent of figure legend(s).
    inv = fig.transFigure.inverted()
    legend_bboxs = [inv.transform(legend.get_window_extent(renderer=r))
                    for legend in fig.legends]
    legend_miny = min(bbox[0][1] for bbox in legend_bboxs)
    # Find maximum bbox y-extent of axes titles.
    titles = [ax.title for ax in fig.axes]
    title_bboxs = [inv.transform(title.get_window_extent(renderer=r))
                   for title in titles]
    offending_title_bboxs = [title_bbox for title_bbox in title_bboxs
                             if title_bbox[1][1] > legend_miny]
    # If there are no offending bboxes, then we can skip ahead.
    if offending_title_bboxs == []:
        axes_maxy = legend_miny - padding
    # Otherwise, we need to find which of them is the largest.
    else:
        max_offending_height = max(bbox[1][1] - bbox[0][1]
                                   for bbox in offending_title_bboxs)
        axes_maxy = legend_miny - padding - max_offending_height
    # Resize
    plt.subplots_adjust(top=axes_maxy)


def move_ylabel(ax: Any,
                pos: str,
                remove_ticks: int = 0,
                tight_layout: bool = True,
                ) -> None:
    if pos == "topright":
        # move yticks to right
        ax.yaxis.tick_right()
        # remove the first remove_ticks ticks within the ylims
        max, min = ax.get_ylim()
        for ytick in ax.yaxis.get_major_ticks():
            if remove_ticks == 0:
                break
            else:
                ypos = ytick.label2.get_position()[1]
                if min < ypos and ypos < max:
                    ytick.label2.set_visible(False)
                    remove_ticks -= 1
        # Move the label
        ax.yaxis.label.set_rotation(0)  # right way up
        ax.yaxis.label.set_horizontalalignment("left")
        ax.yaxis.label.set_verticalalignment("top")
        ax.yaxis.set_label_coords(1.03, 1)
    else:
        raise ValueError(f"Invalid position '{pos}' provided.")

    if tight_layout:
        plt.tight_layout()
