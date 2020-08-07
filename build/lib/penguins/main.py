from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Tuple, Any

import matplotlib.pyplot as plt    # type: ignore

from . import dataset as ds
from . import pgplot
from .pgplot import (get_pha, get_properties,
                     style_axes)


# -- READ -----------------------------------------------

def read(path: Union[str, Path],
         expno: int,
         procno: int) -> ds.TDatasetnD:
    """
    Creates a Dataset object from the spectrum name folder, expno, and procno.
    """
    p = Path(path) / str(expno) / "pdata" / str(procno)
    return read_abs(p)


def read_abs(path: Union[str, Path]
             ) -> ds.TDatasetnD:
    """
    Creates a Dataset object from the spectrum procno folder.
    """
    p = Path(path)
    # Figure out which type of spectrum it is.
    if not (p / "procs").exists() or not (p.parents[1] / "acqus").exists():
        raise ValueError(f"Invalid path to spectrum {p}: procs or acqus not found")
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


# -- PLOT ----------------------------------------------

def mkplot(ax: Any = None,
           figsize: Optional[Tuple[float, float]] = None,
           close: bool = True,
           empty_pha: bool = True,
           **kwargs):
    """
    Delegates to _mkplot1d() or _mkplot2d() as necessary.
    """
    # Close open figures, *unless* an axes was given, in which case
    # we assume that the user isn't keen on having that closed!
    # Useful e.g. when doing subplots.
    if close and ax is None:
        plt.close("all")
    # Reset plot properties
    pgplot._reset_properties()

    PHA = get_pha()
    if len(PHA.plot_queue) == 0:
        raise ValueError("No spectra have been staged yet.")
    else:
        if close and figsize is not None and ax is None:
            plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.gca()
        if isinstance(PHA.plot_queue[0], pgplot.PlotObject1D):
            fig, ax = pgplot._mkplot1d(PHA, ax=ax, **kwargs)
        elif isinstance(PHA.plot_queue[0], pgplot.PlotObject2D):
            fig, ax = pgplot._mkplot2d(PHA, ax=ax, **kwargs)
        else:
            raise TypeError("Plot holding area has invalid entries.")
    # Reset the PHA to being empty
    if empty_pha:
        pgplot._reset_pha()
    return (fig, ax)


def mkinset(pos: Tuple[float, float],
            size: Tuple[float, float],
            ax: Any = None,
            parent_corners: Tuple[str, str] = ("sw", "se"),
            inset_corners: Tuple[str, str] = ("sw", "se"),
            transform: Any = None,
            plot_options: Optional[dict] = None,
            inset_options: Optional[dict] = None,
            show_zoom: bool = True,
            ) -> Any:
    """
    Constructs an inset plot, abstracting away the matplotlib interface,
    which is especially clunky with inverted axes.
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


def show(*args, **kwargs) -> None:
    return plt.show(*args, **kwargs)


def savefig(*args, **kwargs) -> None:
    return plt.savefig(*args, **kwargs)


def pause(*args, **kwargs) -> None:
    return plt.pause(*args, **kwargs)


def subplots(nrows=1, ncols=1, *args, **kwargs) -> Tuple[Any, Any]:
    # This implementation captures nrows and ncols so that we can set figsize
    # automatically. We don't care about the rest of the arguments, so those
    # can just be passed on directly.
    if "figsize" not in kwargs:
        kwargs["figsize"] = (ncols * 4, nrows * 4)
    return plt.subplots(nrows=nrows, ncols=ncols, *args, **kwargs)
