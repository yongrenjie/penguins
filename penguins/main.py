from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Tuple

import matplotlib.pyplot as plt    # type: ignore

from . import dataset as ds
from . import pgplot
from .pgplot import get_pha, get_properties


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
    elif (p / "1r").exists() and not (p / "1i").exists() \
        and (p / "used_from").exists():
        return ds.Dataset1DProj(p)
    elif (p.parents[1] / "fid").exists() and (p / "1r").exists():
        return ds.Dataset1D(p)
    else:
        raise ValueError(f"Invalid path to spectrum {p}: data files not found")


# -- PLOT ----------------------------------------------

def mkplot(axes: Any = None,
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
    if close and axes is None:
        plt.close("all")
    # Reset plot properties
    pgplot._reset_properties()

    # If axis is provided, set it to the current one
    # plt.plot() and plt.contour() will just use the current axes
    if axes is not None:
        plt.sca(axes)

    PHA = get_pha()
    if len(PHA.plot_queue) == 0:
        raise ValueError("No spectra have been staged yet.")
    else:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if isinstance(PHA.plot_queue[0], pgplot.PlotObject1D):
            fig, ax = pgplot._mkplot1d(PHA, **kwargs)
        elif isinstance(PHA.plot_queue[0], pgplot.PlotObject2D):
            fig, ax = pgplot._mkplot2d(PHA, **kwargs)
        else:
            raise TypeError("Plot holding area has invalid entries.")
    # Reset the PHA to being empty
    if empty_pha:
        pgplot._reset_pha()
    return (fig, ax)


def show(*args, **kwargs):
    return plt.show(*args, **kwargs)


def savefig(*args, **kwargs):
    return plt.savefig(*args, **kwargs)


def pause(*args, **kwargs):
    return plt.pause(*args, **kwargs)


def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs)
