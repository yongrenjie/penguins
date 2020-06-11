from __future__ import annotations

from pathlib import Path
from typing import Union

from . import dataset as ds
from .pgplot import *

# sp.read() is the main entry point for users.
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
