import os
from pathlib import Path

from .main import *
from ._version import __version__


# These are pure convenience routines for my personal use.
# The average user should never use these.
# Default save location for plots
_dsl = Path("/Users/yongrenjie/Desktop/a_plot.png")
# Path to NMR spectra. The $nmrd environment variable should resolve to
# .../dphil/expn/nmr. On my Mac this is set to my SSD.
def __getenv(key):
    if os.getenv(key) is not None:
        x = Path(os.getenv(key))
        if x.exists():
            return x
    raise FileNotFoundError("$nmrd does not point to a valid location.")
_nmrd = __getenv("nmrd")
