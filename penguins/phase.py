"""
phase.py
--------

Experimental module for phase correction of spectra.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from . import dataset as ds
from .exportdeco import export
from .type_aliases import *
from .pgplot import *


DEG_TO_RAD = np.pi / 180


def get_corr_fid(ds: ds.Dataset1D) -> (np.ndarray, float):
    """
    Returns a FID where the group delay has been shifted to the end, plus a
    second number indicating the 'extra' first-order phase correction required,
    in units of degrees.
    """
    grpdly = ds['grpdly']
    shift_points = int(np.around(grpdly))
    return np.roll(ds.fid, -shift_points), 360 * (grpdly - shift_points)


def phase_1d_manual(ds: ds.Dataset1D,
                    phc0: float,
                    phc1: float) -> None:
    """
    Performs in-place phase correction on a 1D dataset, i.e. it doesn't create
    and return a new dataset! This overwrites the private '_real' and '_imag'
    attributes on the dataset.

    phc0 and phc1 should be given in units of degrees. Each point in the
    spectrum is multiplied by the phase factor exp(ix), where the phase x =
    phc0 + k * phc1, with k ranging from 0 at the left end of the spectrum
    (downfield / high shift) to 1 at the right end of the spectrum (upfield /
    low shift). This matches the order in which the ndarrays in the dataset are
    stored.
    """
    complex_spec = ds.real + (1j * ds.imag)
    si = ds['si']
    phases = DEG_TO_RAD * (phc0 + phc1 * np.linspace(0, 1, si))
    phased_spec = complex_spec * np.exp(1j * phases)
    ds._real = phased_spec.real
    ds._imag = phased_spec.imag


def phase_1d_interactive(ds: ds.Dataset1D) -> (ds.Dataset1D, float, float):
    """
    Opens a matplotlib slider for interactive phase correction.

    Returns the final dataset as well as the values of phc0 and phc1.
    """
    from copy import deepcopy
    plot_ds = deepcopy(ds)

    fig, plot_axes = plt.subplots()
    plot_ds.stage(plot_axes)
    mkplot(plot_axes, tight_layout=False)
    orig_xlim = plot_axes.get_xlim()
    orig_ylim = plot_axes.get_ylim()
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Generate sliders
    from matplotlib.widgets import Slider, Button     # type: ignore
    phc0_axes = plt.axes([0.17, 0.15, 0.73, 0.03], facecolor="lavender")
    phc0_slider = Slider(phc0_axes, "", -180, +180, 0, color="purple")
    phc0_axes.text(-0.02, 0.5, "phc0", transform=phc0_axes.transAxes,
                   horizontalalignment="right",
                   verticalalignment="center")

    phc1_axes = plt.axes([0.17, 0.1, 0.73, 0.03], facecolor="lavender")
    phc1_slider = Slider(phc1_axes, "", -100, +100, 0, color="purple")
    phc1_axes.text(-0.02, 0.5, "phc1", transform=phc1_axes.transAxes,
                   horizontalalignment="right",
                   verticalalignment="center")

    # Define the behaviour when redrawn
    def redraw() -> None:
        nonlocal plot_ds
        phc0 = phc0_slider.val
        phc1 = phc1_slider.val
        plot_ds = deepcopy(ds)
        phase_1d_manual(plot_ds, phc0, phc1)
        # Replot
        xlim = plot_axes.get_xlim()
        ylim = plot_axes.get_ylim()
        plot_axes.cla()
        plot_ds.stage(plot_axes)
        mkplot(plot_axes, tight_layout=False)
        plot_axes.set_xlim(xlim)
        plot_axes.set_ylim(ylim)
    phc0_slider.on_changed(lambda val: redraw())
    phc1_slider.on_changed(lambda val: redraw())

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

    plt.show()

    # Return info if OK was hit
    if okay:
        return plot_ds, phc0_slider.val, phc1_slider.val


def phase_2d_manual(ds: ds.Dataset2D,
                    f1_phc0: float,
                    f1_phc1: float,
                    f2_phc0: float,
                    f2_phc1: float) -> None:
    """
    Performs manual 2D phasing.

    Just like `phase_1d_manual`, this doesn't create a new dataset, it modifies
    the existing dataset in place!
    """
    si = ds['si']
    # F1 phasing
    # Form complex pairs. Note that phasing along F1 should not affect the
    # "real/imag" parts along the F2 dimension.
    real_in_f2 = ds.rr + 1j * ds.ri
    imag_in_f2 = ds.ir + 1j * ds.ii
    # Apply phases
    phases_f1 = DEG_TO_RAD * (f1_phc0 + f1_phc1 * np.linspace(0, 1, si[0]))
    phases_f1 = phases_f1[:, np.newaxis]
    phased_real_in_f2 = real_in_f2 * np.exp(1j * phases_f1)
    phased_imag_in_f2 = imag_in_f2 * np.exp(1j * phases_f1)
    # Separate complex pairs
    ds._rr = phased_real_in_f2.real
    ds._ri = phased_real_in_f2.imag
    ds._ir = phased_imag_in_f2.real
    ds._ii = phased_imag_in_f2.imag

    # F2 phasing, follows the same principles
    real_in_f1 = ds.rr + 1j * ds.ir
    imag_in_f1 = ds.ri + 1j * ds.ii
    phases_f2 = DEG_TO_RAD * (f2_phc0 + f2_phc1 * np.linspace(0, 1, si[1]))
    phases_f2 = phases_f2[np.newaxis, :]
    phased_real_in_f1 = real_in_f1 * np.exp(1j * phases_f2)
    phased_imag_in_f1 = imag_in_f1 * np.exp(1j * phases_f2)
    ds._rr = phased_real_in_f1.real
    ds._ir = phased_real_in_f1.imag
    ds._ri = phased_imag_in_f1.real
    ds._ii = phased_imag_in_f1.imag


def phase_2d_interactive(ds: ds.Dataset2D) -> (ds.Dataset2D,
                                               float, float,
                                               float, float):
    """
    Opens a matplotlib slider for interactive phase correction.

    Returns the final dataset as well as the values of f1_phc0, f1_phc1,
    f2_phc0, and f2_phc1.
    """
    from copy import deepcopy
    plot_ds = deepcopy(ds)

    # contour levels, choose just 5 contours so that it's snappier
    baselev = ds.ts_baselev
    incr = 1.5
    nlev = 4

    fig, plot_axes = plt.subplots()
    plot_ds.stage(plot_axes, levels=(baselev, incr, nlev))
    mkplot(plot_axes, tight_layout=False)
    orig_xlim = plot_axes.get_xlim()
    orig_ylim = plot_axes.get_ylim()
    plt.subplots_adjust(left=0.3, bottom=0.3)

    # Generate sliders, that's a lot of boilerplate...
    from matplotlib.widgets import Slider, Button     # type: ignore
    f1_phc0_axes = plt.axes([0.10, 0.27, 0.03, 0.63], facecolor="lavender")
    f1_phc0_slider = Slider(f1_phc0_axes, "", -180, +180, 0, color="purple",
                            orientation='vertical')
    f1_phc0_axes.text(0.5, -0.10, "f1_phc0", transform=f1_phc0_axes.transAxes,
                      horizontalalignment="right",
                      verticalalignment="center")
    f1_phc1_axes = plt.axes([0.15, 0.27, 0.03, 0.63], facecolor="lavender")
    f1_phc1_slider = Slider(f1_phc1_axes, "", -100, +100, 0, color="purple",
                            orientation='vertical')
    f1_phc1_axes.text(0.5, -0.10, "f1_phc1", transform=f1_phc1_axes.transAxes,
                      horizontalalignment="left",
                      verticalalignment="center")
    f2_phc0_axes = plt.axes([0.17, 0.15, 0.73, 0.03], facecolor="lavender")
    f2_phc0_slider = Slider(f2_phc0_axes, "", -180, +180, 0, color="purple")
    f2_phc0_axes.text(-0.02, 0.5, "f2_phc0", transform=f2_phc0_axes.transAxes,
                      horizontalalignment="right",
                      verticalalignment="center")
    f2_phc1_axes = plt.axes([0.17, 0.10, 0.73, 0.03], facecolor="lavender")
    f2_phc1_slider = Slider(f2_phc1_axes, "", -100, +100, 0, color="purple")
    f2_phc1_axes.text(-0.02, 0.5, "f2_phc1", transform=f2_phc1_axes.transAxes,
                      horizontalalignment="right",
                      verticalalignment="center")

    # Define the behaviour when redrawn
    def redraw() -> None:
        nonlocal plot_ds
        plot_ds = deepcopy(ds)
        phase_2d_manual(plot_ds,
                        f1_phc0_slider.val, f1_phc1_slider.val,
                        f2_phc0_slider.val, f2_phc1_slider.val)
        # Replot
        xlim = plot_axes.get_xlim()
        ylim = plot_axes.get_ylim()
        plot_axes.cla()
        plot_ds.stage(plot_axes, levels=(baselev, incr, nlev))
        mkplot(plot_axes, tight_layout=False)
        plot_axes.set_xlim(xlim)
        plot_axes.set_ylim(ylim)
    f1_phc0_slider.on_changed(lambda val: redraw())
    f1_phc1_slider.on_changed(lambda val: redraw())
    f2_phc0_slider.on_changed(lambda val: redraw())
    f2_phc1_slider.on_changed(lambda val: redraw())

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

    plt.show()

    # Return info if OK was hit
    if okay:
        return (plot_ds,
                f1_phc0_slider.val, f1_phc1_slider.val,
                f2_phc0_slider.val, f2_phc1_slider.val)
