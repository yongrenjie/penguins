from pathlib import Path

import pytest
import numpy as np
import penguins as pg
import seaborn as sns
import matplotlib.pyplot as plt

datadir = Path(__file__).parents[1].resolve() / "penguins-testdata"


def test_1d_stage():
    """Tests the stage() method on 1D datasets."""
    fig, (empty_ax, plot_ax) = pg.subplots2d(1, 2) 
    proton = pg.read(datadir, 1)
    proton.stage(ax=plot_ax)
    # Check that the PHA was created on the right axes
    assert not hasattr(empty_ax, "pha")
    assert len(plot_ax.pha.plot_objs) == 1
    # Check the properties of the plot object
    po = plot_ax.pha.plot_objs[0]
    assert po.dataset is proton
    assert po.scale == 1
    assert po.bounds == ""
    assert np.allclose(po.ppm_scale, proton.ppm_scale())
    assert po.options["linewidth"] == 1
    assert po.options["color"] == sns.color_palette("deep")[0]
    # It would be horribly boring to test every parameter to stage()
    # individually...
    # Stage a second dataset (well, the same thing)
    proton2 = pg.read(datadir, 1)
    proton2.stage(ax=plot_ax, scale=2, linestyle="--", label="again")
    assert len(plot_ax.pha.plot_objs) == 2
    po2 = plot_ax.pha.plot_objs[1]
    assert np.allclose(po2.proc_data, po.proc_data * 2)
    assert np.allclose(po2.ppm_scale, po.ppm_scale)
    assert po2.options["linestyle"] == "--"
    assert po2.options["color"] == sns.color_palette("deep")[1]
    assert po2.options["label"] == "again"
    # Check dfilter. Reject any point not in [-1e4, 1e4]
    proton3 = pg.read(datadir, 1)
    proton3.stage(ax=plot_ax, dfilter=lambda t: np.abs(t) <= 1e4,
                  bounds="2..6")
    assert len(plot_ax.pha.plot_objs) == 3
    po3 = plot_ax.pha.plot_objs[2]
    # greater_equal() returns False for NaN comparisons.
    assert np.all(np.greater_equal(1e4, np.abs(po3.proc_data)) |
                  np.isnan(po3.proc_data))
    # Check that staging a 2D dataset causes errors
    with pytest.raises(TypeError) as exc_info:
        pg.read(datadir, 2).stage(ax=plot_ax)
        assert "Plot queue already contains 1D spectra." in str(exc_info)


def test_2d_stage():
    """Tests the stage() method on 2D datasets."""
    fig, (empty_ax, plot_ax) = pg.subplots2d(1, 2) 
    cosy = pg.read(datadir, 2)
    cosy.stage(ax=plot_ax, levels=(1e5, 1.2, 10))
    # Check that the PHA was created on the right axes
    assert not hasattr(empty_ax, "pha")
    assert len(plot_ax.pha.plot_objs) == 1
    # Check the properties of the plot object
    po = plot_ax.pha.plot_objs[0]
    assert po.dataset is cosy
    assert po.f1_bounds == ""
    assert po.f2_bounds == ""
    assert po.clevels == ([-1e5 * (1.2 ** i) for i in range(9, -1, -1)] +
                          [1e5 * (1.2 ** i) for i in range(10)])
    assert po.ccolors == ["#E8000B"] * 10 + ["#023EFF"] * 10 
    assert po.label is None
    assert np.allclose(po.f1_ppm_scale, cosy.ppm_scale(axis=0))
    assert np.allclose(po.f2_ppm_scale, cosy.ppm_scale(axis=1))
    assert np.allclose(po.f1_hz_scale, cosy.hz_scale(axis=0))
    assert np.allclose(po.f2_hz_scale, cosy.hz_scale(axis=1))

    # Check that staging a 2D dataset causes errors
    with pytest.raises(TypeError) as exc_info:
        pg.read(datadir, 1).stage(ax=plot_ax)
        assert "Plot queue already contains 2D spectra." in str(exc_info)
