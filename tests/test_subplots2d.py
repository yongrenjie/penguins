import numpy as np
import penguins as pg
import pytest


def test_subplots2d():
    """Tests that pg.subplots2d() generates figures with the correct size."""

    # No arguments, should just be 4x4
    fig, _ = pg.subplots2d()
    assert np.array_equal([4, 4], fig.get_size_inches())

    # ncols, nrows passed, each should be 4x4
    fig, _ = pg.subplots2d(1, 2)
    assert np.array_equal([8, 4], fig.get_size_inches())
    fig, _ = pg.subplots2d(3, 1)
    assert np.array_equal([4, 12], fig.get_size_inches())
    fig, _ = pg.subplots2d(5, 9)
    assert np.array_equal([36, 20], fig.get_size_inches())

    # height_ratios passed only
    fig, _ = pg.subplots2d(2, 2, height_ratios=[0.5, 1])
    assert np.array_equal([8, 6], fig.get_size_inches())
    # wrong number of elements
    with pytest.raises(ValueError):
        fig, _ = pg.subplots2d(2, 2, height_ratios=[0.5, 1, 0.2])

    # width_ratios passed only
    fig, _ = pg.subplots2d(2, 2, width_ratios=[0.5, 1])
    assert np.array_equal([6, 8], fig.get_size_inches())
    # wrong number of elements
    with pytest.raises(ValueError):
        fig, _ = pg.subplots2d(2, 2, width_ratios=[0.5, 1, 0.2])

    # check that figsize overrides width_ratios
    fig, _ = pg.subplots2d(2, 2, width_ratios=[0.5, 1], figsize=(100, 100))
    assert np.array_equal([100, 100], fig.get_size_inches())
    # check that figsize overrides height_ratios
    fig, _ = pg.subplots2d(2, 2, height_ratios=[0.5, 1], figsize=(100, 100))
    assert np.array_equal([100, 100], fig.get_size_inches())
    # check that figsize overrides both of these
    fig, _ = pg.subplots2d(2, 2, width_ratios=[0.5, 1],
                           height_ratios=[0.5, 1], figsize=(100, 100))
    assert np.array_equal([100, 100], fig.get_size_inches())
