from pathlib import Path

import penguins as pg

datadir = Path(__file__).parent.resolve() / "data"


def test_read_1D():
    # test 1D datasets
    path = datadir / "pt2"
    proton = pg.read(path, 1, 1)
    assert isinstance(proton, pg.dataset.Dataset1D)
    # make sure that strings are ok too
    proton = pg.read(str(path), 1, 1)
    assert isinstance(proton, pg.dataset.Dataset1D)


def test_read_1Dprojections():
    # test 1D projections
    path = datadir / "rot1"
    proj = pg.read(path, 8, 179)
    assert isinstance(proj, pg.dataset.Dataset1DProj)
    # make sure that strings are ok too
    proj = pg.read(str(path), 8, 179)
    assert isinstance(proj, pg.dataset.Dataset1DProj)


def test_read_2D():
    # test 2D datasets
    path = datadir / "rot1"
    hsqmbc = pg.read(path, 8, 1)
    assert isinstance(hsqmbc, pg.dataset.Dataset2D)
    # make sure that strings are ok too
    hsqmbc = pg.read(str(path), 8, 1)
    assert isinstance(hsqmbc, pg.dataset.Dataset2D)


def test_read_abs_1D():
    # test 1D datasets
    path = datadir / "pt2" / "1" / "pdata" / "1"
    proton = pg.read_abs(path)
    assert isinstance(proton, pg.dataset.Dataset1D)
    # make sure that strings are ok too
    proton = pg.read_abs(str(path))
    assert isinstance(proton, pg.dataset.Dataset1D)


def test_read_abs_1Dprojections():
    # test 1D projections
    path = datadir / "rot1" / "8" / "pdata" / "179"
    proj = pg.read_abs(path)
    assert isinstance(proj, pg.dataset.Dataset1DProj)
    # make sure that strings are ok too
    proj = pg.read_abs(str(path))
    assert isinstance(proj, pg.dataset.Dataset1DProj)


def test_read_abs_2D():
    # test 2D datasets
    path = datadir / "rot1" / "8" / "pdata" / "1"
    hsqmbc = pg.read_abs(path)
    assert isinstance(hsqmbc, pg.dataset.Dataset2D)
    # make sure that strings are ok too
    hsqmbc = pg.read_abs(str(path))
    assert isinstance(hsqmbc, pg.dataset.Dataset2D)


# Tests for the plotting functions are elsewhere.
