from pathlib import Path

import penguins as pg

datadir = Path(__file__).parent.resolve() / "nmrdata"


def test_read_api():
    """
    Tests API of read() for correctness.
    """
    # Test that string arguments are OK too
    proton = pg.read(str(datadir), 1, 1)
    assert isinstance(proton, pg.dataset.Dataset1D)
    assert proton.path == datadir / "1" / "pdata" / "1"

    # Test that keyword arguments are OK
    proton = pg.read(path=datadir, expno=1, procno=1)
    assert isinstance(proton, pg.dataset.Dataset1D)
    assert proton.path == datadir / "1" / "pdata" / "1"

    # Test that leaving out procno defaults to 1
    proton = pg.read(datadir, 1)
    assert isinstance(proton, pg.dataset.Dataset1D)
    assert proton.path == datadir / "1" / "pdata" / "1"


def test_read_instance():
    """
    Tests that the correct Dataset subclasses are created according to the type
    of spectrum.
    """
    # 1D
    proton = pg.read(datadir, 1, 1)
    assert isinstance(proton, pg.dataset.Dataset1D)
    assert proton.path == datadir / "1" / "pdata" / "1"
    # 1D projections
    proj = pg.read(datadir, 2, 1001)
    assert isinstance(proj, pg.dataset.Dataset1DProj)
    assert proj.path == datadir / "2" / "pdata" / "1001"
    # 1D slice. But note that slice is a keyword, hence 'slic'
    slic = pg.read(datadir, 2, 1002)
    assert isinstance(slic, pg.dataset.Dataset1DProj)
    assert slic.path == datadir / "2" / "pdata" / "1002"
    # 2D 
    cosy = pg.read(datadir, 2, 1)
    assert isinstance(cosy, pg.dataset.Dataset2D)
    assert cosy.path == datadir / "2" / "pdata" / "1"
