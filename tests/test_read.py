import sys
from pathlib import Path

import penguins as pg

datadir = Path(__file__).parent.resolve() / "nmrdata"


def get_size(ds):
    """
    Gets the sum of sizes of each object in vars(ds).
    """
    return sum(sys.getsizeof(getattr(ds, it)) for it in vars(ds).keys())


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


def test_read_lazy_1d():
    """Tests that read() is truly being lazy on 1D datasets."""
    proton = pg.read(datadir, 1)
    assert "_real" not in vars(proton)
    sz0 = get_size(proton)
    # Read in the real part
    proton.real
    assert "_real" in vars(proton)
    sz1 = get_size(proton)
    # The nparray is 524384 bytes in size.
    assert sz1 > sz0 + 5e5
    # Read in the imaginary part
    proton.imag
    assert "_imag" in vars(proton)
    sz2 = get_size(proton)
    assert sz2 > sz1 + 5e5
    # Read in the FID
    proton.fid
    assert "_fid" in vars(proton)
    sz3 = get_size(proton)
    assert sz3 > sz2 + 5e5


def test_read_lazy_2d():
    """Tests that read() is truly being lazy on 2D datasets."""
    cosy = pg.read(datadir, 2)
    assert "_rr" not in vars(cosy)
    sz0 = get_size(cosy)

    cosy.rr
    assert "_rr" in vars(cosy)
    sz1 = get_size(cosy)
    assert sz1 > sz0 + 1.5e7

    cosy.ri
    assert "_ri" in vars(cosy)
    sz2 = get_size(cosy)
    assert sz2 > sz1 + 1.5e7

    cosy.ir
    assert "_ir" in vars(cosy)
    sz3 = get_size(cosy)
    assert sz3 > sz2 + 1.5e7

    cosy.ii
    assert "_ii" in vars(cosy)
    sz4 = get_size(cosy)
    assert sz4 > sz3 + 1.5e7
