import sys
from pathlib import Path

import pytest
import numpy as np
import penguins as pg

datadir = Path(__file__).parents[1].resolve() / "penguins-testdata"


# -- Tests on _Dataset() base class -----------------

def test_dataset_initialisation():
    """Tests that parameters are being initialised correctly by the various
    __init__() methods.
    """
    # Parameters that are supposed to be read in upon initialisation.
    # The rest are lazily read in.
    initial_pars = ["aq", "td", "sw", "sfo1", "dw",
                    "o1p", "o1", "bf1", "si", "nuc1"]
    # Check 1D
    proton = pg.read(datadir, 1)
    assert all(par in proton.pars for par in initial_pars)
    assert proton["aq"] == pytest.approx(2.9360127)
    assert proton["td"] == 65536
    assert proton["sw"] == pytest.approx(15.9440, rel=1e-4)
    assert proton["sfo1"] == pytest.approx(699.9935)
    assert proton["dw"] == pytest.approx(44.800)
    assert proton["bf1"] == pytest.approx(699.99)
    assert proton["o1p"] == pytest.approx(5.00)
    assert proton["o1"] == pytest.approx(3499.95)
    assert proton["si"] == 65536
    assert proton["nuc1"] == "1H"

    # Check 2D
    cosy = pg.read(datadir, 2)
    assert all(par in cosy.pars for par in initial_pars)
    assert np.allclose(cosy["aq"], np.array([0.0182784, 0.0731136]))
    assert np.array_equal(cosy["td"], np.array([256, 1024]))
    assert np.allclose(cosy["sw"], np.array([10.0041, 10.0041]))
    assert np.allclose(cosy["sfo1"], np.array([699.9928, 699.9928]))
    assert cosy["inf1"] == pytest.approx(142.8001)
    assert cosy["dw"] == pytest.approx(71.400)
    assert np.allclose(cosy["bf1"], np.array([699.99, 699.99]))
    assert np.allclose(cosy["o1p"], np.array([4.00, 4.00]))
    assert np.allclose(cosy["o1"], np.array([2799.96, 2799.96]))
    assert np.array_equal(cosy["si"], np.array([1024, 2048]))
    assert cosy["nuc1"] == ("1H", "1H")


def test_parDict():
    """Tests the implementation of parDict, i.e. that it is reading values
    lazily and accurately.
    """
    # Check 1D
    proton = pg.read(datadir, 1)
    assert "ns" not in proton.pars
    assert proton["ns"] == 16
    assert "pulprog" not in proton.pars
    assert proton["pulprog"] == "zg60"
    assert "rg" not in proton.pars
    assert proton["rg"] == pytest.approx(7.12)
    assert "lb" not in proton.pars
    assert proton["lb"] == pytest.approx(0.3)
    assert "nc_proc" not in proton.pars
    assert proton["nc_proc"] == pytest.approx(-6)

    # Check 2D
    cosy = pg.read(datadir, 2)
    assert "ns" not in cosy.pars
    assert cosy["ns"] == 2
    assert "pulprog" not in cosy.pars
    assert cosy["pulprog"] == "jy-clipcosy"
    assert "rg" not in cosy.pars
    assert cosy["rg"] == pytest.approx(36)
    assert "lb" not in cosy.pars
    assert np.allclose(cosy["lb"], np.array([0, 0]))
    assert "phc0" not in cosy.pars
    assert np.allclose(cosy["phc0"], np.array([25.363, 90.363]))
    assert "nc_proc" not in cosy.pars
    assert cosy["nc_proc"] == pytest.approx(-3)
    assert "gpnam12" not in cosy.pars
    assert cosy["gpnam12"] == "SMSQ10.100"
    # Check parameter with space and caps
    assert "GPZ 12" not in cosy.pars
    assert cosy["GPZ 12"] == pytest.approx(43)
    assert "gpz12" not in cosy.pars
    assert cosy["gpz12"] == cosy["GPZ 12"]
    # Check deletion
    assert "gpz12" in cosy.pars
    del cosy["gpz12"]
    assert "gpz12" not in cosy.pars

    # Check errors
    with pytest.raises(KeyError):
        cosy["dontexist"]
        proton["dontexist"]


# -- Tests on raw data handling ---------------------

def test_1d_raw_fid():
    """Test 1D raw data readin."""
    proton = pg.read(datadir, 1)
    fid = proton.fid
    # /2 because fid is complex points and TD is both real + imag
    assert fid.shape == (proton["td"] / 2,)
    # Check raw_data accessor
    assert proton.raw_data() is fid

    proc_psyche = pg.read(datadir, 4)
    fid = proc_psyche.fid
    assert fid.shape == (proc_psyche["td"] / 2,)
    assert proc_psyche.raw_data() is fid


def test_2d_raw_ser():
    """Test 2D raw data readin."""
    cosy = pg.read(datadir, 2)
    ser = cosy.ser
    # Note that only TD2 needs to be divided by 2, because there isn't the real
    # + imag -> complex combination in the indirect dimension. (Well, in a way
    # there *could*, but traditionally (for example) the cosine- and
    # sine-modulated FIDs in the States method are kept separate for
    # appropriate processing.)
    assert ser.shape == (cosy["td"][0], cosy["td"][1] / 2)
    # Test raw_data() accessor
    rawdata = cosy.raw_data()
    assert rawdata is ser

    # Test the case where TD is not a multiple of 256
    hsqc = pg.read(datadir, 6)
    assert (hsqc["td"][1] / 2) % 256 != 0
    assert hsqc.ser.shape == (hsqc["td"][0], hsqc["td"][1] / 2)


def test_projection_raw_ser():
    """Verify that "raw data" for projections and slices is indeed the original
    2D ser file."""
    proj = pg.read(datadir, 2, 1001)
    ser1 = proj.raw_data()
    assert ser1.shape == (proj["td"][0], proj["td"][1] / 2)

    slic = pg.read(datadir, 2, 1002)
    ser2 = slic.raw_data()
    assert np.allclose(ser1, ser2)


# -- Tests on low-level processed data handling -----

def test_1d_proc_data():
    """Test 1D processed data readin."""
    proton = pg.read(datadir, 1)
    r = proton.real
    assert r.shape == (proton["si"], )
    # Check proc_data() accessor. They don't point to the same object because
    # proc_data() takes bounds, hence we need np.allclose().
    assert np.allclose(proton.proc_data(), r)

    # Check the bounds on proc_data(). This assumes that ppm_to_index() works
    # correctly.
    good_bounds = "4..6"
    bad_bounds = "-10..2"
    assert np.allclose(proton.proc_data(bounds=good_bounds),
                       r[proton.ppm_to_index(6):proton.ppm_to_index(4) + 1])
    with pytest.raises(ValueError) as exc_info:
        proton.proc_data(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)


def test_1d_ppm_to_index():
    """Test ppm_to_index(). Note that there are often small discrepancies
    versus the indices that TopSpin reports, for some odd reason. For example,
    if you navigate to 11.96 ppm in this spectrum, TopSpin claims that it is
    index 4160.
    """
    proton = pg.read(datadir, 1)
    assert proton.ppm_to_index(11.96) == 4161
    assert proton.ppm_to_index(0.66729) == 50577
    with pytest.raises(ValueError) as exc_info:
        proton.ppm_to_index(25)
        assert "outside spectral window" in str(exc_info)


def test_1d_ppm_hz_scales():
    """Test ppm_scale() and hz_scale(). Note that both of these output chemical
    shifts or frequencies in descending order, so that their indices match
    those in proc_data() (which are read in in descending order).
    """
    proton = pg.read(datadir, 1)
    good_bounds = "4..6"
    bad_bounds = "-10..2"

    ppm = proton.ppm_scale()
    assert ppm.shape == (proton["si"], )
    assert np.min(ppm) == proton["offset"] - (proton["sw_p"] / proton["sfo1"])
    assert np.max(ppm) == proton["offset"]
    # for this dataset offset and sw_p should be almost the same as O1 and SW_h
    assert np.allclose(ppm, np.linspace(proton["o1p"] + proton["sw"]/2,
                                        proton["o1p"] - proton["sw"]/2,
                                        proton["si"]),
                       atol=1e-4)
    # check that ppm scale is in the right order
    assert np.min(ppm) == pytest.approx(ppm[-1])
    assert np.max(ppm) == pytest.approx(ppm[0])
    # check bounds argument
    assert np.allclose(proton.ppm_scale(bounds=good_bounds), ppm[28658:36880])
    # check outside spectral window
    with pytest.raises(ValueError) as exc_info:
        proton.ppm_scale(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)

    hz = proton.hz_scale()
    assert hz.shape == (proton["si"], )
    assert np.min(hz) == pytest.approx((proton["offset"] * proton["sfo1"])
                                       - proton["sw_p"],
                                       rel=1e-4)
    assert np.max(hz) == pytest.approx(proton["offset"] * proton["sfo1"],
                                       rel=1e-4)
    # for this dataset offset and sw_p should be almost the same as O1 and SW_h
    assert np.min(hz) == pytest.approx(proton["o1"]
                                       - (proton["sw"] * proton["sfo1"] / 2),
                                       rel=1e-4)
    assert np.max(hz) == pytest.approx(proton["o1"]
                                       + (proton["sw"] * proton["sfo1"] / 2),
                                       rel=1e-4)
    assert np.min(hz) == pytest.approx(hz[-1])
    assert np.max(hz) == pytest.approx(hz[0])
    assert np.allclose(proton.hz_scale(bounds=good_bounds), hz[28658:36880])
    with pytest.raises(ValueError) as exc_info:
        proton.hz_scale(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)


def test_2d_proc_data():
    """Test 2D processed data readin."""
    cosy = pg.read(datadir, 2)
    rr = cosy.rr
    assert rr.shape == (cosy["si"][0], cosy["si"][1])
    ri = cosy.ri
    assert ri.shape == (cosy["si"][0], cosy["si"][1])
    ir = cosy.ir
    assert ir.shape == (cosy["si"][0], cosy["si"][1])
    ii = cosy.ii
    assert ii.shape == (cosy["si"][0], cosy["si"][1])
    # Check proc_data() accessor
    assert np.allclose(cosy.proc_data(), rr)
    # Check bounds on proc_data(). This assumes that ppm_to_index() works
    # correctly.
    assert np.allclose(cosy.proc_data(f1_bounds="4..6", f2_bounds="5..7"),
                       rr[cosy.ppm_to_index(0, 6):cosy.ppm_to_index(0, 4) + 1,
                          cosy.ppm_to_index(1, 7):cosy.ppm_to_index(1, 5) + 1])
    # Check for bad f1 and f2 bounds
    with pytest.raises(ValueError) as exc_info:
        cosy.proc_data(f1_bounds="-4..2", f2_bounds="4..5")
        assert "outside spectral window" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        cosy.proc_data(f2_bounds="4..20")
        assert "outside spectral window" in str(exc_info)


def test_2d_ppm_to_index():
    """Test ppm_to_index() for 2D datasets."""
    cosy = pg.read(datadir, 2)
    assert cosy.ppm_to_index(axis=0, ppm=6.629) == 244
    assert cosy.ppm_to_index(axis=1, ppm=6.629) == 487
    # Test out of bounds
    with pytest.raises(ValueError) as exc_info:
        cosy.ppm_to_index(axis=0, ppm=-2)
        assert "outside spectral window" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        cosy.ppm_to_index(axis=1, ppm=15)
        assert "outside spectral window" in str(exc_info)


def test_2d_ppm_hz_scales():
    """Test ppm_scale() and hz_scale() for 2D datasets."""
    cosy = pg.read(datadir, 2)

    # f1 ppm_scale
    ppm = cosy.ppm_scale(axis=0)
    assert ppm.shape == (cosy["si"][0], )
    assert np.min(ppm) == cosy["offset"][0] - (cosy["sw_p"][0] / cosy["sfo1"][0])
    assert np.max(ppm) == cosy["offset"][0]
    # for this dataset offset and sw_p should be almost the same as O1 and SW_h
    assert np.allclose(ppm, np.linspace(cosy["o1p"][0] + cosy["sw"][0]/2,
                                        cosy["o1p"][0] - cosy["sw"][0]/2,
                                        cosy["si"][0]),
                       atol=1e-4)
    # Check that ppm scale is in the right order
    assert np.min(ppm) == pytest.approx(ppm[-1])
    assert np.max(ppm) == pytest.approx(ppm[0])
    # Check bounds argument
    assert np.allclose(cosy.ppm_scale(axis=0, bounds="4..6"),
                       ppm[cosy.ppm_to_index(0, 6):cosy.ppm_to_index(0, 4) + 1])
    # Check outside spectral window
    with pytest.raises(ValueError) as exc_info:
        cosy.ppm_scale(axis=0, bounds="-10..2")
        assert "outside spectral window" in str(exc_info)

    # f2 ppm_scale
    ppm = cosy.ppm_scale(axis=1)
    assert ppm.shape == (cosy["si"][1], )
    assert np.min(ppm) == cosy["offset"][1] - (cosy["sw_p"][1] / cosy["sfo1"][1])
    assert np.max(ppm) == cosy["offset"][1]
    # for this dataset offset and sw_p should be almost the same as O1 and SW_h
    assert np.allclose(ppm, np.linspace(cosy["o1p"][1] + cosy["sw"][1]/2,
                                        cosy["o1p"][1] - cosy["sw"][1]/2,
                                        cosy["si"][1]),
                       atol=1e-4)
    # Check that ppm scale is in the right order
    assert np.min(ppm) == pytest.approx(ppm[-1])
    assert np.max(ppm) == pytest.approx(ppm[0])
    # Check bounds argument
    assert np.allclose(cosy.ppm_scale(axis=1, bounds="4..6"),
                       ppm[cosy.ppm_to_index(1, 6):cosy.ppm_to_index(1, 4) + 1])
    # Check outside spectral window
    with pytest.raises(ValueError) as exc_info:
        cosy.ppm_scale(axis=1, bounds="-10..2")
        assert "outside spectral window" in str(exc_info)

    # f1 hz_scale
    hz = cosy.hz_scale(axis=0)
    assert hz.shape == (cosy["si"][0], )
    assert np.min(hz) == pytest.approx((cosy["offset"][0] * cosy["sfo1"][0]) -
                                       cosy["sw_p"][0],
                                       rel=1e-4)
    assert np.max(hz) == pytest.approx(cosy["offset"][0] * cosy["sfo1"][0],
                                       rel=1e-4)
    # check that hz scale is in the right order
    assert np.min(hz) == pytest.approx(hz[-1])
    assert np.max(hz) == pytest.approx(hz[0])
    assert np.allclose(cosy.hz_scale(axis=0, bounds="4..6"),
                       hz[cosy.ppm_to_index(0, 6):cosy.ppm_to_index(0, 4) + 1])
    with pytest.raises(ValueError) as exc_info:
        cosy.hz_scale(axis=0, bounds="-10..2")
        assert "outside spectral window" in str(exc_info)
    # f2 hz_scale
    hz = cosy.hz_scale(axis=1)
    assert hz.shape == (cosy["si"][1], )
    assert np.min(hz) == pytest.approx((cosy["offset"][1] * cosy["sfo1"][1]) -
                                       cosy["sw_p"][1])
    assert np.max(hz) == pytest.approx(cosy["offset"][1] * cosy["sfo1"][1])
    assert np.min(hz) == pytest.approx(hz[-1])
    assert np.max(hz) == pytest.approx(hz[0])
    assert np.allclose(cosy.hz_scale(axis=1, bounds="4..6"),
                       hz[cosy.ppm_to_index(1, 6):cosy.ppm_to_index(1, 4) + 1])
    with pytest.raises(ValueError) as exc_info:
        cosy.hz_scale(axis=1, bounds="-10..2")
        assert "outside spectral window" in str(exc_info)


# -- Tests on 2D projection generation --------------

def test_2d_project():
    """Tests penguins' generation of virtual projections."""
    cosy = pg.read(datadir, 2)
    # Projections onto f2
    f2 = cosy.f2projp()
    assert f2.real.shape == (cosy["si"][1], )
    assert np.all(np.greater_equal(f2.real, 0))
    # Projections onto f1
    f1 = cosy.f1projp()
    assert f1.real.shape == (cosy["si"][0], )
    assert np.all(np.greater_equal(f1.real, 0))
    # Test projection bounds
    # There are no peaks between 7 and 8 ppm
    f2_bounded = cosy.f2projp(bounds="7..8")
    assert f2_bounded.real.shape == (cosy["si"][1], )
    assert np.max(f2_bounded.real) < np.max(f2.real)
    # All the peaks are between 0.5 and 7 ppm
    f2_bounded = cosy.f2projp(bounds="0.5..7")
    assert f2_bounded.real.shape == (cosy["si"][1], )
    assert np.max(f2_bounded.real) == np.max(f2.real)

    # Check that imaginary part of a projection is zero
    assert f1.imag == 0
    assert f2.imag == 0
    assert f2_bounded.imag == 0

    # Check that manual projection is the same as TopSpin projection
    assert np.allclose(f2.real, pg.read(datadir, 2, 1001).real)


def test_2d_slice():
    """Tests penguins' generation of virtual slices."""
    # TODO: improve these tests. I think SI here is (1k, 1k) which makes SI
    # a poor check!
    cosy = pg.read(datadir, 2)
    # Slice along f1 at one point of f2 (i.e. a column)
    along_f1 = cosy.slice(axis="column", ppm=1.38)
    assert along_f1.real.shape == (cosy["si"][0], )
    # Slice along f2 at one point of f1 (i.e. a row)
    along_f2 = cosy.slice(axis="row", ppm=1.38)
    assert along_f2.real.shape == (cosy["si"][1], )
    # Check that they have the same value at (1.38, 1.38)
    assert (along_f2.real[cosy.ppm_to_index(1, 1.38)] ==
            pytest.approx(along_f1.real[cosy.ppm_to_index(0, 1.38)]))

    # Check that slice(f1=x) is the same as slice(axis="row", ppm=x)
    assert np.array_equal(cosy.slice(f1=1.38).real, along_f2.real)
    # Check that slice(f2=y) is the same as slice(axis="column", ppm=y)
    assert np.array_equal(cosy.slice(f2=1.38).real, along_f1.real)

    # Check that penguins and TopSpin slices agree
    cosy_slice = pg.read(datadir, 2, 1002)
    assert np.allclose(along_f2.real, cosy_slice.real)


def test_2d_sum():
    """Tests penguins' generation of virtual slices."""
    cosy = pg.read(datadir, 2)
    # Sum onto f1 (i.e. add up multiple column)
    f1 = cosy.sum(axis=0, bounds="4..6")
    assert f1.real.shape == (cosy["si"][0], )
    # Sum onto f2 (i.e. add up multiple rows)
    f2 = cosy.sum(axis=1, bounds="4..6")
    assert f2.real.shape == (cosy["si"][1], )
    # Check that multiple sub-sums add up to the same sum (after accounting for
    # the double-counting of the row at 5 ppm)
    subsum = (cosy.sum(axis=1, bounds="4..5").real
              + cosy.sum(axis=1, bounds="5..6").real
              - cosy.slice(axis="row", ppm=5).real)
    assert np.allclose(f2.real, subsum)
    # Check that penguins and TopSpin sums are the same
    # For some reason, this is not the same thing. It *looks* like the same
    # thing, but isn't. This is probably because TopSpin uses some weird
    # algorithm which I don't know.
    # assert np.allclose(f2.real, pg.read(datadir, 2, 1003).real)


# -- Tests on TopSpin-like processing functions -----

def test_1d_mc():
    proton = pg.read(datadir, 1)
    r = proton.real
    i = proton.imag
    proton_mc = proton.mc()
    si = proton["si"]
    # Check that it's the right thing
    assert np.all(np.greater_equal(proton_mc.real, 0))
    assert np.allclose(proton_mc.real, np.abs(r + i * 1j))
    # Check that the imaginary part has been removed
    assert proton_mc.imag == 0
    # Check that mc() is idempotent
    proton_mc_mc = proton_mc.mc()
    assert np.allclose(proton_mc.real, proton_mc_mc.real)


def test_2d_xfnm():
    """Tests xf1m(), xf2m(), and xfbm()."""
    cosy = pg.read(datadir, 2)
    rr = cosy.rr
    ri = cosy.ri
    ir = cosy.ir
    ii = cosy.ii
    # Check output of xf1m()
    xf1 = cosy.xf1m()
    assert np.allclose(xf1.rr, np.abs(rr + 1j * ri))
    assert np.allclose(xf1.ir, np.abs(ir + 1j * ii))
    assert xf1.ri == 0
    assert xf1.ii == 0
    # Check output of xf2m()
    xf2 = cosy.xf2m()
    assert np.allclose(xf2.rr, np.abs(rr + 1j * ir))
    assert np.allclose(xf2.ri, np.abs(ri + 1j * ii))
    assert xf2.ir == 0
    assert xf2.ii == 0
    # Check that xf1m() is idempotent
    xf1_xf1m = xf1.xf1m()
    assert np.allclose(xf1.rr, xf1_xf1m.rr)
    assert np.allclose(xf1.ri, xf1_xf1m.ri)
    assert np.allclose(xf1.ir, xf1_xf1m.ir)
    assert np.allclose(xf1.ii, xf1_xf1m.ii)
    # Check that xf2m() is idempotent
    xf2_xf2m = xf2.xf2m()
    assert np.allclose(xf2.rr, xf2_xf2m.rr)
    assert np.allclose(xf2.ri, xf2_xf2m.ri)
    assert np.allclose(xf2.ir, xf2_xf2m.ir)
    assert np.allclose(xf2.ii, xf2_xf2m.ii)
    # Check that xf1m() and xf2m() commute
    xf1_xf2m = xf1.xf2m()
    xf2_xf1m = xf2.xf1m()
    assert np.allclose(xf1_xf2m.rr, xf2_xf1m.rr)
    assert np.allclose(xf1_xf2m.ri, xf2_xf1m.ri)
    assert np.allclose(xf1_xf2m.ir, xf2_xf1m.ir)
    assert np.allclose(xf1_xf2m.ii, xf2_xf1m.ii)
    # Check that ds.xfbm() == ds.xf1m().xf2m()
    xfbm = cosy.xfbm()
    assert np.allclose(xfbm.rr, xf2_xf1m.rr)
    assert np.allclose(xfbm.ri, xf2_xf1m.ri)
    assert np.allclose(xfbm.ir, xf2_xf1m.ir)
    assert np.allclose(xfbm.ii, xf2_xf1m.ii)


# -- Miscellaneous tests ----------------------------

def test_rawdata_relationships():
    """Check relationships between raw data of different datasets, to make sure
    that raw data is being read in a consistent manner.
    """
    # Check that the first chunk of the processed PSYCHE data is the same as
    # the original 2D data
    unproc_psyche = pg.read(datadir, 3)
    ser = unproc_psyche.ser
    proc_psyche = pg.read(datadir, 4)
    fid = proc_psyche.fid
    dp = int(unproc_psyche["cnst50"])  # drop points in NOAH PSYCHE is cnst50
    assert np.allclose(fid[:32], ser[0, :32])  # group delay
    assert np.allclose(fid[70:110], ser[0, 70+dp:110+dp])  # first chunk, ish


def test_procdata_relationships():
    """Check relationships between processed data of different datasets, to
    make sure that processed data is being read in a consistent manner.
    """
    # Check that TopSpin's COSY slice is the same as our COSY slice
    cosy = pg.read(datadir, 2)
    cosy_slice = pg.read(datadir, 2, 1002)
    assert np.allclose(cosy.rr[780,:], cosy_slice.real)
