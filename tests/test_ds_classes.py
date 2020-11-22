import sys
from pathlib import Path

import pytest
import numpy as np
import penguins as pg

datadir = Path(__file__).parent.resolve() / "nmrdata"


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
    assert repr(proton) == f"Dataset1D('{str(proton.path)}')"
    assert all(par in proton.pars for par in initial_pars)
    assert proton["aq"] == pytest.approx(2.9360127)
    assert proton["td"] == 65536
    assert proton["sw"] == pytest.approx(15.9440, abs=0.0001)
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


def test_projection_raw_ser():
    """Test that projections and slices read in the ser file."""
    proj = pg.read(datadir, 2, 1001)
    ser1 = proj.raw_data()
    assert ser1.shape == (proj["td"][0], proj["td"][1] / 2)

    slic = pg.read(datadir, 2, 1002)
    ser2 = slic.raw_data()
    assert np.allclose(ser1, ser2)


# -- Tests on processed data handling ---------------

def test_1d_proc_data():
    proton = pg.read(datadir, 1)
    r = proton.real
    assert r.shape == (proton["si"], )
    # Check proc_data() accessor. They don't point to the same object because
    # proc_data() takes bounds, hence we need np.allclose().
    assert np.allclose(proton.proc_data(), r)

    # Check the bounds on proc_data().
    good_bounds = "4..6"
    bad_bounds = "-10..2"
    # In TopSpin it's 28657, but for some weird reason TopSpin's numbers are
    # always slightly off. I had a similar problem with nmrpoise.
    assert np.allclose(proton.proc_data(bounds=good_bounds), r[28658:36880])
    with pytest.raises(ValueError) as exc_info:
        proton.proc_data(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)


def test_1d_mc():
    proton = pg.read(datadir, 1)
    r = proton.real
    i = proton.imag
    proton_mc = proton.mc()
    si = proton["si"]
    # Check that it's the right thing
    assert np.all(np.greater_equal(proton_mc.real, np.zeros(si)))
    assert np.allclose(proton_mc.real, np.abs(r + i * 1j))
    # Check that the imaginary part has been removed
    assert np.array_equal(proton_mc.imag, np.zeros(si))
    # Check that mc() is idempotent
    proton_mc_mc = proton_mc.mc()
    assert np.allclose(proton_mc.real, proton_mc_mc.real)


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
    assert np.min(ppm) == proton["o1p"] - (proton["sw"] / 2)
    assert np.max(ppm) == proton["o1p"] + (proton["sw"] / 2)
    assert np.min(ppm) == ppm[-1]
    assert np.max(ppm) == ppm[0]
    assert np.allclose(proton.ppm_scale(bounds=good_bounds), ppm[28658:36880])
    with pytest.raises(ValueError) as exc_info:
        proton.ppm_scale(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)

    hz = proton.hz_scale()
    assert hz.shape == (proton["si"], )
    assert np.min(hz) == proton["o1"] - (proton["sw"] * proton["sfo1"] / 2)
    assert np.max(hz) == proton["o1"] + (proton["sw"] * proton["sfo1"] / 2)
    assert np.min(hz) == hz[-1]
    assert np.max(hz) == hz[0]
    assert np.allclose(proton.hz_scale(bounds=good_bounds), hz[28658:36880])
    with pytest.raises(ValueError) as exc_info:
        proton.hz_scale(bounds=bad_bounds)
        assert "outside spectral window" in str(exc_info)


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
