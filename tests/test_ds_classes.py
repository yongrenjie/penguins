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
