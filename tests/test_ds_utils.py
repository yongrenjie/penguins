import sys
from pathlib import Path

import pytest
import numpy as np
import penguins as pg
from penguins.dataset import _try_convert, _parse_bounds

datadir = Path(__file__).parents[1].resolve() / "penguins-testdata"


# -- Tests on utility functions ---------------------

def test_try_convert():
    """Tests _try_convert()."""
    # Conversion of one element
    assert _try_convert(4, str) == "4"
    assert _try_convert("hello", int) == "hello"
    # Conversion of tuples and lists. We need to make sure that it preserves
    # the right structure.
    assert _try_convert(("4.3", "5.1"), float) == (4.3, 5.1)
    assert _try_convert(["4.3", "5.1"], float) == [4.3, 5.1]
    assert _try_convert(["a", "b"], int) == ["a", "b"]
    assert _try_convert(("a", "b"), int) == ("a", "b")
    # Numpy array that will work.
    x = np.array(["4", "5"])
    assert x.dtype == np.dtype("<U1")
    assert _try_convert(x, int).dtype == np.dtype("int64")
    # Numpy array that will fail.
    x2 = np.array(["X", "Y"])
    assert x2.dtype == np.dtype("<U1")
    assert _try_convert(x2, int) is x2


def test_parse_bounds():
    """Tests _parse_bounds()."""
    # Test typical tuple input
    assert _parse_bounds((4.3, 5.1)) == (4.3, 5.1)
    assert _parse_bounds((None, 5.1)) == (None, 5.1)
    assert _parse_bounds((4.3, None)) == (4.3, None)
    assert _parse_bounds((None, None)) == (None, None)
    # Test a list
    assert _parse_bounds([4.3, 5.1]) == (4.3, 5.1)
    # Test typical string input
    assert _parse_bounds("4.3..5.1") == (4.3, 5.1)
    assert _parse_bounds("..5.1") == (None, 5.1)
    assert _parse_bounds("4.3..") == (4.3, None)
    assert _parse_bounds("") == (None, None)
    assert _parse_bounds("..") == (None, None)
    # Test pathological string input
    assert _parse_bounds(".4...6") == (0.4, 0.6)
    # Test some minus signs
    assert _parse_bounds("-100..1") == (-100, 1)
    assert _parse_bounds("..-3") == (None, -3)
    # Test errors
    # Bounds specified the wrong way round
    with pytest.raises(ValueError) as exc_info:
        _parse_bounds("5.1..4.3")
        assert "Use '4.3..5.1', not '5.1..4.3'" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        _parse_bounds((5.1, 4.3))
        assert "Use (4.3, 5.1), not (5.1, 4.3)" in str(exc_info)
    # Tuple given with wrong number of elements
    with pytest.raises(ValueError) as exc_info:
        _parse_bounds(())
        _parse_bounds((1))
        _parse_bounds((1, 2, 3))
        _parse_bounds((1, 2, 3, 4, 5, 6, 7, 8, 9))
    # String with rubbish inside
    with pytest.raises(ValueError) as exc_info:
        _parse_bounds("hello there")
        _parse_bounds(".....")
        _parse_bounds("a..b")
