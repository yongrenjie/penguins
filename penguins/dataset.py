from __future__ import annotations

from collections import UserDict
from pathlib import Path
from typing import (Any, Union, Tuple, Optional,
                    TypeVar, Callable, overload)
from typing_extensions import Protocol   # "from typing" in 3.8+

import numpy as np               # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from . import pgplot
from .type_aliases import *


class _DSProtocol(Protocol):
    """
    Specifies the protocol for a generic dataset object.

    The problem is that in our mixins, Mixin1 references attributes and methods
    which are provided in Mixin2. Thus, in general, it is not possible to
    guarantee that any class deriving Mixin1 has the necessary attributes and
    methods. Thus, mypy throws lots of errors: "_Dataset has no attribute X",
    and so on. This is one way of remedying it.

    See: mypy documentation on "Protocols and structural subtypings", and also
    on "More types" > "Advanced uses of self-types".
    """
    path: Path
    pars: _parDict
    _p_procs: Path
    _p_proc2s: Path
    _p_acqus: Path
    _p_acqu2s: Path
    _p_fid: Path
    _p_real: Path
    _p_imag: Path
    _p_ser: Path
    _p_rr: Path
    _p_ir: Path
    _p_ri: Path
    _p_ii: Path
    fid: np.ndarray
    ser: np.ndarray
    real: np.ndarray
    imag: np.ndarray
    rr: np.ndarray
    ir: np.ndarray
    ri: np.ndarray
    ii: np.ndarray
    proj_axis: int
    proj_type: str
    index: Optional[int]
    bounds: Optional[slice]
    sign: Optional[str]

    __getitem__: Callable
    __setitem__: Callable
    _initialise_pars: Callable
    _find_raw_data_paths: Callable
    _find_proc_data_paths: Callable
    _find_param_file_paths: Callable
    _read_raw_data: Callable
    _read_spec: Callable

    @overload
    def _ppm_to_index(self, ppm: Optional[float]) -> Optional[int]: ...
    @overload
    def _ppm_to_index(self, axis: int, ppm: Optional[float]) -> Optional[int]: ...

    @overload
    def _ppm_to_slice(self, bounds: Optional[TBounds1D]) -> slice: ...
    @overload
    def _ppm_to_slice(self, axis: int, bounds: Optional[TBounds1D]) -> slice: ...


def _try_convert(x: Any, type: Any):
    # Tries to convert an input to a specific type.
    # Returns the original input if it fails.
    # Works with tuples and lists too: tries to convert individual elements.
    # Works with np.ndarrays using ndarray.astype()
    try:
        return type(x)
    except ValueError:
        return x
    except TypeError:
        if isinstance(x, tuple) or isinstance(x, list):
            y = [_try_convert(i, type) for i in x]
            if isinstance(x, tuple):
                return tuple(y)
            else:
                return y
        elif isinstance(x, np.ndarray):
            return x.astype(dtype=type)
        else:
            return x


class _parDict(UserDict):
    """Dictionary for storing acquisition & processing parameters.

    Initialise within a Dataset class as self.pars = _parDict(self.path).

    Parameter names are stored as lower-case strings. When looking up a
    parameter, if its value is not already stored, the dictionary will look
    it up in the associated TopSpin parameter files. Subsequently, the value
    is cached.

    Therefore, the dictionary can be treated *as if* it were already fully
    populated when initialised; but this avoids the time spent parsing the
    entire file for parameters, the vast majority of which are useless."""

    def _editkey(self, key: object):
        """Method for converting keys before lookup."""
        return str(key).lower()

    def __init__(self, path: Path):
        self.path = path
        self._p_acqus = path.parents[1] / "acqus"
        self._p_procs = path / "procs"
        self._p_acqu2s = path.parents[1] / "acqu2s" \
                         if (path.parents[1] / "acqu2s").exists() else None
        self._p_proc2s = path / "proc2s" \
                         if (path / "proc2s").exists() else None
        super().__init__()

    def _getpar(self, par: str):
        # Get the direct dimension
        val = self._get_acqus_par(par, self._p_acqus)
        if val is None:
            val = self._get_procs_par(par, self._p_procs)
        val = _try_convert(val, float)
        # Check if there's also an indirect dimension
        if self._p_acqu2s is not None:
            val_indirect = self._get_acqus_par(par, self._p_acqu2s)
            if val_indirect is None and self._p_proc2s is not None:
                val_indirect = self._get_procs_par(par, self._p_proc2s)
            val_indirect = _try_convert(val_indirect, float)
            # Make a tuple if the indirect dimension was found
            if val_indirect is not None:
                if isinstance(val, float) and isinstance(val_indirect, float):
                    # ndarrays are good, so that we can do elementwise stuff
                    # like self["o1p] = self["o1"] / self["sfo1"]
                    val = np.array([val_indirect, val])
                else:
                    val = (val_indirect, val)
        # Some parameters must be ints, this will save the user headaches
        int_pars = ["TD", "SI"]
        if par.upper() in int_pars:
            val = _try_convert(val, int)
        return val

    def _get_acqus_par(self, par: str, fp: Union[str, Path]):
        """Get the value of an acquisition parameter and return as string."""
        # Capitalise and remove any spaces from par
        par = par.upper()
        if len(par.split()) > 1:
            par = "".join(par.split())
        # Split par into number-less bit and number bit
        parl = par.rstrip("1234567890")
        parr = par[len(parl):]
        params_with_space = ["CNST", "D", "P", "PLW", "PCPD", "GPX", "GPY",
                             "GPZ", "SPW", "SPOAL", "SPOFFS", "L", "IN",
                             "INP", "PHCOR"]
        # Get the parameter
        if (parr != "") and (parl in params_with_space):  # e.g. cnst2
            with open(fp, "r") as file:
                # Read up to the line declaring the parameters
                for line in file:
                    if line.upper().startswith(f"##${parl}="):
                        break
                # Grab the values and put them in a list
                s = ""
                # Read until next parameter
                line = file.readline()
                while not line.startswith("##"):
                    s = s + line + " "
                    line = file.readline()
                # Pick out the desired value and return it
                return s.split()[int(parr)]
        else:                                             # e.g. sfo1 or rga
            with open(fp, "r") as file:
                for line in file:
                    if line.upper().startswith(f"##${par}="):
                        val = line.split(maxsplit=1)[-1].strip()
                        # strip away surrounding angle brackets
                        if val[0] == '<' and val[-1] == '>':
                            val = val[1:-1]
                        return val
        # If it hasn't been found
        return None

    def _get_procs_par(self, par: str, fp: Union[str, Path]):
        """Get the value of an processing parameter and return as string."""
        # Capitalise and remove any spaces from par
        par = par.upper()
        if len(par.split()) > 1:
            par = "".join(par.split())
        # Get the value (for processing parameters there aren't any lists like
        # CNST/D/P)
        with open(fp, "r") as file:
            for line in file:
                if line.upper().startswith(f"##${par}="):
                    val = line.split(maxsplit=1)[-1].strip()
                    # strip away surrounding angle brackets
                    if val[0] == '<' and val[-1] == '>':
                        val = val[1:-1]
                    return val

    def __getitem__(self, key: object):
        k = self._editkey(key)
        if k in self.data:
            return self.data[k]
        else:
            val = self._getpar(k)
            if val is not None:
                self.data[k] = val
                return val
            else:
                raise KeyError(key)

    def __setitem__(self, key: object, val: Any):
        k = self._editkey(key)
        self.data.__setitem__(k, val)

    def __delitem__(self, key: object):
        k = self._editkey(key)
        self.data.__delitem__(k)

    def __contains__(self, key: object):
        k = self._editkey(key)
        return k in self.data

    def __repr__(self):
        keys = sorted(list(self))
        s = "{" + ",\n".join(f"'{k}': {self[k]}" for k in keys) + "}"
        return s


# -- Fundamental Dataset methods ------------------------

class _Dataset():
    """
    Defines behaviour that is common to all datasets.
    """

    def __init__(self: _DSProtocol,
                 path: Union[str, Path],
                 **kwargs
                 ) -> None:
        # Set up file path information
        self.path = Path(path).resolve().expanduser()
        self.expno = self.path.parents[1].name
        self.procno = self.path.name
        # Initialise _parDict and some key parameters
        self._initialise_pars()
        # Get paths to data and parameter files
        self._find_raw_data_paths()
        self._find_proc_data_paths()
        self._find_param_file_paths()
        # Read in the spectral data
        self._read_raw_data()
        self._read_spec()

    def _initialise_pars(self: _DSProtocol):
        self.pars = _parDict(self.path)
        self["aq"] = (self["td"] / 2) / (self["sw"] * self["sfo1"])
        self["dw"] = self["aq"] * 1000000 / self["td"]
        self["o1p"] = self["o1"] / self["sfo1"]
        self["si"]
        self["nuc1"]

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.path}')"

    def __getitem__(self, par: str):
        return self.pars[par]

    def __setitem__(self, par: str, val: Any):
        self.pars.__setitem__(par, val)

    def __delitem__(self, par:str):
        self.pars.__delitem__(par)


# -- 1D mixins ------------------------------------------

class _1D_RawDataMixin():
    def _find_raw_data_paths(self: _DSProtocol) -> None:
        self._p_fid = self.path.parents[1] / "fid"

    def _find_param_file_paths(self: _DSProtocol) -> None:
        self._p_procs = self.path / "procs"
        self._p_acqus = self.path.parents[1] / "acqus"

    def _read_raw_data(self: _DSProtocol) -> np.ndarray:
        fid = np.fromfile(self._p_fid, dtype=np.int32)
        fid = fid.reshape(int(self["td"]/2), 2)
        fid = np.transpose(fid) * (2 ** self["nc"])
        self.fid = fid[0] + (1j * fid[1])

    def raw_data(self: _DSProtocol) -> np.ndarray:
        return self.fid


class _1D_ProcDataMixin():

    def _find_proc_data_paths(self: _DSProtocol):
        self._p_real = self.path / "1r"
        if (self.path / "1i").exists():
            self._p_imag = self.path / "1i"

    def _read_spec(self: _DSProtocol) -> np.ndarray:
        self.real = np.fromfile(self._p_real, dtype=np.int32) * (2 ** self["nc_proc"])
        if hasattr(self, "_p_imag"):
            self.imag = np.fromfile(self._p_imag, dtype=np.int32) * (2 ** self["nc_proc"])

    def proc_data(self: _DSProtocol,
                  bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  ) -> np.ndarray:
        return self.real[self._ppm_to_slice(bounds)]

    def _ppm_to_slice(self: _DSProtocol,
                      bounds: Optional[TBounds1D] = None,
                      ) -> slice:
        """
        Converts a tuple of chemical shifts (upper, lower) to a slice object.
        Note that upper must be greater than lower. Either upper or lower can
        be None, in which case the spectrum limit is used.
        """
        bounds = bounds or (None, None)  # allow None as a synonym for (None, None)
        upper, lower = bounds
        if upper is not None and lower is not None and upper < lower:
            raise ValueError("Bounds should be specified as (upper, lower), not the other way around.")
        return slice(self._ppm_to_index(upper) or 0,
                     (self._ppm_to_index(lower) or self["si"] - 1) + 1)


class _1D_PlotMixin():

    def plot(self, *args, **kwargs):
        pgplot.plot1d(self, *args, **kwargs)


# -- 2D mixins ------------------------------------------

class _2D_RawDataMixin():

    def _find_raw_data_paths(self: _DSProtocol) -> None:
        self._p_ser = self.path.parents[1] / "ser"

    def _find_param_file_paths(self: _DSProtocol) -> None:
        self._p_procs = self.path / "procs"
        self._p_acqus = self.path.parents[1] / "acqus"
        self._p_proc2s = self.path / "proc2s"
        self._p_acqu2s = self.path.parents[1] / "acqu2s"

    def _read_raw_data(self: _DSProtocol) -> np.ndarray:
        # TODO
        # ser = np.fromfile(self._p_ser, dtype=??)
        pass

    def raw_data(self: _DSProtocol):
        # TODO
        raise NotImplementedError


class _2D_ProcDataMixin():

    def _find_proc_data_paths(self: _DSProtocol) -> None:
        # TODO: not all of these will exist for NUS data
        self._p_rr = self.path / "2rr"
        self._p_ri = self.path / "2ri"
        self._p_ir = self.path / "2ir"
        self._p_ii = self.path / "2ii"

    def _read_spec(self: _DSProtocol) -> np.ndarray:
        rr = np.fromfile(self._p_rr, dtype=np.int32)
        rr = rr.reshape(int(self["si"][0]), int(self["si"][1]))
        self.rr = rr * (2 ** self["nc_proc"][1])
        # TODO: the remainder (ri, ir, ii)

    def proc_data(self: _DSProtocol,
                  f1_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  f2_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  ) -> np.ndarray:
        f1_slice = self._ppm_to_slice(axis=0, bounds=f1_bounds)
        f2_slice = self._ppm_to_slice(axis=1, bounds=f2_bounds)
        return self.rr[f1_slice, f2_slice]

    def _ppm_to_slice(self: _DSProtocol,
                      axis: int,
                      bounds: Optional[TBounds1D] = None,
                      ) -> slice:
        """
        Converts a tuple of chemical shifts (upper, lower) to a slice object.
        Note that upper must be greater than lower. Either upper or lower can
        be None, in which case the spectrum limit is used.
        Axis = 0 for f1, 1 for f2.
        """
        bounds = bounds or (None, None)  # allow None as a synonym for (None, None)
        upper, lower = bounds
        if upper is not None and lower is not None and upper < lower:
            raise ValueError("Bounds should be specified as (upper, lower), not the other way around.")
        return slice(self._ppm_to_index(axis, upper) or 0,
                     (self._ppm_to_index(axis, lower) or self["si"][axis] - 1) + 1)


class _2D_PlotMixin():

    def plot(self, *args, **kwargs):
        pgplot.plot2d(self, *args, **kwargs)


# -- Actual dataset classes -----------------------------

class Dataset1D(_1D_RawDataMixin,
                _1D_ProcDataMixin,
                _1D_PlotMixin,
                _Dataset):

    def _ppm_to_index(self: _DSProtocol,
                      ppm: Optional[float]
                      ) -> Optional[int]:
        """
        Finds the index of the spectrum which is closest to the given
        chemical shift. Returns None if ppm is None.
        """
        if ppm is None:
            return None
        max_ppm = self["o1p"] + self["sw"]/2
        min_ppm = self["o1p"] - self["sw"]/2
        if ppm > max_ppm or ppm < min_ppm:
            raise ValueError(f"Chemical shift {ppm} is outside spectral window.")
        spacing = (max_ppm - min_ppm)/(self["si"] - 1)
        x = 1 + round((max_ppm - ppm)/spacing)
        return int(x)

    def ppm_scale(self: _DSProtocol,
                  bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  ) -> np.ndarray:
        max_ppm = self["o1p"] + self["sw"]/2
        min_ppm = self["o1p"] - self["sw"]/2
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self._ppm_to_slice(bounds)]

    def hz_scale(self: _DSProtocol) -> np.ndarray:
        max_hz = self["o1"] + self["sw"]/(2 * self["sfo1"])
        min_hz = self["o1"] - self["sw"]/(2 * self["sfo1"])
        return np.linspace(max_hz, min_hz, self["si"])


class Dataset1DProj(_2D_RawDataMixin,
                    _1D_ProcDataMixin,
                    _1D_PlotMixin,
                    _Dataset):

    def _initialise_pars(self):
        # Initialise _parDict and common pars as before.
        super()._initialise_pars()
        # Then figure out which axis we have been projected onto.
        projdimen = self.pars._get_acqus_par("curexp",
                                             self.path / "used_from")
        if projdimen == "column":   # projection onto f1
            self.proj_axis = 0
        elif projdimen == "row":    # projection onto f2
            self.proj_axis = 1
        else:
            raise ValueError("Projection dimension was not found in "
                             "'used_from' file.")

    def _ppm_to_index(self: _DSProtocol,
                      ppm: Optional[float]
                      ) -> Optional[int]:
        """
        Finds the index of the spectrum which is closest to the given
        chemical shift. Returns None if ppm is None.
        """
        if ppm is None:
            return None
        max_ppm = self["o1p"][self.proj_axis] + self["sw"][self.proj_axis]/2
        min_ppm = self["o1p"][self.proj_axis] - self["sw"][self.proj_axis]/2
        if ppm > max_ppm or ppm < min_ppm:
            raise ValueError(f"Chemical shift {ppm} is outside spectral window.")
        spacing = (max_ppm - min_ppm)/(self["si"] - 1)
        x = 1 + round((max_ppm - ppm)/spacing)
        return int(x)

    def ppm_scale(self: _DSProtocol,
                  bounds: Optional[TBounds1D] = None,
                  ) -> np.ndarray:
        max_ppm = self["o1p"][self.proj_axis] + self["sw"][self.proj_axis]/2
        min_ppm = self["o1p"][self.proj_axis] - self["sw"][self.proj_axis]/2
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self._ppm_to_slice(bounds)]

    def hz_scale(self: _DSProtocol) -> np.ndarray:
        max_hz = self["o1"][self.proj_axis] + self["sw"][self.proj_axis]/(2 * self["sfo1"][self.proj_axis])
        min_hz = self["o1"][self.proj_axis] - self["sw"][self.proj_axis]/(2 * self["sfo1"][self.proj_axis])
        return np.linspace(max_hz, min_hz, self["si"])


class Dataset2D(_2D_RawDataMixin,
                _2D_ProcDataMixin,
                _2D_PlotMixin,
                _Dataset):

    def _ppm_to_index(self: _DSProtocol,
                      axis: int,
                      ppm: Optional[float]
                      ) -> Optional[int]:
        """
        Finds the index of the spectrum which is closest to the given
        chemical shift. Returns None if ppm is None.
        Axis = 0 for f1, 1 for f2.
        """
        if ppm is None:
            return None
        max_ppm = self["o1p"][axis] + self["sw"][axis]/2
        min_ppm = self["o1p"][axis] - self["sw"][axis]/2
        if ppm > max_ppm or ppm < min_ppm:
            raise ValueError(f"Chemical shift {ppm} is outside spectral window.")
        spacing = (max_ppm - min_ppm)/(self["si"][axis] - 1)
        x = 1 + round((max_ppm - ppm)/spacing)
        return int(x)

    def ppm_scale(self: _DSProtocol,
                  axis: int,
                  bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  ) -> np.ndarray:
        max_ppm = self["o1p"][axis] + (self["sw"][axis] / 2)
        min_ppm = self["o1p"][axis] - (self["sw"][axis] / 2)
        full_scale = np.linspace(max_ppm, min_ppm, int(self["si"][axis]))
        return full_scale[self._ppm_to_slice(axis, bounds)]

    def hz_scale(self: _DSProtocol,
                 axis: int
                 ) -> np.ndarray:
        max_hz = self["o1"][axis] + (self["sw"][axis] / (2 * self["sfo1"][axis]))
        min_hz = self["o1"][axis] - (self["sw"][axis] / (2 * self["sfo1"][axis]))
        return np.linspace(max_hz[axis], min_hz[axis], int(self["si"][axis]))

    def project(self,
                type: str,
                axis: Union[int, str],
                sign: str,
                bounds: Optional[TBounds1D] = None,
                ) -> Dataset1DProjVirtual:
        if type not in ["projection", "sum"]:
            raise ValueError(f"Invalid value for type '{type}'")
        # Convenience for people who can't remember (like me)
        if axis == "column":  # sum / projection of columns
            axis = 0
        elif axis == "row":   # sum / projection of rows
            axis = 1
        if axis not in [0, 1]:
            raise ValueError(f"Invalid value for axis '{axis}'")
        # Allow some short forms...
        if sign not in ["positive", "negative", "pos", "neg"]:
            raise ValueError(f"Invalid value for sign '{sign}'")
        if sign == "pos":
            sign = "positive"
        elif sign == "neg":
            sign = "negative"
        # For some reason mypy doesn't realise that axis must be an int by here.
        index_bounds = self._ppm_to_slice(axis=(1 - axis), bounds=bounds)  # type: ignore
        return Dataset1DProjVirtual(self.path, proj_type=type,
                                    proj_axis=axis, sign=sign,
                                    bounds=index_bounds)

    def slice(self,
              axis: Union[int, str],
              ppm: float,
              ) -> Dataset1DProjVirtual:
        # convenience for people who can't remember (like me)
        if axis == "column":  # extract a column
            axis = 0
        elif axis == "row":   # extract a row
            axis = 1
        if axis not in [0, 1]:
            raise ValueError(f"Invalid value for axis '{axis}'")
        # For some reason mypy doesn't realise that axis must be an int by here.
        index = self._ppm_to_index(axis=(1 - axis), ppm=ppm)   # type: ignore
        return Dataset1DProjVirtual(self.path, proj_type="slice",
                                    proj_axis=axis, sign=None, index=index)


class Dataset1DProjVirtual(Dataset1DProj):
    """
    Class for a 1D projection generated from a 2D dataset.
    Note that the projection won't have any baseline correction, therefore
    integrals may not be entirely accurate.
    """

    def __init__(self: _DSProtocol,
                 path: Union[str, Path],
                 **kwargs
                 ) -> None:
        # Set some flags so that our overriding _read_spec() method can
        # calculate the projection
        self.proj_type = kwargs["proj_type"]  # "sum", "projection", or "slice"
        self.proj_axis = kwargs["proj_axis"]  # 0 or 1
        self.sign = kwargs.get("sign", None)  # "positive" or "negative"
        self.bounds = kwargs.get("bounds", None)  # only for sum or projection
        self.index = kwargs.get("index", None) # only for slice
        # Carry out the same initialisation tasks.
        # This calls _initialise_pars.
        Dataset1DProj.__init__(self, path)
        # Pars have been initialised at this point, so we can overwrite SI.
        # For 2D spectra SI is an array, for a standard projection it's a
        # single number. Not sure what other parameters need to be overwritten.
        self["si"] = self["si"][self.proj_axis]

    def _initialise_pars(self):
        # We don't want to inherit this from Dataset1DProj, because that
        # tries to look for the used_from file, and we don't have that.
        _Dataset._initialise_pars(self)

    def _read_spec(self: _DSProtocol) -> np.ndarray:
        # There's no 1r or 1i to read here, since the data isn't
        # from TopSpin.
        del self._p_real
        # First get the real, real part.
        _p_rr = self.path / "2rr"
        rr = np.fromfile(_p_rr, dtype=np.int32)
        rr = rr.reshape(int(self["si"][0]), int(self["si"][1]))
        rr = rr * (2 ** self["nc_proc"][1])
        # First check if it's a slice; that's the easiest case
        if self.proj_type == "slice":
            if self.proj_axis == 0:  # a column
                self.real = rr[:, self.index]
            elif self.proj_axis == 1:  # a row
                self.real = rr[self.index, :]
            return
        # Then check if there are bounds
        if self.bounds is not None:
            if self.proj_axis == 0:  # columns
                rr = rr[:, self.bounds]
            elif self.proj_axis == 1:  # rows
                rr = rr[self.bounds, :]
        # Zero all entries that are of the wrong sign
        if self.sign == "positive":
            rr[rr < 0] = 0
            projection_fn = np.amax
        elif self.sign == "negative":
            rr[rr > 0] = 0
            projection_fn = np.amin
        # Then make the projection / sum
        if self.proj_type == "projection":
            self.real = projection_fn(rr, axis=(1 - self.proj_axis))
        elif self.proj_type == "sum":
            self.real = np.sum(rr, axis=(1 - self.proj_axis))


TDataset1D = Union[Dataset1D, Dataset1DProj, Dataset1DProjVirtual]
TDatasetnD = Union[Dataset1D, Dataset1DProj, Dataset2D, Dataset1DProjVirtual]

