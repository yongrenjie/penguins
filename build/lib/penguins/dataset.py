from __future__ import annotations

from collections import UserDict, abc
from pathlib import Path
from typing import (Any, Union, Tuple, Optional,
                    TypeVar, Callable, overload)

import numpy as np               # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from . import pgplot
from .type_aliases import *


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
            # We make an exception for NS because for some reason, TopSpin stores
            # an "NS" in the indirect dimension which is just 1.
            if val_indirect is not None and par.lower() != "ns":
                if isinstance(val, float) and isinstance(val_indirect, float):
                    # ndarrays are good, so that we can do elementwise stuff
                    # like self["o1p] = self["o1"] / self["bf1"]
                    val = np.array([val_indirect, val])
                else:
                    val = (val_indirect, val)
        # Some parameters must be ints, this will save the user (and me) headaches
        int_pars = ["td", "si", "ns", "xdim", "nustd", "bytorda", "bytordp",
                    "dtypa", "dtypp"]
        if par.lower() in int_pars:
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
                else:   # triggers if didn't break -- i.e. parameter was not found
                    return None
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


def _parse_bounds(b: TBounds = "",
                  ) -> Tuple[OF, OF]:
    if isinstance(b, str):
        if b == "":
            return None, None
        elif b.startswith(".."):   # "..5" -> (None, 5)
            return None, float(b[2:])
        elif b.endswith(".."):   # "3.." -> (3, None)
            return float(b[:-2]), None
        elif ".." in b:
            x, y = b.split("..")
            xf, yf = float(x), float(y)  # let TypeError propagate
            if xf >= yf:
                raise ValueError(f"Use '{yf}..{xf}', not '{xf}..{yf}'.")
            return xf, yf
        else:
            raise ValueError(f"Invalid value {b} provided for bounds.")
    else:
        if len(b) != 2:
            raise ValueError(f"Invalid value {b} provided for bounds.")
        elif b[0] is not None and b[1] is not None and b[0] > b[1]:
            raise ValueError(f"Please use {(b[1], b[0])}, not {b}.")
        else:
            return b[0], b[1]

# -- Fundamental Dataset methods ------------------------

class _Dataset():
    """
    Defines behaviour that is common to all datasets.
    """

    def __init__(self,
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
        self._find_raw_data_paths()     # type: ignore # mixin
        self._find_proc_data_paths()    # type: ignore # mixin
        self._find_param_file_paths()   # type: ignore # mixin
        # Read in the spectral data
        self._read_raw_data()           # type: ignore # mixin
        self._read_spec()               # type: ignore # mixin

    def _initialise_pars(self) -> None:
        self.pars = _parDict(self.path)
        self["aq"] = (self["td"] / 2) / (self["sw"] * self["sfo1"])  # This is sfo1
        self["dw"] = self["aq"] * 1000000 / self["td"]
        self["o1p"] = self["o1"] / self["bf1"]   # This is BF1
        self["si"]
        self["nuc1"]
        # print(self.pars)

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
    def _find_raw_data_paths(self) -> None:
        self.path: Path
        self._p_fid = self.path.parents[1] / "fid"

    def _find_param_file_paths(self) -> None:
        self._p_procs = self.path / "procs"
        self._p_acqus = self.path.parents[1] / "acqus"

    def _read_raw_data(self) -> np.ndarray:
        datatype = "<" if self["bytorda"] == 0 else ">"  # type: ignore # mixin
        datatype += "i4"
        fid = np.fromfile(self._p_fid, dtype=datatype)
        fid = fid.reshape(int(self["td"]/2), 2)          # type: ignore # mixin
        fid = np.transpose(fid) * (2 ** self["nc"])      # type: ignore # mixin
        self.fid = fid[0] + (1j * fid[1])

    def raw_data(self) -> np.ndarray:
        return self.fid


class _1D_ProcDataMixin():

    def _find_proc_data_paths(self) -> None:
        self._p_real = self.path / "1r"        # type: ignore # mixin
        if (self.path / "1i").exists():        # type: ignore # mixin
            self._p_imag = self.path / "1i"    # type: ignore # mixin

    def _read_spec(self) -> np.ndarray:
        if self["dtypp"] == 0:                            # type: ignore # mixin
            dt = "<" if self["bytordp"] == 0 else ">"     # type: ignore # mixin
            dt += "i4"
        else:
            raise NotImplementedError("float data not yet accepted")
        self.real = np.fromfile(self._p_real, dtype=np.dtype(dt)) * (2 ** self["nc_proc"])  # type: ignore # mixin
        if hasattr(self, "_p_imag"):
            self.imag = np.fromfile(self._p_imag, dtype=np.int32) * (2 ** self["nc_proc"])  # type: ignore # mixin

    def proc_data(self,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        return self.real[self._ppm_to_slice(bounds)]

    def _ppm_to_slice(self,
                      bounds: TBounds = "",
                      ) -> slice:
        """
        Converts a tuple of chemical shifts (upper, lower) to a slice object.
        Note that upper must be greater than lower. Either upper or lower can
        be None, in which case the spectrum limit is used.
        """
        lower, upper = _parse_bounds(bounds)
        return slice(self._ppm_to_index(upper) or 0,                      # type: ignore # mixin
                     (self._ppm_to_index(lower) or self["si"] - 1) + 1)   # type: ignore # mixin

    def integrate(self,
                  peak: OF = None,
                  margin: OF = None,
                  mode: str = "sum",
                  bounds: TBounds = None,
                  ) -> float:
        """
        Integrates a region of a spectrum.
        Regions can either be defined via peak and margin, which leads to the region of
        (peak - margin) to (peak + margin), or manually via the bounds parameter. Note that
        specifying (peak, margin) overrules the bounds parameter if both are passed.

        Default mode is "sum", which simply adds up all points in the region. Alternatives are
        "max" which returns the highest peak, and "min" which returns the lowest peak.
        """
        integration_functions = {"sum": np.sum, "max": np.amax, "min": np.amin}
        if mode not in integration_functions:
            raise ValueError(f"Invalid value '{mode}' specified for integration mode.")
        # Process (peak, margin)
        if peak is not None and margin is not None:
            bounds = f"{peak-margin}..{peak+margin}"
        # We will run into this if user has not specified enough information
        if bounds is None:
            raise ValueError("Please specify integration region.")
        # Perform the integration
        finteg = integration_functions[mode]
        return finteg(self.proc_data(bounds))

    def nuclei_to_str(self
                      ) -> str:
        """
        Returns a string with the nucleus nicely formatted for use as the
        xlabel in a plot.
        """
        nuc = self["nuc1"]          # type: ignore
        elem = nuc.lstrip("1234567890")
        mass = nuc[:-len(elem)]
        return rf"$^{{{mass}}}${elem}"


class _1D_PlotMixin():

    def stage(self, *args, **kwargs) -> None:
        pgplot.stage1d(self, *args, **kwargs)  # type: ignore # mixin


# -- 2D mixins ------------------------------------------

class _2D_RawDataMixin():

    def _find_raw_data_paths(self) -> None:
        self.path: Path
        self._p_ser = self.path.parents[1] / "ser"

    def _find_param_file_paths(self) -> None:
        self._p_procs = self.path / "procs"
        self._p_acqus = self.path.parents[1] / "acqus"
        self._p_proc2s = self.path / "proc2s"
        self._p_acqu2s = self.path.parents[1] / "acqu2s"

    def _read_raw_data(self) -> np.ndarray:
        if self["dtypa"] == 0:                             # type: ignore # mixin
            dtype = "<" if self["bytorda"] == 0 else ">"   # type: ignore # mixin
            dtype += "i4"
        else:
            raise NotImplementedError("float data not yet accepted")
        # TODO
        # ser = np.fromfile(self._p_ser, dtype=??)
        pass

    def raw_data(self):
        # TODO
        raise NotImplementedError


class _2D_ProcDataMixin():

    def _find_proc_data_paths(self) -> None:
        self.path: Path
        self._p_rr = self.path / "2rr"
        if (self.path / "2ri").exists():
            self._p_ri = self.path / "2ri"
        if (self.path / "2ir").exists():
            self._p_ir = self.path / "2ir"
        if (self.path / "2ii").exists():
            self._p_ii = self.path / "2ii"

    def _read_spec(self) -> np.ndarray:
        # Helper function
        def _read_one_spec(spec_path = Path) -> np.ndarray:
            if np.all(self["dtypp"] == 0):                           # type: ignore # mixin
                dt = "<" if np.all(self["bytordp"] == 0) else ">"    # type: ignore # mixin
                dt += "i4"
            else:
                raise NotImplementedError("float data not yet accepted")
            sp = np.fromfile(spec_path, dtype=np.dtype(dt))
            sp = sp.reshape(self["si"])                              # type: ignore # mixin
            # Format according to xdim. See TopSpin "data format" manual.
            # See also http://docs.nmrfx.org/viewer/files/datasets.
            nrows, ncols = self["si"] / self["xdim"]                 # type: ignore # mixin
            submatrix_size = np.prod(self["xdim"])                   # type: ignore # mixin
            nrows = int(nrows); ncols = int(ncols)
            sp = sp.reshape(self["si"][0] * ncols, self["xdim"][1])  # type: ignore # mixin
            sp = np.vsplit(sp, nrows * ncols)
            sp = np.concatenate(sp, axis=1)
            sp = np.hsplit(sp, nrows)
            sp = np.concatenate(sp, axis=0)
            sp = sp.reshape(self["si"])                              # type: ignore # mixin
            return sp * (2 ** self["nc_proc"][1])                    # type: ignore # mixin

        # Read the spectra
        self.rr = _read_one_spec(self._p_rr)
        # Calculate a suitable baselev
        self._tsbaselev = self["s_dev"][1] * 35 * (2 ** self["nc_proc"][1])    # type: ignore # mixin
        if hasattr(self, "_p_ri"):
            self.ri = _read_one_spec(self._p_ri)
        if hasattr(self, "_p_ir"):
            self.ir = _read_one_spec(self._p_ir)
        if hasattr(self, "_p_ii"):
            self.ii = _read_one_spec(self._p_ii)

    def proc_data(self,
                  f1_bounds: TBounds = "",
                  f2_bounds: TBounds = "",
                  ) -> np.ndarray:
        f1_slice = self._ppm_to_slice(axis=0, bounds=f1_bounds)
        f2_slice = self._ppm_to_slice(axis=1, bounds=f2_bounds)
        return self.rr[f1_slice, f2_slice]

    def _ppm_to_slice(self,
                      axis: int,
                      bounds: TBounds = "",
                      ) -> slice:
        """
        Converts a tuple of chemical shifts (upper, lower) to a slice object.
        Note that upper must be greater than lower. Either upper or lower can
        be None, in which case the spectrum limit is used.
        Axis = 0 for f1, 1 for f2.
        """
        lower, upper = _parse_bounds(bounds)
        return slice(self._ppm_to_index(axis, upper) or 0,                           # type: ignore # mixin
                     (self._ppm_to_index(axis, lower) or self["si"][axis] - 1) + 1)  # type: ignore # mixin

    def integrate(self,
                  peak: Optional[Tuple[float, float]] = None,
                  margin: Optional[Tuple[float, float]] = None,
                  mode: str = "sum",
                  f1_bounds: TBounds = None,
                  f2_bounds: TBounds = None,
                  ) -> float:
        """
        Integrates a region of a spectrum.
        Regions can either be defined via peak and margin, which leads to the region of
        (peak - margin) to (peak + margin) in both dimensions, or manually via the (f1,f2)_bounds
        parameter. Note that specifying (peak, margin) overrules the bounds parameter if both are passed.

        Default mode is "sum", which simply adds up all points in the region. Alternatives are
        "max" which returns the highest peak, and "min" which returns the lowest peak.
        """
        integration_functions = {"sum": np.sum, "max": np.amax, "min": np.amin}
        if mode not in integration_functions:
            raise ValueError(f"Invalid value '{mode}' specified for integration mode.")
        # Process (peak, margin)
        if peak is not None and margin is not None:
            f1_bounds = f"{peak[0]-margin[0]}..{peak[0]+margin[0]}"
            f2_bounds = f"{peak[1]-margin[1]}..{peak[1]+margin[1]}"
        # We will run into this if user has not specified enough information
        if f1_bounds is None or f2_bounds is None:
            raise ValueError("Please specify integration region.")
        # Perform the integration
        finteg = integration_functions[mode]
        return finteg(self.proc_data(f1_bounds, f2_bounds))

    def nuclei_to_str(self,
                      ) -> Tuple[str, str]:
        """
        Returns a tuple of two strings with the nuclei nicely formatted for use
        as xlabels or ylabels in plots.
        """
        f1, f2 = self["nuc1"]              # type: ignore
        f1_elem = f1.lstrip("1234567890")
        f1_mass = f1[:-len(f1_elem)]
        f2_elem = f2.lstrip("1234567890")
        f2_mass = f2[:-len(f2_elem)]
        return (rf"$^{{{f1_mass}}}${f1_elem}", rf"$^{{{f2_mass}}}${f2_elem}")


class _2D_PlotMixin():

    def stage(self, *args, **kwargs) -> None:
        pgplot.stage2d(self, *args, **kwargs)  # type: ignore # mixin

    def find_baselev(self, *args, **kwargs):
        pgplot._make_contour_slider(self, *args, **kwargs)


# -- Actual dataset classes -----------------------------

class Dataset1D(_1D_RawDataMixin,
                _1D_ProcDataMixin,
                _1D_PlotMixin,
                _Dataset):

    def _ppm_to_index(self,
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

    def ppm_scale(self,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        max_ppm = self["o1p"] + self["sw"]/2
        min_ppm = self["o1p"] - self["sw"]/2
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self._ppm_to_slice(bounds)]

    def hz_scale(self) -> np.ndarray:
        # These use SFO, not BF
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

    def _ppm_to_index(self,
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

    def ppm_scale(self,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        max_ppm = self["o1p"][self.proj_axis] + self["sw"][self.proj_axis]/2
        min_ppm = self["o1p"][self.proj_axis] - self["sw"][self.proj_axis]/2
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self._ppm_to_slice(bounds)]

    def hz_scale(self) -> np.ndarray:
        # These use SFO, not BF
        max_hz = self["o1"][self.proj_axis] + self["sw"][self.proj_axis]/(2 * self["sfo1"][self.proj_axis])
        min_hz = self["o1"][self.proj_axis] - self["sw"][self.proj_axis]/(2 * self["sfo1"][self.proj_axis])
        return np.linspace(max_hz, min_hz, self["si"])


class Dataset2D(_2D_RawDataMixin,
                _2D_ProcDataMixin,
                _2D_PlotMixin,
                _Dataset):

    def _ppm_to_index(self,
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

    def ppm_scale(self,
                  axis: int,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        max_ppm = self["o1p"][axis] + (self["sw"][axis] / 2)
        min_ppm = self["o1p"][axis] - (self["sw"][axis] / 2)
        full_scale = np.linspace(max_ppm, min_ppm, int(self["si"][axis]))
        return full_scale[self._ppm_to_slice(axis, bounds)]

    def hz_scale(self,
                 axis: int
                 ) -> np.ndarray:
        # These use SFO, not BF
        max_hz = self["o1"][axis] + (self["sw"][axis] / (2 * self["sfo1"][axis]))
        min_hz = self["o1"][axis] - (self["sw"][axis] / (2 * self["sfo1"][axis]))
        return np.linspace(max_hz[axis], min_hz[axis], int(self["si"][axis]))

    def project(self,
                type: str,
                axis: Union[int, str],
                sign: str,
                bounds: TBounds = "",
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
                                    index_bounds=index_bounds)

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

    def __init__(self,
                 path: Union[str, Path],
                 **kwargs
                 ) -> None:
        # Set some flags so that our overriding _read_spec() method can
        # calculate the projection
        self.proj_type = kwargs["proj_type"]  # "sum", "projection", or "slice"
        self.proj_axis = kwargs["proj_axis"]  # 0 or 1
        self.sign = kwargs.get("sign", None)  # "positive" or "negative"
        self.index_bounds = kwargs.get("index_bounds", None)  # only for sum or projection
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

    def _read_spec(self) -> np.ndarray:
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
        if self.index_bounds is not None:
            if self.proj_axis == 0:  # columns
                rr = rr[:, self.index_bounds]
            elif self.proj_axis == 1:  # rows
                rr = rr[self.index_bounds, :]
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
