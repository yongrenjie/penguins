from __future__ import annotations

import math
from copy import deepcopy
from collections import UserDict, abc
from pathlib import Path
from typing import (Any, Union, Tuple, Optional,
                    TypeVar, Callable, overload)

import numpy as np               # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from . import pgplot
from .exportdeco import export
from .type_aliases import *


# -- Reading in data ------------------------------------

@export
def read(path: Union[str, Path],
         expno: int,
         procno: int = 1) -> TDatasetnD:
    """Create a Dataset object from a spectrum folder, expno, and procno.

    The subclass of Dataset returned is determined by what files are available
    in the spectrum folder. It can be either `Dataset1D`, `Dataset1DProj`, or
    `Dataset2D`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the spectrum name folder.
    expno : int
        Expno of experiment of interest.
    procno : int (optional)
        Procno of processed data. Defaults to 1.

    Returns
    -------
    Dataset
        A `Dataset1D`, `Dataset1DProj`, or `Dataset2D` object depending on the
        detected spectrum dimensionality.
    """
    p = Path(path) / str(expno) / "pdata" / str(procno)
    return read_abs(p)


@export
def read_abs(path: Union[str, Path]
             ) -> TDatasetnD:
    """Create a Dataset object directly from a procno folder.

    There is likely no reason to ever use this because `read()` is far easier
    to use, especially if you are plotting multiple spectra with different
    expnos from the same folder.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the procno folder.

    Returns
    -------
    Dataset
        A `Dataset1D`, `Dataset1DProj`, or `Dataset2D` object depending on the
        detected spectrum dimensionality.

    See Also
    --------
    read : The preferred interface for importing datasets.
    """
    p = Path(path)
    # Figure out which type of spectrum it is.
    if not (p / "procs").exists() or not (p.parents[1] / "acqus").exists():
        raise FileNotFoundError(f"Invalid path to spectrum {p}:"
                                " either procs or acqus were not found")
    if (p.parents[1] / "ser").exists():
        if (p / "used_from").exists():
            return Dataset1DProj(p)
        else:
            return Dataset2D(p)
    elif (p.parents[1] / "fid").exists():
        return Dataset1D(p)
    else:
        raise FileNotFoundError(f"Invalid path to spectrum {p}:"
                                " no raw data was found. Have you acquired the"
                                " spectrum yet?")


# -- Utility objects ------------------------------------

def _try_convert(x: Any, type: Any):
    """
    Tries to convert an input, or each element of a sequence of inputs, to a
    specific type. Returns the original input if it fails. The sequence can be
    a tuple, list, or np.ndarray (in which case it delegates to the
    np.ndarray.astype() method).

    Examples:
        >>> _try_convert(4, str)
        "4"
        >>> _try_convert(("4.3", "5.1"), float)
        (4.3, 5.1)
        >>> _try_convert(["a", "b"], int)
        ["a", "b"]
    """
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
            try:
                return x.astype(dtype=type)
            except ValueError:
                return x
        else:
            return x


def _parse_bounds(b: TBounds = "",
                  ) -> Tuple[OF, OF]:
    """
    Parses a bounds string or tuple, checking that it is valid. Returns (lower,
    upper).
    """
    if isinstance(b, str):
        if b == "" or b == "..":
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


# Parameters that are lists, i.e. can be followed by a number.
_list_pars = """AMP AMPCOIL BWFAC CAGPARS CNST CPDPRG D FCUCHAN FN_INDIRECT FS
GPNAM GPX GPY GPZ HGAIN HPMOD IN INF INP INTEGFAC L MULEXPNO P PACOIL PCPD
PEXSEL PHCOR PL PLW PLWMAX PRECHAN PROBINPUTS RECCHAN RECPRE RECPRFX RECSEL
RSEL S SELREC SP SPNAM SPOAL SPOFFS SPPEX SPW SUBNAM SWIBOX TD_INDIRECT TE_STAB
TL TOTROT XGAIN""".split()

# Parameters that should be integers, not floats. Taken from TopSpin 4.10 AU
# programming documentation.
_int_pars = """ABSG AQORDER AQSEQ AQ_MOD BC_MOD BYTORDA BYTORDP DATMOD DIGMOD
DIGTYP DS EXPNO2 EXPNO3 FNMODE FT_MOD HGAIN HOLDER HPMOD HPPRGN INTBC L LOCSHFT
LPBIN MASR MC2 ME_MOD NBL NC NCOEF NC_PROC NLEV NS NSP NZP OVERFLW PARMODE
PH_MOD PKNL POWMOD PPARMOD PRGAIN PSCAL PSIGN PROCNO2 PROCNO3 QNP REVERSE RO
RSEL SI STSI STSR SYMM TD TD0 TDEFF TDOFF TILT WBST WDW XDIM XGAIN YMAX_P
YMIN_P""".split()

# Extra integer parameters that weren't documented.
_int_pars = _int_pars + """DTYPA DTYPP GRPDLY""".split()


# -- Fundamental Dataset methods ------------------------

class _parDict(UserDict):
    """Modified dictionary for storing acquisition & processing parameters.

    Parameter names are stored as lower-case strings. When looking up a
    parameter, if its value is not already stored, the dictionary will look it
    up in the associated TopSpin parameter files. Subsequently, the value is
    cached.

    Therefore, the dictionary can be treated *as if* it were already fully
    populated when initialised. However, because values are only read and
    stored on demand, we avoid also cluttering it with a whole range of useless
    parameters upon initialisation.

    The lookup function attempts to be clever and convert the parameters to
    floats if possible; otherwise, the parameters are stored as strings. There
    are a number of exceptions to this rule, such as ``TD`` and ``SI``, which
    should be ints and not floats. The list of such parameters is extracted
    from the TopSpin documentation.

    For 2D spectra, string parameters are stored as a tuple of *(f1_value,
    f2_value)*. Float and int parameters are stored as a |ndarray| to
    facilitate elementwise manipulations (e.g. calculating ``O1P`` in both
    dimensions at one go).
    """

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
        if (par.upper() in _int_pars
                or par.rstrip("1234567890").upper() in _int_pars):
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
        # Get the parameter
        if (parr != "") and (parl in _list_pars):  # e.g. cnst2
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
                # Pick out the desired value
                val = s.split()[int(parr)]
                # Strip away surrounding angle brackets
                if val[0] == '<' and val[-1] == '>':
                    val = val[1:-1]
                return val
        else:                                             # e.g. sfo1 or rga
            with open(fp, "r") as file:
                for line in file:
                    if line.upper().startswith(f"##${par}="):
                        val = line.split(maxsplit=1)[-1].strip()
                        # Strip away surrounding angle brackets
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


class _Dataset():
    """Defines behaviour that is common to all datasets. Specifically, this
    class defines the dictionary-style lookup of NMR parameters.

    This should **never** be instantiated directly! Use :func:`~penguins.read`
    for that instead.

    Attributes
    ----------
    pars : _parDict
        Case-insensitive dictionary in which the parameters are stored. See the
        `_parDict` documentation for more details. This should never be
        accessed directly as the _Dataset special methods (``__getitem__``,
        etc.) are routed to the underlying `_parDict`.
    """

    def __init__(self,
                 path: Union[str, Path],
                 **kwargs
                 ) -> None:
        # Set up file path information
        self.path = Path(path).resolve().expanduser()
        self.expno = int(self.path.parents[1].name)
        self.procno = int(self.path.name)
        # Initialise _parDict and some key parameters
        self._initialise_pars()
        # Get paths to data and parameter files
        self._find_raw_data_paths()     # type: ignore # mixin
        self._find_proc_data_paths()    # type: ignore # mixin
        self._find_param_file_paths()   # type: ignore # mixin
        # Don't read in the actual data, *until* the user asks for it. In other
        # words, data reading is lazy. Haskell is really growing on me...

    def _initialise_pars(self) -> None:
        self.pars = _parDict(self.path)
        self["aq"] = (self["td"] / 2) / (self["sw"] * self["sfo1"])  # This is sfo1
        if isinstance(self["aq"], np.ndarray):  # 2D
            self["dw"] = self["aq"][1] * 1000000 / self["td"][1]
            self["inf1"]
        else:  # 1D
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

    def __delitem__(self, par: str):
        self.pars.__delitem__(par)


# -- 1D mixins ------------------------------------------

class _1D_RawDataMixin():
    """Defines behaviour that is applicable for 1D raw data, i.e. ``fid``
    files. Also contains a few private methods which initialise (for example)
    the paths to the parameter files.

    Attributes
    ----------
    fid : ndarray
        Complex |ndarray| of the FID. No preprocessing (e.g. removal of the
        group delay) is carried out.

        Note also that for Bruker data, real and imaginary components are
        sampled sequentially instead of simultaneously. Therefore, the
        imaginary component of ``fid[0]`` is actually acquired *after* the real
        component of ``fid[0]`` (the exact time difference is the dwell width,
        ``DW).`` In principle this does not cause any issues as it can be
        corrected for post-Fourier transformation using the
        :func:`~numpy.fft.fftshift()` function.
    """

    @property
    def fid(self):
        """
        Complex-valued FID. This function exists only to make the FID lazily
        read, i.e. don't read it in unless the user asks for it explicitly
        either with ds.fid or ds.raw_data().
        """
        try:
            return self._fid
        except AttributeError:
            self._read_raw_data()
            return self._fid

    def _find_raw_data_paths(self) -> None:
        self.path: Path
        self._p_fid = self.path.parents[1] / "fid"

    def _find_param_file_paths(self) -> None:
        self._p_procs = self.path / "procs"
        self._p_acqus = self.path.parents[1] / "acqus"

    def _read_raw_data(self) -> np.ndarray:
        # Determine datatype of FID
        if self["dtypa"] == 0:                               # type: ignore # mixin
            datatype = "<" if self["bytorda"] == 0 else ">"  # type: ignore # mixin
            datatype += "i4"
            fid = np.fromfile(self._p_fid, dtype=datatype)
            fid = fid[:self["td"]]                           # type: ignore # mixin
            fid = fid.reshape(int(self["td"]/2), 2)          # type: ignore # mixin
            fid = np.transpose(fid) * (2 ** self["nc"])      # type: ignore # mixin
            self._fid = fid[0] + (1j * fid[1])
        elif self["dtypa"] == 2:
            datatype = "<" if self["bytorda"] == 0 else ">"  # type: ignore # mixin
            datatype += "d"
            fid = np.fromfile(self._p_fid, dtype=datatype)
            fid = fid[:self["td"]]                           # type: ignore # mixin
            fid = fid.reshape(int(self["td"]/2), 2)          # type: ignore # mixin
            fid = np.transpose(fid)
            self._fid = fid[0] + (1j * fid[1])

    def raw_data(self) -> np.ndarray:
        """
        Returns the FID as a complex |ndarray|.
        """
        return self.fid


class _1D_ProcDataMixin():
    """Defines behaviour that is applicable for 1D processed data.

    Attributes
    ----------
    real : ndarray
        Real-valued |ndarray| containing the real spectrum. ``real[0]``
        contains the left-most point of the spectrum, i.e. the greatest
        chemical shift.

    imag : ndarray
        This attribute only applies to `Dataset1D` instances. The projection
        classes do not have this attribute.

        Real-valued |ndarray| containing the imaginary spectrum. ``imag[0]``
        contains the left-most point of the spectrum, i.e. the greatest
        chemical shift.
    """

    @property
    def real(self):
        try:
            return self._real
        except AttributeError:
            self._read_spec(spectype="real")
            return self._real

    @property
    def imag(self):
        try:
            return self._imag
        except AttributeError:
            self._read_spec(spectype="imag")
            return self._imag

    def _find_proc_data_paths(self) -> None:
        self._p_real = self.path / "1r"        # type: ignore # mixin
        self._p_imag = self.path / "1i"        # type: ignore # mixin
        # if the imaginary part does not exist, the error will be caught later,
        # specifically at _read_one_spec().

    def _read_spec(self, spectype: str) -> None:
        def _read_one_spec(spec_path: Path) -> np.ndarray:
            """Helper function."""
            # Determine datatype
            if self["dtypp"] == 0:                            # type: ignore # mixin
                dt = "<" if self["bytordp"] == 0 else ">"     # type: ignore # mixin
                dt += "i4"
            else:
                raise NotImplementedError("float data not yet accepted")
            # Return the spectrum as an ndarray
            if not spec_path.exists():
                raise FileNotFoundError(f"processed data file {spec_path}"
                                        " not found.")
            return (np.fromfile(spec_path, dtype=np.dtype(dt))
                    * (2 ** float(self["nc_proc"])))          # type: ignore # mixin

        if spectype == "real":
            self._real = _read_one_spec(self._p_real)
        elif spectype == "imag":
            self._imag = _read_one_spec(self._p_imag)
        else:
            raise ValueError("_read_spec(): invalid spectype")

    def proc_data(self,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        """Returns the real part of the spectrum as a real-valued |ndarray|.

        Note that if (for example) a magnitude mode calculation has been
        performed, then the "real" part is actually the magnitude mode
        spectrum. In short, the "real" part is whatever is stored in the ``1r``
        file.

        This is used in constructing the *y*-values to be plotted.

        Parameters
        ----------
        bounds : str or (float, float), optional
            Bounds can be specified as a string ``lower..upper`` or a tuple of
            floats ``(lower, upper)``, upon which the appropriate slice of the
            spectrum will be taken.

        Returns
        -------
        ndarray
            The real spectrum or the slice of interest.
        """
        return self.real[self.bounds_to_slice(bounds)]

    def integrate(self,
                  peak: OF = None,
                  margin: OF = None,
                  bounds: TBounds = None,
                  mode: str = "sum",
                  ) -> float:
        """Integrates a region of a spectrum.

        Regions can either be defined via *peak* and *margin*, which leads to
        the region of *(peak - margin)* to *(peak + margin)*, or manually via
        the *bounds* parameter. Note that specifying *(peak, margin)* overrules
        the *bounds* parameter if both are passed.

        Parameters
        ----------
        peak : float, optional
            The chemical shift of the peak of interest.
        margin : float, optional
            The integration margin which extends on either side of the peak.
        bounds : str or (float, float), optional
            Integration bounds which can be directly specified in the usual
            format. Note that passing *(peak, margin)* will overrule this
            parameter.
        mode : {"sum", "max", "min"}, optional
            Mode of integration. ``sum`` (the default) directly adds up all
            points in the region, ``max`` finds the greatest intensity, and
            ``min`` finds the lowest intensity.

        Returns
        -------
        float
            The value of the integral.
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

    def bounds_to_slice(self,
                        bounds: TBounds = "",
                        ) -> slice:
        """Converts a string ``lower..upper`` or a tuple of chemical shifts
        ``(upper, lower)`` to a slice object, which can be used to slice a
        spectrum |ndarray|. Note that ``upper`` must be greater than
        ``lower``.
        """
        lower, upper = _parse_bounds(bounds)
        return slice(self.ppm_to_index(upper) or 0,                      # type: ignore # mixin
                     (self.ppm_to_index(lower) or self["si"] - 1) + 1)   # type: ignore # mixin

    def to_magnitude(self) -> _1D_ProcDataMixin:
        """
        Calculates the magnitude mode spectrum and returns it as a new
        Dataset1D object.
        """
        new_ds = deepcopy(self)
        try:
            new_ds._real = np.abs(new_ds.real + 1j * new_ds.imag)
        except AttributeError:   # no imag
            raise TypeError("The imaginary part of the spectrum was not"
                            " found.") from None
        new_ds._imag = 0
        return new_ds

    def mc(self) -> _1D_ProcDataMixin:
        """
        Alias for `to_magnitude()`.
        """
        return self.to_magnitude()


class _1D_PlotMixin():
    """Defines 1D plotting methods."""

    def stage(self, *args, **kwargs) -> None:
        """Calls :func:`penguins.pgplot._stage1d` on the dataset."""
        pgplot._stage1d(self, *args, **kwargs)  # type: ignore # mixin


# -- 2D mixins ------------------------------------------

class _2D_RawDataMixin():
    """Defines behaviour that is applicable for 2D raw data, i.e. ``ser``
    files.

    There are no functions which actually read the ``ser`` file (I haven't
    implemented those yet, as I've never needed it), but this mixin defines a
    few private methods which initialise (for example) the paths to the
    parameter files, so it's not useless at all.
    """

    @property
    def ser(self):
        """
        Complex-valued 2D raw data matrix (ser file). This function exists only
        to make the ser lazily read, i.e. don't read it in unless the user asks
        for it explicitly either with ds.ser or ds.raw_data().
        """
        try:
            return self._ser
        except:
            self._read_raw_data()
            return self._ser

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
        # Read in the data from the ser file.
        ser = np.fromfile(self._p_ser, dtype=dtype)
        # Reshape the matrix according to TD. Note that in the ser file, each
        # new FID always begins at a new block of 256 data points. Effectively,
        # this means that TD2 is rounded up to the nearest multiple of 256.
        td1, td2 = self["td"]   # type: ignore # mixin
        if td2 % 256 != 0:
            td2_eff = math.ceil(td2 / 256) * 256
            ser = ser.reshape((td1, td2_eff))
            ser = ser[:, :td2]
        else:
            ser = ser.reshape((td1, td2))
        # Combine real and imaginary parts. The -1 allows numpy to
        # automatically calculate the value that should go there.
        ser = ser.astype(np.complex128)  # otherwise the imaginary part is lost
        ser = ser.reshape((td1, -1, 2))
        ser[:,:,1] = ser[:,:,1] * 1j
        self._ser = ser.sum(axis=2) * (2 ** self["nc"])  # type: ignore # mixin

    def raw_data(self):
        return self.ser


class _2D_ProcDataMixin():

    @property
    def rr(self):
        try:
            return self._rr
        except AttributeError:
            self._read_spec(spectype="rr")
            return self._rr

    @property
    def ri(self):
        try:
            return self._ri
        except AttributeError:
            self._read_spec(spectype="ri")
            return self._ri

    @property
    def ir(self):
        try:
            return self._ir
        except AttributeError:
            self._read_spec(spectype="ir")
            return self._ir

    @property
    def ii(self):
        try:
            return self._ii
        except AttributeError:
            self._read_spec(spectype="ii")
            return self._ii

    @property
    def ts_baselev(self):
        """TopSpin base level for contours."""
        return (self["s_dev"][1]
                * 35
                * (2 ** float(self["nc_proc"][1])))   # type: ignore # mixin

    def _find_proc_data_paths(self) -> None:
        self.path: Path
        self._p_rr = self.path / "2rr"
        self._p_ri = self.path / "2ri"
        self._p_ir = self.path / "2ir"
        self._p_ii = self.path / "2ii"
        # if the imaginary parts do not exist, the error will be caught later,
        # specifically at _read_one_spec().

    def _read_spec(self, spectype: str) -> None:
        # Helper function
        def _read_one_spec(spec_path = Path) -> np.ndarray:
            if not spec_path.exists():
                raise FileNotFoundError(f"processed data file {spec_path}"
                                        " not found.")
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
            return sp * (2 ** float(self["nc_proc"][1]))             # type: ignore # mixin

        # Read the spectra
        if spectype == "rr":
            self._rr = _read_one_spec(self._p_rr)
        elif spectype == "ri":
            self._ri = _read_one_spec(self._p_ri)
        elif spectype == "ir":
            self._ir = _read_one_spec(self._p_ir)
        elif spectype == "ii":
            self._ii = _read_one_spec(self._p_ii)

    def proc_data(self,
                  f1_bounds: TBounds = "",
                  f2_bounds: TBounds = "",
                  ) -> np.ndarray:
        """Returns the real part of the spectrum (the 'RR quadrant') as a
        two-dimensional, real-valued |ndarray|.

        This is used in constructing the *z*-values to be plotted.

        Note that if a magnitude mode calculation has been performed, this will
        return the magnitude mode spectrum (i.e. it returns whatever is in
        TopSpin's ``2rr`` file).

        Parameters
        ----------
        f1_bounds : str or (float, float), optional
            Bounds for the indirect dimension.
        f2_bounds : str or (float, float), optional
            Bounds for the direct dimension.

        Returns
        -------
        ndarray
            The doubly real spectrum, or the section of interest.
        """
        f1_slice = self.bounds_to_slice(axis=0, bounds=f1_bounds)
        f2_slice = self.bounds_to_slice(axis=1, bounds=f2_bounds)
        return self.rr[f1_slice, f2_slice]

    def integrate(self,
                  peak: Optional[Tuple[float, float]] = None,
                  margin: Optional[Tuple[float, float]] = None,
                  f1_bounds: TBounds = None,
                  f2_bounds: TBounds = None,
                  mode: str = "sum",
                  ) -> float:
        """Integrates a region of a spectrum.

        The interface is exactly analogous to the 1D version
        (:meth:`~_1D_ProcDataMixin.integrate`), except that *peak* and *margin*
        now need to be specified as tuples of *(f1_shift, f2_shift)*, or
        *bounds* must be specified as *f1_bounds* and *f2_bounds* separately.

        Parameters
        ----------
        peak : (float, float), optional
            The chemical shifts of the peak of interest.
        margin : (float, float), optional
            The integration margins which extends on all sides of the peak.
            The first number refers to the margin in the indirect dimension,
            the second the margin in the direct dimension.
        f1_bounds : str or (float, float), optional
            Integration bounds for the indirect dimension which can be directly
            specified in the usual format. Note that passing *(peak, margin)*
            will overrule the *f1_bounds* and *f2_bounds* parameters.
        f2_bounds : str or (float, float), optional
            Integration bounds for the direct dimension which can be directly
            specified in the usual format.
        mode : {"sum", "max", "min"}, optional
            Mode of integration. ``sum`` (the default) directly adds up all
            points in the region, ``max`` finds the greatest intensity, and
            ``min`` finds the lowest intensity.

        Returns
        -------
        float
            The value of the integral.
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

    def bounds_to_slice(self,
                        axis: int,
                        bounds: TBounds = "",
                        ) -> slice:
        """Converts a string ``lower..upper`` or a tuple of chemical shifts
        ``(upper, lower)`` to a slice object, which can be used to slice a
        spectrum |ndarray|.

        Parameters
        ----------
        axis : int from {0, 1}
            0 for indirect dimension, 1 for direct dimension.

        bounds : str or (float, float), optional
            Bounds given in the usual format.

        Returns
        -------
        slice
            Slice object for the requested axis.
        """
        lower, upper = _parse_bounds(bounds)
        return slice(self.ppm_to_index(axis, upper) or 0,                           # type: ignore # mixin
                     (self.ppm_to_index(axis, lower) or self["si"][axis] - 1) + 1)  # type: ignore # mixin

    def nuclei_to_str(self,
                      ) -> Tuple[str, str]:
        """Returns a tuple of strings with the nuclei nicely formatted in LaTeX
        syntax.  Can be directly used with e.g. matplotlib.
        """
        f1, f2 = self["nuc1"]              # type: ignore
        f1_elem = f1.lstrip("1234567890")
        f1_mass = f1[:-len(f1_elem)]
        f2_elem = f2.lstrip("1234567890")
        f2_mass = f2[:-len(f2_elem)]
        return (rf"$\rm ^{{{f1_mass}}}{f1_elem}$", rf"$\rm ^{{{f2_mass}}}{f2_elem}$")

    def to_magnitude(self, axis: int) -> _2D_ProcDataMixin:
        """
        Calculates the magnitude mode spectrum along the specified axis and
        returns it as a new Dataset2D object.

        Parameters
        ----------
        axis : int from {0, 1}
            The axis along which to perform the magnitude calculation. 0 for
            f1, or 1 for f2.
        """
        new_ds = deepcopy(self)
        try:
            if axis == 0:
                new_ds._rr = np.abs(new_ds.rr + 1j * new_ds.ri)
                new_ds._ir = np.abs(new_ds.ir + 1j * new_ds.ii)
                new_ds._ri = 0
                new_ds._ii = 0
            elif axis == 1:
                new_ds._rr = np.abs(new_ds.rr + 1j * new_ds.ir)
                new_ds._ri = np.abs(new_ds.ri + 1j * new_ds.ii)
                new_ds._ir = 0
                new_ds._ii = 0
            else:
                raise ValueError("to_magnitude(): axis must be 0 (for"
                                 " magnitude mode in f1) or 1 (for f2).")
        except AttributeError:
            raise TypeError("The imaginary part of the spectrum was not"
                            " found.") from None
        return new_ds

    def xf1m(self) -> _2D_ProcDataMixin:
        """
        Alias for ``to_magnitude(axis=0)``, i.e. magnitude mode calculation
        along f1.
        """
        return self.to_magnitude(axis=0)

    def xf2m(self) -> _2D_ProcDataMixin:
        """
        Alias for ``to_magnitude(axis=1)``, i.e. magnitude mode calculation
        along f2.
        """
        return self.to_magnitude(axis=1)

    def xfbm(self) -> _2D_ProcDataMixin:
        """
        Performs magnitude mode calculation along both axes. ds.xfbm() is
        equivalent to ds.xf1m().xf2m(). It is manually implemented here for
        efficiency reasons.
        """
        new_ds = deepcopy(self)
        try:
            new_ds._rr = np.sqrt(new_ds.rr ** 2 + new_ds.ri ** 2
                                 + new_ds.ir ** 2 + new_ds.ii ** 2)
            new_ds._ri = 0
            new_ds._ir = 0
            new_ds._ii = 0
        except AttributeError:
            raise TypeError("The imaginary part of the spectrum was not"
                            " found.") from None
        return new_ds


class _2D_PlotMixin():
    """Defines 2D plotting methods."""

    def stage(self, *args, **kwargs) -> None:
        """Calls :func:`penguins.pgplot._stage2d` on the dataset."""
        pgplot._stage2d(self, *args, **kwargs)  # type: ignore # mixin

    def find_baselev(self, *args, **kwargs):
        """Calls :func:`penguins.pgplot._find_baselev` on the dataset."""
        pgplot._find_baselev(self, *args, **kwargs)


# -- Actual dataset classes -----------------------------

class Dataset1D(_1D_RawDataMixin,
                _1D_ProcDataMixin,
                _1D_PlotMixin,
                _Dataset):
    """Dataset object representing 1D spectra.

    Inherits from: `_1D_RawDataMixin`, `_1D_ProcDataMixin`, `_1D_PlotMixin`,
    and `_Dataset`.
    """

    def ppm_to_index(self,
                     ppm: Optional[float]
                     ) -> Optional[int]:
        """Converts a chemical shift into the index which is closest to the
        chemical shift.

        Parameters
        ----------
        ppm : float (optional)
            The chemical shift of interest.

        Returns
        -------
        index : int
            The index, or None if ppm is None.
        """
        if ppm is None:
            return None
        ppm_scale = self.ppm_scale()
        max_ppm, min_ppm = ppm_scale[0], ppm_scale[-1]
        if ppm > max_ppm or ppm < min_ppm:
            raise ValueError(f"Chemical shift {ppm} is outside spectral window.")
        spacing = (max_ppm - min_ppm)/(self["si"] - 1)
        x = 1 + round((max_ppm - ppm)/spacing)
        return int(x)

    def ppm_scale(self,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        """Constructs an |ndarray| of the chemical shifts at each point of the
        spectrum, in descending order of chemical shift.

        This is used in generating the *x*-values for plotting.

        Parameters
        ----------
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of chemical shifts.
        """
        max_ppm = self["offset"]
        min_ppm = max_ppm - (self["sw_p"] / self["sfo1"])
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self.bounds_to_slice(bounds)]

    def hz_scale(self,
                 bounds: TBounds = "",
                 ) -> np.ndarray:
        """Constructs an |ndarray| of the frequencies (in units of Hz) at each
        point of the spectrum, in descending order of frequency.

        Parameters
        ----------
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of frequencies.
        """
        full_hz_scale = self.ppm_scale() * self["sfo1"]
        return full_hz_scale[self.bounds_to_slice(bounds)]

    def nuclei_to_str(self
                      ) -> str:
        """Returns a string with the nucleus nicely formatted in LaTeX syntax.
        Can be directly used with e.g. matplotlib.
        """
        nuc = self["nuc1"]          # type: ignore
        elem = nuc.lstrip("1234567890")
        mass = nuc[:-len(elem)]
        return rf"$\rm ^{{{mass}}}{elem}$"


class Dataset1DProj(_2D_RawDataMixin,
                    _1D_ProcDataMixin,
                    _1D_PlotMixin,
                    _Dataset):
    """Dataset object representing 1D projections or slices of 2D spectra,
    which have been generated inside TopSpin.

    Inherits from: `_2D_RawDataMixin`, `_1D_ProcDataMixin`, `_1D_PlotMixin`,
    and `_Dataset`.

    Notes
    -----
    The implementation of these methods has to be different from the equivalent
    methods on `Dataset1D`, because the parameters (e.g. O1, SW) are read as
    2-element arrays (for both dimensions) but the returned value must select
    the correct projection axis.
    """

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

    def ppm_to_index(self,
                     ppm: Optional[float]
                     ) -> Optional[int]:
        """Converts a chemical shift into the index which is closest to the
        chemical shift.

        Parameters
        ----------
        ppm : float (optional)
            The chemical shift of interest.

        Returns
        -------
        index : int
            The index, or None if ppm is None.
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
        """Constructs an |ndarray| of the chemical shifts at each point of the
        spectrum, in descending order of chemical shift.

        This is used in generating the *x*-values for plotting.

        Parameters
        ----------
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of chemical shifts.
        """
        max_ppm = self["o1p"][self.proj_axis] + self["sw"][self.proj_axis]/2
        min_ppm = self["o1p"][self.proj_axis] - self["sw"][self.proj_axis]/2
        full_scale = np.linspace(max_ppm, min_ppm, self["si"])
        return full_scale[self.bounds_to_slice(bounds)]

    def hz_scale(self,
                 bounds: TBounds = "",
                 ) -> np.ndarray:
        """Constructs an |ndarray| of the frequencies (in units of Hz) at each
        point of the spectrum, in descending order of frequency.

        Parameters
        ----------
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of frequencies.
        """
        # These use SFO, not BF
        max_hz = self["o1"][self.proj_axis] + (self["sw"][self.proj_axis]
                                               * self["sfo1"][self.proj_axis]
                                               / 2)
        min_hz = self["o1"][self.proj_axis] - (self["sw"][self.proj_axis]
                                               * self["sfo1"][self.proj_axis]
                                               / 2)
        full_hz_scale = np.linspace(max_hz, min_hz, self["si"])
        return full_hz_scale[self.bounds_to_slice(bounds)]

    def nuclei_to_str(self
                      ) -> str:
        """Returns a string with the nucleus nicely formatted in LaTeX syntax.
        Can be directly used with e.g. matplotlib.
        """
        nuc = self["nuc1"][self.proj_axis]   # type: ignore
        elem = nuc.lstrip("1234567890")
        mass = nuc[:-len(elem)]
        return rf"$\rm ^{{{mass}}}{elem}$"


class Dataset2D(_2D_RawDataMixin,
                _2D_ProcDataMixin,
                _2D_PlotMixin,
                _Dataset):
    """Dataset object representing 2D spectra.

    Inherits from: `_2D_RawDataMixin`, `_2D_ProcDataMixin`, `_2D_PlotMixin`,
    and `_Dataset`.
    """

    def ppm_to_index(self,
                     axis: int,
                     ppm: Optional[float]
                     ) -> Optional[int]:
        """Converts a chemical shift into the index which is closest to the
        chemical shift.

        Parameters
        ----------
        axis : int
            0 for f1 (indirect dimension), 1 for f2 (direct dimension).
        ppm : float (optional)
            The chemical shift of interest.

        Returns
        -------
        index : int
            The index, or None if ppm is None.
        """
        if ppm is None:
            return None
        ppm_scale = self.ppm_scale(axis=axis)
        max_ppm, min_ppm = ppm_scale[0], ppm_scale[-1]
        if ppm > max_ppm or ppm < min_ppm:
            raise ValueError(f"Chemical shift {ppm} is outside spectral window.")
        spacing = (max_ppm - min_ppm)/(self["si"][axis] - 1)
        x = 1 + round((max_ppm - ppm)/spacing)
        return int(x)

    def ppm_scale(self,
                  axis: int,
                  bounds: TBounds = "",
                  ) -> np.ndarray:
        """Constructs an |ndarray| of the chemical shifts at each point of the
        spectrum, in descending order of chemical shift.

        This is used in generating the *x*- and *y*-values for plotting.

        Parameters
        ----------
        axis : int
            0 for f1 (indirect dimension), 1 for f2 (direct dimension).
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of chemical shifts.
        """
        max_ppm = self["offset"][axis]
        min_ppm = max_ppm - (self["sw_p"][axis] / self["sfo1"][axis])
        full_scale = np.linspace(max_ppm, min_ppm, self["si"][axis])
        return full_scale[self.bounds_to_slice(axis, bounds)]

    def hz_scale(self,
                 axis: int,
                 bounds: TBounds = "",
                 ) -> np.ndarray:
        """Constructs an |ndarray| of the frequencies (in units of Hz) at each
        point of the spectrum, in descending order of frequency.

        Parameters
        ----------
        axis : int
            0 for f1 (indirect dimension), 1 for f2 (direct dimension).
        bounds : str or (float, float), optional
            Bounds specified in the usual manner.

        Returns
        -------
        scale : ndarray
            The appropriate slice of frequencies.
        """
        # These use SFO, not BF
        full_hz_scale = self.ppm_scale(axis=axis) * self["sfo1"][axis]
        return full_hz_scale[self.bounds_to_slice(axis, bounds)]

    def project(self,
                axis: Union[int, str],
                sign: str,
                bounds: TBounds = "",
                ) -> Dataset1DProjVirtual:
        """Make a 1D projection from a 2D spectrum.

        Parameters
        ----------
        axis : int or str from {0, "column", 1, "row"}
            The axis to project *onto*, 0 / "column" being f1 and 1 / "row"
            being f2. This can be very confusing, so an example will help.

            Projections *onto* f1 will collapse multiple columns into one
            column. This should be done by passing ``0`` or ``column`` as the
            *axis* argument. For example, if you used this on a C–H HSQC, you
            would get a projection with <sup>13</sup>C chemical shifts.
        sign : str from {"positive", "pos", "negative", "neg"}
            The sign desired. Using ``positive`` (or the short form ``pos``)
            means that the greatest point along the collapsed axis will be
            taken, and vice versa for ``negative``/``neg``.
        bounds : str or (float, float), optional
            Bounds specified in the usual manner, representing the segment of
            chemical shifts that should be collapsed. That is to say, if you
            are projecting *onto* f2, then *bounds* would represent the section
            of f1 chemical shifts to collapse. If not provided, then defaults
            to the entire range of chemical shifts along the collapsed axis.

        Returns
        -------
        proj : Dataset1DProjVirtual
            A `Dataset1DProjVirtual` object that is similar in every way to a
            typical `Dataset1DProj` and can be plotted, integrated, etc. in the
            same manner. The actual projection can be accessed using
            `_1D_ProcDataMixin.proc_data`, which `Dataset1DProj` inherits.
        """
        # Determine the axis
        if axis == "column":  # sum / projection of columns, or onto f1
            axis = 0
        elif axis == "row":   # sum / projection of rows, or onto f2
            axis = 1
        if axis not in [0, 1]:
            raise ValueError(f"Invalid value for axis '{axis}'")
        # Allow some short forms for sign
        if sign not in ["positive", "negative", "pos", "neg"]:
            raise ValueError(f"Invalid value for sign '{sign}'")
        if sign == "pos":
            sign = "positive"
        elif sign == "neg":
            sign = "negative"
        # For some reason mypy doesn't realise that axis must be an int by here.
        index_bounds = self.bounds_to_slice(axis=(1 - axis), bounds=bounds)  # type: ignore
        return Dataset1DProjVirtual(self.path, proj_type="projection",
                                    proj_axis=axis, sign=sign,
                                    index_bounds=index_bounds,
                                    rr=self.rr)

    def f1projp(self,
                bounds: TBounds = ""
                ) -> Dataset1DProjVirtual:
        """Alias for ``project(axis="column", sign="pos")``. See `project`."""
        return self.project(axis="column", sign="pos", bounds=bounds)

    def f1projn(self,
                bounds: TBounds = ""
                ) -> Dataset1DProjVirtual:
        """Alias for ``project(axis="column", sign="neg")``. See `project`."""
        return self.project(axis="column", sign="neg", bounds=bounds)

    def f2projp(self,
                bounds: TBounds = ""
                ) -> Dataset1DProjVirtual:
        """Alias for ``project(axis="row", sign="pos")``. See `project`."""
        return self.project(axis="row", sign="pos", bounds=bounds)

    def f2projn(self,
                bounds: TBounds = ""
                ) -> Dataset1DProjVirtual:
        """Alias for ``project(axis="row", sign="neg")``. See `project`."""
        return self.project(axis="row", sign="neg", bounds=bounds)

    def sum(self,
            axis: Union[int, str],
            bounds: TBounds = "",
            ) -> Dataset1DProjVirtual:
        """Make a 1D sum from a 2D spectrum.

        Parameters
        ----------
        axis : int or str from {0, "column", 1, "row"}
            The axis to sum onto. ``0`` / ``column`` is f1 (i.e. adding up
            multiple columns) and ``1`` / ``row`` is f2 (i.e. adding up
            multiple rows).
        bounds : str or (float, float), optional
            Bounds specified in the usual manner, representing the segment of
            chemical shifts that should be collapsed. That is to say, if you
            are projecting *onto* f2, then *bounds* would represent the section
            of f1 chemical shifts to collapse. If not provided, then defaults
            to the entire range of chemical shifts along the collapsed axis.

        Returns
        -------
        proj : Dataset1DProjVirtual
            A `Dataset1DProjVirtual` object that is similar in every way to a
            typical `Dataset1DProj` and can be plotted, integrated, etc. in the
            same manner. The actual sum can be accessed using
            `_1D_ProcDataMixin.proc_data`, which `Dataset1DProj` inherits.
        """
        # Determine the axis
        if axis == "column":  # sum / projection of columns, or onto f1
            axis = 0
        elif axis == "row":   # sum / projection of rows, or onto f2
            axis = 1
        if axis not in [0, 1]:
            raise ValueError(f"Invalid value for axis '{axis}'")
        # For some reason mypy doesn't realise that axis must be an int by here.
        # Dynamic typing problems...
        index_bounds = self.bounds_to_slice(axis=(1 - axis), bounds=bounds)  # type: ignore
        return Dataset1DProjVirtual(self.path, proj_type="sum",
                                    proj_axis=axis, index_bounds=index_bounds,
                                    rr=self.rr)

    def f1sum(self,
              bounds: TBounds = ""
              ) -> Dataset1DProjVirtual:
        """Alias for ``sum(axis="column")``. See `sum`."""
        return self.sum(axis="column", bounds=bounds)

    def f2sum(self,
              bounds: TBounds = ""
              ) -> Dataset1DProjVirtual:
        """Alias for ``sum(axis="row")``. See `sum`."""
        return self.sum(axis="row", bounds=bounds)

    def slice(self,
              axis: Union[int, str],
              ppm: float,
              ) -> Dataset1DProjVirtual:
        """Extract a 1D slice from a 2D spectrum.

        Parameters
        ----------
        axis : int or str from {0, "column", 1, "row"}
            Axis to slice along. To extract a column (which corresponds to a
            slice along f1, at one specific value of f2), use ``0`` or
            ``column``, and vice versa for ``row``.
        ppm : float
            The chemical shift of the other axis to slice at. For example, if
            you are extracting a column, then this would be the f2 chemical
            shift of interest.

        Returns
        -------
        proj : Dataset1DProjVirtual
            A `Dataset1DProjVirtual` object that is similar in every way to a
            typical `Dataset1DProj` and can be plotted, integrated, etc. in the
            same manner. The actual projection or sum can be accessed using
            `_1D_ProcDataMixin.proc_data`, which `Dataset1DProj` inherits.
        """
        # Find axis
        if axis == "column":  # extract a column
            axis = 0
        elif axis == "row":   # extract a row
            axis = 1
        if axis not in [0, 1]:
            raise ValueError(f"Invalid value for axis '{axis}'")
        # For some reason mypy doesn't realise that axis must be an int by here.
        index = self.ppm_to_index(axis=(1 - axis), ppm=ppm)   # type: ignore
        return Dataset1DProjVirtual(self.path, proj_type="slice",
                                    proj_axis=axis, index=index,
                                    rr=self.rr)


class Dataset1DProjVirtual(Dataset1DProj):
    """Dataset representing 1D projections which have been constructed by
    calling the `project`, `slice`, or `sum` methods (or their short forms) on
    `Dataset2D` objects.

    This is a subclass of `Dataset1DProj`, so the available methods are exactly
    the same.

    See the __init__() docstring for implementation details.
    """

    def __init__(self,
                 path: Union[str, Path],
                 rr: np.ndarray,
                 sign: OS = None,
                 index_bounds: Optional[slice] = None,
                 index: OI = None,
                 **kwargs
                 ) -> None:
        """
        The construction of this class is necessarily slightly tortuous. The
        call to the constructor (in Dataset2D.project() / .slice() / .sum())
        will contain the doubly real part of the parent dataset, together with
        all the information needed to figure out the correct data for this
        dataset (e.g. proj_type, proj_axis, sign...).

        The sequence of events in __init__ is broadly as follows:

         1. Get all of this data from **kwargs into instance variables.

         2. Perform generic dataset initialisation. This sets instance
            variables such as self.path, etc. and also instantiates the
            _parDict instance. Because _parDict uses the path to the original
            2D dataset, any parameters read here will be those of the original
            2D dataset.

         3. Overwrite the SI parameter manually. This is necessary to maintain
            compatibility with the ppm_scale() and hz_scale() methods inherited
            from Dataset1DProj. At an even deeper level, the reason why the
            Dataset1DProj methods are like this is because in "genuine" 1D
            projections made in TopSpin, the SI parameter is one-dimensional.
            So, by manually setting SI, we are actually mimicking this TopSpin
            behaviour.

        The instance variables stored in step (1) are not used instantaneously,
        because the data is lazily calculated. Eventually, it will be: see the
        _read_spec() method.
        """
        # Step 1 of the docstring
        self.proj_type = kwargs["proj_type"]  # "sum", "projection", or "slice"
        self.proj_axis = kwargs["proj_axis"]  # 0 or 1
        self.sign = sign  # "positive" or "negative", or None for sums/slices
        self.index_bounds = index_bounds  # only for projections/sums
        self.index = index                # only for slice
        self._rr = rr
        # Step 2 of the docstring
        Dataset1DProj.__init__(self, path)
        # Step 3 of the docstring
        self["si"] = self["si"][self.proj_axis]

    def _initialise_pars(self):
        # We don't want to inherit this from Dataset1DProj, because that
        # tries to look for the used_from file, and we don't have that.
        _Dataset._initialise_pars(self)

    @property
    def real(self):
        try:
            return self._real
        except AttributeError:
            self._read_spec(spectype="real")
            return self._real

    def _read_spec(self, spectype: str) -> np.ndarray:
        """
        This method does not actually *read* a spectrum, but rather calculates
        it from the parent 2D rr part, using the instance variables that were
        stored during initialisation.
        """
        # The imaginary part can just be 0. It does not exist, but having a
        # zero value allows us to do things like mc() on a projection.
        if spectype == "imag":
            self._imag = 0
        # This is the real part of this function. Ha ha.
        elif spectype == "real":
            # Otherwise, we can proceed with the actual calculation.
            # For a slice, the projection axis and the index specifying which
            # row/column to take are all we need.
            if self.proj_type == "slice":
                if self.proj_axis == 0:  # a column
                    self._real = self._rr[:, self.index]
                elif self.proj_axis == 1:  # a row
                    self._real = self._rr[self.index, :]
                return
            # If we reach here, then it's a projection or sum. First we need to
            # check the bounds to make sure we take the projection/sum of the
            # appropriate spectral region. We make sure to not override self._rr
            # JUST IN CASE the user ever wants it.
            if self.index_bounds is not None:
                if self.proj_axis == 0:  # columns
                    rr = self._rr[:, self.index_bounds]
                elif self.proj_axis == 1:  # rows
                    rr = self._rr[self.index_bounds, :]
            # Then make the projection / sum.
            if self.proj_type == "projection":
                # For positive projections, we zero out any negative elements, then
                # take the maximum along the axis being projected along..
                if self.sign == "positive":
                    rr[rr < 0] = 0
                    self._real = np.amax(rr, axis=(1 - self.proj_axis))
                # And the opposite for negative projections.
                elif self.sign == "negative":
                    rr[rr > 0] = 0
                    self._real = np.amin(rr, axis=(1 - self.proj_axis))
            # For a sum, we just add up both positive and negative numbers without
            # zeroing one of them. This is in line with TopSpin's behaviour.
            elif self.proj_type == "sum":
                self._real = np.sum(rr, axis=(1 - self.proj_axis))
            # We should never reach here, unless the user manually instantiates a
            # Dataset1DProjVirtual class.
            else:
                raise ValueError(f"invalid projection type '{self.proj_type}'")
        else:
            raise ValueError("_read_spec(): invalid spectype")


TDataset1D = Union[Dataset1D, Dataset1DProj, Dataset1DProjVirtual]
TDatasetnD = Union[Dataset1D, Dataset1DProj, Dataset2D, Dataset1DProjVirtual]
