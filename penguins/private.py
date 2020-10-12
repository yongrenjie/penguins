"""
private.py
----------

This is only really meant for my personal use. It contains convenience
functions that help me do repeated processing on datasets.
"""


import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from . import dataset as ds


class NHsqc:
    """
    For 15N HSQC experiments. Just set peaks and margin.
    """
    def __init__(self,
                 peaks: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.4, 0.05),
                 ):
        self.peaks = peaks
        self.margin = margin

    def integrate(self,
                  dataset: ds.Dataset2D,
                  ) -> np.ndarray:
        # Get absolute peak intensities for a given dataset.
        return np.array([dataset.integrate(peak=peak,
                                           margin=self.margin,
                                           mode="max")
                         for peak in self.peaks])


class Hsqc:
    """
    For 13C HSQC experiments. The variables ch, ch2, and ch3 should be
    lists of 2-tuples (f1_shift, f2_shift) which indicate, well, CH, CH2,
    and CH3 peaks respectively.
    """
    def __init__(self,
                 ch: List[Tuple[float, float]],
                 ch2: List[Tuple[float, float]],
                 ch3: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.5, 0.02),
                 ):
        self.ch = ch
        self.ch2 = ch2
        self.ch3 = ch3
        self.margin = margin

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        """
        Returns a list of all peaks.
        """
        return self.ch + self.ch2 + self.ch3

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "mult".
        """
        _chdf, _ch2df, _ch3df = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.ch, self.ch2, self.ch3)
        )
        _chdf["mult"] = "ch"
        _ch2df["mult"] = "ch2"
        _ch3df["mult"] = "ch3"
        return pd.concat((_chdf, _ch2df, _ch3df), ignore_index=True)

    def integrate(self,
                  dataset: ds.Dataset2D,
                  edited: bool = False,
                  ) -> np.ndarray:
        """
        Calculates the absolute integral of each peak in the HSQC.
        """
        if edited:
            # We need self.df here as it contains multiplicity information.
            return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                               margin=self.margin,
                                               mode=("max"
                                                     if peak.mult == "ch2"
                                                     else "min"))
                             for peak in self.df.itertuples()])
        else:
            return np.array([dataset.integrate(peak=peak,
                                               margin=self.margin,
                                               mode=("max"))
                             for peak in self.peaks])

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    label: str,
                    ref_dataset: ds.Dataset2D,
                    edited: bool = False,
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns (f1, f2, mult) just like self.df,
        but will also have "expt" which is a string indicating the type of
        experiment being ran, and "int" which is the relative integral vs a
        reference dataset.
        """
        df = pd.DataFrame()
        df["int"] = (self.integrate(dataset, edited=edited) /
                     self.integrate(ref_dataset, edited=edited))
        df["expt"] = label
        df["mult"] = self.df["mult"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class Cosy:
    """
    For COSY experiments. The variables diagonal and cross_half should be
    lists of 2-tuples (f1_shift, f2_shift). cross_half should only contain
    half the peaks, i.e. only at (f1, f2) and not at (f2, f1). These will
    be automatically reflected.
    """
    def __init__(self,
                 diagonal: List[Tuple[float, float]],
                 cross_half: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.02, 0.02),
                 ):
        self.diagonal = diagonal
        self.cross_half = cross_half
        self.margin = margin

    @property
    def cross(self) -> List[Tuple[float, float]]:
        cross_otherhalf = [(t[1], t[0]) for t in self.cross_half]
        # All crosspeaks
        return self.cross_half + cross_otherhalf

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        return self.diagonal + self.cross

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "type".
        """
        _diagdf, _crossdf = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.diagonal, self.cross)
        )
        _diagdf["type"] = "diagonal"
        _crossdf["type"] = "cross"
        return pd.concat((_diagdf, _crossdf), ignore_index=True)

    def integrate(self,
                  dataset: ds.Dataset2D,
                  ) -> np.ndarray:
        # Get absolute peak intensities for a given dataset.
        return np.array([dataset.integrate(peak=peak,
                                           margin=self.margin,
                                           mode="max")
                         for peak in self.peaks])

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    label: str,
                    ref_dataset: ds.Dataset2D,
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns (f1, f2, type) just like self.df,
        but will also have "expt" which is a string indicating the type of
        experiment being ran, and "int" which is the relative integral vs a
        reference dataset.
        """
        df = pd.DataFrame()
        df["int"] = self.integrate(dataset) / self.integrate(ref_dataset)
        df["expt"] = label
        df["type"] = self.df["type"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


class Andrographolide():
    """
    Andrographolide in DMSO.
    """
    hsqc_ch = [(146.6206, 6.6294), (65.0306, 4.9207), (78.9220, 3.2353),
               (54.8319, 1.2222), (56.0628, 1.8698)]
    hsqc_ch2 = [(108.6390, 4.8192), (108.6390, 4.6320), (74.7018, 4.3979),
                (74.7018, 4.0390), (63.0964, 3.8517), (63.0964, 3.2587),
                (24.5873, 2.5096), (37.0720, 1.2144), (37.9512, 1.9400),
                (36.8961, 1.7138), (37.9512, 2.3380), (24.4115, 1.3705),
                (24.4115, 1.7528), (28.4558, 1.6435)]
    hsqc_ch3 = [(23.5323, 1.0896), (15.0919, 0.6682)]
    hsqc = Hsqc(hsqc_ch, hsqc_ch2, hsqc_ch3)

    cosy_diagonal = [(6.6303, 6.6303), (5.7112, 5.7112), (4.9191, 4.9191),
                     (4.8213, 4.8213), (4.6355, 4.6355), (4.4009, 4.4009),
                     (4.0391, 4.0391), (3.8435, 3.8435), (3.2666, 3.2666),
                     (3.2372, 3.2372), (2.4745, 2.4745), (2.5136, 2.5136),
                     (2.3279, 2.3279), (1.9367, 1.9367), (1.8683, 1.8683),
                     (1.7509, 1.7509), (1.7020, 1.7020), (1.6434, 1.6434),
                     (1.3598, 1.3598), (1.2131, 1.2131), (1.0958, 1.0958),
                     (0.6655, 0.6655)]
    cosy_cross_half = [(2.4941, 6.6303), (4.9191, 5.7112), (4.4009, 4.9191),
                       (4.0391, 4.4009), (3.2666, 3.8435), (1.6531, 3.2275),
                       (1.8683, 2.5039), (1.9367, 2.3279), (1.7411, 1.9367),
                       (1.3598, 1.9367), (1.3598, 1.7412), (1.2131, 1.7020),
                       (1.2131, 1.6434), (1.2131, 1.3500), (1.3696, 2.3181)]
    cosy = Cosy(cosy_diagonal, cosy_cross_half)


class Zolmitriptan():
    """
    55 mM zolmitriptan in DMSO.
    """
    nhsqc_peaks = [(89.2165, 7.7742), (129.5334, 10.6982)]
    nhsqc = NHsqc(nhsqc_peaks)
    hsqc_ch = [(119.3186, 7.3631), (111.7687, 7.2574), (123.1813, 7.1165),
               (123.0057, 6.9345), (53.6525, 4.0573)]
    hsqc_ch2 = [(68.5766, 4.2335), (68.5766, 4.0280),   # diastereotopic
                (41.1865, 2.8947), (41.0110, 2.7891),   # diastereotopic
                (60.5001, 2.5131), (23.6288, 2.8008)]   # CH2CH2NMe2
    hsqc_ch3 = [(45.5760, 2.2195)]
    hsqc_margin = (0.5, 0.02)
    hsqc = Hsqc(hsqc_ch, hsqc_ch2, hsqc_ch3, hsqc_margin)


# These are pure convenience routines for my personal use.
# Default save location for plots
dsl = Path("/Users/yongrenjie/Desktop/a_plot.png")
# Path to NMR spectra. The $nmrd environment variable should resolve to
# .../dphil/expn/nmr. On my Mac this is set to my SSD.
def __getenv(key):
    if os.getenv(key) is not None:
        x = Path(os.getenv(key))
        if x.exists():
            return x
    raise FileNotFoundError("$nmrd does not point to a valid location.")
nmrd = lambda: __getenv("nmrd")
