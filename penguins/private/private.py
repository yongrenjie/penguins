"""
private.py
----------

This is only really meant for my personal use. It contains convenience
functions that help me do repeated processing on datasets.
"""


import os
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Any, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from ..pgplot import style_axes
from .. import dataset as ds
from ..exportdeco import export


# -- Experiment types -----------------------------------

class Experiment:
    """
    Generic interface for experiments.
    """
    default_margin = (0.5, 0.02)   # applicable for 13C experiments
    #            use (0.02, 0.02) for 1H experiments
    #            use (0.4, 0.05) for 15N experiments
    def __init__(self,
                 peaks: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = None,
                 ):
        self.peaks = peaks
        self.margin = margin or self.default_margin

    def integrate(self,
                  dataset: ds.Dataset2D,
                  ) -> np.ndarray:
        # Get absolute peak intensities for a given dataset.
        return np.array([dataset.integrate(peak=peak,
                                           margin=self.margin,
                                           mode="max")
                         for peak in self.peaks])

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1" and "f2".
        """
        return pd.DataFrame.from_records(self.peaks, columns=("f1", "f2"))

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    ) -> pd.DataFrame:
        """
        Construct a dataframe of relative intensities vs a reference
        dataset.

        This DataFrame will have columns "f1", "f2", "expt", and "int".
        """
        df = pd.DataFrame()
        df["int"] = self.integrate(dataset) / self.integrate(ref_dataset)
        df["expt"] = label
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


@export
class Hmbc(Experiment):
    """
    For 13C HMBC experiments. Just call hmbc(peaks, margin) to instantiate.
    """
    default_margin = (0.5, 0.02)


@export
class NHsqc(Experiment):
    """
    For 15N HSQC experiments. Just call nhsqc(peaks, margin) to instantiate.
    """
    default_margin = (0.4, 0.05)


@export
class Hsqc(Experiment):
    """
    For 13C HSQC experiments. The variables ch, ch2, and ch3 should be
    lists of 2-tuples (f1_shift, f2_shift) which indicate, well, CH, CH2,
    and CH3 peaks respectively.

    None of the methods from Experiment are actually inherited.
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
        Calculates the absolute integral of each peak in the HSQC. Assumes that
        CH/CH3 is phased to negative and CH2 to positive.
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
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
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


@export
class HsqcCosy(Experiment):
    """
    For 13C HSQC-COSY experiments. The variables hsqc and cosy should be lists
    of 2-tuples (f1_shift, f2_shift) which indicate the direct (HSQC) and
    indirect (HSQC-COSY) responses respectively.

    None of the methods from Experiment are actually inherited.
    """
    def __init__(self,
                 hsqc: List[Tuple[float, float]],
                 cosy: List[Tuple[float, float]],
                 margin: Optional[Tuple[float, float]] = (0.5, 0.02),
                 ):
        self.hsqc = hsqc
        self.cosy = cosy
        self.margin = margin

    @property
    def peaks(self) -> List[Tuple[float, float]]:
        """
        Returns a list of all peaks.
        """
        return self.hsqc + self.cosy

    @property
    def df(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all the peaks. This DF has
        columns "f1", "f2", and "type".
        """
        hsqc_df, cosy_df = (
            pd.DataFrame.from_records(peaklist, columns=("f1", "f2"))
            for peaklist in (self.hsqc, self.cosy)
        )
        hsqc_df["type"] = "hsqc"
        cosy_df["type"] = "cosy"
        return pd.concat((hsqc_df, cosy_df), ignore_index=True)

    def integrate(self,
                  dataset: ds.Dataset2D,
                  edited: bool = True,
                  ) -> np.ndarray:
        """
        Calculates the absolute integral of each peak in the HSQC. If editing
        is enabled, assumes that HSQC peaks are positive and HSQC-COSY peaks
        negative.
        """
        if edited:
            # We need self.df here as it contains multiplicity information.
            return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                               margin=self.margin,
                                               mode=("max"
                                                     if peak.type == "hsqc"
                                                     else "min"))
                             for peak in self.df.itertuples()])
        else:
            return np.array([dataset.integrate(peak=peak,
                                               margin=self.margin,
                                               mode=("max"))
                             for peak in self.peaks])

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
                    edited: bool = True,
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
        df["type"] = self.df["type"]
        df["f1"] = self.df["f1"]
        df["f2"] = self.df["f2"]
        return df


@export
class Cosy(Experiment):
    """
    For COSY experiments. The variables diagonal and cross_half should be
    lists of 2-tuples (f1_shift, f2_shift). cross_half should only contain
    half the peaks, i.e. only at (f1, f2) and not at (f2, f1). These will
    be automatically reflected.
    
    Only integrate() is actually inherited from Experiment.
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

    def rel_ints_df(self,
                    dataset: ds.Dataset2D,
                    ref_dataset: ds.Dataset2D,
                    label: str = "",
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


@export
class Tocsy(Cosy):
    """
    For TOCSY experiments. The variables diagonal and cross_half should be
    lists of 2-tuples (f1_shift, f2_shift). cross_half should only contain
    half the peaks, i.e. only at (f1, f2) and not at (f2, f1). These will
    be automatically reflected.

    The code is derived from the Cosy code, but for Tocsy classes there is
    additionally a `mixing_time` attribute which gives the TOCSY mixing time in
    milliseconds.
    """
    def __init__(self,
                 diagonal: List[Tuple[float, float]],
                 cross_half: List[Tuple[float, float]],
                 mixing_time: float,
                 margin: Optional[Tuple[float, float]] = (0.02, 0.02),
                 ):
        super().__init__(diagonal, cross_half, margin)
        self.mixing_time = mixing_time


# -- Molecules ------------------------------------------

@export
class Andrographolide():
    """
    40 mM in DMSO. HSQC and COSY data available.
    """
    hsqc_ch = [(146.6206, 6.6294), (65.0306, 4.9207), (78.9220, 3.2353),
               (54.8319, 1.2222), (56.0628, 1.8698)]
    hsqc_ch2 = [(108.6390, 4.8192), (108.6390, 4.6320),
                (74.7018, 4.3979), (74.7018, 4.0390),
                (63.0964, 3.8517), (63.0964, 3.2587),
                (37.9512, 2.3380), (37.9512, 1.9400),
                (37.0720, 1.2144), (36.8961, 1.7138),
                (28.4558, 1.6435), (24.5873, 2.5096),
                (24.4115, 1.7528), (24.4115, 1.3705)]
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

    # HSQC-COSY spectrum, we treat it as a HSQC-COSY for now
    hsqc_cosy_hsqc = hsqc_ch + hsqc_ch2 + hsqc_ch3
    hsqc_cosy_cosy = [(24.3311, 1.2059), (28.3694, 1.2156), (24.5067, 1.9386),
                      (24.5067, 1.8702), (28.3694, 1.7090), (36.9727, 1.6455),
                      (38.0262, 1.3524), (54.8816, 1.3671), (55.9351, 2.5101),
                      (28.3694, 3.2331), (24.5067, 2.3147), (37.8506, 1.7383),
                      (24.5067, 6.6280), (64.8896, 5.7146), (74.7219, 4.9183),
                      (65.0651, 4.4006), (78.9358, 1.6455), (146.7087, 2.4889)]
    hsqc_cosy = HsqcCosy(hsqc_cosy_hsqc, hsqc_cosy_cosy)


@export
class Zolmitriptan():
    """
    50 mM in DMSO. HMBC, 15N HSQC, 13C HSQC data available.
    """
    # I redid the peak picking on 21/5/21 because the ones above didn't seem
    # accurate enough.
    hmbc_peaks = [
        (135.6473, 10.7204), (123.1813, 10.7204), (112.4710, 10.7204),
        (127.9219, 10.7204), (41.0110, 6.9396), (53.6525, 7.7798),
        (41.0110, 7.3694), (68.5766, 7.7798), (112.4710, 7.3646),
        (127.9219, 7.2620), (112.4710, 7.1203), (127.9219, 7.1252),
        (135.6473, 7.1252), (126.3417, 7.2620), (119.1430, 6.9347),
        (123.0057, 7.3694), (135.6473, 7.3694), (135.6473, 6.9396),
        (159.1747, 7.7798), (41.0110, 4.0380), (41.0110, 4.2334),
        (53.6525, 4.0282), (68.5766, 2.8070), (68.5766, 2.7679),
        (53.6525, 2.8998), (53.6525, 2.7679), (53.6525, 2.8070),
        (68.5766, 2.8998), (60.1489, 2.8217), (45.4004, 2.5872),
        (60.1489, 2.2746), (45.4004, 2.2746), (159.1747, 4.2334),
        (159.1747, 4.0380), (123.0057, 2.8998), (119.1430, 2.8998),
        (119.3186, 2.8070), (123.0057, 2.7679), (112.4710, 2.5872),
        (126.3417, 2.8119), (126.3417, 2.8998), (119.1430, 2.7679),
        (126.3417, 2.7728), (127.9219, 2.8217), (123.3569, 2.8217),
        (112.4710, 2.8217), (23.2776, 2.5872)
    ]
    hmbc = Hmbc(hmbc_peaks)

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

    cosy_diagonal = [
        (10.7041, 10.7041), (7.3602, 7.3602), (6.9345, 6.9345),
        (4.2335, 4.2335), (7.1136, 7.1136), (7.2545, 7.2545),
        (7.7771, 7.7771), (4.0398, 4.0398), (3.3616, 3.3616),
        (2.8948, 2.8948), (2.8008, 2.8008), (2.5278, 2.5278),
        (2.2283, 2.2283)
    ]
    cosy_cross_half = [
        (7.2516, 6.9345), (4.0456, 4.2277), (2.7891, 4.0515),
        (2.8830, 4.0515), (2.7891, 2.8889), (2.5307, 2.8008),
    ]
    cosy = Cosy(cosy_diagonal, cosy_cross_half)

    # TOCSY with 35 ms mixing
    tocsy_35ms_cross_half = [
        (7.1156, 10.7150), (7.2565, 6.9336), (4.0506, 4.2326),
        (2.8998, 4.2326), (2.7941, 4.2326), (2.7941, 4.0389),
        (2.8998, 4.0447), (2.7941, 2.8939), (2.5592, 2.8117),
        (6.9395, 7.3622)
    ]
    tocsy_35ms = Tocsy(cosy_diagonal, tocsy_35ms_cross_half, 35)


@export
class Gramicidin():
    """
    40 mM in DMSO. 15N HSQC, 13C HSQC data available.
    """
    nhsqc_peaks = [(127.9990, 9.0926),  # Phe NH
                   (125.4206, 8.6639),  # Orn NH
                   (123.2816, 8.3292),  # Leu NH
                   (113.2316, 7.2254)   # Val NH
                   ]  # Orn epsilon-NH2 is folded in most spectra.
    nhsqc = NHsqc(nhsqc_peaks)

    hsqc_ch = [(129.8390, 7.2606),  # Phe ortho
               (128.7822, 7.2900),  # Phe meta
               (127.3291, 7.2488),  # Phe para
               (60.3538, 4.3071),   # Pro alpha
               (57.3154, 4.4070),   # Val alpha
               (54.4092, 4.3600),   # Phe alpha
               (51.3709, 4.7651),   # Orn alpha
               (50.0499, 4.5714),   # Leu alpha
               (31.5556, 2.0759),   # Val beta
               (24.4222, 1.4065),   # Leu delta
               ]
    hsqc_ch2 = [(46.4831, 3.5908), (46.4831, 2.4987),  # Pro delta
                (41.4633, 1.3537), (41.4633, 1.3008),  # Leu beta
                (39.0855, 2.8392), (39.0855, 2.7805),  # Orn delta
                (36.1792, 2.9743), (36.1792, 2.8803),  # Phe beta
                (30.1026, 1.7471), (30.1026, 1.6062),  # Orn beta
                (29.4421, 1.9526), (29.4421, 1.4761),  # Pro beta
                (23.6296, 1.5122),  # Pro gamma
                (23.4975, 1.6473),  # Orn gamma
                ]
    hsqc_ch3 = [(23.1012, 0.8018),  # Leu delta
                (19.4024, 0.7665),  # Val gamma
                (18.4777, 0.8076),  # Val gamma
                ]
    hsqc_margin = (0.5, 0.02)
    hsqc = Hsqc(hsqc_ch, hsqc_ch2, hsqc_ch3, hsqc_margin)


# -- Personal functions ---------------------------------

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


@export
def nmrd():
    return __getenv("nmrd")


@export
def hsqc_stripplot(molecule: Any,
                   datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                   ref_dataset: ds.Dataset2D,
                   expt_labels: Union[str, Sequence[str]],
                   xlabel: str = "Experiment",
                   ylabel: str = "Intensity",
                   title: str = "",
                   edited: bool = False,
                   show_averages: bool = True,
                   ncol: int = 3,
                   loc: str = "upper center",
                   ax: Optional[Any] = None,
                   **kwargs: Any,
                   ) -> Tuple[Any, Any]:
    """
    Plot HSQC strip plots (i.e. plot relative intensities, split by
    multiplicity).

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the hsqc attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.hsqc.rel_ints_df(dataset=ds,
                                              ref_dataset=ref_dataset,
                                              label=label,
                                              edited=edited)
                    for (ds, label) in zip(datasets, expt_labels)]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by multiplicity
    avgd_ints = pd.concat((df.groupby("mult").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="mult",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=all_dfs, ax=ax, **kwargs)
    if show_averages:
        sns.pointplot(x="expt", y="int", hue="mult", zorder=1,
                      dodge=0.5, data=all_dfs, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:3],
                  labels=["CH", r"CH$_2$", r"CH$_3$"])
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(),
                                                  sns.color_palette("deep"))):
            ax.text(x=x-0.25+i*0.25, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    style_axes(ax, "plot")
    return plt.gcf(), ax


@export
def cosy_stripplot(molecule: Any,
                   datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                   ref_dataset: ds.Dataset2D,
                   expt_labels: Union[str, Sequence[str]],
                   xlabel: str = "Experiment",
                   ylabel: str = "Intensity",
                   title: str = "",
                   ncol: int = 2,
                   separate_type: bool = True,
                   loc: str = "upper center",
                   ax: Optional[Any] = None,
                   **kwargs: Any,
                   ) -> Tuple[Any, Any]:
    """
    Plot COSY strip plots (i.e. plot relative intensities, split by peak type).

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the cosy attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.cosy.rel_ints_df(dataset=ds,
                                              ref_dataset=ref_dataset,
                                              label=label)
                    for (ds, label) in zip(datasets, expt_labels)]
    if not separate_type:
        rel_ints_dfs = [rel_int_df.assign(type="cosy")
                        for rel_int_df in rel_ints_dfs]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by type
    avgd_ints = pd.concat((df.groupby("type").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    sns.stripplot(x="expt", y="int", hue="type",
                  dodge=True, data=all_dfs, ax=ax,
                  palette=sns.color_palette("deep")[3:], **kwargs)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if separate_type:
        ax.legend(ncol=ncol, loc=loc,
                  labels=["diagonal", "cross"]).set(title=None)
    else:
        ax.legend().set_visible(False)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    offset = -0.2 if separate_type else 0
    dx = 0.4 if separate_type else 1
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(
                expt_avgs.items(), sns.color_palette("deep")[3:])):
            ax.text(x=x-offset+i*dx, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    style_axes(ax, "plot")
    return plt.gcf(), ax


@export
def hsqc_cosy_stripplot(molecule: Any,
                        datasets: Sequence[ds.Dataset2D],
                        ref_datasets: Sequence[ds.Dataset2D],
                        xlabel: str = "Experiment",
                        ylabel: str = "Intensity",
                        title: str = "",
                        edited: bool = False,
                        show_averages: bool = True,
                        separate_mult: bool = True,
                        ncol: int = 4,
                        loc: str = "upper center",
                        ax: Optional[Any] = None,
                        font_kwargs: Optional[dict] = None,
                        **kwargs: Any,
                        ) -> Tuple[Any, Any]:
    """
    Plot HSQC and COSY relative intensities on the same Axes. HSQC peaks are
    split by multiplicity, COSY peaks are not split.

    Parameters
    ----------
    molecule : pg.private.Andrographolide or pg.private.Zolmitriptan
        The class from which the hsqc and cosy attributes will be taken from
    datasets : (pg.Dataset2D, pg.Dataset2D)
        HSQC and COSY dataset(s) to analyse intensities of
    ref_datasets : (pg.Dataset2D, pg.Dataset2D)
        Reference HSQC and COSY datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing in the HSQC is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 4.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Set up default font_kwargs if not provided.
    font_kwargs = font_kwargs or {}
    # Calculate dataframes of relative intensities.
    hsqc_rel_ints_df = molecule.hsqc.rel_ints_df(dataset=datasets[0],
                                                 ref_dataset=ref_datasets[0],
                                                 edited=edited)
    # Rename mult -> type to match COSY
    hsqc_rel_ints_df = hsqc_rel_ints_df.rename(columns={"mult": "type"})
    # Remove multiplicity information if separation is not desired
    if not separate_mult:
        hsqc_rel_ints_df = hsqc_rel_ints_df.assign(type="hsqc")
    cosy_rel_ints_df = molecule.cosy.rel_ints_df(dataset=datasets[1],
                                                 ref_dataset=ref_datasets[1])
    cosy_rel_ints_df = cosy_rel_ints_df.assign(type="cosy")
    rel_ints_df = pd.concat((hsqc_rel_ints_df, cosy_rel_ints_df))

    # Calculate the average integrals by multiplicity
    avgd_ints = rel_ints_df.groupby("type").mean()
    # Fix the order if we need to (because by default it would be alphabetical)
    if not separate_mult:
        avgd_ints = avgd_ints.reindex(["hsqc", "cosy"])
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="type",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=rel_ints_df, ax=ax, **kwargs)
    if show_averages:
        dodge = 0.6 if separate_mult else 0.4
        sns.pointplot(x="expt", y="int", hue="type", zorder=1,
                      dodge=dodge, data=rel_ints_df, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)

    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[])
    # Setting the handles manually ensures that we get stripplot handles
    # rather than the pointplot ones (if present).
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:4],
                  labels=["HSQC CH", r"HSQC CH$_2$", r"HSQC CH$_3$", "COSY"])
    l.set(title=None)
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))

    # Add the text and averages
    x0 = -0.3 if separate_mult else -0.2
    dx = 0.2 if separate_mult else 0.4
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), deep) in enumerate(zip(expt_avgs.items(),
                                                 sns.color_palette("deep"))):
            ax.text(x=x+x0+i*dx, y=0.02, s=f"({avg:.2f})",
                    color=deep, horizontalalignment="center",
                    transform=ax.get_xaxis_transform(),
                    **font_kwargs)
    style_axes(ax, "plot")
    return plt.gcf(), ax


@export
def hsqcc_stripplot(molecule: Any,
                    datasets: Union[ds.Dataset2D, Sequence[ds.Dataset2D]],
                    ref_dataset: ds.Dataset2D,
                    expt_labels: Union[str, Sequence[str]],
                    xlabel: str = "Experiment",
                    ylabel: str = "Intensity",
                    title: str = "",
                    edited: bool = True,
                    show_averages: bool = True,
                    ncol: int = 2,
                    loc: str = "upper center",
                    ax: Optional[Any] = None,
                    **kwargs: Any,
                    ) -> Tuple[Any, Any]:
    """
    Plot HSQC-COSY strip plots (i.e. plot relative intensities, split by peak
    type).

    Parameters
    ----------
    molecule : pg.private.Andrographolide
        The class from which the hsqc attribute will be taken from
    datasets : pg.Dataset2D or sequence of pg.Dataset2D
        Dataset(s) to analyse intensities of
    ref_dataset : pg.Dataset2D
        Reference dataset
    expt_labels : str or sequence of strings
        Labels for the analysed datasets
    xlabel : str, optional
        Axes x-label, defaults to "Experiment"
    ylabel : str, optional
        Axes y-label, defaults to "Intensity"
    title : str, optional
        Axes title, defaults to empty string
    edited : bool, default False
        Whether editing is enabled or not.
    show_averages : bool, default True
        Whether to indicate averages in each category using sns.pointplot.
    ncol : int, optional
        Passed to ax.legend(). Defaults to 2.
    loc : str, optional
        Passed to ax.legend(). Defaults to "upper center".
    ax : matplotlib.axes.Axes, optional
        Axes instance to plot on. If not provided, uses plt.gca().
    kwargs : dict, optional
        Keywords passed on to sns.stripplot().

    Returns
    -------
    (fig, ax).
    """
    # Stick dataset/label into a list if needed
    if isinstance(datasets, ds.Dataset2D):
        datasets = [datasets]
    if isinstance(expt_labels, str):
        expt_labels = [expt_labels]
    # Calculate dataframes of relative intensities.
    rel_ints_dfs = [molecule.hsqc_cosy.rel_ints_df(dataset=ds,
                                                   ref_dataset=ref_dataset,
                                                   label=label,
                                                   edited=edited)
                    for (ds, label) in zip(datasets, expt_labels)]
    all_dfs = pd.concat(rel_ints_dfs)

    # Calculate the average integrals by multiplicity
    avgd_ints = pd.concat((df.groupby("type").mean() for df in rel_ints_dfs),
                          axis=1)
    avgd_ints.drop(columns=["f1", "f2"], inplace=True)

    # Get currently active axis if none provided
    if ax is None:
        ax = plt.gca()

    # Plot the intensities.
    stripplot_alpha = 0.3 if show_averages else 0.8
    sns.stripplot(x="expt", y="int", hue="type",
                  zorder=0, alpha=stripplot_alpha,
                  dodge=True, data=all_dfs, ax=ax, **kwargs)
    if show_averages:
        sns.pointplot(x="expt", y="int", hue="type", zorder=1,
                      dodge=0.4, data=all_dfs, ax=ax, join=False,
                      markers='_', palette="dark", ci=None, scale=1.25)
    # Customise the plot
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    handles, _ = ax.get_legend_handles_labels()
    l = ax.legend(ncol=ncol, loc=loc,
                  markerscale=0.4,
                  handles=handles[0:3],
                  labels=["direct", "indirect"])
    ax.axhline(y=1, color="grey", linewidth=0.5, linestyle="--")
    # Set y-limits. We need to expand it by ~20% to make space for the legend,
    # as well as the averaged values.
    EXPANSION_FACTOR = 1.2
    ymin, ymax = ax.get_ylim()
    ymean = (ymin + ymax)/2
    ylength = (ymax - ymin)/2
    new_ymin = ymean - (EXPANSION_FACTOR * ylength)
    new_ymax = ymean + (EXPANSION_FACTOR * ylength)
    ax.set_ylim((new_ymin, new_ymax))
    # add the text
    for x, (_, expt_avgs) in enumerate(avgd_ints.items()):
        for i, ((_, avg), color) in enumerate(zip(expt_avgs.items(),
                                                  sns.color_palette("deep"))):
            ax.text(x=x-0.2+i*0.4, y=0.02, s=f"({avg:.2f})",
                    color=color, horizontalalignment="center",
                    transform=ax.get_xaxis_transform())
    style_axes(ax, "plot")
    return plt.gcf(), ax
