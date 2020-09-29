"""
This is only really meant for my personal use. It contains peak lists for
compounds that I use in my DPhil on a day-to-day basis, and quick methods for
integrating spectra of these compounds.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from . import dataset as ds


class Andrographolide:
    """
    Andrographolide in DMSO.
    Usually apk2d turns CH/CH3 negative and CH2 positive.
    """
    class hsqc:
        ch = [(146.6206, 6.6294), (65.0306, 4.9207), (78.9220, 3.2353),
              (54.8319, 1.2222), (56.0628, 1.8698)]
        ch2 = [(108.6390, 4.8192), (108.6390, 4.6320), (74.7018, 4.3979),
               (74.7018, 4.0390), (63.0964, 3.8517), (63.0964, 3.2587),
               (24.5873, 2.5096), (37.0720, 1.2144), (37.9512, 1.9400),
               (36.8961, 1.7138), (37.9512, 2.3380), (24.4115, 1.3705),
               (24.4115, 1.7528), (28.4558, 1.6435)]
        ch3 = [(23.5323, 1.0896), (15.0919, 0.6682)]
        peaks = ch + ch2 + ch3
        _ch_df, _ch2_df, _ch3_df = (pd.DataFrame.from_records(peaklist,
                                                              columns=("f1", "f2"))
                                    for peaklist in (ch, ch2, ch3))
        _ch_df["mult"] = "ch"
        _ch2_df["mult"] = "ch2"
        _ch3_df["mult"] = "ch3"
        df = pd.concat((_ch_df, _ch2_df, _ch3_df), ignore_index=True)
        margin = (0.5, 0.02)

        @classmethod
        def integrate(cls,
                      dataset: ds.Dataset2D,
                      edited: bool = False,
                      ) -> np.ndarray:
            # Get absolute peak intensities for a given dataset.
            if edited:
                return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                                   margin=cls.margin,
                                                   mode=("max"
                                                         if peak.mult == "ch2"
                                                         else "min"))
                                 for peak in cls.df.itertuples()])
            else:
                return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                                   margin=cls.margin,
                                                   mode=("max"))
                                 for peak in cls.df.itertuples()])

        @classmethod
        def make_df(cls,
                    dataset: ds.Dataset2D,
                    label: str,
                    ref_dataset: ds.Dataset2D,
                    edited: bool = False,
                    ) -> pd.DataFrame:
            """
            Construct a dataframe of relative intensities vs a reference
            dataset (in the correct form for seaborn plotting).
            """
            # Reference intensities
            ref_ints = cls.integrate(ref_dataset, edited=edited)
            # Actual intensities
            this_ints = cls.integrate(dataset, edited=edited)
            # Relative intensities
            rel_ints = this_ints / ref_ints
            # Construct dataframe
            df = pd.DataFrame()
            df["int"] = rel_ints
            df["expt"] = label
            df["mult"] = cls.df["mult"]
            df["f1"] = cls.df["f1"]
            df["f2"] = cls.df["f2"]
            return df


    class cosy:
        # Diagonal peaks
        diagonal = [(6.6303, 6.6303), (5.7112, 5.7112), (4.9191, 4.9191),
                    (4.8213, 4.8213), (4.6355, 4.6355), (4.4009, 4.4009),
                    (4.0391, 4.0391), (3.8435, 3.8435), (3.2666, 3.2666),
                    (3.2372, 3.2372), (2.4745, 2.4745), (2.5136, 2.5136),
                    (2.3279, 2.3279), (1.9367, 1.9367), (1.8683, 1.8683),
                    (1.7509, 1.7509), (1.7020, 1.7020), (1.6434, 1.6434),
                    (1.3598, 1.3598), (1.2131, 1.2131), (1.0958, 1.0958),
                    (0.6655, 0.6655)]
        # Crosspeaks above the diagonal
        _cross_half = [(2.4941, 6.6303), (4.9191, 5.7112), (4.4009, 4.9191),
                       (4.0391, 4.4009), (3.2666, 3.8435), (1.6531, 3.2275),
                       (1.8683, 2.5039), (1.9367, 2.3279), (1.7411, 1.9367),
                       (1.3598, 1.9367), (1.3598, 1.7412), (1.2131, 1.7020),
                       (1.2131, 1.6434), (1.2131, 1.3500), (1.3696, 2.3181)]
        # Crosspeaks below the diagonal
        _cross_otherhalf = [(t[1], t[0]) for t in _cross_half]
        # All crosspeaks
        cross = _cross_half + _cross_otherhalf
        peaks = diagonal + cross
        # Dataframes
        _diagonal_df, _cross_df = (pd.DataFrame.from_records(peaklist,
                                                             columns=("f1", "f2"))
                                   for peaklist in (diagonal, cross))
        _diagonal_df["type"] = "diagonal"
        _cross_df["type"] = "cross"
        df = pd.concat((_diagonal_df, _cross_df), ignore_index=True)
        margin = (0.02, 0.02)

        @classmethod
        def integrate(cls,
                      dataset: ds.Dataset2D,
                      ) -> np.ndarray:
            # Get absolute peak intensities for a given dataset.
            return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                               margin=cls.margin,
                                               mode="max")
                             for peak in cls.df.itertuples()])

        @classmethod
        def make_df(cls,
                    dataset: ds.Dataset2D,
                    label: str,
                    ref_dataset: ds.Dataset2D,
                    ) -> pd.DataFrame:
            """
            Construct a dataframe of relative intensities vs a reference
            dataset (in the correct form for seaborn plotting).
            """
            # Reference intensities
            ref_ints = cls.integrate(ref_dataset)
            # Actual intensities
            this_ints = cls.integrate(dataset)
            # Relative intensities
            rel_ints = this_ints / ref_ints
            # Construct dataframe
            df = pd.DataFrame()
            df["int"] = rel_ints
            df["expt"] = label
            df["type"] = cls.df["type"]
            df["f1"] = cls.df["f1"]
            df["f2"] = cls.df["f2"]
            return df


class Zolmitriptan:
    """
    55 mM zolmitriptan in DMSO.
    """

    class nhsqc:
        margin = (0.4, 0.05)
        # two peaks only! how nice.
        peaks = [(89.2165, 7.7742), (129.5334, 10.6982)]

        @classmethod
        def integrate(cls,
                      dataset: ds.Dataset2D,
                      ) -> np.ndarray:
            # Get absolute peak intensities for a given dataset.
            return np.array([dataset.integrate(peak=peak,
                                               margin=cls.margin,
                                               mode="max")
                             for peak in cls.peaks])

    class chsqc:
        ch = [(119.3186, 7.3631), (111.7687, 7.2574), (123.1813, 7.1165),
              (123.0057, 6.9345), (53.6525, 4.0573)]
        ch2 = [(68.5766, 4.2335), (68.5766, 4.0280),   # diastereotopic
               (41.1865, 2.8947), (41.0110, 2.7891),   # diastereotopic
               (60.5001, 2.5131), (23.6288, 2.8008)]   # CH2CH2NMe2
        ch3 = [(45.5760, 2.2195)]
        peaks = ch + ch2 + ch3
        _ch_df, _ch2_df, _ch3_df = (pd.DataFrame.from_records(peaklist,
                                                              columns=("f1", "f2"))
                                    for peaklist in (ch, ch2, ch3))
        _ch_df["mult"] = "ch"
        _ch2_df["mult"] = "ch2"
        _ch3_df["mult"] = "ch3"
        df = pd.concat((_ch_df, _ch2_df, _ch3_df), ignore_index=True)
        margin = (0.5, 0.02)

        @classmethod
        def integrate(cls,
                      dataset: ds.Dataset2D,
                      edited: bool = False,
                      ) -> np.ndarray:
            # Get absolute peak intensities for a given dataset.
            if edited:
                return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                                   margin=cls.margin,
                                                   mode=("max"
                                                         if peak.mult == "ch2"
                                                         else "min"))
                                 for peak in cls.df.itertuples()])
            else:
                return np.array([dataset.integrate(peak=(peak.f1, peak.f2),
                                                   margin=cls.margin,
                                                   mode=("max"))
                                 for peak in cls.df.itertuples()])

        @classmethod
        def make_df(cls,
                    dataset: ds.Dataset2D,
                    label: str,
                    ref_dataset: ds.Dataset2D,
                    edited: bool = False,
                    ) -> pd.DataFrame:
            """
            Construct a dataframe of relative intensities vs a reference
            dataset (in the correct form for seaborn plotting).
            """
            # Reference intensities
            ref_ints = cls.integrate(ref_dataset, edited=edited)
            # Actual intensities
            this_ints = cls.integrate(dataset, edited=edited)
            # Relative intensities
            rel_ints = this_ints / ref_ints
            # Construct dataframe
            df = pd.DataFrame()
            df["int"] = rel_ints
            df["expt"] = label
            df["mult"] = cls.df["mult"]
            df["f1"] = cls.df["f1"]
            df["f2"] = cls.df["f2"]
            return df
