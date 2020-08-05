try:
    import pandas as pd
except ImportError:
    raise ImportError("penguins.peaklists requires the pandas package.")


class Andrographolide():
    """
    Andrographolide in DMSO.
    Usually apk2d turns CH/CH3 negative and CH2 positive.
    """
    hsqc_ch = [(146.6206, 6.6294), (65.0306, 4.9207), (78.9220, 3.2353),
               (54.8319, 1.2222), (56.0628, 1.8698)]
    hsqc_ch2 = [(108.6390, 4.8192), (108.6390, 4.6320), (74.7018, 4.3979),
                (74.7018, 4.0390), (63.0964, 3.8517), (63.0964, 3.2587),
                (24.5873, 2.5096), (37.0720, 1.2144), (37.9512, 1.9400),
                (36.8961, 1.7138), (37.9512, 2.3380), (24.4115, 1.3705),
                (24.4115, 1.7528), (28.4558, 1.6435)]
    hsqc_ch3 = [(23.5323, 1.0896), (15.0919, 0.6682)]
    _ch_df = pd.DataFrame.from_records(hsqc_ch, columns=("f1", "f2"))
    _ch_df["mult"] = "ch"
    _ch2_df = pd.DataFrame.from_records(hsqc_ch2, columns=("f1", "f2"))
    _ch2_df["mult"] = "ch2"
    _ch3_df = pd.DataFrame.from_records(hsqc_ch3, columns=("f1", "f2"))
    _ch3_df["mult"] = "ch3"
    hsqc = pd.concat((_ch_df, _ch2_df, _ch3_df), ignore_index=True)
