"""Table of Contents:
    -remove_inf_cols
"""

import pandas as pd
import numpy as np

from param import Param


def remove_inf_cols(df):
    """
    Removes columns with infinite values.

    Looks for infinite values twice because the columns with infinite values in the training set may be different than in the test set. (fills these values with column means)
    """
    assert isinstance(df, pd.DataFrame)

    def get_no_inf():
        return np.isinf(df).sum(axis=0) == 0

    no_inf = Param.f("no_inf", get_no_inf)
    df_no_inf = df.ix[:, no_inf]
    df_no_inf[np.isinf(df_no_inf)] = 0
    return df_no_inf
