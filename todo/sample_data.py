from sys import argv
from os import path
import numpy as np
import pandas as pd


def new_filename(filename, prefix):
    dirname = path.dirname(filename)
    basename = path.basename(filename)
    return path.join(dirname, prefix + "_" + basename)


def sample_data(filename, target_col=-1, test_ratio=0.2, shuffle=True):
    """
    Takes in the filename of a csv file, and splits it into training and test data.

    Assumes that target_col is either the name of a column, or an integer index.
    """

    assert 0 < test_ratio < 1, "test ratio must be between 0 and 1"

    data = pd.read_csv(filename)

    columns = list(data.columns)

    """ make sure target_col is valid """
    if target_col not in columns:
        target_col = columns[target_col]

    num_test_rows = int(data.shape[0] * test_ratio)
    rows = data.index

    if shuffle:
        rows = np.random.permutation(rows)

    train = data.ix[rows[:-num_test_rows]]
    test = data.ix[rows[-num_test_rows:]]

    target = test[[target_col]]
    del test[target_col]

    train.to_csv(new_filename(filename, "cv_train"), index=False)
    test.to_csv(new_filename(filename, "cv_test"), index=False)
    target.to_csv(new_filename(filename, "cv_target"), index=False)

    return map(lambda x: new_filename(filename, x), ["cv_train", "cv_test", "cv_target"])

if __name__ == "__main__":
    filename = argv[1]
    sample_data(filename)
