"""
TODO:
    -imputation using classifiers (k-nn, etc.)
    -category frequencies

Table of Contents:
    -to_float
    -generic_string_transform
    -scale
    -pairwise_transforms
    -binarize
    -basic_numeric_transform
    -impute
    -category_output_average
    -split_into_quantiles
    -cluster
    -group_transforms
    -generic_feature_creation

"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from string import punctuation
from functools import partial
from collections import defaultdict

from param import Param
from decorators import timer, default_catcher
from utils import column_append, df_identifier, first_col, column_apply, combine_dfs
from helper.gap_statistic import gap_statistic


@default_catcher(np.nan)
def to_float(s):
    return float(s)


@timer
def generic_string_transform(df):
    """
    Tries to extract generic features from strings.
    """
    assert isinstance(df, pd.DataFrame)

    def apply_only(description, f):
        return column_append(description, str_df.applymap(f))

    def apply_count(description, f):
        return column_append(description, str_df.applymap(lambda x: sum(map(f, x))))

    def apply_mean(description, f):
        return column_append(description, str_df.applymap(lambda x: sum(map(f, x)) / float(len(x))))

    def is_punct(c):
        return c in punctuation

    def word_count(s):
        return len(s.split())

    @default_catcher(0)
    def len_longest_word(s):
        return max(map(len, s.split()))

    @default_catcher(0)
    def avg_word_len(s):
        return sum(map(len, s.split())) / len(s.split())

    str_df = df.applymap(str)

    len_df = apply_only("len", len)
    capital_df = apply_count("nCap", str.isupper)
    pcnt_capital_df = apply_mean("%Cap", str.isupper)
    alpha_df = apply_count("nAlpha", str.isalpha)
    pcnt_alpha_df = apply_mean("%Alpha", str.isalpha)
    digit_df = apply_count("nDig", str.isdigit)
    pcnt_digit_df = apply_mean("%Dig", str.isdigit)
    punctuation_df = apply_count("nPunct", is_punct)
    pcnt_punctuation_df = apply_mean("%Punct", is_punct)
    word_count_df = apply_only("wordCnt", word_count)
    len_longest_word_df = apply_only("longWord", len_longest_word)
    avg_word_len_df = apply_only("avgWord", avg_word_len)

    transforms = []
    transforms.append(len_df)
    transforms.append(capital_df)
    transforms.append(pcnt_capital_df)
    transforms.append(alpha_df)
    transforms.append(pcnt_alpha_df)
    transforms.append(digit_df)
    transforms.append(pcnt_digit_df)
    transforms.append(punctuation_df)
    transforms.append(pcnt_punctuation_df)
    transforms.append(word_count_df)
    transforms.append(len_longest_word_df)
    transforms.append(avg_word_len_df)

    return combine_dfs(transforms)


@timer
def scale(df):
    """
    Scales columns to 0 mean and unit variance.
    """
    assert isinstance(df, pd.DataFrame)

    def get_scaler():
        scaler = StandardScaler()
        scaler.fit(df)
        return scaler

    scaler = Param.f("scaler", get_scaler)
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df


@timer
def pairwise_transforms(df):
    """
    Use on columns that measure similar attributes to compute various pairwise transforms (+,*,/,-).
    """
    assert isinstance(df, pd.DataFrame)
    float_df = df.applymap(to_float)
    assert not np.any(pd.isnull(float_df))

    pairwise_results = []

    for i, colA in enumerate(float_df):
        for j, colB in enumerate(float_df):

            def apply_binary_function(func, name):
                tmp_df = pd.DataFrame(func(float_df[colA], float_df[colB]), columns=["{}{}{}".format(colA, name, colB)], index=df.index)
                pairwise_results.append(tmp_df)

            if i != j:
                apply_binary_function(lambda x, y: x / y, "/")
            if i > j:
                apply_binary_function(lambda x, y: x + y, "+")
                apply_binary_function(lambda x, y: x - y, "-")
                apply_binary_function(lambda x, y: x * y, "*")

    return combine_dfs(pairwise_results)


def binarize_col(df):
    """
    Creates a binary column for each value in a column.
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1

    identifier = df_identifier(df) + "_binarize"

    unique = Param.f(identifier, pd.unique, first_col(df))

    binary_cols = []
    for val in unique:
        """ adding 0.0 to convert boolean values to float """
        binarized = (df == val) + 0.0
        binarized.columns = [identifier + str(val)]
        binary_cols.append(binarized)

    return combine_dfs(binary_cols)


@timer
def auto_binarize(df, binarization_threshold=50):
    """
    Binarizes columns that have less than binarization_threshold unique values.
    """
    assert isinstance(df, pd.DataFrame)

    binarization_columns = []
    for col in df:
        if len(set(df[col])) < binarization_threshold:
            binarization_columns.append(col)

    if len(binarization_columns) == 0:
        return pd.DataFrame()
    else:
        return binarize(df[binarization_columns])


def basic_numeric_transform_col(df):
    """
    Creates features from various unary transformations.
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1

    transforms = []
    transforms.append(df)
    transforms.append(column_append("square", df ** 2))
    transforms.append(column_append("abs.log+1", np.log(np.abs(df) + 1)))
    transforms.append(column_append("abs.sqrt", np.sqrt(np.abs(df))))
    transforms.append(column_append("1/1+abs", 1 / (1 + np.abs(df))))

    return combine_dfs(transforms)


def impute_col(df):
    """
    Fills in the missing values in a dataframe.
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1

    if not np.any(pd.isnull(df)):
        return df

    identifier = df_identifier(df)

    mean = Param.v(identifier + "_mean", df.mean())
    median = Param.v(identifier + "_median", df.median())

    zero_impute = column_append("zeroImp", df.fillna(0))
    mean_impute = column_append("meanImp", df.fillna(mean))
    median_impute = column_append("medianImp", df.fillna(median))

    imputed = []
    imputed.append(zero_impute)
    imputed.append(mean_impute)
    imputed.append(median_impute)

    return combine_dfs(imputed)


@timer
def split_categorical_numerical(df, numerical_nan_percent=0.5):
    """
    Divides a dataframe into it's categorical and numerical columns, based on the number of entries that can't be converted to a float and numerical_nan_percent.
    """
    assert isinstance(df, pd.DataFrame)

    categorical_cols, numerical_cols = list(), list()
    cutoff = df.shape[0] * numerical_nan_percent
    for col in df:
        if pd.isnull(df[col]).sum() >= cutoff:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    return df[categorical_cols], df[numerical_cols]


def category_output_average_col(df, target):
    """
    Replaces each categorical variable with the average target of the value of the categorical variable.
    Example. Dataframe is for colors. This function replaces "red" with the average of all rows with color as "red".
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1

    identifier = df_identifier(df) + "_category_output_average"

    def get_output_avg_dict():
        output_dict = defaultdict(list)

        for value, output in zip(first_col(df), target):
            output_dict[value].append(output)

        output_avg_dict = {}
        for k, v in output_dict.items():
            output_avg_dict[k] = float(sum(v)) / len(v)
        return output_avg_dict

    output_avg_dict = Param.f(identifier, get_output_avg_dict)
    output_avg = Param.f(identifier + "_mean", np.mean, target)

    def get_output_avg(key):
        return output_avg_dict[key] if key in output_avg_dict else output_avg

    transformed = [get_output_avg(k) for k in first_col(df)]
    transformed_df = pd.DataFrame(transformed, columns=[identifier], index=df.index)
    return transformed_df


def category_frequency_col(df):
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1


def split_into_quantiles_col(df):
    """
    Splits a numerical variable into categorical variable for quantiles of various size. The first column contains information on top half vs bottom half.
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1

    identifier = df_identifier(df) + "_split_into_quantiles"

    sorted_vals = Param.f(identifier, np.sort, first_col(df))
    num_sorted_vals = len(sorted_vals)

    quantiles = 2
    quantile_dfs = []
    while 2 * quantiles < num_sorted_vals:
        step_size = int(num_sorted_vals / quantiles)
        rankings = df * 0
        for cutoff in sorted_vals[step_size::step_size]:
            rankings += (df > cutoff)
        quantile_dfs.append(pd.DataFrame(first_col(rankings), columns=["{}_{}".format(identifier, quantiles)], index=df.index))
        quantiles *= 2

    return combine_dfs(quantile_dfs)


def cluster(df):
    """
    Converts to cluster-distance space using the optimal number of clusters from the gap statistic.
    """
    assert isinstance(df, pd.DataFrame)

    identifier = df_identifier(df) + "_cluster"

    def get_kmeans():
        clusters = gap_statistic(df.as_matrix())
        kmeans = KMeans(clusters)
        kmeans.fit(df)
        return kmeans

    kmeans = Param.f(identifier, get_kmeans)
    clustered = kmeans.transform(df)

    return pd.DataFrame(clustered, columns=["{}_{}".format(identifier, i) for i in range(clustered.shape[1])], index=df.index)


def group_transforms(df, groups):
    """
    Returns the dataframe resulting from calling pairwise_transforms and cluster on each group.
    """
    assert isinstance(df, pd.DataFrame)

    grouped_transforms = []
    for group in groups:
        assert isinstance(group, (list, tuple))
        tmp_df = df[group]
        grouped_transforms.append(cluster(tmp_df))
        if len(group) > 1:
            grouped_transforms.append(pairwise_transforms(tmp_df))

    return combine_dfs(grouped_transforms)

binarize = partial(column_apply, func=binarize_col)

basic_numeric_transform = partial(column_apply, func=basic_numeric_transform_col)

impute = partial(column_apply, func=impute_col)

# category_output_average = partial(column_apply, func=category_output_average_col)


def category_output_average(df, target):
    return column_apply(df, category_output_average_col, target)

split_into_quantiles = partial(column_apply, func=split_into_quantiles_col)


def generic_feature_creation(df, target, column_groups=[]):
    assert isinstance(df, pd.DataFrame)

    categorical_df, numerical_df = split_categorical_numerical(df)
    imputed_df = impute(numerical_df)
    quantile_df = split_into_quantiles(imputed_df)
    CAO_df = category_output_average(combine_dfs((quantile_df, categorical_df)), target=target)
    binarized_df = binarize(CAO_df)
    group_df = group_transforms(combine_dfs((CAO_df, imputed_df)), [])
    numeric_transformed_df = basic_numeric_transform(combine_dfs((CAO_df, imputed_df, group_df)))
    final_df = combine_dfs((binarized_df, numeric_transformed_df))

    return final_df

from utils import random_df

x = random_df(5, 4)
