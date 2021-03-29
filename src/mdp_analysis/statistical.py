import pandas as pd
import numpy as np
import seaborn as sn


######## Input functions ########

def check_if_significant(data, thresh=1e-5):
    """
    trim data based on threshold.

    Args:
        pandas DataFrame: data needs to be trimmed.

        threshold (float): a constant. Default = 1e-5.

    Returns:
        pandas DataFrame: data after trimming.
        pandas DataFrame: corrsponding index of data.
    """

    data_out = data.drop(data.var()[data.var()<thresh].index.values, axis=1)
    indices = data.var()[data.var() > thresh].index.values
    return data_out, indices

def get_correlation_measure(df):
    """
    Get correlation of data.

    Args:
        pandas DataFrame: data needs for correlation.

    Returns:
        pandas DataFrame: correlation.
    """
    drop_values = set() # an unordered collection of items
    cols = df.columns # get the column labels
    print(cols)
    for i in range(0, df.shape[1]):
        for j in range(0, i+1): # get rid of all diagonal entries and the lower triangular
            drop_values.add((cols[i], cols[j]))
    corr2 = df.corr().unstack() # pivot the correlation matrix
    corr2 = corr2.drop(labels=drop_values).sort_values(ascending=False, key=lambda col: col.abs()) # sort by absolute values but keep sign
    return corr2

def euclidean_distance(list_ref, list_comp, vectors):
    """
    Get correlation of data.

    Args:
        List: index of column for reference.
        List: index of column for comparison.
        Numpy DataFrame: Data stored in Matrix.

    Returns:
        Numpy DataFrame: Euclidean_distance.
    """
    distances = np.zeros(len(list_ref))
    for i in range(len(list_ref)):
        distances[i] = np.linalg.norm(vectors[list_comp[i]] - vectors[list_ref[i]])
    return distances
