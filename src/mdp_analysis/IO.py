import pandas as pd
import numpy as np


# Input functions #

def read_df(filedir, filename):
    """
    Reads pandas dataframes.

    Args:
        filedir (string): the directory in which the file can be found.

        filename (string): the name of the file.

    Returns:
        pandas DataFrame: containing the data from the file.
    """

    name = '{}{}'.format(filedir, filename)
    print('Reading from file {} - pandas'.format(name))
    data = pd.read_csv(name, r'\s+')
    return data


def read_np(filedir, filename):
    """
    Reads numpy arrays.

    Args:
        filedir (string): the directory in which the file can be found

        filename (string): the name of the file.

    Returns:
        numpy array: containing the data from the file.
    """

    name = '{}{}'.format(filedir, filename)
    print('Reading from file {} - numpy'.format(name))
    data = np.loadtxt(name, skiprows=1)
    data = data.T
    return data


# Output functions #

def save_plot(figure, plotdir, filename):
    """
    Saves a figure to file

    Args:
        figure (seaborn Figure): the plot to save.

        plotdir (string): the directory in which the plot should be saved.

        filename (string): the name of the file to save in (with extension).

    Returns:
        Nothing
    """

    fig = figure.get_figure()
    print('Saving figure at {}{}'.format(plotdir, filename))
    fig.savefig('{}{}'.format(plotdir, filename))
    return


def save_df_data(df, filedir, filename):
    """
    Saves a pandas DataFrame to .csv-file

    Args:
        df (pandas Dataframe): the data to save.

        filedir (string): the directory in which the data should be saved.

        filename (string): the name of the file to save in (with extension).

    Returns:
        Nothing
    """

    print('Saving data at {}{}'.format(filedir, filename))
    df.to_csv('{}{}'.format(filedir, filename), index=False)
    return


def save_np_data(arr, filedir, filename):
    """
    Saves a numpy array to .csv-file

    Args:
        arr (numpy array): the data to save.

        filedir (string): the directory in which the data should be saved.

        filename (string): the name of the file to save in (with extension).

    Returns:
        Nothing
    """

    print('Saving data at {}{}'.format(filedir, filename))
    np.savetxt(fname='{}{}'.format(filedir, filename), X=arr)
    return
