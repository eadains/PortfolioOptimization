import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


def tsplot(data, lags=20, dist=stats.norm):
    """
    Shows useful plots for time series analysis: original data, autocorrelation, partial autocorrelation, and QQ Plot
    :param data: numpy array, pandas series, or pandas dataframe with single column
    :param lags: lags for use in autocorrelation plots. Default 20
    :param dist: scipy stats distribution. Default normal
    """
    layout = (3, 2)
    acf_ax = plt.subplot2grid(layout, (0, 0))
    pacf_ax = plt.subplot2grid(layout, (0, 1))
    qqplot_ax = plt.subplot2grid(layout, (1, 0), colspan=2, rowspan=2)
    qqplot_ax.set_title('QQ Plot')

    sm.graphics.tsa.plot_acf(data, lags=lags, ax=acf_ax, zero=False)
    sm.graphics.tsa.plot_pacf(data, lags=lags, ax=pacf_ax, zero=False)
    sm.graphics.qqplot(data, dist=dist, fit=True, line='45', ax=qqplot_ax)


def time_series_split(x, y, test_reserve=0.20):
    """
    Train/Test split for time series data
    :param x: features
    :param y: target values
    :param test_reserve: percentage of data to hold out. Holds out most recent 20% of data by default
    :return: 4 dataframes/Series
    """
    if type(x.index) == pd.MultiIndex:
        # Assumes first level of index are dates
        # iloc ignores multiindex structure so we have to replicate behavior based on desired index level
        # Coerce to int because indexing only takes integers
        train_size = int(x.index.levels[0].shape[0] - (x.index.levels[0].shape[0] * test_reserve))
        x_train = x.loc[x.index.levels[0][:train_size]]
        y_train = y.loc[y.index.levels[0][:train_size]]
        x_test = x.loc[x.index.levels[0][train_size:]]
        y_test = y.loc[y.index.levels[0][train_size:]]
    else:
        train_size = int(x.shape[0] - (x.shape[0] * test_reserve))
        x_train = x.iloc[:train_size]
        y_train = y.iloc[:train_size]
        x_test = x.iloc[train_size:]
        y_test = y.iloc[train_size:]
    return x_train, x_test, y_train, y_test
