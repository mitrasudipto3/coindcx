from utils.data import get_full_path
import matplotlib.pyplot as plt
import scipy
import numpy as np
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed

reference_date = datetime(2018, 10, 1)  # index value considered 100 as reference on this qtr start date # a Monday


def sm_data_path():
    return get_full_path('')


def enlist(x):
    """
    make a list if not a list else nothing
    """
    if not isinstance(x, list):
        x = [x]
    return x


def plot(s):
    """ plot a series """
    plt.plot(s.sort_index())
    plt.show()
    plt.clf()


def normalize(df, ub=None, lb=None):
    """
    pass a wide frame (of positive numbers only) and it will normalize from 0 to 1 such that they sum up to 1 on a date.
    date is level 0 (row), cols are coins (level 1)
    ub and lb are upper and lower bounds on the final returned numbers - achieved by recursion
    """
    df = df.div(df.sum(axis=1), axis=0)
    if ub is not None:
        if df.max().max() > ub:  # note: df.max() is a series
            # any number above ub gets capped at ub
            # capping at ub-0.005 to give some room of optimisation else recursion is called too many times due to
            # stringency
            # passing a_min since we have to. It is set at 0 to do nothing
            return normalize(df.clip(lower=0, upper=ub - 0.05), ub=ub, lb=lb)
    if lb is not None:
        if df.min().min() < lb:  # note: df.min() is a series
            # any number below lb gets converted to 0
            return normalize(df.mask(df < lb, np.nan), ub=ub, lb=lb)
    return df


def create_pivot(df, col1='date', col2='target'):
    """
    df must have 3 cols exactly
    """
    cols = list(df.columns)
    cols.remove(col1)
    cols.remove(col2)
    df = df.pivot(index=col1, columns=col2, values=cols[0])
    return df.sort_index()


def create_pivot_index(df, col1='date', col2='target'):
    """
    df must have 3 cols exactly
    """
    cols = list(df.columns)
    cols.remove(col1)
    cols.remove(col2)
    df = df.pivot(index=col1, columns=col2, values=cols[0])
    # drop coins with all nan
    df = df.dropna(axis=1, how='all')
    # keep only after reference date
    df = df[df.index >= reference_date]
    # add all cal dates
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
    return df.sort_index()


def pmap(fct, jobs, num_procs=-1, require=None):
    """
    parallelize using joblib
    """
    return Parallel(n_jobs=num_procs, require=require)(
        delayed(fct)(job) for job in jobs)


def unstack(s):
    """
    s is series/frame
    """
    return s.unstack().dropna().reset_index().rename(columns={'level_0': 'target', 'level_1': 'date'})
