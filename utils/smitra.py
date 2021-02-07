from utils.data import get_full_path
import matplotlib.pyplot as plt
import scipy
import numpy as np


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


def normalize(s, ub=None, lb=None):
    """
    pass a series (of positive numbers only) and it will normalize from 0 to 1 such that they sum up to 1.
    ub and lb are upper and lower bounds on the final returned numbers - achieved by recursion
    """
    assert s.min() > 0
    s = s / s.sum()
    if ub is not None:
        if s.max() > ub:
            # any number above ub gets capped at ub
            # capping at ub-0.005 to give some room of optimisation else recursion is called too many times due to
            # stringency
            # passing a_min since we have to. It is set at 0 to do nothing
            return normalize(np.clip(s, a_max=ub-0.005, a_min=0),ub=ub,lb=lb)
    if lb is not None:
        if s.min() < lb:
            # any number below lb gets converted to 0
            return normalize(s.mask(s<lb,np.nan),ub=ub,lb=lb)
    # final returned series must not have nan weights. zero weight is okay
    return s.replace(np.nan,0)