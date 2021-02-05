from utils.data import get_full_path
import matplotlib.pyplot as plt

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