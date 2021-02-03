from utils.data import get_full_path


def sm_data_path():
    return get_full_path('')

def enlist(x):
    """
    make a list if not a list else nothing
    """
    if not isinstance(x, list):
        x = [x]
    return x

