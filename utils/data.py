import pandas as pd
import os
from pathlib import Path
import pyarrow.parquet as pq


def read_pq(path):
    path = path.replace(' ', '_')  # replace space by '_'
    if not is_full_path(path):
        path = get_full_path(path)  # get data path if not full path
    print(f'Reading {path}')
    return pd.read_parquet(path)


def write_pq(df, path):
    """
    if path contains folder that do not exist then create them
    """
    if isinstance(df, pd.Series):
        name = path.split('/')[-1].split('.')[0]
        df = df.to_frame(name=name)
    path = path.replace(' ', '_')  # replace space by '_'
    if not is_full_path(path):
        path = get_full_path(path)  # get data path if not full path
    print(f'Writing {path}')
    dir_path = Path(path).parent
    os.makedirs(dir_path.as_posix(), exist_ok=True)
    df.to_parquet(path)


def get_full_path(path):
    """
    pass a child. parent path prefix will be added. path is in string
    """
    if is_full_path(path):
        # if already a full path then do nothing
        return path
    if os.name == 'posix':
        return 'not yet decided'
    elif os.name == 'nt':
        return f'C:\\Users\\mitra\\PycharmProjects\\coindcx\\' + path
    else:
        raise Exception('What OS am I in?')


def is_full_path(path):
    if os.name == 'posix':
        return path.startswith('/') or path.startswith("\\")
    elif os.name == 'nt':
        return path.startswith('C:')
    else:
        raise Exception('What OS am I in?')


def create_pivot(df, col1='time', col2='stock'):
    """
    df must have 3 cols exactly
    """
    cols = list(df.columns)
    cols.remove(col1)
    cols.remove(col2)
    df = df.pivot(index=col1, columns=col2, values=cols[0])
    return df.sort_index()


def exists(path):
    return os.path.exists(get_full_path(path))


def is_corrupt_pq(path):
    """
    checks if parquet is corrupt or not
    returns True if corrupt
    """
    try:
        # this should fail for corrupt or non parquet file
        parquet_file = pq.ParquetFile(get_full_path(path)).metadata
        return False
    except:
        print(f'{path} is corrupt or else not a parquet!')
        return True

def lvals(df,name):
    """
    short hand to get pandas series for one level within a multi-index
    """
    return df.index.get_level_values(name)