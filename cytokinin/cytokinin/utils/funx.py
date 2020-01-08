from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from cytokinin.config import ERROR

## pandas

def infer_file_cols_dtypes(filepath, ftype='csv', skipna=True):
    '''
        Returns:
            (dict): inferred dtypes of columns, {COLNAME: DTYPE}
    '''
    fpath = Path(filepath)
    if not fpath.exists():
        msg = f'You must provide an existing and valid path, instead {fpath} was not.'
        ERROR['error_invalid_path'](msg)
    err = lambda x,t: ERROR['error_invalid_path'](f'{x} is not a {t.upper()} file.')
    
    if ftype == 'csv':
        if not fpath.suffix.lower() == '.csv': err(fpath, 'csv')
        temp_df = pd.read_csv(fpath, nrows=5)
    
    types = {}
    for c in temp_df.columns:
        types[c] = infer_dtype(temp_df[c], skipna=skipna)
    return types

def ar1hot(n, mx):
    '''
        Args:
            n (int): class
            mx (int): max class or classes amount
        
        Returns:
            (np.array): The 1-hot encoded number n
        
        Example:
            ar1hot(1,4)
            # [0,1,0,0,0]
    '''
    ar = np.zeros(int(mx+1)).astype(int)
    ar[n] = 1
    return ar
