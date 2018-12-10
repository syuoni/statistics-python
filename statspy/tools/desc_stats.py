# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def desc_stats(df, desc_vars=None):
    if desc_vars is None:
        desc_vars = df.columns.tolist()
    
    desc_data = []
    for var in desc_vars:
        values = df.loc[df[var].notnull(), var].values
        try:
            this_desc = [values.shape[0], values.mean(), values.std(ddof=1), values.min(), values.max()]
        except:
            this_desc = [values.shape[0], np.nan, np.nan, np.nan, np.nan]
        
        desc_data.append(this_desc)
    desc_df = pd.DataFrame(desc_data, columns=['Obs', 'Mean', 'Std.Dev', 'Min', 'Max'], index=desc_vars)
    return desc_df

