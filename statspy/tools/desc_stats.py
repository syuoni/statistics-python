import numpy as np
import pandas as pd

def desc_stats(df, desc_vars=None):
    if desc_vars is None:
        desc_vars = df.columns
    
    desc_data = []
    for var in desc_vars:
        values = df.loc[pd.notnull(df[var]), var].values
        try:
            this_desc = [len(values), np.mean(values), np.std(values), np.min(values), np.max(values)]
        except Exception as e:
            this_desc = [len(values), np.nan, np.nan, np.nan, np.nan]
        
        desc_data.append(this_desc)
    desc_df = pd.DataFrame(desc_data, columns=['obs', 'mean', 'std_dev', 'min', 'max'], index=desc_vars)
    return desc_df
