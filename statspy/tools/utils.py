import numpy as np
import pandas as pd

def show_model_res(model):
    print('=' * 70)
    if hasattr(model, 'res_stats'):
        print(model.res_stats)
        print('=' * 70)
    if hasattr(model, 'res_table'):
        print(model.res_table)
        print('=' * 70)

def format_res_table(res_table_list, in_parenth='t-statistic', digit=3, digit_in_parenth=2):
    all_params = []
    for res_table in res_table_list:
        for param in res_table.index:
            if param not in all_params:
                all_params.append(param)
    all_params = np.array(all_params, dtype=str)
    
    index = []
    for param in all_params:
        index.append(param)
        index.append(param + '-pth')
    
    formatted = pd.DataFrame(index=index)
    formatted['index'] = ''
    formatted['index'][0::2] = all_params
    
    for k, res_table in enumerate(res_table_list, 1):
        coef_seq = np.array([('%%.%df%%s' % digit) % (c, sig_star(p)) for c, p in zip(res_table['coef'], res_table['p-value'])])
        parenth_seq = np.array([('(%%.%df)' % digit_in_parenth) % c for c in res_table[in_parenth]])
        formatted['(%d)' % k] = ''
        formatted.loc[res_table.index, '(%d)' % k] = coef_seq
        formatted.loc[[par + '-pth' for par in res_table.index], '(%d)' % k] = parenth_seq
    return formatted

def sig_star(p):
    if pd.isnull(p):
        return ''
    elif p <= 0.01:
        return '***'
    elif p <= 0.05:
        return '**'
    elif p <= 0.1:
        return '*'
    else:
        return ''

