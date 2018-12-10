# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats

from statspy.base import BaseModel, coef_test, calc_R_sq
from statspy.tools import show_model_res


class WeightedLeastSquare(BaseModel):
    '''Ordinary least square estimation
    '''
    def __init__(self, df, y_var, x_vars, wt_var, has_const=True):
        super(WeightedLeastSquare, self).__init__(df, y_var, x_vars, wt_var=wt_var, has_const=has_const)
        self._clean_data()
    
    def predict(self, df):
        assert self._fitted
        if self.has_const:
            X = df.assign(_const=1)[self.x_vars]
        else:
            X = df[self.x_vars]
        return X.values.dot(self.res_table['Coef'].values)
    
    def fit(self, show_res=True):
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        wt = self.reg_df[self.wt_var].values
        n, K = X.shape
        
        wXt = X.T * wt
        wXtX = wXt.dot(X)
        wXtX_inv = np.linalg.inv(wXtX)
        
        b_hat = wXtX_inv.dot(wXt.dot(y))
        y_hat = X.dot(b_hat)
        e_hat = y - y_hat
        s_sq = (e_hat ** 2).dot(wt ** 2) / (wt ** 2).sum() * n / (n-K)
        s = s_sq ** 0.5
        
        # Calculate R-squared
        R_sq, adj_R_sq, SSE, SSR, SST = calc_R_sq(y, y_hat, n, K, wt=wt, return_SS=True)
        
        # sqrt(n)*(b-beta) ~ N(0, Avarb)
        # b-beta = (X'WX)-1 * X' * (wt * epsilon)
        S_hat = X.T.dot((e_hat[:, None]**2)*(wt[:, None]**2) * X) / n
        Avarb_hat = (n*wXtX_inv).dot(S_hat).dot(n*wXtX_inv)
        
        # Freedom adjusted as n-K to be consistent with stata
        Avarb_hat = Avarb_hat * n / (n-K)
            
        # Estimated Var-Cov matrix of b_hat
        est_cov = Avarb_hat / n
        t_dist = stats.t(n-K)
        est_std_err, t_stat, p_value, CI_lower, CI_upper = coef_test(b_hat, est_cov, t_dist, CI_alpha=0.05)
        
        self.res_table = pd.DataFrame(OrderedDict([('Coef', b_hat), 
                                                   ('Std.Err', est_std_err),
                                                   ('t', t_stat),
                                                   ('p', p_value),
                                                   ('CI.lower', CI_lower),
                                                   ('CI.upper', CI_upper)]), index=self.x_vars)

        self.res_stats = pd.Series(OrderedDict([('method', 'OLS'),
                                                ('obs', n),
                                                ('RMSE', s),
                                                ('SSE', SSE),
                                                ('SSR', SSR),
                                                ('SST', SST),
                                                ('R-sq', R_sq),
                                                ('adj-R-sq', adj_R_sq)]))
        self._fitted = True   
        if show_res:
            show_model_res(self)
        