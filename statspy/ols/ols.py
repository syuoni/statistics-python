# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats

from statspy.base import BaseModel, F_test, coef_test, calc_R_sq
from statspy.tools import show_model_res


class OrdinaryLeastSquare(BaseModel):
    '''Ordinary least square estimation
    '''
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(OrdinaryLeastSquare, self).__init__(df, y_var, x_vars, wt_var=None, has_const=has_const)
        self._clean_data()
        
    def predict(self, df):
        assert self._fitted
        if self.has_const:
            X = df.assign(_const=1)[self.x_vars]
        else:
            X = df[self.x_vars]
        return X.values.dot(self.res_table['Coef'].values)
        
    def fit(self, robust=False, show_res=True):
        self.robust = robust
        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        XtX = X.T.dot(X)
        XtX_inv = np.linalg.inv(XtX)
        
        b_hat = XtX_inv.dot(X.T).dot(y)
        y_hat = X.dot(b_hat)
        e_hat = y - y_hat
        s_sq = (e_hat**2).sum() / (n-K)
        s = s_sq ** 0.5
        
        # Calculate R-squared
        R_sq, adj_R_sq, SSE, SSR, SST = calc_R_sq(y, y_hat, n, K, return_SS=True)
        
        if not robust:
            Avarb_hat = n * XtX_inv * s_sq
        else:
            # sqrt(n)*(b-beta) ~ N(0, Avarb)
            # b-beta = (X'X)-1 * X' * epsilon
            S_hat = X.T.dot((e_hat[:, None]**2) * X) / n
            Avarb_hat = (n*XtX_inv).dot(S_hat).dot(n*XtX_inv)
        
            # Freedom adjusted as n-K to be consistent with stata
            Avarb_hat = Avarb_hat * n / (n-K)
            
        # Estimated Var-Cov matrix of b_hat
        est_cov = Avarb_hat / n
        # Use t(n-K) distribution to be consistent with stata
        # Option: use standard normal distribution, if robust?
        t_dist = stats.t(n-K)
        est_std_err, t_stat, p_value, CI_lower, CI_upper = coef_test(b_hat, est_cov, t_dist, CI_alpha=0.05)
        
        # F-test
        F_stat, F_p_value = F_test(X, b_hat, Avarb_hat, np.identity(K)[:-1], robust=robust)
        
        self.res_table = pd.DataFrame(OrderedDict([('Coef', b_hat), 
                                                   ('Std.Err', est_std_err),
                                                   ('t', t_stat),
                                                   ('p', p_value),
                                                   ('CI.lower', CI_lower),
                                                   ('CI.upper', CI_upper)]), index=self.x_vars)

        self.res_stats = pd.Series(OrderedDict([('method', 'OLS'),
                                                ('robust', robust),
                                                ('obs', n),
                                                ('RMSE', s),
                                                ('SSE', SSE),
                                                ('SSR', SSR),
                                                ('SST', SST),
                                                ('R-sq', R_sq),
                                                ('adj-R-sq', adj_R_sq),
                                                ('F-stat', F_stat),
                                                ('Prob(F)', F_p_value)]))
        self._fitted = True   
        if show_res:
            show_model_res(self)
            
            
