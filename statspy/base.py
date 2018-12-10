# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

class BaseModel(object):
    '''The base model
    '''
    def __init__(self, df, y_var, x_vars, wt_var=None, has_const=True):
        self.df = df
        self.y_var = y_var
        self.x_vars = x_vars.copy()
        self.wt_var = wt_var
        self.has_const = has_const
        self._fitted = False
        
    def _make_reg_vars(self):
        if self.has_const and ('_const' not in self.x_vars):
            self.x_vars.append('_const')
            
        reg_vars = []
        if self.y_var is not None:
            reg_vars.append(self.y_var)
        reg_vars.extend(self.x_vars)
        if self.wt_var is not None:
            reg_vars.append(self.wt_var)
        return reg_vars
        
    def _clean_data(self):
        # self.df should NOT be modified in any case
        # self.reg_df only contains variables needed in the regression
        self.reg_vars = self._make_reg_vars()
        self.reg_df = self.df.assign(_const=1.0)[self.reg_vars].dropna().copy()
        
    def fit(self):
        raise NotImplementedError
        
    def predict(self, df):
        raise NotImplementedError
        
        
def F_test(X, coef, Avarb_hat, R, robust=False):
    n, K = X.shape
    if K <= 1:
        return np.nan, np.nan
    
    Rb = R.dot(coef)
    R_Avarb_hat_Rt_inv = np.linalg.inv(R.dot(Avarb_hat).dot(R.T))
    if robust:
        # F-test for all coef=0 (except for the constant)
        F_stat = n * Rb.dot(R_Avarb_hat_Rt_inv).dot(Rb) / (K-1)
    else:
        # Wald-test and F-test are equivalent in large sample
        Wald_statistic = n * Rb.dot(R_Avarb_hat_Rt_inv).dot(Rb)
        F_stat = Wald_statistic / (K-1)
    F_dist = stats.f(K-1, n-K)
    p_value = 1 - F_dist.cdf(F_stat)
    return F_stat, p_value


def calc_R_sq(y_true, y_hat, n, K, wt=None, return_SS=True):
    if wt is None:
        wt = np.ones_like(y_true)
        
    y_mean = y_true.dot(wt) / wt.sum()
    # Sum of squares total, explained, residual
    SST = ((y_true - y_mean)**2).dot(wt**2)
    SSE = ((y_hat - y_mean)**2).dot(wt**2)
    SSR = ((y_true - y_hat)**2).dot(wt**2)
    R_sq = 1 - SSR / SST
    adj_R_sq = 1 - (SSR/(n-K)) / (SST/(n-1))
    
    if return_SS:
        return R_sq, adj_R_sq, SSE, SSR, SST
    else:
        return R_sq, adj_R_sq
    
    
def coef_test(coef, est_cov, dist, CI_alpha=0.05):
    est_std_err = np.diag(est_cov) ** 0.5
    t_stat = coef / est_std_err
    p_value = (1 - dist.cdf(np.abs(t_stat))) * 2
    
    dist_thres = dist.ppf(1 - CI_alpha/2)
    CI_lower = coef - dist_thres * est_std_err
    CI_upper = coef + dist_thres * est_std_err
    return est_std_err, t_stat, p_value, CI_lower, CI_upper

    
