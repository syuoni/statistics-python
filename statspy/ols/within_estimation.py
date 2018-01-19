# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from statspy.ols import OrdinaryLeastSquare
from statspy.tools import show_model_res


class WithinEstimation(OrdinaryLeastSquare):
    def __init__(self, df, y_var, x_vars, group_var):
        super(WithinEstimation, self).__init__(df, y_var, x_vars, has_const=False)
        self.group_var = group_var
        
    def _clean_data(self):
        reg_vars = [self.y_var] + self.x_vars + [self.group_var]
        self.reg_df = self.df[reg_vars].dropna()
        
    def fit(self, robust=False, show_res=True):
        self.robust = robust
        self._clean_data()
        
        self.exp_df = self.reg_df.groupby(self.group_var).transform(np.mean)
        self.resid_df = self.reg_df[[self.y_var] + self.x_vars] - self.exp_df
        
        self.ols = OrdinaryLeastSquare(self.resid_df, self.y_var, self.x_vars, has_const=False)
        self.ols.fit(robust=robust, show_res=False)
        
        n, K = self.reg_df[self.x_vars].shape
        
        y = self.reg_df[self.y_var].values
        y_hat = self.exp_df[self.y_var].values + self.ols.predict(self.resid_df)
        e_hat = y - y_hat
        s_sqr = np.sum(e_hat**2) / (n-K)
        s = s_sqr ** 0.5
        
        SST = np.sum((y - y.mean())**2)
        SSR = np.sum(e_hat ** 2)
        R_sqr = 1 - SSR / SST
        adj_R_sqr = 1 - (SSR/(n-K)) / (SST/(n-1))
        
        F_statistic = self.ols.res_stats['F-stats']
        F_p_value = self.ols.res_stats['Prob(F)']
        
        self.res_table = self.ols.res_table
        self.res_stats = pd.Series(      ['with-estimation', robust,   n,     s,      SSR,   SST,   R_sqr,  adj_R_sqr,  F_statistic, F_p_value],
                                   index=['method',         'robust', 'obs', 'RMSE', 'SSR', 'SST', 'R-sq', 'adj-R-sq', 'F-stats',   'Prob(F)'])
        
        self._fitted = True
        if show_res: show_model_res(self)
        
    def predict(self, df):
        pass
        
        

