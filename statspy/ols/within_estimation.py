# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd

from statspy.base import BaseModel, calc_R_sq
from statspy.ols import OrdinaryLeastSquare
from statspy.tools import show_model_res


class WithinEstimation(BaseModel):
    def __init__(self, df, y_var, x_vars, group_var):
        super(WithinEstimation, self).__init__(df, y_var, x_vars, has_const=False)
        self.group_var = group_var
        
        self._clean_data()
    
    def _make_reg_vars(self):
        # NO `_const' for within-estimation
        reg_vars = [self.y_var] + self.x_vars + [self.group_var]
        return reg_vars
    
    def predict(self, df):
        pass
        
    def fit(self, robust=False, show_res=True):
        self.robust = robust
        
        self.exp_df = self.reg_df.groupby(self.group_var).transform(np.mean)
        self.resid_df = self.reg_df[[self.y_var] + self.x_vars] - self.exp_df
        
        self.ols = OrdinaryLeastSquare(self.resid_df, self.y_var, self.x_vars, has_const=False)
        self.ols.fit(robust=robust, show_res=False)
        
        n, K = self.reg_df[self.x_vars].shape
        K = K + self.reg_df[self.group_var].nunique()
        
        y = self.reg_df[self.y_var].values
        y_hat = self.exp_df[self.y_var].values + self.ols.predict(self.resid_df)
        e_hat = y - y_hat
        s_sq = (e_hat**2).sum() / (n-K)
        s = s_sq ** 0.5
        
        # Calculate R-squared
        R_sq, adj_R_sq, SSE, SSR, SST = calc_R_sq(y, y_hat, n, K, return_SS=True)
        
        self.res_table = self.ols.res_table
        # NO F-stat here, since the full Cov-matrix cannot be estimated 
        self.res_stats = pd.Series(OrderedDict([('method', 'within-estimation'),
                                                ('robust', robust),
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
        
    
        
        

