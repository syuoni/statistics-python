# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from statspy.mle import MaximumLikelihoodEstimation
from statspy.tools import show_model_res

class ExponentialModel(MaximumLikelihoodEstimation):
    '''
    y is duration time
    d=0 indicates sample censored, d=1 indicates sample failed
    '''
    def __init__(self, df, y_var, x_vars, d_var, has_const=True):
        super(ExponentialModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.d_var = d_var
        self.hess = None
        self._init()
        self._clean_data()
        
    def _clean_data(self):
        reg_vars = [self.y_var] + self.x_vars + [self.d_var]
        self.reg_df = self.df[reg_vars].dropna()
        if self.has_const:
            if '_const' not in self.reg_df: self.reg_df['_const'] = 1
            if '_const' not in self.x_vars: self.x_vars.append('_const')
    
    def neg_loglikelihood(self, params):
        beta = params        
        t = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        d = self.reg_df[self.d_var].values
        
        Xb = X.dot(beta)
        
        loglikelihood = d*Xb - np.exp(Xb)*t
        return -np.sum(loglikelihood)
        
    def jac(self, params):
        beta = params        
        t = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        d = self.reg_df[self.d_var].values
        
        Xb = X.dot(beta)
        gr = np.sum((d-np.exp(Xb)*t)[:, None] * X, axis=0)
        return -gr
    
    def linear_predict(self, df):
        assert self._fitted
        if self.has_const:
            X = df.assign(_const=1)[self.x_vars]
        else:
            X = df[self.x_vars]
        return X.values.dot(self.res_table['Coef'].values)
        
    def predict(self, df):
        linear_predict = self.linear_predict(df)
        return np.exp(linear_predict)
        
    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars)) * 1e-5
        params0 = pd.Series(params0, index=self.x_vars)
        self._optimize(params0, params0_limit=1e-5, **kwargs)
        self._fitted = True   
        if show_res: show_model_res(self)


# TODO: weibull model cannot converge because of unrobust optimization method (sensitive to initial params)
class WeibullModel(MaximumLikelihoodEstimation):
    '''
    y is duration time
    d=0 indicates sample censored, d=1 indicates sample failed
    '''
    def __init__(self, df, y_var, x_vars, d_var, has_const=True):
        super(WeibullModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.d_var = d_var
        self.jac = None
        self.hess = None
        self._init()
        self._clean_data()
        
    def _clean_data(self):
        reg_vars = [self.y_var] + self.x_vars + [self.d_var]
        self.reg_df = self.df[reg_vars].dropna()
        if self.has_const:
            if '_const' not in self.reg_df: self.reg_df['_const'] = 1
            if '_const' not in self.x_vars: self.x_vars.append('_const')
    
    def neg_loglikelihood(self, params):
        beta, lnp = params[:-1], params[-1]
        t = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        d = self.reg_df[self.d_var].values
        
        p = np.exp(lnp)
        Xb = X.dot(beta)
        
        loglikelihood = d*(Xb+lnp+(p-1)*np.log(t)) - np.exp(Xb)*(t**p)
#        loglikelihood = d*(Xb+lnp) - np.exp(Xb)*t**p
        return -np.sum(loglikelihood)
        
    def jac(self, params):
        beta, lnp = params[:-1], params[-1]
        t = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        d = self.reg_df[self.d_var].values
        
        p = np.exp(lnp)
        Xb = X.dot(beta)
        
        beta_gr = np.sum((d-np.exp(Xb)*t**p)[:, None] * X, axis=0)
        lnp_gr = np.sum(d*(1+np.log(t)*p) - np.exp(Xb)*t**p*np.log(t)*p)
        return -np.append(beta_gr, lnp_gr)
#    
#    def linear_predict(self, df):
#        assert self._fitted
#        if self.has_const:
#            df['_const'] = 1
#        X = df[self.x_vars]
#        return X.dot(self.res_table['coef'])
#        
#    def predict(self, df):
#        linear_predict = self.linear_predict(df)
#        return np.exp(linear_predict)
        
    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars) + 1) * 1e-5
        params0 = pd.Series(params0, index=np.append(self.x_vars, 'lnp'))
        self._optimize(params0, params0_limit=1e-5, **kwargs)
        self._fitted = True   
        if show_res: show_model_res(self)


