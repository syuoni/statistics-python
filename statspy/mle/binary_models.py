# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import norm

from statspy.mle import MaximumLikelihoodEstimation
from statspy.tools import show_model_res

class ProbitModel(MaximumLikelihoodEstimation):
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(ProbitModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.hess = None
        self._init()
        self._clean_data()
    
    def neg_loglikelihood(self, params):
        beta = params        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        
        loglikelihood = y*norm.logcdf(Xb) + (1-y)*norm.logcdf(-Xb)
        return -np.sum(loglikelihood)
        
    def jac(self, params):
        beta = params        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        tmp = y*norm.pdf(Xb)/norm.cdf(Xb) - (1-y)*norm.pdf(-Xb)/norm.cdf(-Xb)
        gr = np.sum(tmp.reshape((-1, 1)) * X, axis=0)
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
        return np.where(linear_predict >= 0, 1, 0)
        
    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars))
        params0 = pd.Series(params0, index=self.x_vars)
        self._optimize(params0, **kwargs)
        self._fitted = True   
        if show_res:
            show_model_res(self)

        
class LogitModel(MaximumLikelihoodEstimation):
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(LogitModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.hess = None
        self._init()
        self._clean_data()
    
    def neg_loglikelihood(self, params):
        beta = params        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        eXb = np.exp(Xb)
        
        loglikelihood = y*np.log(eXb/(1+eXb)) + (1-y)*np.log(1/(1+eXb))
        return -np.sum(loglikelihood)
        
    def jac(self, params):
        beta = params        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        eXb = np.exp(Xb)
        
        tmp = (y - (1-y)*eXb)/(1+eXb)
        gr = np.sum(tmp.reshape((-1, 1)) * X, axis=0)
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
        return np.where(linear_predict >= 0, 1, 0)
        
    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars))
        params0 = pd.Series(params0, index=self.x_vars)
        self._optimize(params0, **kwargs)
        self._fitted = True   
        if show_res:
            show_model_res(self)

