# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from statspy.mle import MaximumLikelihoodEstimation
from statspy.tools import show_model_res

class LinearModel(MaximumLikelihoodEstimation):
    # If not call the initial function of super-class explicitly, it would NOT be called
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(LinearModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.hess = None
        self._init()
        self._clean_data()
    
    def neg_loglikelihood(self, params):
        beta, logsigma = params[:-1], params[-1]
        sigma = np.exp(logsigma)
        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        e_hat = y - Xb
        
        loglikelihood = -0.5*(e_hat)**2/sigma**2 - 0.5*np.log(2*np.pi) - logsigma
        return -np.sum(loglikelihood)
        
    def jac(self, params):
        beta, logsigma = params[:-1], params[-1]
        sigma = np.exp(logsigma)
        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        e_hat = y - Xb
        
        beta_gr = np.sum(e_hat.reshape((-1, 1))*X, axis=0) / sigma**2
        logsigma_gr = np.sum(e_hat**2/sigma**2 - 1)
        return -np.append(beta_gr, logsigma_gr)
    
    def predict(self, df):
        assert self._fitted
        if self.has_const:
            X = df.assign(_const=1)[self.x_vars]
        else:
            X = df[self.x_vars]
        return X.values.dot(self.res_table['Coef'].values[:-1])
        
    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars) + 1)
        params0 = pd.Series(params0, index=np.append(self.x_vars, 'logsigma'))
        self._optimize(params0, **kwargs)
        self._fitted = True   
        if show_res:
            show_model_res(self)
        