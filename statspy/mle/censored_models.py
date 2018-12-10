# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import norm

from statspy.mle import MaximumLikelihoodEstimation
from statspy.tools import show_model_res

class TobitModel(MaximumLikelihoodEstimation):
    def __init__(self, df, y_var, x_vars, lower=None, upper=None, has_const=True):
        assert not (lower is None and upper is None)
        
        super(TobitModel, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.jac = None
        self.hess = None        
        self._init()
        self._clean_data()
        
        # Cencored variable: Left  cencored is 0,
        #                    Not   cencored is 1,
        #                    Right cencored is 2.
        self.reg_df['_c'] = 1
        if lower is not None:
            self.reg_df.loc[self.reg_df[y_var] <= lower, '_c'] = 0
        if upper is not None:
            self.reg_df.loc[self.reg_df[y_var] >= upper, '_c'] = 2
    
    def neg_loglikelihood(self, params):
        beta, logsigma = params[:-1], params[-1]
        sigma = np.exp(logsigma)
        
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        
        Xb = X.dot(beta)
        e_hat = y - Xb
        
        loglikelihood = np.zeros((3, len(y)))
        loglikelihood[0] = norm.logcdf(e_hat/sigma)
        loglikelihood[1] = -0.5*(e_hat)**2/sigma**2 - 0.5*np.log(2*np.pi) - logsigma
        loglikelihood[2] = norm.logcdf(-e_hat/sigma)
        
        loglikelihood = loglikelihood[self.reg_df['_c'], np.arange(len(y))]
        return -np.sum(loglikelihood)
        
#    def jac(self, params, data):
#        beta, logsigma = params[:-1], params[-1]
#        sigma = np.exp(logsigma)
#        
#        y = data['y']
#        X = data['X']
#        
#        Xb = X.dot(beta)
#        e_hat = y - Xb
#        
#        beta_gr = np.sum(e_hat.values.reshape((-1, 1))*X, axis=0) / sigma**2
#        logsigma_gr = np.sum(e_hat**2/sigma**2 - 1)
#        return -np.append(beta_gr, logsigma_gr)
    
    def predict(self, df):
        assert self._fitted
        if self.has_const:
            X = df.assign(_const=1)[self.x_vars]
        else:
            X = df[self.x_vars]
        return X.values.dot(self.res_table['Coef'].values[:-1])

    def fit(self, show_res=True, **kwargs):
        params0 = np.ones(len(self.x_vars)+1)
        params0 = pd.Series(params0, index=np.append(self.x_vars, 'logsigma'))
        self._optimize(params0, **kwargs)
        self._fitted = True   
        if show_res: show_model_res(self)
        
