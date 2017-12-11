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
        
        # cencored variable: left  cencored is 0,
        #                    not   cencored is 1,
        #                    right cencored is 2.
        self.df['_c'] = 1
        if lower is not None:
            self.df['_c'] = np.where(self.df[y_var] <= lower, 0, self.df['_c'])
        if upper is not None:
            self.df['_c'] = np.where(self.df[y_var] >= upper, 2, self.df['_c'])
    
    def neg_loglikelihood(self, params):
        beta, logsigma = params[:-1], params[-1]
        sigma = np.exp(logsigma)
        
        y = self.reg_df[self.y_var]
        X = self.reg_df[self.x_vars]
        
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
            df['_const'] = 1
        X = df[self.x_vars]
        return X.dot(self.res_table['coef'][:-1])

    def fit(self, show_res=True, **kwargs):
        self._clean_data()
        
        params0 = np.ones(len(self.x_vars)+1)
        params0 = pd.Series(params0, index=np.append(self.x_vars, 'logsigma'))
        self._optimize(params0, **kwargs)
        self._fitted = True   
        if show_res: show_model_res(self)
        
