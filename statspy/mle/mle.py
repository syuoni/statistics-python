# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from statspy.base import BaseModel, coef_test
from statspy.tools import gen_hessian_with_func

class MaximumLikelihoodEstimation(BaseModel):
    '''Maximum Likelihood Estimation Model Base
    
    neg_loglikelihood must be implemented in sub-classes!
    If jac or hess not implemented in sub-classes, MUST set it to be None in the initial function!
    '''
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(MaximumLikelihoodEstimation, self).__init__(df, y_var, x_vars, has_const=has_const)
            
    def neg_loglikelihood(self, params):
        raise NotImplementedError
        
    def jac(self, params):
        raise NotImplementedError
    
    def hess(self, params):
        raise NotImplementedError
        
    def _init(self):
        '''Additioanal initialization, may called by sub-classes'''
        if self.hess is None:
            self.numeric_hess = gen_hessian_with_func(self.neg_loglikelihood)
        else:
            self.numeric_hess = None
    
    def _rescale_params(self, params):
        if (params == 0).all():
            return params
        else:
            res = minimize_scalar(lambda c: self.neg_loglikelihood(c*params))
            return res.fun, res.x
    
    def _initial_params(self, params0, params0_limit=0.1):
        '''Rescale and randomly alternative to initialize parameters'''
        n_params = len(params0)
        
        alt_params = np.random.uniform(low=-params0_limit, high=params0_limit, size=(5, n_params))
        alt_params[0] = params0
        alt_params[1] = params0_limit
        
        alt_nll_scale = np.array([self._rescale_params(params) for params in alt_params])
        idx = np.argmin(alt_nll_scale[:, 0])
        return alt_nll_scale[idx, 0], alt_nll_scale[idx, 1]*alt_params[idx]
    
    def _optimize(self, params0, params0_limit=0.1, robust=False, method='bfgs', log=True):
        n, K = self.reg_df[self.x_vars].shape
        
        nll0 = self.neg_loglikelihood(params0)
        if log:
            print('Initial: negative log-likelihood = %.5f' % nll0)
        
        nll1, param1 = self._initial_params(params0, params0_limit=params0_limit)
        if log:
            print('Rescale & Alternative: negative log-likelihood = %.5f' % nll1)
        
        res = minimize(self.neg_loglikelihood, param1, method=method, jac=self.jac)
        nll2 = res.fun
        if log:
            print('Optimized: negative log-likelihood = %.5f' % nll2)
        max_jac = np.max(np.abs(res.jac))
        if log:
            print('Optimized: max jacobian = %.5f' % max_jac)
        
        coef = res.x
        # DO NOT use the hess_inv returned by minimize
        if self.hess is not None:
            hess_inv = np.linalg.inv(self.hess(coef))
        else:
            hess_inv = np.linalg.inv(self.numeric_hess(coef))
            
        Avarb = n * hess_inv
        est_cov = Avarb / n
        est_std_err, z_stat, p_value, CI_lower, CI_upper = coef_test(coef, est_cov, norm, CI_alpha=0.05)
        
        self.res_table = pd.DataFrame(OrderedDict([('Coef', coef), 
                                                   ('Std.Err', est_std_err),
                                                   ('z', z_stat),
                                                   ('p', p_value),
                                                   ('CI.lower', CI_lower),
                                                   ('CI.upper', CI_upper)]), index=params0.index)
        self.res_stats = pd.Series(OrderedDict([('method', 'MLE'),
                                                ('max.jac', max_jac),
                                                ('obs', n),
                                                ('neg_loglikelihood', nll2)]))
    