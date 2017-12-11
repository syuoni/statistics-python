import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from statspy.base import BaseModel
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
        self.numeric_hess = gen_hessian_with_func(self.neg_loglikelihood) if self.hess is None else None
    
    def _rescale_params(self, params):
        if (params == 0).all():
            return params
        else:
            res = minimize_scalar(lambda c: self.neg_loglikelihood(c*params))
            return res.fun, res.x
    
    def _initial_params(self, params0):
        '''rescale and randomly alternative to initialize parameters'''
        n_params = len(params0)
        
        alt_params = np.random.uniform(low=-1, high=1, size=(5, n_params))
        alt_params[0] = params0
        alt_params[1] = 1
        
        alt_nll_scale = np.array([self._rescale_params(params) for params in alt_params])
        idx = np.argmin(alt_nll_scale[:, 0])
        return alt_nll_scale[idx, 0], alt_nll_scale[idx, 1]*alt_params[idx]
    
    def _optimize(self, params0, robust=False, method='bfgs', log=True):
        n, K = self.reg_df[self.x_vars].shape
        
        nll0 = self.neg_loglikelihood(params0)
        if log: print('Initial: negative log-likelihood = %.5f' % nll0)
        
        nll1, param1 = self._initial_params(params0)
        if log: print('Rescale & Alternative: negative log-likelihood = %.5f' % nll1)
        
        res = minimize(self.neg_loglikelihood, param1, method=method, jac=self.jac)
        nll2 = res.fun
        if log: print('Optimized: negative log-likelihood = %.5f' % nll2)
        max_jac = np.max(np.abs(res.jac))
        print('Optimized: max jacobian = %.5f' % max_jac)
        
        coef = res.x
        # DO NOT use the hess_inv returned by minimize
        if self.hess is not None:
            hess_inv = np.linalg.inv(self.hess(coef))
        else:
            hess_inv = np.linalg.inv(self.numeric_hess(coef))
            
        Avarb = n*hess_inv
        est_cov = Avarb / n
        est_std_err = np.diag(est_cov) ** 0.5
        z_statistic = coef / est_std_err
        
        p_value = 2 * (1-norm.cdf(np.abs(z_statistic)))
        z95 = norm.ppf(0.975)
        conf_int_lower = coef - z95*est_std_err
        conf_int_upper = coef + z95*est_std_err
        
        self.res_table = pd.DataFrame({'coef':        res.x,
                                       'est.std.err': est_std_err,
                                       'z-statistic': z_statistic,
                                       'p-value':     p_value,
                                       'conf.int.lower': conf_int_lower,
                                       'conf.int.upper': conf_int_upper},
                                       columns=['coef', 'est.std.err', 'z-statistic', 'p-value', 'conf.int.lower', 'conf.int.upper'],
                                       index  =params0.index)
        
        self.res_stats = pd.Series(      ['MLE',     max_jac,   n,     nll2],
                                   index=['method', 'max.jac', 'obs', 'neg_loglikelihood'])
        
        