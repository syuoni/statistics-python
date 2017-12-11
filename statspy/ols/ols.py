import numpy as np
import pandas as pd
from scipy import stats

from statspy.base import BaseModel
from statspy.tools import show_model_res

class OrdinaryLeastSquare(BaseModel):
    '''Ordinary Least Square Estimation
    '''
    def __init__(self, df, y_var, x_vars, has_const=True):
        super(OrdinaryLeastSquare, self).__init__(df, y_var, x_vars, has_const=has_const)
    
    def predict(self, df):
        assert self._fitted
        if self.has_const:
            df['_const'] = 1
        X = df[self.x_vars]
        return X.dot(self.res_table['coef'])
    
    def F_test(self, R):
        X = self.reg_df[self.x_vars]
        n, K = X.shape
        
        Rb = R.dot(self.coef)
        R_Avarb_hat_Rt_inv = np.linalg.inv(R.dot(self.Avarb_hat).dot(R.T))
        if self.robust:
            # F-test for all coef=0 (except for the constant)
            F_statistic = n * Rb.dot(R_Avarb_hat_Rt_inv).dot(Rb) / (K-1)
        else:
            # Wald-test and F-test are equivalent in large sample
            Wald_statistic = n * Rb.dot(R_Avarb_hat_Rt_inv).dot(Rb)
            F_statistic = Wald_statistic / (K-1)
        F_dist = stats.f(K-1, n-K)
        F_p_value = 1 - F_dist.cdf(F_statistic)                
        return F_statistic, F_p_value        
        
    def fit(self, robust=False, show_res=True):
        self.robust = robust
        self._clean_data()
        
        y = self.reg_df[self.y_var]
        X = self.reg_df[self.x_vars]
        n, K = X.shape
        
        XtX = X.T.dot(X)
        XtX_inv = np.linalg.inv(XtX)
        
        b_hat = XtX_inv.dot(X.T).dot(y)
        y_hat = X.dot(b_hat)
        e_hat = y - y_hat
        s_sqr = np.sum(e_hat**2) / (n-K)
        s = s_sqr ** 0.5
        
        # calculate R-square
        y_mean = np.mean(y)
        SST = np.sum((y - y_mean)**2)
        SSE = np.sum((y_hat - y_mean)**2)
        SSR = np.sum(e_hat**2)        
        R_sqr = 1 - SSR / SST        
        adj_R_sqr = 1 - (SSR/(n-K)) / (SST/(n-1))
        
        self.coef = b_hat
        
        if not self.robust:
            # stanard errors and t-test
            self.Avarb_hat = n * XtX_inv * s_sqr
        else:
            # sqrt(n)*(b-beta) ~ N(0, Avarb)
            # b-beta = (X'X)-1 * X' * epsilon
            S_hat = X.T.dot((e_hat**2).values.reshape((-1, 1)) * X) / n
            Avarb_hat = (n*XtX_inv).dot(S_hat).dot(n*XtX_inv)
        
            # freedom adjusted as n-K to be consistent with stata
            self.Avarb_hat = Avarb_hat * n / (n-K)
            
        est_cov = self.Avarb_hat / n
        est_std_err = np.diag(est_cov) ** 0.5
        t_statistic = b_hat / est_std_err
        
        # use t(n-K) distribution to be consistent with stata
        # option: use standard normal distribution, if robust
        t_dist = stats.t(n-K)
        p_value = (1 - t_dist.cdf(np.abs(t_statistic))) * 2
        
        t95 = t_dist.ppf(0.9750)
        conf_int_lower = b_hat - t95 * est_std_err
        conf_int_upper = b_hat + t95 * est_std_err
                
        F_statistic, F_p_value = self.F_test(np.identity(K)[:-1])
        
        self.res_table = pd.DataFrame({'coef':        b_hat,
                                       'est.std.err': est_std_err,
                                       't-statistic': t_statistic,
                                       'p-value':     p_value,
                                       'conf.int.lower': conf_int_lower,
                                       'conf.int.upper': conf_int_upper},
                                       columns=['coef', 'est.std.err', 't-statistic', 'p-value', 'conf.int.lower', 'conf.int.upper'],
                                       index  =self.x_vars)

        self.res_stats = pd.Series(      ['OLS',     robust,   n,     s,      SSE,   SSR,   SST,   R_sqr,  adj_R_sqr,  F_statistic, F_p_value],
                                   index=['method', 'robust', 'obs', 'RMSE', 'SSE', 'SSR', 'SST', 'R-sq', 'adj-R-sq', 'F-stats',   'Prob(F)'])
        
        self._fitted = True   
        if show_res: show_model_res(self)
