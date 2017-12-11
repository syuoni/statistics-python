import numpy as np
import pandas as pd
from scipy import stats

from statspy.ols import OrdinaryLeastSquare
from statspy.tools import show_model_res

class TwoStepLeastSquare(OrdinaryLeastSquare):
    def __init__(self, df, y_var, x_vars, endog_var, exog_vars, has_const=True):
        super(TwoStepLeastSquare, self).__init__(df, y_var, x_vars, has_const=has_const)
        self.endog_var = endog_var
        self.exog_vars = exog_vars.copy()
    
    def _clean_data(self):
        reg_vars = [self.y_var] + self.x_vars + [self.endog_var] + self.exog_vars
        self.reg_df = self.df[reg_vars].dropna()
        if self.has_const:
            if '_const' not in self.reg_df: self.reg_df['_const'] = 1
            if '_const' not in self.x_vars: self.x_vars.append('_const')
    
    def fit(self, robust=False, show_res=True):
        self.robust = robust
        self._clean_data()
        
        # First Step
        self.endog_var_hat = self.endog_var + '_hat'
        self.step1 = OrdinaryLeastSquare(self.reg_df, self.endog_var, self.exog_vars+self.x_vars)
        self.step1.fit(robust=robust, show_res=False)
        self.reg_df[self.endog_var_hat] = self.step1.predict(self.df)
        
        # Second Step
        y = self.reg_df[self.y_var]
        X = self.reg_df[[self.endog_var] + self.x_vars]
        X_hat = self.reg_df[[self.endog_var_hat] + self.x_vars]
        n, K = X_hat.shape
        
        XtX = X_hat.T.dot(X_hat)
        XtX_inv = np.linalg.inv(XtX)
        
        b_hat = XtX_inv.dot(X_hat.T).dot(y)
        # NOTE: use X, rather than X_hat, to calculate prediction and error
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
            S_hat = X_hat.T.dot((e_hat**2).values.reshape((-1, 1)) * X_hat) / n
            Avarb_hat = (n*XtX_inv).dot(S_hat).dot(n*XtX_inv)
        
            # freedom adjusted as n-K 
            # stata use freedom=n as default, use "small" option to make adjustment
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
                                       index  =[self.endog_var] + self.x_vars)

        self.res_stats = pd.Series(      ['2SLS',    robust,   n,     s,      SSE,   SSR,   SST,   R_sqr,  adj_R_sqr,  F_statistic, F_p_value],
                                   index=['method', 'robust', 'obs', 'RMSE', 'SSE', 'SSR', 'SST', 'R-sq', 'adj-R-sq', 'F-stats',   'Prob(F)'])
        
        self._fitted = True   
        if show_res: show_model_res(self)
        