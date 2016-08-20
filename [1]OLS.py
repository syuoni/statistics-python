# coding: utf-8
from __future__ import division
from scipy import stats
import pandas as pd
import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm

class OLS(object):
    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.n, self.K = X.shape
        
    def fit(self, robust=False):
        y, X = self.y, self.X
        n, K = self.n, self.K
        
        XtX = X.T.dot(X)
        XtXi = inv(XtX)
        B = XtXi.dot(X.T)
        # projection matrix
        P = X.dot(B)
        # annihilator matrix
        M = np.identity(n) - P
        
        self.b = B.dot(y)
        self.y_hat = P.dot(y)
        self.e = M.dot(y)
        
        self.s_square = np.sum(self.e**2) / (n-K)
        self.s = self.s_square ** 0.5
        
        # R-square
        y_mean = np.mean(y)
        SST = np.sum((y - y_mean)**2)
        SSE = np.sum((self.y_hat - y_mean)**2)
        SSR = np.sum(self.e**2)        
        self.R_square = SSE / SST        
        self.adj_R_square = 1 - (SSR/(n-K)) / (SST/(n-1))
        
        if not robust:
            self.covariance_type = 'non-robust'
            # stanard errors and t-test
            self.est_cov = self.s_square * XtXi
            self.est_std_err = (np.diag(self.est_cov))**0.5
            self.t_statistic = self.b / self.est_std_err
            
            t_dist = stats.t(n-K)
            self.pvalue = (1 - t_dist.cdf(np.abs(self.t_statistic))) * 2
            
            t_95 = t_dist.ppf(97.5/100)
            self.conf_int_lower = self.b - t_95 * self.est_std_err
            self.conf_int_upper = self.b + t_95 * self.est_std_err        
            
            # F-test for all coef=0 (except for the constant)
            R = np.identity(K)[:-1]
            Rb = R.dot(self.b)
            RXtXiRt = inv(R.dot(XtXi).dot(R.T))
            self.F_statistic = Rb.dot(RXtXiRt).dot(Rb) / (K-1) / self.s_square
            
            F_dist = stats.f(K-1, n-K)
            self.F_pvalue = 1 - F_dist.cdf(self.F_statistic)        
        else:
            self.covariance_type = 'robust'
            # sqrt(n)*(b-beta) ~ N(0, Avarb)
            # b-beta = (X'X)-1 * X' * epsilon
            Avarb_hat = n * B.dot(np.diag(self.e**2)).dot(B.T)
            # freedom adjusted as n-K to be consistent with stata
            Avarb_hat = Avarb_hat * n / (n-K)
            self.est_cov = Avarb_hat / n
            self.est_std_err = (np.diag(self.est_cov))**0.5
            self.t_statistic = self.b / self.est_std_err
            
            # use t(n-K) distribution to be consistent with stata
            # option: use standard normal distribution
            t_dist = stats.t(n-K)
            self.pvalue = (1 - t_dist.cdf(np.abs(self.t_statistic))) * 2
            
            t_95 = t_dist.ppf(97.5/100)
            self.conf_int_lower = self.b - t_95 * self.est_std_err
            self.conf_int_upper = self.b + t_95 * self.est_std_err
            
            # Wald-test and F-test are equivalent in large sample
            R = np.identity(K)[:-1]
            Rb = R.dot(self.b)
            RAvarb_hatRti = inv(R.dot(Avarb_hat).dot(R.T))
            self.Wald_statistic = n * Rb.dot(RAvarb_hatRti).dot(Rb)
            self.F_statistic = self.Wald_statistic / (K-1)
            
            F_dist = stats.f(K-1, n-K)
            self.F_pvalue = 1 - F_dist.cdf(self.F_statistic)                    
            
        self.variable_res_table = pd.DataFrame({'coef': self.b,
                                                'est.std.err': self.est_std_err,
                                                't-statistic': self.t_statistic,
                                                'p-value': self.pvalue,
                                                'conf.int.lower': self.conf_int_lower,
                                                'conf.int.upper': self.conf_int_upper},
                                                columns=['coef', 'est.std.err', 't-statistic', 'p-value',
                                                         'conf.int.lower', 'conf.int.upper'],
                                                index=X.columns)
     
    def print_summary(self):
        print '=' * 70
        print 'Covariance:\t%s' % self.covariance_type
        print 'Observations:\t%d' % self.n
        print 'R-square:\t%.4f' % self.R_square
        print 'Adj-R-square:\t%.4f' % self.adj_R_square
        print 'F-statitic:\t%.4f' % self.F_statistic
        print 'Prob(F):\t%.4f' % self.F_pvalue
        print '=' * 70
        print self.variable_res_table
        print '=' * 70

if __name__ == '__main__':    
    df = pd.read_csv('reg.csv')
    n, m = df.shape
    const = np.ones(n)
    df['const'] = const
    
    y = df['y']
    X = df[['x1', 'x3', 'x4', 'const']]
    
    md = OLS(y, X)
    md.fit(robust=False)
    md.print_summary()

#    md = sm.OLS(y, X)
#    res = md.fit()
#    print res.summary()
