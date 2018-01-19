# import module
# form module import class/function...
import unittest
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from statspy.base import BaseModel
from statspy.ols import OrdinaryLeastSquare, TwoStepLeastSquare, WithinEstimation
from statspy.tools import gen_hessian_with_func, clean4reg, show_model_res, format_res_table, desc_stats
from statspy.mle import MaximumLikelihoodEstimation, ProbitModel, LogitModel, LinearModel, TobitModel
from statspy.nonparam import KernelDensity, KernelFunction, LocalPolyRegression, RobinsonRegression
from statspy.mle import ExponentialModel, WeibullModel

class TestOLS(unittest.TestCase):
    def test_ols_result(self):
        df = pd.read_stata(r'example-data\womenwk.dta')
        ols_md = OrdinaryLeastSquare(df, 'work', ['age', 'married', 'children', 'education'])
        ols_md.fit(robust=True, show_res=False)
        
        self.assertEqual(ols_md.res_stats['method'], 'OLS')
        self.assertTrue(ols_md.res_stats['robust'])        
        self.assertAlmostEqual(ols_md.res_stats['adj-R-sq'], 0.20102407308444759)
        self.assertAlmostEqual(ols_md.res_stats['F-stats'], 192.57650255037254)
        
        self.assertTrue(np.allclose(ols_md.res_table['coef'].values, 
                                    np.array([0.01025522, 0.11111163, 0.11530842, 0.01860109, -0.2073227])))
        self.assertTrue(np.allclose(ols_md.res_table['est.std.err'].values, 
                                    np.array([0.00122355, 0.02267192, 0.0056978 , 0.00330056, 0.05345809])))
        self.assertTrue(np.allclose(ols_md.res_table['t-statistic'].values, 
                                    np.array([8.381508  , 4.90084717,20.23737107, 5.63573275, -3.87822915])))
        self.assertTrue(np.allclose(ols_md.predict(df).values[:5],
                                    np.array([0.31541467, 0.45898772, 0.37694598, 0.46924294,  0.6050618 ])))
        
class Test2SLS(unittest.TestCase):
    def test_2sls_result(self):
        df = pd.read_stata(r'example-data\grilic.dta')
        tsls_md = TwoStepLeastSquare(df, 'lw', ['s', 'expr', 'tenure', 'rns', 'smsa'], 'iq', ['med', 'kww', 'mrt', 'age'])
        tsls_md.fit(robust=True, show_res=False)
        
        self.assertEqual(tsls_md.res_stats['method'], '2SLS')
        self.assertTrue(tsls_md.res_stats['robust'])        
        self.assertAlmostEqual(tsls_md.res_stats['adj-R-sq'], 0.19370465812528215)
        self.assertAlmostEqual(tsls_md.res_stats['F-stats'], 70.456250828834001)
        
        self.assertTrue(np.allclose(tsls_md.res_table['coef'].values, 
                                    np.array([-0.01155218, 0.13734069, 0.03380146, 0.0405657, -0.11771223, 0.1499839, 4.83838987])))
        self.assertTrue(np.allclose(tsls_md.step1.res_table['coef'].values[:6], 
                                    np.array([0.334796190, 0.370970577, -0.901282787, -0.637618780, 2.92112398, 0.0130364895])))
        
#if __name__ == '__main__':
#    unittest.main(verbosity=2)




# mle-linear model
#df = pd.read_stata('example-data\\womenwk.dta')
#lm = LinearModel(df, 'work', ['age', 'married', 'children', 'education'])
#lm.fit()

# mle-probit
#df = pd.read_stata('example-data\\womenwk.dta')
#pm = ProbitModel(df, 'work', ['age', 'married', 'children', 'education'])
#pm.fit()

# mle-logit
#df = pd.read_stata('example-data\\womenwk.dta')
#lm = LogitModel(df, 'work', ['age', 'married', 'children', 'education'])
#lm.fit()
    
# mle-tobit
#df = pd.read_stata('example-data\\womenwk.dta')
#tm = TobitModel(df, 'lwf', ['age', 'married', 'children', 'education'], lower=0)
#tm.fit()


# kernel-density 
#n_sample = 2000
#df = pd.DataFrame({'x': 2*np.random.randn(n_sample),
#                   'y': np.random.randn(n_sample)-1})
#
#kd = KernelDensity(df, ['x', 'y'], kernel=KernelFunction.gaussian)
#kd.fit(h=None)
#res_panel = kd.predict_with_bounds()
#print(stats.norm.pdf(0)**2/2, max(res_panel['density_hat']))
#print(stats.norm.pdf(1)**2/2, kd.predict_obs([2, 0]))

# kernel-regression
#n_sample = 2000
#data = np.random.randn(n_sample, 2)
#df = pd.DataFrame(data, columns=['x', 'y'])

#f = lambda x: np.sin(2*x) + 2*np.exp(-16*x**2)
#df['z'] = f(df['x']) + df['y']
#kr = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.quadratic, degree=0)
#kr.fit(h=[0.2], self_fit=True)
#lr = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.quadratic, degree=1)
#lr.fit(h=[0.2], self_fit=True)
#lw0 = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.tricubic, degree=1)
#lw0.fit(local_ratio=0.15, robust_wt_updates=0, self_fit=True)
#lw3 = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.tricubic, degree=1)
#lw3.fit(local_ratio=0.15, robust_wt_updates=3, self_fit=True)
#
#x_seq = np.linspace(-2.5, 2.5, 100)
#y_seq_true = f(x_seq)
#y_seq_kr = kr.predict_obs_seq(x_seq[:, None])['z_hat'].values
#y_seq_lr = lr.predict_obs_seq(x_seq[:, None])['z_hat'].values
#y_seq_lw0 = lw0.predict_obs_seq(x_seq[:, None])['z_hat'].values
#y_seq_lw3 = lw3.predict_obs_seq(x_seq[:, None])['z_hat'].values
#
#fig, ax = plt.subplots(figsize=(8, 5))
##ax.scatter(df['x'], df['z'], s=1)
#ax.plot(x_seq, y_seq_true, label='true')
#ax.plot(x_seq, y_seq_kr, label='kr')
#ax.plot(x_seq, y_seq_lr, label='lr')
#ax.plot(x_seq, y_seq_lw0, label='lw0')
#ax.plot(x_seq, y_seq_lw3, label='lw3')
#ax.legend(loc='upper right')
#plt.show()

# interval plotting
#md = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.quadratic, degree=1)
#md.fit(h=np.array([0.4]))
##md.fit(local_ratio=0.3)
#fig, ax = plt.subplots(figsize=(8, 5))
#ax.scatter(df['x'], df['z'], s=1)
#y_seq_lr = md.predict_obs_seq(x_seq[:, None], stat_inf=True)
#ax.plot(x_seq, y_seq_lr['z_hat'])
#ax.plot(x_seq, y_seq_lr['_conf_int_lower'])
#ax.plot(x_seq, y_seq_lr['_conf_int_upper'])
#plt.show()


#df['z'] = df['x'] + df['y']
#kr = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=KernelFunction.trunc_gaussian, degree=0)
#kr.fit(h=[0.5, 0.5], local_ratio=0.3)
#res_panel_kr = kr.predict_with_bounds()
#print(kr.predict_obs([-5, -5], stat_inf=True))
#
#lr = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=KernelFunction.trunc_gaussian, degree=2)
#lr.fit(h=[0.5, 0.5], local_ratio=0.3)
#res_panel_lr = lr.predict_with_bounds()
#print(lr.predict_obs([-5, -5], stat_inf=True))
#
#lw = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=KernelFunction.trunc_gaussian, degree=2)
#lw.fit(local_ratio=0.3)
#res_panel_lw = lw.predict_with_bounds()
#print(lw.predict_obs([-5, -5], stat_inf=True))


# semi-parameter estimation
#df = pd.read_stata('example-data\\nerlove.dta')
#print(desc_stats(df))
#
#ols = OrdinaryLeastSquare(df, 'lntc', ['lnpf', 'lnq', 'lnpl', 'lnpk'])
#ols.fit(robust=True)
#
#rb = RobinsonRegression(df, 'lntc', ['lnpf'], ['lnq', 'lnpl', 'lnpk'])
#rb.fit(robust=True, kernel=KernelFunction.clust_gaussian, degree=1, h=[0.15], local_ratio=0.2, robust_wt_updates=0)
#
#pred_df = pd.DataFrame()
#pred_df['lnpf'] = np.linspace(2, 4, 50)
##pred_df['lnq'] = np.mean(df['lnq'])
##pred_df['lnpl'] = np.mean(df['lnpl'])
##pred_df['lnpk'] = np.mean(df['lnpk'])
#y_hat, std_dev, conf_int_lower, conf_int_upper = rb.predict(pred_df, predict_type='nonpar', stat_inf=True)
#
#fig, ax = plt.subplots(figsize=(8, 5))
#ax.scatter(rb.reg_df['lnpf'], rb.reg_df['_eta'])
#ax.plot(pred_df['lnpf'], y_hat)
#ax.plot(pred_df['lnpf'], conf_int_lower)
#ax.plot(pred_df['lnpf'], conf_int_upper)
#plt.show()

# duration models 
# TODO: weibull model cannot converge because of unrobust optimization method (sensitive to initial params)
#df = pd.read_stata('example-data\\recid.dta')
#df['d'] = 1 - df['cens']
#exp_md = ExponentialModel(df, y_var='durat', x_vars=['workprg', 'priors', 'tserved', 'felon', 'alcohol',
#                                                     'drugs', 'black', 'married', 'educ', 'age'], d_var='d')
#exp_md.fit()

#wb_md = WeibullModel(df, y_var='durat', x_vars=['workprg', 'priors', 'tserved', 'felon', 'alcohol',
#                                                'drugs', 'black', 'married', 'educ', 'age'], d_var='d')
#wb_md.fit()


df = pd.read_stata(r'example-data\womenwk.dta')
md = WithinEstimation(df, 'work', ['age', 'married', 'children', 'education'], 'county')
md.fit(robust=True, show_res=True)



