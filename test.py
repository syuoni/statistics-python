# -*- coding: utf-8 -*-
# import module
# form module import class/function...
import unittest
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from statspy.ols import OrdinaryLeastSquare, TwoStepLeastSquare, WithinEstimation, WeightedLeastSquare
from statspy.nonparam import KernelDensity, KernelFunction, LocalPolyRegression, RobinsonRegression
from statspy.tools import gen_jacobian, gen_hessian_with_func
from statspy.tools import show_model_res, format_res_table, desc_stats
from statspy.mle import MaximumLikelihoodEstimation, ProbitModel, LogitModel, LinearModel, TobitModel
from statspy.mle import ExponentialModel, WeibullModel


class TestTools(unittest.TestCase):
    def test_desc_stats(self):
        df = pd.read_stata('example-data/nerlove.dta')
        desc_df = desc_stats(df)
        self.assertEqual(desc_df.loc['lntc', 'Obs'], 145)
        self.assertAlmostEqual(desc_df.loc['lntc', 'Mean'], 1.724663, delta=1e-6)
        self.assertAlmostEqual(desc_df.loc['lntc', 'Std.Dev'], 1.421723, delta=1e-6)
        
    def test_numerical_deriv(self):
        # func(x0, x1) = y*x0 + 3x1 + exp(x1) + 2x0x1
        def func(x, y=2):
            return x[0] * y + x[1] ** 3 + np.exp(x[1]) + x[0]*x[1]**2
        
        def jac(x, y=2):
            return np.array([y+x[1]**2, 3*x[1]**2+np.exp(x[1])+2*x[0]*x[1]])
            
        def hess(x, y=2):
            return np.array([[0,      2*x[1]],
                             [2*x[1], 6*x[1]+np.exp(x[1])+2*x[0]]])
        
        for x0, x1, y in [(1, 3, 3), (0, -5, 5), (np.pi, 1.234, 4.321)]:
            jac_res_true = jac(np.array([x0, x1]), y)
            jac_num = gen_jacobian(func)
            jac_res_num = jac_num(np.array([x0, x1]), y)
            self.assertTrue(np.allclose(jac_res_num, jac_res_true, atol=1e-8))
            
            jac_num = gen_jacobian(func, h='nash')
            jac_res_num = jac_num(np.array([x0, x1]), y)
            self.assertTrue(np.allclose(jac_res_num, jac_res_true, atol=1e-8))
            
            hess_res_true = hess(np.array([x0, x1]), y)
            hess_num = gen_hessian_with_func(func)
            hess_res_num = hess_num(np.array([x0, x1]), y)
            self.assertTrue(np.allclose(hess_res_num, hess_res_true, atol=0.05))
            
            hess_num = gen_hessian_with_func(func, h='nash')
            hess_res_num = hess_num(np.array([x0, x1]), y)
            self.assertTrue(np.allclose(hess_res_num, hess_res_true, atol=0.05))
            
            
class TestOLS(unittest.TestCase):
    def test_ols_result(self):
        df = pd.read_stata('example-data/womenwk.dta')
        ols_md = OrdinaryLeastSquare(df, 'work', ['age', 'married', 'children', 'education'])
        ols_md.fit(robust=False, show_res=False)
        
        self.assertEqual(ols_md.res_stats['method'], 'OLS')
        self.assertFalse(ols_md.res_stats['robust'])
        self.assertAlmostEqual(ols_md.res_stats['RMSE'], 0.41992, delta=1e-5)
        self.assertAlmostEqual(ols_md.res_stats['adj-R-sq'], 0.20102401, delta=1e-5)
        self.assertAlmostEqual(ols_md.res_stats['F-stat'], 126.73813, delta=1e-5)
        
        self.assertTrue(np.allclose(ols_md.res_table['Coef'].values, 
                                    np.array([0.0102552, 0.1111116, 0.1153084, 0.0186011, -0.2073227]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(ols_md.res_table['Std.Err'].values, 
                                    np.array([0.0012269, 0.0219477, 0.0067715, 0.0032499, 0.054111]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(ols_md.res_table['t'].values, 
                                    np.array([8.36, 5.06, 17.03, 5.72, -3.83]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(ols_md.predict(df)[:5],
                                    np.array([0.3154147, 0.4589877, 0.376946, 0.4692429, 0.6050618])))
        
        # Fit with robust standard errors
        ols_md.fit(robust=True, show_res=False)
        
        self.assertEqual(ols_md.res_stats['method'], 'OLS')
        self.assertTrue(ols_md.res_stats['robust'])
        self.assertAlmostEqual(ols_md.res_stats['RMSE'], 0.41992, delta=1e-5)
        self.assertAlmostEqual(ols_md.res_stats['adj-R-sq'], 0.20102401, delta=1e-5)
        self.assertAlmostEqual(ols_md.res_stats['F-stat'], 192.576503, delta=1e-5)
        
        self.assertTrue(np.allclose(ols_md.res_table['Coef'].values, 
                                    np.array([0.0102552, 0.1111116, 0.1153084, 0.0186011, -0.2073227]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(ols_md.res_table['Std.Err'].values, 
                                    np.array([0.0012236, 0.0226719, 0.0056978, 0.0033006, 0.0534581]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(ols_md.res_table['t'].values, 
                                    np.array([8.38, 4.90, 20.24, 5.64, -3.88]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(ols_md.predict(df)[:5],
                                    np.array([0.3154147, 0.4589877, 0.376946, 0.4692429, 0.6050618])))


class TestWLS(unittest.TestCase):
    def test_wls_result(self):
        rng = np.random.RandomState(123)
        n_sample = 10000
        h_sample = n_sample//2
        
        df = pd.DataFrame(rng.randn(n_sample, 2), columns=['x', 'y'])
        df['z'] = df[['x', 'y']].dot([1, 1]) + 1
        df.loc[h_sample:, 'z'] = df.loc[h_sample:, ['x', 'y']].dot([2, 1]) + 2
        df['wt0'] = 1
        df['wt1'] = 1
        df.loc[h_sample:, 'wt1'] = 9
        
        # Regression on the constant
        ols = OrdinaryLeastSquare(df, 'z', [])
        ols.fit(show_res=False)
        self.assertAlmostEqual(ols.res_table.loc['_const', 'Coef'], df['z'].mean())
        
        wls0 = WeightedLeastSquare(df, 'z', [], wt_var='wt0')
        wls0.fit(show_res=False)
        self.assertEqual(wls0.res_stats['method'], 'WLS')
        self.assertAlmostEqual(wls0.res_table.loc['_const', 'Coef'], df['z'].mean())
        
        wls1 = WeightedLeastSquare(df, 'z', [], wt_var='wt1')
        wls1.fit(show_res=False)
        self.assertAlmostEqual(wls1.res_table.loc['_const', 'Coef'], df['z'].values.dot(df['wt1'].values) / df['wt1'].sum())
        
        # Regression on a variable
        ols0 = OrdinaryLeastSquare(df, 'z', ['x'])
        ols0.fit(show_res=False)
        
        wls0 = WeightedLeastSquare(df, 'z', ['x'], wt_var='wt0')
        wls0.fit(show_res=False)
        self.assertAlmostEqual(wls0.res_table.loc['x', 'Coef'], ols0.res_table.loc['x', 'Coef'])
        self.assertAlmostEqual(wls0.res_table.loc['x', 'Coef'], 1.5, delta=0.1)
        
        # Simulate weight with frequency
        ext_df = pd.concat([df.loc[h_sample:] for _ in range(8)], axis=0, ignore_index=True)
        ext_df = pd.concat([df, ext_df], axis=0, ignore_index=True)
        ols1 = OrdinaryLeastSquare(ext_df, 'z', ['x'])
        ols1.fit(show_res=False)
        
        wls1 = WeightedLeastSquare(df, 'z', ['x'], wt_var='wt1')
        wls1.fit(show_res=False)
        self.assertAlmostEqual(wls1.res_table.loc['x', 'Coef'], ols1.res_table.loc['x', 'Coef'])
        self.assertAlmostEqual(wls1.res_table.loc['x', 'Coef'], 1.9, delta=0.1)


class TestWithinEst(unittest.TestCase):
    def test_we_result(self):
        df = pd.read_stata('example-data/womenwk.dta')
        md = WithinEstimation(df, 'work', ['age', 'married', 'children', 'education'], 'county')
        md.fit(robust=True, show_res=False)
        
        self.assertEqual(md.res_stats['method'], 'within-estimation')
        self.assertTrue(md.res_stats['robust'])
        self.assertAlmostEqual(md.res_stats['RMSE'], 0.41953126, delta=1e-5)
        self.assertAlmostEqual(md.res_stats['adj-R-sq'], 0.20250116, delta=1e-5)
        
        self.assertTrue(np.allclose(md.res_table['Coef'].values, 
                                    np.array([0.0106925, 0.1137088, 0.1166422, 0.0190964]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(md.res_table['Std.Err'].values, 
                                    np.array([0.0013438, 0.026916, 0.0060536, 0.0035119]), 
                                    atol=1e-3))
        
        
class Test2SLS(unittest.TestCase):
    def test_2sls_result(self):
        df = pd.read_stata('example-data/grilic.dta')
        tsls_md = TwoStepLeastSquare(df, 'lw', ['s', 'expr', 'tenure', 'rns', 'smsa'], 'iq', ['med', 'kww', 'mrt', 'age'])
        tsls_md.fit(robust=True, show_res=False)
        
        self.assertEqual(tsls_md.res_stats['method'], '2SLS')
        self.assertTrue(tsls_md.res_stats['robust'])
        self.assertAlmostEqual(tsls_md.res_stats['RMSE'], 0.38514284, delta=1e-5)
        self.assertAlmostEqual(tsls_md.res_stats['adj-R-sq'], 0.19382078, delta=1e-5)
        self.assertAlmostEqual(tsls_md.res_stats['F-stat'], 58.740133, delta=1e-5)
        
        self.assertTrue(np.allclose(tsls_md.res_table['Coef'].values, 
                                    np.array([-0.0115468, 0.1373477, 0.0338041, 0.040564, -0.1176984, 0.149983, 4.837875]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(tsls_md.res_table['Std.Err'].values, 
                                    np.array([0.0056638, 0.0175802, 0.0075192, 0.0096294, 0.0361254, 0.0323774, 0.3817098]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(tsls_md.res_table['t'].values, 
                                    np.array([-2.04, 7.81, 4.50, 4.21, -3.26, 4.63, 12.67]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(tsls_md.predict(df)[:5],
                                    np.array([5.577791, 5.892475, 5.71853, 5.579354, 5.795889])))
        
        self.assertTrue(np.allclose(tsls_md.step1.res_table['Coef'].values, 
                                    np.array([0.3348112, 0.370977, -0.9012916, -0.6376273, 2.921092, 0.0130216, 0.5044333, 
                                              -2.474132, 0.1990152, 61.44371]), 
                                    atol=1e-6))
    
    
class TestKernelDensity(unittest.TestCase):
    def test_kernel_util_func(self):
        index_seq = list(KernelDensity.gen_index_seq(2, 3))
        self.assertTrue(np.all(index_seq[0][1] == np.array([0, 0])))
        self.assertTrue(np.all(index_seq[5][1] == np.array([1, 2])))
        
        index_seq = list(KernelDensity.gen_index_seq(3, 3))
        self.assertTrue(np.all(index_seq[0][1] == np.array([0, 0, 0])))
        self.assertTrue(np.all(index_seq[15][1] == np.array([1, 2, 0])))
        
    def test_kernel_density(self):
        rng = np.random.RandomState(123)
        n_sample = 1000
        df = pd.DataFrame({'x': rng.randn(n_sample) * 2,
                           'y': rng.randn(n_sample) - 1, 
                           'wt1': 1, 
                           'wt2': 2,
                           'wt3': rng.uniform(size=n_sample)})
        
        kd = KernelDensity(df, ['x', 'y'], wt_var='wt1', kernel=KernelFunction.gaussian)
        kd.fit(h=None)
        res_panel = kd.predict_with_bounds()
        self.assertAlmostEqual(res_panel['density_hat'].max(), stats.norm.pdf(0)**2/2, delta=1e-2)
        self.assertAlmostEqual(kd.predict_obs([2, 0]), stats.norm.pdf(1)**2/2, delta=1e-3)
        
        kd = KernelDensity(df, ['x', 'y'], wt_var='wt2', kernel=KernelFunction.gaussian)
        kd.fit(h=None)
        self.assertAlmostEqual(kd.predict_obs([2, 0]), stats.norm.pdf(1)**2/2 * 2, delta=1e-3)
        
        kd = KernelDensity(df, ['x', 'y'], wt_var='wt3', kernel=KernelFunction.gaussian)
        kd.fit(h=None)
        self.assertAlmostEqual(kd.predict_obs([2, 0]), stats.norm.pdf(1)**2/2 / 2, delta=1e-3)
        
    
class TestKernelReg(unittest.TestCase):
    def test_kernel_reg(self):
        rng = np.random.RandomState(123)
        n_sample = 2000
        df = pd.DataFrame(rng.randn(n_sample, 2), columns=['x', 'y'])
        f = lambda x: np.sin(2*x) + 2*np.exp(-16*x**2)
        df['z'] = f(df['x']) + df['y']
        
        kernel = KernelFunction.quadratic
        kr = LocalPolyRegression(df, 'z', ['x'], kernel=kernel, degree=0)
        kr.fit(h=[0.2], self_fit=True, show_res=False)
        self.assertEqual(kr.res_stats['method'], 'LPR')
        self.assertAlmostEqual(kr.res_stats['RMSE'], 0.963, delta=1e-2)
        self.assertAlmostEqual(kr.res_stats['adj-R-sq'], 0.473, delta=1e-2)
        
        lr = LocalPolyRegression(df, 'z', ['x'], kernel=kernel, degree=1)
        lr.fit(h=[0.2], self_fit=False, show_res=False)
        lp2r = LocalPolyRegression(df, 'z', ['x'], kernel=kernel, degree=2)
        lp2r.fit(h=[0.2], self_fit=False, show_res=False)
        lw0 = LocalPolyRegression(df, 'z', ['x'], kernel=kernel, degree=1)
        lw0.fit(local_ratio=0.15, robust_wt_updates=0, self_fit=False, show_res=False)
        lw3 = LocalPolyRegression(df, 'z', ['x'], kernel=kernel, degree=1)
        lw3.fit(local_ratio=0.15, robust_wt_updates=3, self_fit=False, show_res=False)
        
        models = [kr, lr, lp2r, lw0, lw3]
        model_names = ['kr', 'lr', 'lp2r', 'lw0', 'lw3']
        x_seq = np.linspace(-2.5, 2.5, 51)
        y_seq_true = f(x_seq)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['x'].values, df['z'].values, s=1)
        ax.plot(x_seq, y_seq_true, label='true')
        for md, md_name in zip(models, model_names):
            y_seq_pred = md.predict_obs_seq(x_seq[:, None])['z_hat'].values
            ax.plot(x_seq, y_seq_pred, label=md_name)
        ax.set_xbound(-2.8, 2.8)
        ax.set_ybound(-1.5, 2.5)
        ax.legend(loc='upper right')
        plt.show()
        
    def test_kernel_reg_confidence_interval(self):
        rng = np.random.RandomState(123)
        n_sample = 2000
        df = pd.DataFrame(rng.randn(n_sample, 2), columns=['x', 'y'])
        f = lambda x: np.sin(2*x) + 2*np.exp(-16*x**2)
        df['z'] = f(df['x']) + df['y']
        
        x_seq = np.linspace(-2.5, 2.5, 51)
        md = LocalPolyRegression(df, 'z', ['x'], kernel=KernelFunction.quadratic, degree=1)
        md.fit(h=np.array([0.4]))
        #md.fit(local_ratio=0.3)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['x'].values, df['z'].values, s=1)
        y_seq_pred = md.predict_obs_seq(x_seq[:, None], stat_inf=True)
        ax.plot(x_seq, y_seq_pred['z_hat'].values, 'b-', label='mean')
        ax.plot(x_seq, y_seq_pred['_CI.lower'].values, 'r--', label='CI.lower')
        ax.plot(x_seq, y_seq_pred['_CI.upper'].values, 'r--', label='CI.upper')
        ax.set_xbound(-2.8, 2.8)
        ax.set_ybound(-1.5, 2.5)
        ax.legend(loc='upper right')
        plt.show()
        
    def test_kernel_reg_2d(self):
        rng = np.random.RandomState(123)
        n_sample = 2000
        df = pd.DataFrame(rng.randn(n_sample, 2), columns=['x', 'y'])
        df['z'] = df['x'] + df['y']
        
        kernel = KernelFunction.trunc_gaussian
        kr = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=kernel, degree=0)
        kr.fit(h=[0.5, 0.5], local_ratio=0.3)
        obs_pred = kr.predict_obs([-5, -5], stat_inf=False)
        self.assertAlmostEqual(obs_pred, -0.343549, delta=1e-2)
        
        obs_pred = kr.predict_obs([-5, -5], stat_inf=True)
        self.assertAlmostEqual(obs_pred['z_hat'], -0.343549, delta=1e-2)
        self.assertAlmostEqual(obs_pred['_Std.Err'], 0.032397, delta=1e-2)
        
        # Hihger-degree LPR can yield better results in data boudaries
        lp2r = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=kernel, degree=2)
        lp2r.fit(h=[0.5, 0.5], local_ratio=0.3)
        obs_pred = lp2r.predict_obs([-5, -5], stat_inf=True)
        self.assertAlmostEqual(obs_pred['z_hat'], -10, delta=1e-5)
        
        lw = LocalPolyRegression(df, 'z', ['x', 'y'], kernel=kernel, degree=2)
        lw.fit(local_ratio=0.3)
        obs_pred = lw.predict_obs([-5, -5], stat_inf=True)
        self.assertAlmostEqual(obs_pred['z_hat'], -10, delta=1e-5)
        
    def test_kernel_reg_weight(self):
        df = pd.DataFrame([[1.0, 1, 1], 
                           [1.0, 2, 10], 
                           [0.9, 1, 1], 
                           [1.1, 1, 1]], columns=['x', 'z', 'wt'])
        
        kernel = KernelFunction.quadratic
        kr = LocalPolyRegression(df, 'z', ['x'], wt_var=None, kernel=kernel, degree=0)
        kr.fit(h=[0.2], self_fit=True, show_res=False)
        self.assertAlmostEqual(kr.predict_obs([1]), 1.28571429, delta=1e-6)
        
        kr = LocalPolyRegression(df, 'z', ['x'], wt_var='wt', kernel=kernel, degree=0)
        kr.fit(h=[0.2], self_fit=True, show_res=False)
        self.assertAlmostEqual(kr.predict_obs([1]), 1.8, delta=1e-6)
        
        
class TestRobinson(unittest.TestCase):
    def test_robinson_reg(self):
        df = pd.read_stata('example-data/nerlove.dta')
        
        ols = OrdinaryLeastSquare(df, 'lntc', ['lnq', 'lnpl', 'lnpk', 'lnpf'])
        ols.fit(robust=True, show_res=False)
        self.assertTrue(np.allclose(ols.res_table['Coef'].values, 
                                    np.array([0.7209135, 0.4559645, -0.2151476, 0.4258137, -3.566513]), 
                                    atol=1e-6))
        
        rb = RobinsonRegression(df, 'lntc', ['lnpf'], ['lnq', 'lnpl', 'lnpk'])
        rb.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.0825], 
               local_ratio=None, robust_wt_updates=0, show_res=False)
        self.assertEqual(rb.res_stats['method'], 'Robinson')
        self.assertTrue(np.allclose(rb.step2.res_table['Coef'].values, 
                                    np.array([0.7237013, 0.3398899, -0.3549753]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(rb.step2.res_table['Std.Err'].values, 
                                    np.array([0.030765, 0.2931278, 0.3252955]), 
                                    atol=1e-2))
        
        # Plotting
        plot_df = pd.DataFrame({'lnpf': np.linspace(rb.reg_df['lnpf'].min(), rb.reg_df['lnpf'].max(), 50)})
        y_hat, std_dev, CI_lower, CI_upper = rb.predict(plot_df, predict_type='nonpar', stat_inf=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(rb.reg_df['lnpf'].values, rb.reg_df['_eta'].values, s=5)
        ax.plot(plot_df['lnpf'].values, y_hat, 'b-', label='mean')
        ax.plot(plot_df['lnpf'].values, CI_lower, 'r--', label='CI.lower')
        ax.plot(plot_df['lnpf'].values, CI_upper, 'r--', label='CI.upper')
        ax.set_xbound(2.25, 3.85)
        ax.set_ybound(-2.25, 0)
        ax.legend(loc='upper left')
        plt.show()
        
        # Test weight consistency
        df['wt0'] = 2.0
        rb_wt0 = RobinsonRegression(df, 'lntc', ['lnpf'], ['lnq', 'lnpl', 'lnpk'], wt_var='wt0')
        rb_wt0.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.0825], 
                   local_ratio=None, robust_wt_updates=0, show_res=False)
        
        self.assertTrue(rb_wt0.wt_var == 'wt0')
        self.assertTrue(all([sub.wt_var  == 'wt0' for sub in rb_wt0.step1.values()]))
        self.assertTrue(rb_wt0.step2.wt_var == 'wt0')
        self.assertTrue(rb_wt0.step3.wt_var == 'wt0')
        self.assertTrue(isinstance(rb.step2, OrdinaryLeastSquare))
        self.assertTrue(isinstance(rb_wt0.step2, WeightedLeastSquare))
        self.assertTrue(np.allclose(rb.step2.res_table.values, rb_wt0.step2.res_table.values, 
                                    atol=1e-6))
        
        
    def test_robinson_reg_weight(self):
        rng = np.random.RandomState(123)
        n_sample = 2000
        h_sample = n_sample//2
        
        df = pd.DataFrame(rng.randn(n_sample, 3), columns=['x', 'y', 'e'])
        df['z'] = df[['x', 'e']].dot([1, 1]) + 3*np.sin(df['y']*3) + 1
        df.loc[h_sample:, 'z'] = df.loc[h_sample:, ['x', 'e']].dot([2, 1]) + 3*np.sin(df['y']*3) + 5
        df['wt0'] = 1
        df['wt1'] = 1
        df.loc[h_sample:, 'wt1'] = 9
        
        # Test for weight consistency
        rb0 = RobinsonRegression(df, 'z', ['y'], ['x'])
        rb0.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.1], 
                local_ratio=None, robust_wt_updates=0, show_res=False)
        self.assertAlmostEqual(rb0.step2.res_table.loc['x', 'Coef'], 1.5, delta=0.1)
        
        rb_wt0 = RobinsonRegression(df, 'z', ['y'], ['x'], wt_var='wt0')
        rb_wt0.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.1], 
                   local_ratio=None, robust_wt_updates=0, show_res=False)
        self.assertAlmostEqual(rb_wt0.step2.res_table.loc['x', 'Coef'], rb0.step2.res_table.loc['x', 'Coef'])
        self.assertAlmostEqual(rb_wt0.step2.res_table.loc['x', 'Coef'], 1.5, delta=0.1)
        
        # Simulate weight with frequency
        ext_df = pd.concat([df.loc[h_sample:] for _ in range(8)], axis=0, ignore_index=True)
        ext_df = pd.concat([df, ext_df], axis=0, ignore_index=True)
        rb1 = RobinsonRegression(ext_df, 'z', ['y'], ['x'])
        rb1.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.1], 
                local_ratio=None, robust_wt_updates=0, show_res=False)
        
        rb_wt1 = RobinsonRegression(df, 'z', ['y'], ['x'], wt_var='wt1')
        rb_wt1.fit(robust=True, kernel=KernelFunction.gaussian, degree=1, h=[0.1], 
                   local_ratio=None, robust_wt_updates=0, show_res=False)
        self.assertAlmostEqual(rb_wt1.step2.res_table.loc['x', 'Coef'], rb1.step2.res_table.loc['x', 'Coef'])
        self.assertAlmostEqual(rb_wt1.step2.res_table.loc['x', 'Coef'], 1.9, delta=0.1)
        
        # Plotting
        plot_df = pd.DataFrame({'y': np.linspace(df['y'].min(), df['y'].max(), 50)})
        y_true_low = 3 * np.sin(plot_df['y'].values * 3) + 1
        y_true_high = 3 * np.sin(plot_df['y'].values * 3) + 5
        y_hat_wt0 = rb_wt0.predict(plot_df, predict_type='nonpar')
        y_hat_wt1 = rb_wt1.predict(plot_df, predict_type='nonpar')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['y'].values, df['z'].values, s=1)
        ax.plot(plot_df['y'].values, y_true_high, 'b--', label='true_high')
        ax.plot(plot_df['y'].values, y_true_low, 'b--', label='true_low')
        ax.plot(plot_df['y'].values, y_hat_wt0, 'r-', label='hat_wt0')
        ax.plot(plot_df['y'].values, y_hat_wt1, 'g-', label='hat_wt1')
        ax.legend(loc='upper left')
        plt.show()
        
        
class TestMLE(unittest.TestCase):
    def test_mle_linear(self):
        df = pd.read_stata('example-data/womenwk.dta')
        lm = LinearModel(df, 'work', ['age', 'married', 'children', 'education'])
        lm.fit(show_res=False, log=False)
        
        self.assertEqual(lm.res_stats['method'], 'MLE')
        self.assertAlmostEqual(lm.res_stats['max.jac'], 0, delta=1e-3)
        self.assertAlmostEqual(lm.res_stats['neg_loglikelihood'], 1099.99, delta=1e-2)
        
        self.assertTrue(np.allclose(lm.res_table['Coef'].values, 
                                    np.array([0.0102552, 0.1111116, 0.1153084, 0.0186011, -0.2073227, -0.8689435]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(lm.res_table['Std.Err'].values, 
                                    np.array([0.0012254, 0.0219202, 0.006763, 0.0032458, 0.0540433, 0.0158114]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(lm.res_table['z'].values, 
                                    np.array([8.37, 5.07, 17.05, 5.73, -3.84, -54.96]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(lm.predict(df)[:5],
                                    np.array([0.3154147, 0.4589877, 0.376946, 0.4692429, 0.6050618])))
        
        
    def test_probit(self):
        df = pd.read_stata('example-data/womenwk.dta')
        pm = ProbitModel(df, 'work', ['age', 'married', 'children', 'education'])
        pm.fit(show_res=False, log=False)
        
        self.assertEqual(pm.res_stats['method'], 'MLE')
        self.assertAlmostEqual(pm.res_stats['max.jac'], 0, delta=1e-3)
        self.assertAlmostEqual(pm.res_stats['neg_loglikelihood'], 1027.0616, delta=1e-2)
        
        self.assertTrue(np.allclose(pm.res_table['Coef'].values, 
                                    np.array([0.0347211, 0.4308575, 0.4473249, 0.0583645, -2.467365]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(pm.res_table['Std.Err'].values, 
                                    np.array([0.0042293, 0.074208, 0.0287417, 0.0109742, 0.1925635]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(pm.res_table['z'].values, 
                                    np.array([8.21, 5.81, 15.56, 5.32, -12.81]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(pm.linear_predict(df)[:5],
                                    np.array([-0.6889973, -0.2029016, -0.4806706, -0.1681804, 0.3485867])))
        
        
    def test_logit(self):
        df = pd.read_stata('example-data/womenwk.dta')
        lm = LogitModel(df, 'work', ['age', 'married', 'children', 'education'])
        lm.fit(show_res=False, log=False)
        
        self.assertEqual(lm.res_stats['method'], 'MLE')
        self.assertAlmostEqual(lm.res_stats['max.jac'], 0, delta=1e-3)
        self.assertAlmostEqual(lm.res_stats['neg_loglikelihood'], 1027.9144, delta=1e-2)
        
        self.assertTrue(np.allclose(lm.res_table['Coef'].values, 
                                    np.array([0.0579303, 0.7417775, 0.7644882, 0.0982513, -4.159247]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(lm.res_table['Std.Err'].values, 
                                    np.array([0.007221, 0.1264705, 0.0515289, 0.0186522, 0.3320401]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(lm.res_table['z'].values, 
                                    np.array([8.02, 5.87, 14.84, 5.27, -12.53]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(lm.linear_predict(df)[:5],
                                    np.array([-1.16049, -0.3494655, -0.8129081, -0.2915352, 0.5888137])))
        
        
    def test_tobit(self):
        df = pd.read_stata('example-data/womenwk.dta')
        tm = TobitModel(df, 'lwf', ['age', 'married', 'children', 'education'], lower=0)
        tm.fit(show_res=False, log=False)
        
        self.assertEqual(tm.res_stats['method'], 'MLE')
        self.assertAlmostEqual(tm.res_stats['max.jac'], 0, delta=1e-2)
        self.assertAlmostEqual(tm.res_stats['neg_loglikelihood'], 3349.9685, delta=1e-2)
        
        self.assertTrue(np.allclose(tm.res_table['Coef'].values[:-1], 
                                    np.array([0.052157, 0.4841801, 0.4860021, 0.1149492, -2.807696]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(tm.res_table['Std.Err'].values[:-1], 
                                    np.array([0.0057457, 0.1035188, 0.0317054, 0.0150913, 0.2632565]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(tm.res_table['z'].values[:-1], 
                                    np.array([9.08, 4.68, 15.33, 7.62, -10.67]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(tm.predict(df)[:5],
                                    np.array([-0.0265698, 0.7036285, 0.2863723, 0.7557855, 1.346102]), 
                                    atol=1e-4))
        
    def test_exponential(self):
        df = pd.read_stata('example-data/recid.dta')
        df['d'] = 1 - df['cens']
        exp_md = ExponentialModel(df, y_var='durat', x_vars=['workprg', 'priors', 'tserved', 'felon', 'alcohol',
                                                             'drugs', 'black', 'married', 'educ', 'age'], d_var='d')
        exp_md.fit(show_res=False, log=False)
        
        self.assertEqual(exp_md.res_stats['method'], 'MLE')
        self.assertAlmostEqual(exp_md.res_stats['max.jac'], 0, delta=1e-2)
        self.assertTrue(np.allclose(exp_md.res_table['Coef'].values[:5], 
                                    np.array([0.0955801, 0.091337, 0.0144009, -0.3122241, 0.4676706]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(exp_md.res_table['Std.Err'].values[:5], 
                                    np.array([0.0905511, 0.0133434, 0.0016599, 0.1056552, 0.1057752]), 
                                    atol=1e-6))
        self.assertTrue(np.allclose(exp_md.res_table['z'].values[:5], 
                                    np.array([1.06, 6.85, 8.68, -2.96, 4.42]), 
                                    atol=1e-2))
        self.assertTrue(np.allclose(exp_md.linear_predict(df)[:5],
                                    np.array([-5.220216, -5.127854, -4.927456, -4.569231, -4.989691]), 
                                    atol=1e-4))
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
    
    # Duration models 
    # TODO: weibull model cannot converge because of unrobust optimization method (sensitive to initial params)
#    df = pd.read_stata('example-data/recid.dta')
#    df['d'] = 1 - df['cens']
#    wb_md = WeibullModel(df, y_var='durat', x_vars=['workprg', 'priors', 'tserved', 'felon', 'alcohol',
#                                                    'drugs', 'black', 'married', 'educ', 'age'], d_var='d')
#    wb_md.fit()
    
#    from scipy.optimize import minimize
#    
#    def neg_loglikelihood(params, t, X, d):
#        beta, lnp = params[:-1], params[-1]
#        
#        Xb = X.dot(beta)
#        p = np.exp(lnp)
#        loglikelihood = d*(Xb+lnp+(p-1)*np.log(t)) - np.exp(Xb)*(t**p)
#    #    loglikelihood = d*(Xb) - np.exp(Xb)*t
#        return -np.sum(loglikelihood)
#    
#    def jac(params, t, X, d):
#        beta, lnp = params[:-1], params[-1]
#        
#        Xb = X.dot(beta)
#        p = np.exp(lnp)
#        
#        beta_gr = np.sum((d-np.exp(Xb)*t**p)[:, None] * X, axis=0)
#        lnp_gr = np.sum(d*(1+np.log(t)*p) - np.exp(Xb)*t**p*np.log(t)*p)
#        return -np.append(beta_gr, lnp_gr)
#    
#    df = pd.read_stata('example-data\\recid.dta')
#    df['_const'] = 1
#    df['d'] = 1 - df['cens']
#    
#    params0 = np.array([0.0472985, -4.03985, -0.2626633])
#    params0 = np.zeros(3)
#    
#    neg_loglikelihood(params0, df['durat'].values, df[['workprg', '_const']], df['d'])
#    res = minimize(neg_loglikelihood, params0, args=(df['durat'].values, df[['workprg', '_const']], df['d']), method='bfgs', options={'disp': True}, jac=jac)
#    
#    print(res)
#    
#    n_jac = gen_jacobian(neg_loglikelihood)
#    print(jac(res['x'], df['durat'].values, df[['workprg', '_const']], df['d']))
#    print(n_jac(res['x'], df['durat'].values, df[['workprg', '_const']], df['d']))
#    
    
    
    # 结论：weibull 模型拟合时，scipy提供的优化算法结果对初值很敏感。



