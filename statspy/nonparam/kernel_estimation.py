# -*- coding: utf-8 -*-
import itertools
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats

from statspy.base import BaseModel, calc_R_sq
from statspy.nonparam import KernelFunction
from statspy.ols import OrdinaryLeastSquare, WeightedLeastSquare
from statspy.tools import show_model_res

class KernelEstimation(BaseModel):
    def __init__(self, df, y_var, x_vars, wt_var=None, has_const=False, kernel=None):
        super(KernelEstimation, self).__init__(df, y_var, x_vars, wt_var=wt_var, has_const=has_const)
        self.wt_var = wt_var
        self.kernel = kernel if kernel is not None else KernelFunction.trunc_gaussian
        
        
    @staticmethod
    def gen_index_seq(dim, max_idx):
        '''Cartesian product of np.arange(`max_idx') for `dim' dimensions. 
        For example, if idx_dim=2, max_idx=3, this function would yield: 
            (0, array([0, 0])),
            (1, array([0, 1])),
            (2, array([0, 2])),
            (3, array([1, 0])),
            (4, array([1, 1])),
            (5, array([1, 2])),
            (6, array([2, 0])),
            (7, array([2, 1])),
            (8, array([2, 2]))
        '''
        for flatten_idx, idx_vec in enumerate(itertools.product(range(max_idx), repeat=dim)):
            yield flatten_idx, np.array(idx_vec)
                    
                    
    @staticmethod
    def gen_space_seq(bounds, n_points=100):
        '''Build x_vars values as panel, in specific bounds. 
        '''
        K = len(bounds)
        
        space = np.array([np.linspace(lower_i, upper_i, n_points) for lower_i, upper_i in bounds])
        space_seq = [space[np.arange(K), idx_vec] for flatten_idx, idx_vec in KernelEstimation.gen_index_seq(K, n_points)]
        return np.array(space_seq)
    

        
class KernelDensity(KernelEstimation):
    def __init__(self, df, x_vars, wt_var=None, kernel=None):
        super(KernelDensity, self).__init__(df, None, x_vars, wt_var=wt_var, has_const=False, kernel=kernel)
        self._clean_data()
        
    def fit(self, h=None):
        '''Decide the window size (h) 
        '''
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        if h is None:
            # Silverman's rule of thumb
            # Ref: https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation
            h = (4/(n*(K+2))) ** (1/(K+4)) * np.std(X, axis=0)
        else:
            if hasattr(h, '__iter__'):
                h = np.array(list(h))
                assert len(h.shape)==1 and h.shape[0] == K
                h = pd.Series(h, index=self.x_vars)
            else:
                raise Exception('Invalid bandwidth input!', h)
                
        self.h = pd.Series(h, index=self.x_vars)
        self._fitted = True
    
    def predict_with_bounds(self, bounds=None, n_points=100):        
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        if bounds is not None:
            assert len(bounds) == K
        else:
            bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))
        res_panel = pd.DataFrame(KernelEstimation.gen_space_seq(bounds, n_points=n_points), 
                                 columns=self.x_vars)
        
        res_panel['density_hat'] = self.predict(res_panel)
        return res_panel
        
    def predict(self, df):
        assert self._fitted
        
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        h = self.h.values
        
        pred_X = df[self.x_vars].values
        n_pred, K = pred_X.shape
        
        # X: (n, K) -> (K, 1, n)
        # pred_X: (n_pred, K) -> (K, n_pred, 1)
        # h: (K, ) -> (K, 1, 1)
        # wts: 3D-tensor with dimension of (K, n_pred, n)
        wts = self.kernel((X.T[:, None, :]-pred_X.T[:, :, None])/h[:, None, None])
        # prod_wts: 2D-matrix with dimension of (n_pred, n)
        prod_wts = np.product(wts, axis=0)
        
        if self.wt_var is not None:
            sample_wts = self.reg_df[self.wt_var].values
            prod_wts = prod_wts * sample_wts
        
        pred_density = np.sum(prod_wts, axis=1) / (n*np.product(h))
        return pred_density
        
    def predict_obs(self, x):
        assert len(x) == len(self.x_vars)
        return self.predict(pd.DataFrame([x], columns=self.x_vars))[0]


class LocalPolyRegression(KernelEstimation):
    '''If call the fit-function with parameter-h not being None and local_ratio being None, 
    it would use a global bandwidth, so it would be Local Polynomial Regression (LPR); 
    
    If Call the fit-function with parameter-h being None and local_ratio not being None, 
    it would use locally adaptive bandwidth with fixed ratio of full sample,     
    so it would be Locally weighted scatterplot smoothing (Lowess);
    
    If Call the fit-function with parameter-h and local_ratio both not being None, 
    it would use a global bandwidth, and complement local sample to satisfy local_ratio when
    local sample size is not sufficient. 
    
    ref: [1]Fan J. Design-adaptive nonparametric regression[J]. Journal of the American statistical Association, 1992, 87(420): 998-1004.
         [2]Fan J, Gijbels I. Local polynomial modelling and its applications: monographs on statistics and applied probability 66[M]. CRC Press, 1996.
         [3]Cleveland W S. Robust locally weighted regression and smoothing scatterplots[J]. Journal of the American statistical association, 1979, 74(368): 829-836.
         [4]陈强. 高级计量经济学及 Stata 应用[M]. 高等教育出版社, 2010.
    '''
    def __init__(self, df, y_var, x_vars, wt_var=None, kernel=None, degree=1):
        super(LocalPolyRegression, self).__init__(df, y_var, x_vars, wt_var=wt_var, has_const=False, kernel=kernel)
        assert isinstance(degree, int) and degree >= 0
        self.degree = degree
        self._clean_data()
        
    def fit(self, h=None, local_ratio=None, robust_wt_updates=0, self_fit=False, show_res=True):
        '''Decide the window size (h) 
        '''
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        if h is None:
            # Fan & Gijbels (1996) only provide ROT bandwidth for univariate model
            # Here we use Lowess when h is None, use LPR when h is ndarray
            assert isinstance(local_ratio, (float, int)) and 0 <= local_ratio <= 1
        else:
            if hasattr(h, '__iter__'):
                h = np.array(list(h))
                assert len(h.shape)==1 and h.shape[0] == K
                h = pd.Series(h, index=self.x_vars)
            else:
                raise Exception('Invalid bandwidth input!', h)
            
        self.h = h
        self.local_ratio = local_ratio
        
        self._fitted = True
        
        # Robustness weights delta initialization for Lowess
        # According to Fan & Gijbels (1996), it would be more robust to outliers
        self.reg_df['_delta'] = 1.
        for _ in range(robust_wt_updates):
            resid = y - self.predict(self.reg_df, stat_inf=False)
            resid_M = np.median(np.abs(resid))
            self.reg_df['_delta'] = KernelFunction.quartic(resid / (6*resid_M))
        
        if self_fit:
            self.self_fit(show_res=show_res)
            
    
    def self_fit(self, show_res=True):
        assert self._fitted
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        # residual square, for conditional variance prediction
        y_hat = self.predict(self.reg_df, stat_inf=False)
        e_hat = y - y_hat        
        self.reg_df[self.y_var+'_hat'] = y_hat
        self.reg_df['_resid'] = e_hat
        
        # remove null predictions
        indic = pd.notnull(y_hat)
        y, y_hat, e_hat = y[indic], y_hat[indic], e_hat[indic]
        
        # calculate statistics
        s_sq = np.sum(e_hat**2) / (n-K)
        s = s_sq ** 0.5        
        
        # Calculate R-squared
        R_sq, adj_R_sq, SSE, SSR, SST = calc_R_sq(y, y_hat, n, K, return_SS=True)
        
        self.res_stats = pd.Series(OrderedDict([('method', 'LPR'),
                                                ('obs', n),
                                                ('RMSE', s),
                                                ('SSE', SSE),
                                                ('SSR', SSR),
                                                ('SST', SST),
                                                ('R-sq', R_sq),
                                                ('adj-R-sq', adj_R_sq)]))
        if show_res:
            show_model_res(self)
    
    
    def predict(self, df, stat_inf=False):
        '''
        stat_inf:
            False: only return y_hat
            True: return (y_hat, std_dev, conf_int_lower, conf_int_upper)
        '''
        assert self._fitted
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        delta = self.reg_df['_delta'].values
        
        # Avoid using df directly
        pred_df = df[self.x_vars].copy()
        # If pred_df is a view of df, using pred_df['_index'] would cause SettingWithCopy warning
        pred_df['_index'] = np.arange(pred_df.shape[0])
        
        # Avoid predicting duplicated observations, from here everything is reduced size...
        red_df = pred_df[self.x_vars].drop_duplicates()
        red_X = red_df[self.x_vars].values
        n_red, K = red_X.shape
        
        y_hat = []
        std_dev = []
        for idx in range(n_red):
            # local_x: (K, )
            local_x = red_X[idx]
            # off_X: (n, K)
            off_X = X - local_x
            
            if self.h is not None:
                h = self.h.values
                if self.local_ratio is None:
                    crit_std_off = 1
                else:
                    # Here, we complement local sample to satisfy local_ratio if it is necessary. 
                    # `max_comp_std_off' is the maximun component of the offset standardized by `h'. 
                    # The idea here is, every point which is located in the bandwidth in all dimensions, 
                    # i.e., in the k-dimensional cube, would have a positive weight in LPR , 
                    # so it is the  that decides whether the point is in the cube. 
                    # So, if `max_comp_std_off' <= 1, the corresponding point is in the cube; 
                    # if `max_comp_std_off' > 1, it is not. 
                    max_comp_std_off = np.abs(off_X / h).max(axis=1)
                    crit_std_off = np.percentile(max_comp_std_off, self.local_ratio*100)
                    # if crit<=1, there are efficient points in bandwidth (the k-D cube)
                    crit_std_off = max(1, crit_std_off)
                    
                # NOTE: We only make sure there are right number of points in bandwidth, but actually the 
                # effective points may exceed the number if we use an gaussian kernel which assigns points 
                # out of bandwidth positive weights
                wt = np.product(self.kernel(off_X / (h*crit_std_off)), axis=1)                    
                #print(np.sum(wt>0)/len(wt), self.local_ratio)
            else:
                dist = ((off_X / X.std(axis=0))**2).sum(axis=1) ** 0.5
                crit_dist = np.percentile(dist, self.local_ratio*100)
                wt = self.kernel(dist / crit_dist)
            
            # Adjust wt with nominated weight
            if self.wt_var is not None:
                sample_wts = self.reg_df[self.wt_var].values
                wt = wt * sample_wts
            # Adjust wt with robustness weights delta
            wt = wt * delta
            
            # Select these observations with positive weight (i.e., weight>0)
            wt_indic = wt > 0
            sel_X = off_X[wt_indic]
            sel_y = y[wt_indic]
            sel_wt = wt[wt_indic]
            
            if self.degree == 0:
                y_wt_mean = sel_y.dot(sel_wt) / sel_wt.sum()
                y_hat.append(y_wt_mean)
                if stat_inf:
                    this_std_dev = ((sel_y - y_wt_mean)**2).dot(sel_wt**2) ** 0.5 / sel_wt.sum()
                    std_dev.append(this_std_dev)
            else:
                # Extend sel_X with a constant variable, maybe higher degree polynimials
                ext_X = np.concatenate([np.ones((sel_X.shape[0], 1)), *[sel_X ** k for k in range(1, self.degree+1)]], axis=1)
                # WLS estimation
                wXt = ext_X.T * sel_wt
                wXtX = wXt.dot(ext_X)
                try:
                    wXtX_inv = np.linalg.inv(wXtX)
                except:
                    y_hat.append(np.nan)
                    if stat_inf:
                        std_dev.append(np.nan)
                else:
                    beta = wXtX_inv.dot(wXt.dot(sel_y))
                    # The cofficient on the constant variable is y_hat
                    y_hat.append(beta[0])
                    if stat_inf:
                        sel_e = sel_y - ext_X.dot(beta)
                        cov = wXtX_inv.dot((wXt * sel_e**2 * sel_wt).dot(ext_X)).dot(wXtX_inv)
                        std_dev.append(cov[0, 0] ** 0.5)
                        
        
        # Map reduced results to original size
        red_df[self.y_var+'_hat'] = y_hat
        if stat_inf:
            red_df['_std_dev'] = std_dev
            
        pred_df = pd.merge(pred_df, red_df, on=self.x_vars)
        pred_df = pred_df.sort_values(by='_index')
        
        y_hat = pred_df[self.y_var+'_hat'].values
        if stat_inf:
            std_dev = pred_df['_std_dev'].values
        
        # From here everything is original size...
        if not stat_inf:
            return y_hat
        else:
            z95 = stats.norm.ppf(0.9750)
            conf_int_lower = y_hat - z95 * std_dev
            conf_int_upper = y_hat + z95 * std_dev
            return y_hat, std_dev, conf_int_lower, conf_int_upper
        
    
    def predict_obs_seq(self, pred_X, stat_inf=False):
        assert isinstance(pred_X, np.ndarray) and pred_X.shape[1] == len(self.x_vars)
        res_panel = pd.DataFrame(pred_X, columns=self.x_vars)
        
        pred_res = self.predict(res_panel, stat_inf=stat_inf)
        if not stat_inf:
            res_panel[self.y_var+'_hat'] = pred_res
        else:
            res_panel[self.y_var+'_hat'] = pred_res[0]
            res_panel['_Std.Err'] = pred_res[1]
            res_panel['_CI.lower'] = pred_res[2]
            res_panel['_CI.upper'] = pred_res[3]
        return res_panel
    
    def predict_obs(self, x, stat_inf=False):
        assert len(x) == len(self.x_vars)
        res_panel = self.predict_obs_seq(np.array([x]), stat_inf=stat_inf)
        
        if stat_inf:
            return res_panel.loc[0, [self.y_var+'_hat', '_Std.Err', '_CI.lower', '_CI.upper']]
        else:
            return res_panel.loc[0, self.y_var+'_hat']
    
    def predict_with_bounds(self, bounds=None, n_points=100, stat_inf=False):
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        if bounds is not None:
            assert len(bounds) == K
        else:
            bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))
        res_panel = self.predict_obs_seq(KernelEstimation.gen_space_seq(bounds, n_points=n_points),
                                         stat_inf=stat_inf)
        return res_panel
    
    
class RobinsonRegression(BaseModel):
    '''Robinson Difference Estimator
    Ref: [1]Robinson P M. Root-N-consistent semiparametric regression[J]. Econometrica: Journal of the Econometric Society, 1988: 931-954. 
         [2]陈强. 高级计量经济学及 Stata 应用[M]. 高等教育出版社, 2010.
    '''
    def __init__(self, df, y_var, x_vars, par_vars, wt_var=None, has_const=False):
        '''const variable is equal to ZERO theoretically, so it may not be used in the most conditions. 
        '''
        super(RobinsonRegression, self).__init__(df, y_var, x_vars, wt_var=wt_var, has_const=has_const)
        self.par_vars = par_vars
        self._clean_data()
        
    def _make_reg_vars(self):
        # Adding const for par_vars would be done in the OLS step
        reg_vars = [self.y_var] + self.x_vars + self.par_vars
        if self.wt_var is not None:
            reg_vars.append(self.wt_var)
        return reg_vars
        
    def fit(self, robust=False, show_res=True, nonpar_method='LPR', kernel=None, degree=1, 
            h=None, local_ratio=None, robust_wt_updates=0, show_proc=False):
        y = self.reg_df[self.y_var].values
        X = self.reg_df[self.x_vars].values
        n, K = X.shape
        
        # TODO: more non-parameter estimation methods
        # First-step: non-parameter estimation
        NonparRegression = LocalPolyRegression
        
        self.step1 = {}
        # Conditional expectation dataframe
        cond_df = pd.DataFrame(index=self.reg_df.index)
        # Residual dataframe
        resid_df = pd.DataFrame(index=self.reg_df.index)
        for this_var in ([self.y_var] + self.par_vars):
            if show_proc: 
                print('step1: fitting variable %s...' % this_var)
            self.step1[this_var] = NonparRegression(self.reg_df, this_var, self.x_vars, wt_var=self.wt_var, kernel=kernel, degree=degree)
            self.step1[this_var].fit(h=h, local_ratio=local_ratio, robust_wt_updates=robust_wt_updates)
            cond_df[this_var] = self.step1[this_var].predict(self.reg_df)
            # NOTE: pd.Series would operate by matching index, which may cause wrong result, so we use the values-attribute
            resid_df[this_var] = self.reg_df[this_var].values - cond_df[this_var].values
            
        # Second-step: parameter estimation
        if show_proc: 
            print('step2: fitting...')
        if self.wt_var is None:
            self.step2 = OrdinaryLeastSquare(resid_df, self.y_var, self.par_vars, has_const=self.has_const)
            self.step2.fit(robust=robust, show_res=show_res)
        else:
            # Adjust wt with nominated weight
            resid_df[self.wt_var] = self.reg_df[self.wt_var]
            self.step2 = WeightedLeastSquare(resid_df, self.y_var, self.par_vars, wt_var=self.wt_var, has_const=self.has_const)
            self.step2.fit(show_res=show_res)
            
        # Third-step: smooth residuals of parametric model (second-step)
        if show_proc: 
            print('step3: fitting...')
        self.reg_df['_eta'] = self.reg_df[self.y_var] - self.step2.predict(self.reg_df)
        self.step3 = NonparRegression(self.reg_df, '_eta', self.x_vars, wt_var=self.wt_var, kernel=kernel, degree=degree)
        self.step3.fit(h=h, local_ratio=local_ratio, robust_wt_updates=robust_wt_updates)
        
        # Repeat smoothing
#        for _ in range(3):
#            cond_df['_eta'] = self.step3.predict(self.reg_df)
#            resid_df['_eta'] = self.reg_df[self.y_var] - cond_df['_eta']
#            self.step2 = OrdinaryLeastSquare(resid_df, '_eta', self.par_vars.copy(), has_const=self.has_const)
#            self.step2.fit(robust=robust, show_res=show_res)
#            self.reg_df['_eta'] = self.reg_df[self.y_var] - self.step2.predict(self.reg_df)
#            self.step3 = NonparRegression(self.reg_df, '_eta', self.x_vars, kernel=kernel, degree=degree)
#            self.step3.fit(h=h, local_ratio=local_ratio, robust_wt_updates=robust_wt_updates)
        
        par_hat = self.step2.predict(self.reg_df)
        nonpar_hat = self.step3.predict(self.reg_df)
        y_hat = nonpar_hat + par_hat
        e_hat = y - y_hat
        self.reg_df['_par_hat'] = par_hat
        self.reg_df['_nonpar_hat'] = nonpar_hat
        self.reg_df[self.y_var+'_hat'] = y_hat
        self.reg_df['_resid'] = e_hat
        
        # Remove null predictions
        indic = pd.notnull(y_hat)
        y, y_hat, e_hat = y[indic], y_hat[indic], e_hat[indic]
        if self.wt_var is not None:
            wt = self.reg_df[self.wt_var].values[indic]
        else:
            wt = np.ones_like(y)
        s_sq = (e_hat ** 2).dot(wt ** 2) / (wt ** 2).sum() * n / (n-K)
        s = s_sq ** 0.5
        
        # Calculate R-squared
        R_sq, adj_R_sq, SSE, SSR, SST = calc_R_sq(y, y_hat, n, K, wt=wt, return_SS=True)
        
        self.res_stats = pd.Series(OrderedDict([('method', 'Robinson'),
                                                ('robust', robust),
                                                ('obs', n),
                                                ('RMSE', s),
                                                ('SSE', SSE),
                                                ('SSR', SSR),
                                                ('SST', SST),
                                                ('R-sq', R_sq),
                                                ('adj-R-sq', adj_R_sq)]))
        self._fitted = True
        if show_res:
            show_model_res(self)
        
        
    def predict(self, df, predict_type='both', stat_inf=False):
        assert self._fitted
        
        if predict_type == 'par':
            return self.step2.predict(df)
        elif predict_type == 'nonpar':
            return self.step3.predict(df, stat_inf=stat_inf)
        elif predict_type == 'both':
            nonpar_res = self.step3.predict(df, stat_inf=stat_inf)
            if not stat_inf:
                return self.step2.predict(df) + nonpar_res
            else:
                # TODO: how to combine variance from step2 and step3?
                return self.step2.predict(df) + nonpar_res[0], nonpar_res[1], nonpar_res[2], nonpar_res[3]
        else:
            raise Exception('Invalid prediction type!', predict_type)
            
    def predict_obs(self, x, predict_type='both', stat_inf=False):
        raise NotImplementedError
        
        
        