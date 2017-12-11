import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statspy.tools import gen_jacobian

def neg_loglikelihood(params, t, X, d):
    beta, lnp = params[:-1], params[-1]
    
    Xb = X.dot(beta)
    p = np.exp(lnp)
    loglikelihood = d*(Xb+lnp+(p-1)*np.log(t)) - np.exp(Xb)*(t**p)
#    loglikelihood = d*(Xb) - np.exp(Xb)*t
    return -np.sum(loglikelihood)

def jac(params, t, X, d):
    beta, lnp = params[:-1], params[-1]
    
    Xb = X.dot(beta)
    p = np.exp(lnp)
    
    beta_gr = np.sum((d-np.exp(Xb)*t**p)[:, None] * X, axis=0)
    lnp_gr = np.sum(d*(1+np.log(t)*p) - np.exp(Xb)*t**p*np.log(t)*p)
    return -np.append(beta_gr, lnp_gr)

df = pd.read_stata('example-data\\recid.dta')
df['_const'] = 1
df['d'] = 1 - df['cens']

params0 = np.array([0.0472985, -4.03985, -0.2626633])
params0 = np.zeros(3)

neg_loglikelihood(params0, df['durat'].values, df[['workprg', '_const']], df['d'])
res = minimize(neg_loglikelihood, params0, args=(df['durat'].values, df[['workprg', '_const']], df['d']), method='bfgs', options={'disp': True}, jac=jac)

print(res)

n_jac = gen_jacobian(neg_loglikelihood)
print(jac(res['x'], df['durat'].values, df[['workprg', '_const']], df['d']))
print(n_jac(res['x'], df['durat'].values, df[['workprg', '_const']], df['d']))



# 结论：weibull 模型拟合时，scipy提供的优化算法结果对初值很敏感。
