import numpy as np

double_eps = 2.2e-16

def get_1st_multip(func, params, ei, h, *args):
    d_params = 0.5 * h * ei
    fv0 = func(params-d_params, *args)
    fv1 = func(params+d_params, *args)
    abs_d_func = np.abs(fv1 - fv0)
    if abs_d_func == 0:
        return np.inf
    abs_func = np.max(np.abs(np.array([fv0, fv1, (fv0+fv1)/2])))
    return abs_func * double_eps**0.5 / abs_d_func

def get_1st_diff(func, params, ei, h='optim', *args):
    # optimize h, to make the func(x+h)-func(x) is half digits of func(x)
    if h == 'optim':
        h = double_eps ** 0.5
        multip = get_1st_multip(func, params, ei, h, *args)
        if multip <= 0.1:
            h *= 0.1
            while h > double_eps:
                multip = get_1st_multip(func, params, ei, h, *args)
                if multip <= 0.1: 
                    h *= 0.1
                else:
                    break
        elif multip > 10:
            h *= 10
            while h < 0.1:
                multip = get_1st_multip(func, params, ei, h, *args)
                if multip > 10: 
                    h *= 10
                else:
                    break
    elif h == 'nash':
        h = (np.abs(func(params, *args)) + double_eps**0.5) * double_eps**0.5
    d_params = 0.5 * h * ei
    fv0 = func(params-d_params, *args)
    fv1 = func(params+d_params, *args)
    d_func = fv1 - fv0        
    return d_func / h

def get_2ed_multip(func, params, ei, ej, h, *args):
    d_params_i = 0.5 * h * ei
    d_params_j = 0.5 * h * ej
    fv00 = func(params-d_params_i-d_params_j, *args)
    fv01 = func(params-d_params_i+d_params_j, *args)
    fv10 = func(params+d_params_i-d_params_j, *args)
    fv11 = func(params+d_params_i+d_params_j, *args)
    abs_d2_func = np.abs(fv00 - fv01 - fv10 + fv11)
    if abs_d2_func == 0:
        return np.inf
    abs_func = np.max(np.abs(np.array([fv00, fv01, fv10, fv11, (fv00+fv01+fv10+fv11)/4])))
    return abs_func * double_eps**0.5 / abs_d2_func

    
def get_2ed_diff(func, params, ei, ej, h='optim', *args):
    # optimize h, to make the func(x+h)-func(x) is half digits of func(x)
    if h == 'optim':
        h = double_eps ** 0.5
        multip = get_2ed_multip(func, params, ei, ej, h, *args)
        if multip <= 0.1:
            h *= 0.1
            while h > double_eps:
                multip = get_2ed_multip(func, params, ei, ej, h, *args)
                if multip <= 0.1: 
                    h *= 0.1
                else:
                    break
        elif multip > 10:
            h *= 10
            while h < 0.1:
                multip = get_2ed_multip(func, params, ei, ej, h, *args)
                if multip > 10: 
                    h *= 10
                else:
                    break
    elif h == 'nash':
        h = (np.abs(func(params, *args)) + double_eps**0.5) * double_eps**0.5
    d_params_i = 0.5 * h * ei
    d_params_j = 0.5 * h * ej
    fv00 = func(params-d_params_i-d_params_j, *args)
    fv01 = func(params-d_params_i+d_params_j, *args)
    fv10 = func(params+d_params_i-d_params_j, *args)
    fv11 = func(params+d_params_i+d_params_j, *args)
    d2_func = fv00 - fv01 - fv10 + fv11        
    return d2_func / h**2

def gen_jacobian(func, h='optim'):
    def jacobian(params, *args):
        n_params = len(params)
        jacobian_vec = np.zeros(n_params)
        for i, ei in enumerate(np.identity(n_params)): 
            jacobian_vec[i] = get_1st_diff(func, params, ei, h, *args)
        return jacobian_vec 
    return jacobian

def gen_hessian_with_func(func, h='optim'):
    def hessian(params, *args):        
        n_params = len(params)
        hessian_matrix = np.zeros((n_params, n_params))
        for i, ei in enumerate(np.identity(n_params)):
            for j, ej in enumerate(np.identity(n_params)[i:], i):
                diff2 = get_2ed_diff(func, params, ei, ej, h, *args)
                hessian_matrix[i, j] = diff2
                if i < j:
                    hessian_matrix[j, i] = diff2
        return hessian_matrix        
    return hessian

def gen_hessian_with_jac(func, h='optim'):
    pass


if __name__ == '__main__':
    def func(x, y=2):
        return x[0] * y + x[1] ** 3 + np.exp(x[1]) + x[0]*x[1]**2
    
    def jac(x, y=2):
        return np.array([y+x[1]**2, 3*x[1]**2+np.exp(x[1])+2*x[0]*x[1]])
        
    def hess(x, y=2):
        return np.array([[0,      2*x[1]],
                         [2*x[1], 6*x[1]+np.exp(x[1])+2*x[0]]])
    
    
    print(jac(np.arange(1, 3), 3))
    n_jac = gen_jacobian(func)
    print(n_jac(np.arange(1, 3), 3))
    n_jac = gen_jacobian(func, h='nash')
    print(n_jac(np.arange(1, 3), 3))
    
    print(hess(np.arange(1, 3), 3))
    n_hess = gen_hessian_with_func(func)
    print(n_hess(np.arange(1, 3), 3))
    n_hess = gen_hessian_with_func(func, h='nash')
    print(n_hess(np.arange(1, 3), 3))

    
    
    
    