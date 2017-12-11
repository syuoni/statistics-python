import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

class KernelFunction(object):
    @staticmethod
    def uniform(z):
        return np.where(np.abs(z)<1, 0.5, 0)
    
    @staticmethod
    def triangular(z):
        return np.where(np.abs(z)<1, 1-np.abs(z), 0)
    
    @staticmethod
    def quadratic(z):
        return np.where(np.abs(z)<1, 0.75 * (1-z**2), 0)
    
    @staticmethod
    def quartic(z):
        return np.where(np.abs(z)<1, 15/16 * (1-z**2)**2, 0)
    
    @staticmethod
    def tricubic(z):
        return np.where(np.abs(z)<1, 70/81 * (1-np.abs(z)**3)**3, 0)
    
    @staticmethod
    def gaussian(z):
        return np.exp(-z**2/2) / (2*np.pi) ** 0.5
    
    @staticmethod
    def trunc_gaussian(z):
        return np.where(np.abs(z)<=3, np.exp(-z**2/2) / (2*np.pi) ** 0.5, 0)
    
    @staticmethod
    def clust_gaussian(z):
        return np.where(np.abs(z)<=1, 3 * np.exp(-(3*z)**2/2) / (2*np.pi) ** 0.5, 0)
        
if __name__ == '__main__':
    kf = KernelFunction.quadratic
#    kf = KernelFunction.clust_gaussian
    
    print(integrate.quad(kf, -3, 3))
    
    x_seq = np.linspace(-2, 2, 201)
    plt.plot(x_seq, kf(x_seq))
    plt.show()