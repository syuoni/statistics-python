from setuptools import setup

setup(name             = 'statspy',
      version          = '0.01',
      description      = 'Statistic methods with Python',
      long_description = open('README.rst').read(),
      url              = 'https://github.com/syuoni/statspy',
      author           = 'syuoni',
      author_email     = 'spiritas@163.com',
      license          = 'MIT',
      packages         = ['statspy', 'statspy.ols', 'statspy.mle', 
                          'statspy.nonparam', 'statspy.tools'],
      install_requires = ['numpy',
                          'pandas',
                          'scipy'],
      zip_safe         = False)