statspy: Statistic methods with Python
=======

By syuoni (https://github.com/syuoni)


Introduction
------
This is an implementation of statistic methods with Python (`numpy`, `pandas`, `scipy`, etc.). 

It currently supports:

* OLS, WLS, 2SLS
* MLE (e.g., binary and duration models)
* Non- and semi-parametric models 

  - Kernel density estimation 
  - Local polynomial regression 
  - Robinson regression 

Demonstration
------

#### Code 
```python
# -*- coding: utf-8 -*-  
import pandas as pd  
from statspy.ols import OrdinaryLeastSquare  

df = pd.read_stata('example-data/womenwk.dta')  
ols_md = OrdinaryLeastSquare(df, 'work', ['age', 'married', 'children', 'education'])  
ols_md.fit()  
```

#### Output
```
======================================================================
method              OLS
robust            False
obs                2000
RMSE            0.41992
SSE             89.3922
SSR             351.783
SST             441.175
R-sq           0.202623
adj-R-sq       0.201024
F-stat          126.738
Prob(F)     1.11022e-16
dtype: object
======================================================================
               Coef   Std.Err          t             p  CI.lower  CI.upper
age        0.010255  0.001227   8.358393  0.000000e+00  0.007849  0.012661
married    0.111112  0.021948   5.062567  4.516572e-07  0.068069  0.154154
children   0.115308  0.006772  17.028476  0.000000e+00  0.102028  0.128588
education  0.018601  0.003250   5.723597  1.200935e-08  0.012228  0.024975
_const    -0.207323  0.054111  -3.831436  1.313376e-04 -0.313443 -0.101203
======================================================================
```
