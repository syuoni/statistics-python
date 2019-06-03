statspy: Statistic methods with Python
=======

By syuoni (https://github.com/syuoni)


Introduction
------
This is an implementation of statistic methods with Python (numpy, pandas, scipy, etc.). 

It currently supports:

* OLS, WLS, 2SLS
* MLE (e.g., binary and duration models)
* Non- and semi-parametric models 

  - Kernel density estimation 
  - Local polynomial regression 
  - Robinson regression 

Demonstration
------
```
# -*- coding: utf-8 -*-  
import pandas as pd  
from statspy.ols import OrdinaryLeastSquare  

df = pd.read_stata('example-data/womenwk.dta')  
ols_md = OrdinaryLeastSquare(df, 'work', ['age', 'married', 'children', 'education'])  
ols_md.fit()  
```