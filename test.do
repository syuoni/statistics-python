cd "D:\Documents\PythonFiles-py3\stats-ml\statspy\statspy\example-data"

// Descriptive
use "nerlove.dta", clear
summarize

// OLS, within-estimator
use "womenwk.dta", clear
reg work age married children education
reg work age married children education, r

reg work age married children education i.county, r

// 2SLS
use "grilic.dta", clear
ivreg2 lw (iq = med kww mrt age) s expr tenure rns smsa, small r first


// Robinson
use "nerlove.dta", clear
reg lntc lnq lnpl lnpk lnpf, r
semipar lntc lnq lnpl lnpk, nonpar(lnpf) robust nograph

// MLE-linear
﻿// 使用正态分布定义似然函数
capture program drop norm_lf
program norm_lf
args lnf mu logsigma
quietly replace `lnf' = ln(normalden($ML_y1, `mu', exp(`logsigma')))
end

use "womenwk.dta", clear
ml model lf norm_lf (mu: work=age married children education) (logsigma: )
ml maximize, nolog

// Probit & Logit
probit work age married children education
logit work age married children education

// Tobit
tobit lwf age married children education, ll(0)


// Duration
use "recid.dta", clear

stset durat, failure(cens=0)
streg workprg priors tserved felon alcohol drugs black married educ age, d(e) nohr nolog noshow
streg workprg priors tserved felon alcohol drugs black married educ age, d(w) nohr nolog noshow
