cd "D:\Documents\PythonFiles-py3\统计和机器学习\statspy\statspy\example-data"
/*
use womenwk.dta, clear

reg work age married children education
reg work age married children education, r
*/

use grilic.dta, clear
reg iq med kww mrt age s expr tenure rns smsa
ivregress 2sls lw s expr tenure rns smsa (iq = med kww mrt age), small
ivregress 2sls lw s expr tenure rns smsa (iq = med kww mrt age), r small

