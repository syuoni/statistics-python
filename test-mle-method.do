cd "D:\Documents\PythonFiles-py3\ͳ�ƺͻ���ѧϰ\statspy\statspy"
use "example-data\\recid.dta", clear

stset durat, failure(cens=0)
streg workprg, d(e) nohr nolog noshow
streg workprg, d(w) nohr nolog noshow
