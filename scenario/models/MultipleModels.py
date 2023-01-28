# excel link : https://blog.naver.com/montrix/221361343611

import sys, os
import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np

filename = './test_multiplemodels.npz'
ref_date = mx.Date.todaysDate()

def test():
	print('multiplemodels test...', filename)
	

	initialValue = 10000
	riskFree = 0.032
	dividend = 0.01
	volatility = 0.15

	gbmconst1 = xen.GBMConst('gbmconst1', initialValue, riskFree, dividend, volatility)
	gbmconst2 = xen.GBMConst('gbmconst2', initialValue, riskFree, dividend, volatility)
	models = [gbmconst1, gbmconst2]

	# corrMatrix = mx.Matrix([[1.0, 0.0],[0.0, 1.0]])
	corrMatrix = mx.IdentityMatrix(len(models))
	timeGrid = mx.TimeDateGrid_Equal(ref_date, 3, 365)

    # random 
	rsg = xen.Rsg(sampleNum=5000)
	results = xen.generate(models, None, corrMatrix, timeGrid, rsg, filename, False)
    # print(results.multiPath(scenCount=10))
	
if __name__ == "__main__":
	
	test()
	#mx.npzee_view(filename)