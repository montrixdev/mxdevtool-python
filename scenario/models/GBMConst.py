# excel link : https://blog.naver.com/montrix/221361343611

import sys, os
import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np

filename = './test_gbmconst.npz'
ref_date = mx.Date.todaysDate()

def model():

	x0 = 100
	rf = 0.032
	div = 0.01
	vol = 0.15

	gbmconst = xen.GBMConst('gbmconst', x0=x0, rf=rf, div=div, vol=vol)

	return gbmconst


def test():
	print('gbmconst test...', filename)
	
	m = model()
	timeGrid = mx.TimeEqualGrid(ref_date, 3, 365)

	# random 
	rsg = xen.Rsg(sampleNum=5000)
	results = xen.generate1d(m, None, timeGrid, rsg, filename, False)
    # print(results.multiPath(scenCount=10))

if __name__ == "__main__":
	
	test()
	#mx.npzee_view(filename)