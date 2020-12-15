# excel link : https://blog.naver.com/montrix/221376083438

import sys, os
import mxdevtool as mx
import mxdevtool.termstructures as ts
import mxdevtool.xenarix as xen
import numpy as np


filename = './test_bk1f.npz'

ref_date = mx.Date.todaysDate()

def model():

    tenor_rates = [('3M', 0.0151),
                ('6M', 0.0152),
                ('9M', 0.0153),
                ('1Y', 0.0154),
                ('2Y', 0.0155),
                ('3Y', 0.0156),
                ('4Y', 0.0157),
                ('5Y', 0.0158),
                ('7Y', 0.0159),
                ('10Y', 0.016),
                ('15Y', 0.0161),
                ('20Y', 0.0162)]

    tenors = []
    zerorates = []
    interpolator1DType = mx.Interpolator1D.Linear
    extrapolator1DType = mx.Extrapolator1D.FlatForward

    for tr in tenor_rates:
        tenors.append(tr[0])
        zerorates.append(tr[1])

    fittingCurve = ts.ZeroYieldCurve(ref_date, tenors, zerorates, interpolator1DType, extrapolator1DType)
    alphaPara = xen.DeterministicParameter([1, 20, 100], [0.1, 0.15, 0.15])
    sigmaPara = xen.DeterministicParameter([20, 100], [0.01, 0.015])

    bk1f = xen.BK1F('bk1f', fittingCurve, alphaPara, sigmaPara)

    return bk1f


def test():
    print('bk1f test...', filename)
    
    m = model()
    timeGrid = mx.TimeEqualGrid(ref_date, 3, 365)

    # random 
    rsg = xen.Rsg(sampleNum=5000)
    results = xen.generate1d(m, None, timeGrid, rsg, filename, False)
    # print(results.multiPath(scenCount=10))
    

if __name__ == "__main__":
    
    test()
	#mx.npzee_view(filename)