# excel link : https://blog.naver.com/montrix/221376083438

import sys, os
import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.termstructures as ts
import numpy as np

filename = './test_hw1f.npz' # result file
ref_date = mx.Date.todaysDate() # referenceDate

def model():
    # zero curve data
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

    # interpolation and extrapolation
    interpolator1DType = mx.Interpolator1D.Linear
    extrapolator1DType = mx.Extrapolator1D.FlatForward

    for tr in tenor_rates:
        tenors.append(tr[0])
        zerorates.append(tr[1])

    # inputs of model
    fittingCurve = ts.ZeroYieldCurve(ref_date, tenors, zerorates, interpolator1DType, extrapolator1DType)
    alphaPara = xen.DeterministicParameter(['1y', '20y', '100y'], [0.1, 0.15, 0.15])
    sigmaPara = xen.DeterministicParameter(['20y', '100y'], [0.01, 0.015])

    # create model
    hw1f = xen.HullWhite1F('hw1f', fittingCurve, alphaPara, sigmaPara)

    return hw1f

def test():
    print('hw1f test...', filename)

    m = model()

    # timegrid
    timeGrid = mx.TimeEqualGrid(ref_date, 3, 365)
    
    # random sequence
    rsg = xen.Rsg(sampleNum=5000)

    # generate scenario of 1 dimension 
    results = xen.generate1d(m, None, timeGrid, rsg, filename, False)

    # results
    # print(results.multiPath(scenCount=10))
    
if __name__ == "__main__":
    
    test()
	#mx.npzee_view(filename)