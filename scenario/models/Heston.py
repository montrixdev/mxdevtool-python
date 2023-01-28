# excel link : https://blog.naver.com/montrix/221376083438

import sys, os
import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.termstructures as ts
import numpy as np

filename = './test_heston.npz'
ref_date = mx.Date.todaysDate()

def model():

    # (period, rf, div)
    tenor_rates = [('3M', 0.0151, 0.01),
                ('6M', 0.0152, 0.01),
                ('9M', 0.0153, 0.01),
                ('1Y', 0.0154, 0.01),
                ('2Y', 0.0155, 0.01),
                ('3Y', 0.0156, 0.01),
                ('4Y', 0.0157, 0.01),
                ('5Y', 0.0158, 0.01),
                ('7Y', 0.0159, 0.01),
                ('10Y', 0.016, 0.01),
                ('15Y', 0.0161, 0.01),
                ('20Y', 0.0162, 0.01)]

    tenors = []
    rf_rates = []
    div_rates = []

    interpolator1DType = mx.Interpolator1D.Linear
    extrapolator1DType = mx.Extrapolator1D.FlatForward

    for tr in tenor_rates:
        tenors.append(tr[0])
        rf_rates.append(tr[1])
        div_rates.append(tr[2])

    x0 = 100
    rfCurve = ts.ZeroYieldCurve(ref_date, tenors, rf_rates, interpolator1DType, extrapolator1DType)
    divCurve = ts.ZeroYieldCurve(ref_date, tenors, div_rates, interpolator1DType, extrapolator1DType)
    v0 = 0.2
    volRevertingSpeed = 0.1
    longTermVol = 0.15
    volOfVol = 0.1
    rho = 0.3

    heston = xen.Heston('heston', x0=x0, rfCurve=rfCurve, divCurve=divCurve, 
                        v0=v0, volRevertingSpeed=volRevertingSpeed, longTermVol=longTermVol, volOfVol=volOfVol, 
                        rho=rho)

    return heston


def test():
    print('heston test...', filename)
    
    m = model()
    timeGrid = mx.TimeDateGrid_Equal(ref_date, 3, 365)

    # random 
    rsg = xen.Rsg(sampleNum=5000)
    results = xen.generate1d(m, None, timeGrid, rsg, filename, False)
    # print(results.multiPath(scenCount=10))
    
if __name__ == "__main__":
    
    test()
    #mx.npzee_view(filename)