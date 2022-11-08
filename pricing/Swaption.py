# python script for IRS_Calculator.xlsm file
# excel link : https://blog.naver.com/montrix/221853410218

import mxdevtool as mx
import mxdevtool.termstructures as ts
import mxdevtool.instruments as mx_i



def yieldCurve():

    ref_date = mx.Date.todaysDate()

    # ref_date
    marketQuotes = [('1D','Cash',0.012779015127),
                ('3M','Cash',0.0146),
                ('6M','Swap',0.014260714286),
                ('9M','Swap',0.014110714286),
                ('1Y','Swap',0.013975),
                ('18M','Swap',0.0138),
                ('2Y','Swap',0.013653571429),
                ('3Y','Swap',0.0137),
                ('4Y','Swap',0.013775),
                ('5Y','Swap',0.013814285714),
                ('6Y','Swap',0.013817857143),
                ('7Y','Swap',0.013835714286),
                ('8Y','Swap',0.013921428571),
                ('9Y','Swap',0.014042857143),
                ('10Y','Swap',0.014185714286),
                ('12Y','Swap',0.014360714286),
                ('15Y','Swap',0.014146428571),
                ('20Y','Swap',0.013175)]

    swap_quote_tenors = []
    swap_quote_types  = []
    swap_quote_values = []

    for q in marketQuotes:
        swap_quote_tenors.append(q[0])
        swap_quote_types.append(q[1])
        swap_quote_values.append(q[2])

    interpolator1DType = mx.Interpolator1D.Linear
    # extrapolation = mx.FlatExtrapolation('forward')
    extrapolation = mx.SmithWilsonExtrapolation(0.14, 0.042)

    family_name = 'irskrw_krccp'
    forSettlement = True

    yield_curve = ts.BootstapSwapCurveCCP(ref_date, swap_quote_tenors, swap_quote_types, swap_quote_values, interpolator1DType, extrapolation, family_name, forSettlement)

    return yield_curve


def test():

    yield_curve = yieldCurve()
    swaption = mx_i.makeSwaption(yieldCurve=yield_curve)

    print('npv : ', swaption.NPV())
    print('blackvol : ', swaption.impliedVolatility(swaption.NPV() * 0.9, yield_curve, 0.3))

if __name__ == "__main__":
    test()