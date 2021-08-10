# excel link : https://blog.naver.com/montrix/221853410218

import mxdevtool as mx
import mxdevtool.termstructures as ts
import mxdevtool.instruments as inst

ref_date = mx.Date.todaysDate()

def yieldCurve():

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
    print('irs pricing test...')

    yield_curve = yieldCurve()

    family_name = 'irskrw_krccp'

    side = mx.VanillaSwap.Receiver
    nominal = 10000000000
    settlementDate = ref_date + 1
    maturityTenor = mx.Period('20Y')
    fixedRate = 0.013175
    spread = 0.0

    swap = inst.makeSwap(side, nominal, maturityTenor, fixedRate, spread, settlementDate, yield_curve, family_name)

    # print(swap.iborIndex.familyName())

    print('npv : ', swap.NPV())
    print('rho : ', swap.rho(mx.LegResultType.Net))
    print('conv : ', swap.convexity(mx.LegResultType.Net))

    print('leg rho(Pay) : ', swap.rho(mx.LegResultType.Pay))
    print('leg rho(Rec) : ', swap.rho(mx.LegResultType.Receive))
    print('leg rho(Fix) : ', swap.rho(mx.LegResultType.Fixed))
    print('leg rho(Flo) : ', swap.rho(mx.LegResultType.Floating))

if __name__ == "__main__":
    test()