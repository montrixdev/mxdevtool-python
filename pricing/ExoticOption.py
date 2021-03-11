# python script for OptionCalculator_v1_1_0.xlsm file
# excel link : https://blog.naver.com/montrix/221378282753
# python link : https://blog.naver.com/montrix/***********

import mxdevtool as mx
import mxdevtool.instruments as mx_i

# vanilla option

def test():
    print('option pricing test...')

    x0 = 255
    strike = 254
    rf = 0.02
    div = 0.0
    vol = 0.16
    maturityDate = mx.Date(2020,8,15)
    exDates = [ mx.Date(2020,8,15), mx.Date(2020,9,15)]

    european_option = mx_i.EuropeanOption(mx.Option.Call, strike, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)
    american_option = mx_i.AmericanOption(mx.Option.Call, strike, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)
    bermudan_option = mx_i.BermudanOption(mx.Option.Call, strike, exDates).setPricingParams_GBMConst(x0, rf, div, vol)

    barrierType = mx.Barrier.UpIn
    barrier = 280
    rebate = 5
    barrier_option = mx_i.BarrierOption(mx.Option.Call, barrierType, barrier, rebate, strike, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)

    options = [european_option, american_option, bermudan_option, barrier_option]

    for option in options:
        print('---------------------------------')
        print('NPV   :', option.NPV())
        print('delta :', option.delta())
        print('gamma :', option.gamma())
        print('vega  :', option.vega())
        print('theta :', option.thetaPerDay())
        print('rho   :', option.rho())
        print('div_rho   :', option.dividendRho())
        print('impliedVolatility   :', option.impliedVolatility(option.NPV()))

    #option1.imvol(1.2)

if __name__ == "__main__":
    test()

