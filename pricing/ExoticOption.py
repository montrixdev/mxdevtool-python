# excel link : https://blog.naver.com/montrix/221378282753

import mxdevtool as mx

# vanilla option

def test():
    print('option pricing test...')

    s0 = 255
    strike = 254
    r = 0.02
    div = 0.0
    vol = 0.16
    maturityDate = mx.Date(2020,8,15)
    exDates = [ mx.Date(2020,8,15), mx.Date(2020,9,15)]

    european_option = mx.EuropeanOption(mx.Option.Call, s0, strike, r, div, vol, maturityDate)
    american_option = mx.AmericanOption(mx.Option.Call, s0, strike, r, div, vol, maturityDate)
    bermudan_option = mx.BermudanOption(mx.Option.Call, s0, strike, r, div, vol, exDates)

    barrierType = mx.Barrier.UpIn
    barrier = 280
    rebate = 5
    barrier_option = mx.BarrierOption(mx.Option.Call, barrierType, barrier, rebate, s0, strike, r, div, vol, maturityDate)

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

