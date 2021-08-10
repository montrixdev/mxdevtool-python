# python script for VanillaOption_v1_1_0.xlsm file
# excel link : https://blog.naver.com/montrix/221378282753
# python link : https://blog.naver.com/montrix/***********

import mxdevtool as mx
import mxdevtool.instruments as mx_i
# import pandas as pd

# vanilla option

def test():
    print('option pricing test...')

    refDate = mx.Date(2020,7,13)
    multiplier = 250000
    mx.Settings.instance().setEvaluationDate(refDate)

    column_names = ['Name', 'Contracts', 'Type', 'X0', 'Strike', 'Rf', 'Div', 'Vol', 'Maturity']
    maturityDate = mx.Date(2020, 8, 13)

    option1 = ['option1', 10, mx.Option.Call, 285, 280, 0.02, 0, 0.16, maturityDate]
    option2 = ['option2', -8, mx.Option.Put, 285, 275, 0.02, 0, 0.16, maturityDate]
    option3 = ['option3', 10, mx.Option.Call, 285, 265, 0.02, 0, 0.16, maturityDate]
    option4 = ['option4', 10, mx.Option.Call, 285, 261.5, 0.02, 0, 0.16, maturityDate]

    option_arr = [option1, option2, option3, option4]

    # option_df = pd.DataFrame(opsion_arr, columns=column_names)

    results = []

    for opt in option_arr:

        option_type = opt[2]
        strike = opt[4]
        maturity = opt[8]
        x0 = opt[3]
        rf = opt[5]
        div = opt[6]
        vol = opt[7]

        option = mx_i.EuropeanOption(option_type, strike, maturity).setPricingParams_GBMConst(x0, rf, div, vol)

        Name = opt[0]
        Contracts = opt[1]
        NPV = multiplier * Contracts * option.NPV()
        Delta = multiplier * Contracts * option.delta()
        Gamma = multiplier * Contracts * option.gamma()
        Vega = multiplier * Contracts * option.vega()
        Theta = multiplier * Contracts * option.thetaPerDay()
        Rho = multiplier * Contracts * option.rho()
        Div_Rho = multiplier * Contracts * option.dividendRho()
        ImVol = option.impliedVolatility(option.NPV())
        UnitNPV = option.NPV()

        results.append([Name, NPV, Delta, Gamma, Vega, Theta, Rho, Div_Rho, ImVol, UnitNPV])

        print(Name + ' ---------------------------------')
        print('NPV   :', NPV)
        print('delta :', Delta)
        print('gamma :', Gamma)
        print('vega  :', Vega)
        print('theta :', Theta)
        print('rho   :', Rho)
        print('div_rho   :', Div_Rho)
        print('impliedVolatility   :', ImVol)
        print('UnitNPV   :', UnitNPV)
        print()

    # print('export csv file')
    # results_df = pd.DataFrame(results, columns=['Name', 'NPV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'DivRho', 'ImVol', 'UnitNPV'])
    # results_df.to_csv('VanillaOptionResults.csv')

if __name__ == "__main__":
    test()

