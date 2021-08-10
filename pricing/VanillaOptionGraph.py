# python script for VanillaOptionGraph_v1_2_0.xlsm file
# excel link : https://blog.naver.com/montrix/222135609534

import mxdevtool as mx
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import mxdevtool.instruments as mx_i

# vanilla option

def test():
    print('option graph test...')

    refDate = mx.Date(2020,7,13)
    mx.setEvaluationDate(refDate)
    multiplier = 250000

    maturityDate = mx.Date(2020, 8, 13)

    x0 = 300
    rf = 0.02
    div = 0.0
    vol = 0.16

    option1 = mx_i.EuropeanOption(mx.Option.Call, 285, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)
    option2 = mx_i.EuropeanOption('c', 270, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)
    option3 = mx_i.EuropeanOption(mx.Option.Put,  283, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)
    option4 = mx_i.EuropeanOption(mx.Option.Call, 310, maturityDate).setPricingParams_GBMConst(x0, rf, div, vol)

    options = [ option1, option2, option3, option4 ]
    multiples = np.array([ 20,-15, 15, 20 ]) * multiplier

    portfolio = mx.Portfolio(multiples.tolist(), options)

    # n = 200
    x_grid = {
        'spot' : np.arange(200, 400, 1) ,
        'rf' : np.arange(0.001, 0.05, 0.00025),
        'div' : np.arange(0.001, 0.05, 0.00025),
        'vol' : np.arange(0.01, 0.8, 0.00395)
    }

    parameter = 'spot' # spot, rf, div, vol
    target = 'npv' # delta, gamma, vega, theta, rho
    results1 = portfolio.calculateMany(parameter, '=', x_grid[parameter].tolist(), target)

    # plt.plot(x_grid[parameter], results1)
    # plt.xlabel(parameter)
    # plt.title(target)
    # plt.show()

    #results1_df = pd.DataFrame(results1)
    #results1_df.to_csv('./excel/pricing/VanillaOptionGraphResults1.csv')

    #results2 = portfolio.calculateMany(['spot','rf'], ['=','='], [spot_grid.tolist(), rf_grid.tolist()], 'npv')
    #results2_df = pd.DataFrame(results2)
    #results2_df.to_csv('./excel/pricing/VanillaOptionGraphResults2.csv')

if __name__ == "__main__":
    test()

