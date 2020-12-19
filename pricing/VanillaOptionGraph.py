# excel link : https://blog.naver.com/montrix/222135609534

import mxdevtool as mx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# vanilla option

def test():
    print('option graph test...')

    refDate = mx.Date(2020,7,13)
    mx.setEvaluationDate(refDate)
    multiplier = 250000

    maturityDate = mx.Date(2020, 8, 13)

    s0 = 300
    r = 0.02
    div = 0.0
    vol = 0.16
    
    option1 = mx.EuropeanOption(mx.Option.Call, s0, 285, r, div, vol, maturityDate)
    option2 = mx.EuropeanOption(mx.Option.Call, s0, 270, r, div, vol, maturityDate)
    option3 = mx.EuropeanOption(mx.Option.Put,  s0, 283, r, div, vol, maturityDate)
    option4 = mx.EuropeanOption(mx.Option.Call, s0, 310, r, div, vol, maturityDate)
    
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

    plt.plot(x_grid[parameter], results1)
    plt.xlabel(parameter)
    plt.title(target)
    plt.show()

if __name__ == "__main__":
    test()

