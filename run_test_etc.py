import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np
import math

def timegrid_test():
    print('timegrid test...')

    ref_date = mx.Date.todaysDate()

    maxYear = 10

    # TimeGrid
    timegrid1  = mx.TimeDateGrid_Equal(refDate=ref_date, maxYear=3, nPerYear=365)
    timegrid2  = mx.TimeDateGrid_Times(refDate=ref_date, times=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    timegrid3  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='day')
    timegrid4  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='week')
    timegrid5  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='month', frequency_day=10)
    timegrid6  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='quarter', frequency_day=10)
    timegrid7  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='semiannual', frequency_day=10)
    timegrid8  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='annual', frequency_month=8, frequency_day=10)
    timegrid9  = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='firstofmonth')
    timegrid10 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='firstofquarter')
    timegrid11 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='firstofsemiannual')
    timegrid12 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='firstofannual')
    timegrid13 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='endofmonth')
    timegrid14 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='endofquarter')
    timegrid15 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='endofsemiannual')
    timegrid16 = mx.TimeDateGrid_Custom(refDate=ref_date, maxYear=maxYear, frequency_type='endofannual')

    timegrids = [ timegrid1, timegrid2, timegrid3, timegrid4, timegrid5, timegrid6, timegrid7, timegrid8, timegrid9, timegrid10, 
                  timegrid11, timegrid12, timegrid13, timegrid14, timegrid15, timegrid16]

    # for tg, i in zip(timegrids, range(len(timegrids))):
    #     print('timeGrid{0} :'.format(i+1), tg.dates()[:10]) 


def correlation_test():
    print('correlation test...')
    # (date, SPX Index,	NKY Index, SHCOMP Index, US0003M Index ) - bloomberg ticker
    mrk_data =[ 
        [ '2020-01-06',3246.28, 23204.86 ,3083.408, 1.87225 ],
        [ '2020-01-07',3237.18, 23575.72 ,3104.802, 1.878 ],
        [ '2020-01-08',3253.05, 23204.76 ,3066.893, 1.834 ],
        [ '2020-01-09',3274.7, 23739.87 ,3094.882, 1.84788 ],
        [ '2020-01-10',3265.35, 23850.57 ,3092.291, 1.83775 ],
        [ '2020-01-14',3283.15, 24025.17 ,3106.82, 1.84263 ],
        [ '2020-01-15',3289.29, 23916.58 ,3090.038, 1.83613 ],
        [ '2020-01-16',3316.81, 23933.13 ,3074.081, 1.82663 ],
        [ '2020-01-17',3329.62, 24041.26 ,3075.496, 1.81913 ],
        [ '2020-01-21',3320.79, 23864.56 ,3052.142, 1.80625 ],
        [ '2020-01-22',3321.75, 24031.35 ,3060.754, 1.80088 ],
        [ '2020-01-23',3325.54, 23795.44 ,2976.528, 1.79413 ],
        [ '2020-02-03',3248.92, 22971.94 ,2746.606, 1.741 ],
        [ '2020-02-04',3297.59, 23084.59 ,2783.288, 1.73738 ],
        [ '2020-02-05',3334.69, 23319.56 ,2818.088, 1.74163 ],
        [ '2020-02-06',3345.78, 23873.59 ,2866.51, 1.73413 ],
        [ '2020-02-07',3327.71, 23827.98 ,2875.964, 1.73088 ],
        [ '2020-02-10',3352.09, 23685.98 ,2890.488, 1.71313 ],
        [ '2020-02-12',3379.45, 23861.21 ,2926.899, 1.70375 ],
        [ '2020-02-13',3373.94, 23827.73 ,2906.073, 1.69163 ],
        [ '2020-02-14',3380.16, 23687.59 ,2917.008, 1.69175 ],
        [ '2020-02-18',3370.29, 23193.8 ,2984.972, 1.69463 ],
        [ '2020-02-19',3386.15, 23400.7 ,2975.402, 1.696 ],
        [ '2020-02-20',3373.23, 23479.15 ,3030.154, 1.68275 ],
        [ '2020-02-21',3337.75, 23386.74 ,3039.669, 1.67925 ],
        [ '2020-02-25',3128.21, 22605.41 ,3013.05, 1.63763 ],
        [ '2020-02-26',3116.39, 22426.19 ,2987.929, 1.61325 ],
        [ '2020-02-27',2978.76, 21948.23 ,2991.329, 1.58038 ],
        [ '2020-02-28',2954.22, 21142.96 ,2880.304, 1.46275 ],
        [ '2020-03-02',3090.23, 21344.08 ,2970.931, 1.25375 ],
        [ '2020-03-03',3003.37, 21082.73 ,2992.897, 1.31425 ],
        [ '2020-03-04',3130.12, 21100.06 ,3011.666, 1.00063 ],
        [ '2020-03-05',3023.94, 21329.12 ,3071.677, 0.99888 ],
        [ '2020-03-06',2972.37, 20749.75 ,3034.511, 0.896 ],
        [ '2020-03-09',2746.56, 19698.76 ,2943.291, 0.76813 ],
        [ '2020-03-10',2882.23, 19867.12 ,2996.762, 0.78413 ],
        [ '2020-03-11',2741.38, 19416.06 ,2968.517, 0.7725 ],
        [ '2020-03-12',2480.64, 18559.63 ,2923.486, 0.7405 ],
        [ '2020-03-13',2711.02, 17431.05 ,2887.427, 0.84313 ],
        [ '2020-03-16',2386.13, 17002.04 ,2789.254, 0.88938 ],
        [ '2020-03-17',2529.19, 17011.53 ,2779.641, 1.05188 ],
        [ '2020-03-18',2398.1, 16726.55 ,2728.756, 1.11575 ],
        [ '2020-03-19',2409.39, 16552.83 ,2702.13, 1.19513 ],
        [ '2020-03-23',2237.4, 16887.78 ,2660.167, 1.21563 ],
        [ '2020-03-24',2447.33, 18092.35 ,2722.438, 1.23238 ],
        [ '2020-03-25',2475.56, 19546.63 ,2781.591, 1.267 ],
        [ '2020-03-26',2630.07, 18664.6 ,2764.911, 1.37463 ],
        [ '2020-03-27',2541.47, 19389.43 ,2772.203, 1.45013 ],
        [ '2020-03-30',2626.65, 19084.97 ,2747.214, 1.43338 ],
        [ '2020-03-31',2584.59, 18917.01 ,2750.296, 1.4505 ] ]

    mrk_data_return = []

    for row, shift_row in zip(mrk_data[:-1], mrk_data[1:]):
        _row = row[1:] # except date
        _shift_row = shift_row[1:] # except date
        mrk_data_return.append([ math.log(v[1]/v[0]) for v in zip(_row, _shift_row) ])

    corr_arr = np.corrcoef(np.transpose(mrk_data_return))    
    corr = mx.Matrix(corr_arr.tolist())
    

if __name__ == "__main__":
    timegrid_test()
    correlation_test()

