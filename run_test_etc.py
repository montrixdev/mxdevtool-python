import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np

def timegrid_test():
    ref_date = mx.Date.todaysDate()

    maxYear = 10

    # TimeGrid
    timeGrid1 = mx.TimeEqualGrid(ref_date, 3, 365)
    timeGrid2 = mx.TimeArrayGrid(ref_date, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    timeGrid3 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofmonth')
    timeGrid4 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='day')
    timeGrid4 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='week')
    timeGrid5 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='month', frequency_day=10)
    timeGrid6 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='quarter', frequency_day=10)
    timeGrid7 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='semiannual', frequency_day=10)
    timeGrid8 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='annual', frequency_month=8, frequency_day=10)
    timeGrid9 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofmonth')
    timeGrid10 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofquarter')
    timeGrid11 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofsemiannual')
    timeGrid12 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofannual')
    timeGrid13 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofmonth')
    timeGrid14 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofquarter')
    timeGrid15 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofsemiannual')
    timeGrid16 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofannual')

    timeGrids = [ timeGrid1, timeGrid2, timeGrid3, timeGrid4, timeGrid5, timeGrid6, timeGrid7, timeGrid8, timeGrid9, timeGrid10, 
                  timeGrid11, timeGrid12, timeGrid13, timeGrid14, timeGrid15, timeGrid16]

    for tg, i in zip(timeGrids, range(len(timeGrids))):
        print('timeGrid{0} :'.format(i+1), tg.dates()[:10]) 

if __name__ == "__main__":
    timegrid_test()