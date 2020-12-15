import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np

def timegrid_test():
    ref_date = mx.Date.todaysDate()

    # TimeGrid
    timeGrid1 = mx.TimeEqualGrid(ref_date, 3, 365)
    print(timeGrid1.times())

    timeGrid2 = mx.TimeArrayGrid(ref_date, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    print(timeGrid2.times())

    timeGrid4 = mx.TimeGrid(refDate=ref_date, maxYear=10, frequency='custom', frequency_month=8, frequency_day=10)
    print(timeGrid4.times())

    timeGrid3 = mx.TimeGrid(ref_date, maxYear=10, frequency='endofmonth')
    print(timeGrid3.times())


if __name__ == "__main__":
    print('core engine version : ' + mx.__version__)

    timegrid_test()




    



