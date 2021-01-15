import numpy as np
import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.termstructures as ts
import mxdevtool.data as mx_d

def test():
    ref_date = mx.Date.todaysDate()

    # (period, rf, div)
    tenor_rates = [('3M', 0.0151, 0.01),
                ('6M', 0.0152, 0.01),
                ('9M', 0.0153, 0.01),
                ('1Y', 0.0154, 0.01),
                ('2Y', 0.0155, 0.01),
                ('3Y', 0.0156, 0.01),
                ('4Y', 0.0157, 0.01),
                ('5Y', 0.0158, 0.01),
                ('7Y', 0.0159, 0.01),
                ('10Y', 0.016, 0.01),
                ('15Y', 0.0161, 0.01),
                ('20Y', 0.0162, 0.01)]

    tenors = []
    rf_rates = []
    div_rates = []
    vol = 0.2

    interpolator1DType = mx.Interpolator1D.Linear
    extrapolator1DType = mx.Extrapolator1D.FlatForward

    for tr in tenor_rates:
        tenors.append(tr[0])
        rf_rates.append(tr[1])
        div_rates.append(tr[2])
        
    rfCurve = ts.ZeroYieldCurve(ref_date, tenors, rf_rates, interpolator1DType, extrapolator1DType)
    divCurve = ts.ZeroYieldCurve(ref_date, tenors, div_rates, interpolator1DType, extrapolator1DType)
    volTs = ts.BlackConstantVol(ref_date, vol)

    # models
    gbmconst = xen.GBMConst('gbmconst', x0=100, rf=0.032, div=0.01, vol=0.15)
    gbm = xen.GBM('gbm', x0=100, rfCurve=rfCurve , divCurve=divCurve, volTs=volTs)
    heston = xen.Heston('heston', x0=100, rfCurve=rfCurve, divCurve=divCurve, v0=0.2, volRevertingSpeed=0.1, longTermVol=0.15, volOfVol=0.1, rho=0.3)

    alphaPara = xen.DeterministicParameter(['1y', '20y', '100y'], [0.1, 0.15, 0.15])
    sigmaPara = xen.DeterministicParameter(['20y', '100y'], [0.01, 0.015])
    
    hw1f = xen.HullWhite1F('hw1f', fittingCurve=rfCurve, alphaPara=alphaPara, sigmaPara=sigmaPara)
    bk1f = xen.BK1F('bk1f', fittingCurve=rfCurve, alphaPara=alphaPara, sigmaPara=sigmaPara)
    cir1f = xen.CIR1F('cir1f', r0=0.02, alpha=0.1, longterm=0.042, sigma=0.03)
    vasicek1f = xen.Vasicek1F('vasicek1f', r0=0.02, alpha=0.1, longterm=0.042, sigma=0.03)
    g2ext = xen.G2Ext('g2ext', fittingCurve=rfCurve, alpha1=0.1, sigma1=0.01, alpha2=0.2, sigma2=0.02, corr=0.5)

    # calcs in models
    hw1f_spot3m = hw1f.spot('hw1f_spot3m', maturityTenor=mx.Period(3, mx.Months), compounding=mx.Compounded)
    hw1f_forward6m3m = hw1f.forward('hw1f_forward6m3m', startTenor=mx.Period(6, mx.Months), maturityTenor=mx.Period(3, mx.Months), compounding=mx.Compounded)
    hw1f_discountFactor = hw1f.discountFactor('hw1f_discountFactor')
    hw1f_discountBond3m = hw1f.discountBond('hw1f_discountBond3m', maturityTenor=mx.Period(3, mx.Months))

    # calcs
    constantValue = xen.ConstantValue('constantValue', 15)
    constantArr = xen.ConstantArray('constantArr', [15,14,13])

    oper1 = gbmconst + gbm
    oper2 = gbmconst - gbm 
    oper3 = (gbmconst * gbm).withName('multiple_gbmconst_gbm')
    oper4 = gbmconst / gbm

    oper5 = gbmconst + 10
    oper6 = gbmconst - 10
    oper7 = gbmconst * 1.1
    oper8 = gbmconst / 1.1

    oper9 = 10 + gbmconst
    oper10 = 10 - gbmconst
    oper11 = 1.1 * gbmconst
    oper12 = 1.1 / gbmconst

    linearOper1 = xen.LinearOper('linearOper1', gbmconst, multiple=1.1, spread=10)
    linearOper2 = gbmconst.linearOper('linearOper2', multiple=1.1, spread=10)

    shiftRight1 = xen.Shift('shiftRight1', hw1f, shift=5)
    shiftRight2 = hw1f.shift('shiftRight2', shift=5, fill_value=0.0)

    shiftLeft1 = xen.Shift('shiftLeft1', cir1f, shift=-5) 
    shiftLeft2 = cir1f.shift('shiftLeft2', shift=-5, fill_value=0.0) 

    returns1 = xen.Returns('returns1', gbm, 'return')
    returns2 = gbm.returns('returns2', 'return')

    logreturns1 = xen.Returns('logreturns1', gbmconst,'logreturn')
    logreturns2 = gbmconst.returns('logreturns2', 'logreturn')

    cumreturns1 = xen.Returns('cumreturns1', heston,'cumreturn')
    cumreturns2 = heston.returns('cumreturns2', 'cumreturn')

    cumlogreturns1 = xen.Returns('cumlogreturns1', gbm,'cumlogreturn')
    cumlogreturns2 = gbm.returns('cumlogreturns2', 'cumlogreturn')

    fixedRateBond = xen.FixedRateBond('fixedRateBond', vasicek1f, notional=10000, fixedRate=0.0, couponTenor=mx.Period(3, mx.Months), maturityTenor=mx.Period(3, mx.Years), discountCurve=rfCurve)

    # timegrid
    timegrid1 = mx.TimeEqualGrid(refDate=ref_date, maxYear=3, nPerYear=365)
    timegrid2 = mx.TimeArrayGrid(refDate=ref_date, times=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    timegrid3 = mx.TimeGrid(refDate=ref_date, maxYear=10, frequency_type='endofmonth')
    timegrid4 = mx.TimeGrid(refDate=ref_date, maxYear=10, frequency_type='annual', frequency_month=8, frequency_day=10)

    # random
    pseudo_rsg = xen.Rsg(sampleNum=1000, dimension=365, seed=0, skip=0, isMomentMatching=False, randomType='pseudo', subType='mersennetwister', randomTransformType='boxmullernormal')
    sobol_rsg = xen.Rsg(sampleNum=1000, dimension=365, seed=0, skip=0, isMomentMatching=False, randomType='sobol', subType='joekuod7', randomTransformType='invnormal')
    
    # single model
    filename1='./single_model.npz'
    results1 = xen.generate1d(model=gbm, calcs=None, timegrid=timegrid1, rsg=pseudo_rsg, filename=filename1, isMomentMatching=False)

    # multiple model
    filename2='./multiple_model.npz'
    models = [gbmconst, gbm, hw1f, cir1f, vasicek1f]
    corrMatrix = mx.IdentityMatrix(len(models))

    results2 = xen.generate(models=models, calcs=None, corr=corrMatrix, timegrid=timegrid3, rsg=sobol_rsg, filename=filename2, isMomentMatching=False)

    # multiple model with calc
    filename3='./multiple_model_with_calc.npz'
    calcs = [oper1, oper3, linearOper1, linearOper2, shiftLeft2, returns1, fixedRateBond, hw1f_spot3m]
    results3 = xen.generate(models=models, calcs=calcs, corr=corrMatrix, timegrid=timegrid4, rsg=sobol_rsg, filename=filename3, isMomentMatching=False)

    all_models = [ gbmconst, gbm, heston, hw1f, bk1f, cir1f, vasicek1f, g2ext ]
    all_calcs = [ hw1f_spot3m, hw1f_forward6m3m, hw1f_discountFactor, hw1f_discountBond3m,
                  constantValue, constantArr, oper1, oper2, oper3, oper4, oper5, oper6, oper7, oper8, oper9, oper10, oper11, oper12,
                  linearOper1, linearOper2, shiftRight1, shiftRight2, shiftLeft1, shiftLeft2, returns1, returns2, logreturns1, logreturns2,
                  cumreturns1, cumreturns2, cumlogreturns1, cumlogreturns2, fixedRateBond ]
    
    filename4='./multiple_model_with_calc_all.npz'
    corrMatrix2 = mx.IdentityMatrix(len(all_models))
    results4 = xen.generate(models=all_models, calcs=all_calcs, corr=corrMatrix2, timegrid=timegrid4, rsg=sobol_rsg, filename=filename4, isMomentMatching=False)

    # results
    results = results3

    genInfo = results.genInfo
    refDate = results.refDate
    maxDate = results.maxDate
    maxTime = results.maxTime
    randomMomentMatch = results.randomMomentMatch
    randomSubtype = results.randomSubtype 
    randomType = results.randomType
    seed = results.seed
    shape = results.shape

    ndarray = results.toNumpyArr() # pre load all scenario data to ndarray
    
    t_pos = 1
    scenCount = 15
    
    # scenario path of selected scenCount
    # ((100.0, 82.94953421561434, 110.87375162324332, 91.96798678908293, 70.29920544659505, ... ), 
    #  (100.0, 96.98838977927142, 97.0643112022828, 91.19803393176569, 104.94407125936456, ... ), 
    #  ...
    #  (200.0, 179.93792399488575, 207.93806282552612, 183.16602072084862, ... ),
    #  (9546.93761943355, 9969.778029330208, 10758.449206155927, 11107.968356394866, ... ))
    multipath = results[scenCount] 
    multipath_arr = ndarray[scenCount]

    # t_pos data
    multipath_t_pos = results.tPosSlice(t_pos=t_pos, scenCount=scenCount) # (82.94953421561434, 96.98838977927142, 0.015097688448292656, 0.02390612251701627, ... )
    multipath_t_pos_arr = ndarray[scenCount,:,t_pos]

    multipath_all_t_pos = results.tPosSlice(t_pos=t_pos) # all t_pos data

    # t_pos data of using date
    t_date = ref_date + 10
    multipath_using_date = results.dateSlice(date=t_date, scenCount=scenCount) # (99.5327905069975, 99.91747715856324, 0.015099936660211026, 0.020107033880707947, ... )
    multipath_all_using_date = results.dateSlice(date=t_date) # all t_pos data

    # t_pos data of using time
    t_time = 1.32
    multipath_using_time = results.timeSlice(time=t_time, scenCount=scenCount) # (91.88967340028992, 97.01269656928498, 0.018200574048792405, 0.02436896520516243, ... )
    multipath_all_using_time = results.timeSlice(time=t_time) # all t_pos data

    # analyticPath and test calculation
    all_pv_list = []
    all_pv_list.extend(all_models)
    all_pv_list.extend(all_calcs)

    for pv in all_pv_list:
        analyticPath = pv.analyticPath(timegrid2)
        
    input_arr = [0.01, 0.02, 0.03, 0.04, 0.05]
    input_arr2d = [[0.01, 0.02, 0.03, 0.04, 0.05],
                   [0.06, 0.07, 0.08, 0.09, 0.1]]
    
    for pv in all_calcs:
        if pv.sourceNum == 1:
            calculatePath = pv.calculatePath(input_arr, timegrid1)
        elif pv.sourceNum == 2:
            calculatePath = pv.calculatePath(input_arr2d, timegrid1)
        else:
            pass
    
    # xenarix manager
    xenrepo_path = './xenrepo'
    xfm_config = { 'location': xenrepo_path }

    xm = xen.XenarixFileManager(xfm_config)

    filename5 = 'scen_all.npz'
    scen_all = xen.Scenario(models=all_models, calcs=all_calcs, corr=corrMatrix2, timegrid=timegrid4, rsg=sobol_rsg, filename=filename5, isMomentMatching=False)
    
    filename6 = 'scen_multiple.npz'
    scen_multiple = xen.Scenario(models=models, calcs=[], corr=corrMatrix, timegrid=timegrid4, rsg=pseudo_rsg, filename=filename6, isMomentMatching=False)

    scen_all_hashCode = scen_all.hashCode() 
    scen_all_hashCode2 = scen_all.fromDict(scen_all.toDict()).hashCode()

    assert scen_all_hashCode == scen_all_hashCode2

    # scenario - save, load, list
    name1 = 'name1'
    xm.save(name=name1, scen=scen_all)
    scen_name1 = xm.load(name=name1)

    scen_name1['scen0'].filename = './reloaded_scenfile.npz'
    scen_name1['scen0'].generate()

    name2 = 'name2'
    xm.save(name=name2, scen=[scen_all, scen_multiple])
    scen_name2 = xm.load(name=name2)

    name3 = 'name3'
    xm.save(name=name3, scen={'scen_all' : scen_all, 'scen_multiple': scen_multiple})
    scen_name3 = xm.load(name=name3)

    scenList = xm.scenList() # ['name1', 'name2', 'name3']
    
    # scenario template builder using market data
    sb = xen.ScenarioJsonBuilder()

    sb.addModel(xen.GBMConst.__name__, 'gbmconst', x0='kospi2', rf='cd91', div=0.01, vol=0.3)
    sb.addModel(xen.GBM.__name__, 'gbm', x0=100, rfCurve='zerocurve1', divCurve=divCurve, volTs=volTs)
    sb.addModel(xen.Heston.__name__, 'heston', x0=100, rfCurve='zerocurve1', divCurve=divCurve, v0=0.2, volRevertingSpeed=0.1, longTermVol=0.15, volOfVol=0.1, rho=0.3)
    sb.addModel(xen.HullWhite1F.__name__, 'hw1f', fittingCurve='zerocurve2', alphaPara=alphaPara, sigmaPara=sigmaPara)
    sb.addModel(xen.BK1F.__name__, 'bk1f', fittingCurve='zerocurve2', alphaPara=alphaPara, sigmaPara=sigmaPara)

    sb.addModel(xen.CIR1F.__name__, 'cir1f', r0='cd91', alpha=0.1, longterm=0.042, sigma=0.03)
    sb.addModel(xen.Vasicek1F.__name__, 'vasicek1f', r0='cd91', alpha='alpha1', longterm=0.042, sigma=0.03)
    sb.addModel(xen.G2Ext.__name__, 'g2ext', fittingCurve=rfCurve, alpha1=0.1, sigma1=0.01, alpha2=0.2, sigma2=0.02, corr=0.5)

    sb.addCalc(xen.SpotRate.__name__, 'hw1f_spot3m', ir_pc='hw1f', maturityTenor='3m', compounding=mx.Compounded)
    sb.addCalc(xen.ForwardRate.__name__, 'hw1f_forward6m3m', ir_pc='hw1f', startTenor=mx.Period(6, mx.Months), maturityTenor=mx.Period(3, mx.Months), compounding=mx.Compounded)
    sb.addCalc(xen.DiscountFactor.__name__, 'hw1f_discountFactor', ir_pc='hw1f')
    sb.addCalc(xen.DiscountBond.__name__, 'hw1f_discountBond3m', ir_pc='hw1f', maturityTenor=mx.Period(3, mx.Months))

    sb.addCalc(xen.ConstantValue.__name__, 'constantValue', v=15)
    sb.addCalc(xen.ConstantArray.__name__, 'constantArr', arr=[15,14,13])
    
    sb.addCalc(xen.AdditionOper.__name__, 'addOper1', pc1='gbmconst', pc2='gbm')
    sb.addCalc(xen.SubtractionOper.__name__, 'subtOper1', pc1='gbmconst', pc2='gbm')
    sb.addCalc(xen.MultiplicationOper.__name__, 'multiple_gbmconst_gbm', pc1='gbmconst', pc2='gbm')
    sb.addCalc(xen.DivisionOper.__name__, 'divOper1', pc1='gbmconst', pc2='gbm')

    sb.addCalc(xen.AdditionOper.__name__, 'addOper2', pc1='gbmconst', pc2=10)
    sb.addCalc(xen.SubtractionOper.__name__, 'subtOper2', pc1='gbmconst', pc2=10)
    sb.addCalc(xen.MultiplicationOper.__name__, 'mulOper2', pc1='gbmconst', pc2=1.1)
    sb.addCalc(xen.DivisionOper.__name__, 'divOper1', pc1='gbmconst', pc2=1.1)

    sb.addCalc(xen.AdditionOper.__name__, 'addOper2', pc1=10, pc2='gbmconst')
    sb.addCalc(xen.SubtractionOper.__name__, 'subtOper2', pc1=10, pc2='gbmconst')
    sb.addCalc(xen.MultiplicationOper.__name__, 'mulOper2', pc1=1.1, pc2='gbmconst')
    sb.addCalc(xen.DivisionOper.__name__, 'divOper1', pc1=1.1, pc2='gbmconst')

    sb.addCalc(xen.LinearOper.__name__, 'linearOper1', pc='gbm', multiple=1.1, spread=10)
    sb.addCalc(xen.Shift.__name__, 'shiftRight1', pc='hw1f', shift=5, fill_value=0.0)
    sb.addCalc(xen.Shift.__name__, 'shiftLeft1', pc='cir1f', shift=-5, fill_value=0.0)

    sb.addCalc(xen.Returns.__name__, 'returns1', pc='gbm', return_type='return')
    sb.addCalc(xen.Returns.__name__, 'logreturns1', pc='gbmconst', return_type='logreturn')
    sb.addCalc(xen.Returns.__name__, 'cumreturns1', pc='heston', return_type='cumreturn')
    sb.addCalc(xen.Returns.__name__, 'cumlogreturns1', pc='gbm', return_type='cumlogreturn')

    sb.addCalc(xen.FixedRateBond.__name__, 'fixedRateBond', ir_pc='vasicek1f', notional=10000, fixedRate=0.0, couponTenor=mx.Period(3, mx.Months), maturityTenor=mx.Period(3, mx.Years), discountCurve=rfCurve)

    mdp = mx_d.SampleMarketDataProvider()
    mrk = mdp.get_data()

    scen = sb.build_scenario(mrk)

    assert scen.hashCode() == scen.fromDict(scen.toDict()).hashCode()
    assert sb.hashCode() == sb.fromDict(sb.toDict()).hashCode()

    res = scen.generate(filename='new_temp.npz')
    # res.show()

    # marketdata clone
    mrk_clone = mrk.clone()


