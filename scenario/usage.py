from mxdevtool.xenarix.core import Scenario
import os
import numpy as np
import mxdevtool as mx
import mxdevtool.shock as mx_s
import mxdevtool.xenarix as xen
import mxdevtool.termstructures as ts
import mxdevtool.quotes as mx_q
import mxdevtool.data.providers as mx_dp
import mxdevtool.data.repositories as mx_dr
import mxdevtool.instruments as mx_i
import mxdevtool.instruments.outputs as mx_io
import mxdevtool.utils as utils

def test():
    ref_date = mx.Date.todaysDate()
    null_calendar = mx.NullCalendar()

    # (period, rf, div)
    tenor_rates = [('3M',  0.0151, 0.01),
                   ('6M',  0.0152, 0.01),
                   ('9M',  0.0153, 0.01),
                   ('1Y',  0.0154, 0.01),
                   ('2Y',  0.0155, 0.01),
                   ('3Y',  0.0156, 0.01),
                   ('4Y',  0.0157, 0.01),
                   ('5Y',  0.0158, 0.01),
                   ('7Y',  0.0159, 0.01),
                   ('10Y', 0.016,  0.01),
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

    x0 = 420

    # yieldCurve
    rfCurve = ts.ZeroYieldCurve(ref_date, tenors, rf_rates, interpolator1DType, extrapolator1DType)
    divCurve = ts.ZeroYieldCurve(ref_date, tenors, div_rates, interpolator1DType, extrapolator1DType)

    utils.check_hashCode(rfCurve, divCurve)

    # variance termstructure
    const_vts = ts.BlackConstantVol(refDate=ref_date, vol=vol)

    periods = [str(i+1) + 'm' for i in range(0, 24)] # monthly upto 2 years
    expirydates = [null_calendar.advance(ref_date, p) for p in periods]
    volatilities = [0.260, 0.223, 0.348, 0.342, 0.328, 0.317, 0.310, 0.302, 0.296, 0.291, 0.286, 0.282, 0.278, 0.275, 0.273, 0.270, 0.267, 0.263, 0.261, 0.258, 0.255, 0.253, 0.252, 0.251]

    curve_vts = ts.BlackVarianceCurve(refDate=ref_date, dates=expirydates, volatilities=volatilities)

    utils.check_hashCode(const_vts, curve_vts)

    # models
    gbmconst = xen.GBMConst('gbmconst', x0=x0, rf=0.032, div=0.01, vol=0.15)
    gbm = xen.GBM('gbm', x0=x0, rfCurve=rfCurve , divCurve=divCurve, volTs=curve_vts)
    heston = xen.Heston('heston', x0=x0, rfCurve=rfCurve, divCurve=divCurve, v0=0.2, volRevertingSpeed=0.1, longTermVol=0.15, volOfVol=0.1, rho=0.3)

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

    # model calculation
    r_t = 0.02 # short rate
    hw1f.model_discountBond(0.0, 1.0, r_t)
    hw1f.model_spot(1.0, 2.0, r_t) # continuous compounding
    hw1f.model_forward(1.0, 2.0, 3.0, r_t) # continuous compounding
    hw1f.model_discount(1.0)

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
    maxYear = 10

    timegrid1  = mx.TimeEqualGrid(refDate=ref_date, maxYear=3, nPerYear=365)
    timegrid2  = mx.TimeArrayGrid(refDate=ref_date, times=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    timegrid3  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='day')
    timegrid4  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='week')
    timegrid5  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='month', frequency_day=10)
    timegrid6  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='quarter', frequency_day=10)
    timegrid7  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='semiannual', frequency_day=10)
    timegrid8  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='annual', frequency_month=8, frequency_day=10)
    timegrid9  = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofmonth')
    timegrid10 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofquarter')
    timegrid11 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofsemiannual')
    timegrid12 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='firstofannual')
    timegrid13 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofmonth')
    timegrid14 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofquarter')
    timegrid15 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofsemiannual')
    timegrid16 = mx.TimeGrid(refDate=ref_date, maxYear=maxYear, frequency_type='endofannual')

    # random
    pseudo_rsg = xen.Rsg(sampleNum=1000, dimension=365, seed=1, skip=0, isMomentMatching=False, randomType='pseudo', subType='mersennetwister', randomTransformType='boxmullernormal')
    sobol_rsg = xen.Rsg(sampleNum=1000, dimension=365, seed=1, skip=0, isMomentMatching=False, randomType='sobol', subType='joekuod7', randomTransformType='invnormal')

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

    corrMatrix2[1][0] = 0.5
    corrMatrix2[0][1] = 0.5

    results4 = xen.generate(models=all_models, calcs=all_calcs, corr=corrMatrix2, timegrid=timegrid4, rsg=sobol_rsg, filename=filename4, isMomentMatching=False)

    # results
    results = results3

    resultsInfo = (results.genInfo, results.refDate, results.maxDate, results.maxTime, results.randomMomentMatch,
                   results.randomSubtype, results.randomType, results.seed, results.shape )

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

    # repository
    repo_path = './xenrepo'
    repo_config = { 'location': repo_path }
    repo = mx_dr.FolderRepository(repo_config)
    mx_dr.settings.set_repo(repo)

    # xenarix manxager
    xm = repo.xenarix_manager

    filename5 = 'scen_all.npz'
    scen_all = xen.Scenario(models=all_models, calcs=all_calcs, corr=corrMatrix2, timegrid=timegrid4, rsg=sobol_rsg, filename=filename5, isMomentMatching=False)

    filename6 = 'scen_multiple.npz'
    scen_multiple = xen.Scenario(models=models, calcs=[], corr=corrMatrix, timegrid=timegrid4, rsg=pseudo_rsg, filename=filename6, isMomentMatching=False)

    utils.check_hashCode(scen_all)

    # scenario - save, load, list
    name1 = 'name1'
    xm.save_xen(name1, scen_all) #
    scen_name1 = xm.load_xen(name=name1)

    scen_name1.filename = './reloaded_scenfile.npz'
    scen_name1.generate()

    name2 = 'name2'
    xm.save_xens(name=name2, scen_all=scen_all, scen_multiple=scen_multiple)
    scen_name2 = xm.load_xens(name=name2)

    scenList = xm.scenList() # ['name1', 'name2']

    # generate in result directory
    xm.generate_xen(scenList[0])

    # scenario template builder using market data
    sb = xen.ScenarioBuilder()

    sb.addModel(xen.GBMConst.__name__, 'gbmconst', x0='kospi2', rf='cd91', div=0.01, vol=0.3)
    sb.addModel(xen.GBM.__name__, 'gbm', x0=100, rfCurve='zerocurve1', divCurve=divCurve, volTs=const_vts)
    sb.addModel(xen.Heston.__name__, 'heston', x0='ni225', rfCurve='zerocurve1', divCurve=divCurve, v0=0.2, volRevertingSpeed=0.1, longTermVol=0.15, volOfVol=0.1, rho=0.3)
    sb.addModel(xen.HullWhite1F.__name__, 'hw1f', fittingCurve='zerocurve2', alphaPara=alphaPara, sigmaPara=sigmaPara)
    sb.addModel(xen.BK1F.__name__, 'bk1f', fittingCurve='zerocurve2', alphaPara=alphaPara, sigmaPara=sigmaPara)

    sb.addModel(xen.CIR1F.__name__, 'cir1f', r0='cd91', alpha=0.1, longterm=0.042, sigma=0.03)
    sb.addModel(xen.Vasicek1F.__name__, 'vasicek1f', r0='cd91', alpha='alpha1', longterm=0.042, sigma=0.03)
    sb.addModel(xen.G2Ext.__name__, 'g2ext', fittingCurve=rfCurve, alpha1=0.1, sigma1=0.01, alpha2=0.2, sigma2=0.02, corr=0.5)

    sb.corr[1][0] = 0.5
    sb.corr[0][1] = 0.5
    sb.corr[0][2] = 'kospi2_ni225_corr'
    sb.corr[2][0] = 'kospi2_ni225_corr'

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

    sb.addCalc(xen.AdditionOper.__name__, 'addOper_for_remove', pc1='gbmconst', pc2='gbm')
    sb.removeCalc('addOper_for_remove')

    # scenarioBuilder - save, load, list

    mdp = mx_dp.SampleMarketDataProvider()
    mrk = mdp.get_data()

    xm.save_xnb('sb1', sb=sb)

    sb.setTimeGridCls(timegrid3)
    sb.setRsgCls(pseudo_rsg)

    xm.save_xnb('sb2', sb=sb)

    sb.setTimeGrid(mx.TimeGrid.__name__, refDate=ref_date, maxYear=10, frequency_type='endofmonth')
    sb.setRsg(xen.Rsg.__name__, sampleNum=1000)

    xm.save_xnb('sb3', sb=sb)
    xm.scenBuilderList() # ['sb1', 'sb2', 'sb3']

    sb1_reload = xm.load_xnb('sb1')
    sb2_reload = xm.load_xnb('sb2')
    sb3_reload = xm.load_xnb('sb3')

    utils.compare_hashCode(sb, sb3_reload)
    utils.check_hashCode(sb, sb1_reload, sb2_reload, sb3_reload)

    xm.generate_xnb('sb1', mrk)
    xm.load_results_xnb('sb1')

    scen = sb.build_scenario(mrk)

    utils.check_hashCode(scen, sb)

    res = scen.generate()
    res1 = scen.generate_clone(filename='new_temp.npz') # clone generate with some change
    # res.show()

    # marketdata
    mrk_clone = mrk.clone()
    utils.compare_hashCode(mrk, mrk_clone)

    zerocurve1 = mrk.get_yieldCurve('zerocurve1')
    zerocurve2 = mrk.get_yieldCurve('zerocurve2')

    # shock definition
    quote1 = mx_q.SimpleQuote('quote1', 100)

    qst_add = mx_s.QuoteShockTrait(name='add_up1', value=10, operand='add')
    qst_mul = mx_s.QuoteShockTrait('mul_up1', 1.1, 'mul')
    qst_ass = mx_s.QuoteShockTrait('assign_up1', 0.03, 'assign')
    qst_add2 = mx_s.QuoteShockTrait('add_down1', 15, 'add')
    qst_mul2 = mx_s.QuoteShockTrait('mul_down2', 0.9, 'mul')

    quoteshocktrait_list = [qst_add, qst_mul, qst_ass, qst_add2, qst_mul2]
    quoteshocktrait_results = [100 + 10, (100 + 10)*1.1, 0.03, 0.03+15, (0.03+15)*0.9]
    quote1_d = quote1.toDict()

    for st, res in zip(quoteshocktrait_list, quoteshocktrait_results):
        st.calculate(quote1_d)
        assert res == quote1_d['v']

    qcst = mx_s.CompositeQuoteShockTrait('comp1', [qst_add2, qst_mul2])

    ycps = mx_s.YieldCurveParallelBpShockTrait('parallel_up1', 10)
    vcps = mx_s.VolTsParallelShockTrait('vol_up1', 0.1)

    shocktrait_list = quoteshocktrait_list + [qcst, ycps, vcps]

    # qcst = mx_s.CompositeQuoteShockTrait('comp2', [qst_add2, vcps])

    # build shock from shocktraits
    shock1 = mx_s.Shock(name='shock1')

    shock1.addShockTrait(target='kospi2', shocktrait=qst_add)
    shock1.addShockTrait(target='spx', shocktrait=qst_add)
    shock1.addShockTrait(target='ni*',  shocktrait=qst_add) # filter expression
    shock1.addShockTrait(target='*', shocktrait=qst_mul)
    shock1.addShockTrait(target='cd91', shocktrait=qst_ass)
    shock1.addShockTrait(target='alpha1', shocktrait=qcst)

    shock1.removeShockTrait(target='cd91')
    shock1.removeShockTrait(shocktrait=qst_mul)
    shock1.removeShockTrait(target='target2', shocktrait=ycps)
    shock1.removeShockTraitAt(3)

    # build shocked market data from shock
    shocked_mrk1 = mx_s.build_shockedMrk(shock1, mrk)
    shock2 = shock1.clone(name='shock2')
    shocked_mrk2 = mx_s.build_shockedMrk(shock2, mrk)

    utils.check_hashCode(shock1, shock2, shocked_mrk1, shocked_mrk2)

    shockedScen_list = mx_s.build_shockedScen([shock1, shock2], sb, mrk)

    shm = mx_s.ShockScenarioModel('shm1', basescen=scen, s_up=shockedScen_list[0], s_down=shockedScen_list[1])

    basescen_name = 'basescen'
    shm.addCompositeScenRes(name='compscen1', basescen_name=basescen_name, gbmconst='s_down')
    # shm.removeCompositeScenRes(name='compscen1')
    shm.compositeScenResList() # ['compscen1']

    # compare ?
    csr = xen.CompositeScenarioResults(shm.shocked_scen_res_d, basescen_name, gbmconst='s_down')

    csr_arr = csr.toNumpyArr()
    base_arr = scen.getResults().toNumpyArr()

    assert base_arr[0][0][0] + qst_add.value == csr_arr[0][0][0] # replaced(gbmconst)
    assert base_arr[0][1][0] == csr_arr[0][1][0] # not replaced(gbm)

    # shock manager - save, load, list
    # extensions : shock(.shk), shocktrait(.sht), shockscenariomodel(.shm)
    sfm = repo.shock_manager

    # shocktrait
    sht_name = 'shocktraits'
    sfm.save_shts(sht_name, *shocktrait_list)
    reloaded_sht_d = sfm.load_shts(sht_name)

    for s in shocktrait_list:
        utils.check_hashCode(s, reloaded_sht_d[s.name])
        utils.compare_hashCode(s, reloaded_sht_d[s.name])

    # shock
    shk_name = 'shocks'
    sfm.save_shks(shk_name, shock1, shock2)
    reloaded_shk_d = sfm.load_shks(shk_name)

    for s in [shock1, shock2]:
        utils.check_hashCode(s, reloaded_shk_d[s.name])
        utils.compare_hashCode(s, reloaded_shk_d[s.name])

    # shock scenario model
    shm_name = 'shockmodel'
    sfm.save_shm(shm_name, shm)
    reloaded_shm = sfm.load_shm(shm_name)

    utils.check_hashCode(shm, reloaded_shm)
    utils.compare_hashCode(shm, reloaded_shm)

    shocked_scen_list = mx_s.build_shockedScen([shock1, shock2], sb, mrk)

    for i, scen in enumerate(shocked_scen_list):
        name = 'shocked_scen{0}'.format(i)
        xm.save_xen(name, scen)
        res = scen.generate_clone(filename=name)

    # bloomberg provider(blpapi) checking to request sample if available
    try: mx_dp.check_bloomberg()
    except: print('fail to check bloomberg')

    # instruments pricing

    # this is built-in instruments
    # option1 = mx_i.EuropeanOption(option_type='c', strike=400, maturityDate=ref_date + 365)

    # this is inherit instrument for user output
    class EuropeanOptionForUserOutput(mx_i.EuropeanOption):
        def userfunc_test(self, scen_data_d, calc_kwargs):
            v = calc_kwargs['calc_arg1']
            return v + 99

    option = EuropeanOptionForUserOutput(option_type='c', strike=400, maturityDate=ref_date + 365)

    # outputs
    delta = mx_io.Delta(up='s_up', down='s_down')
    gamma = mx_io.Gamma(up='s_up', center='basescen', down='s_down')

    npv = mx_io.Npv(scen='basescen', currency='krw')
    discount_cf = mx_io.CashFlow(scen='basescen', currency='krw', discount=None)
    test_output = mx_io.UserFunc(scen='basescen', userfunc=option.userfunc_test, abc=10)

    # calculate from scenario
    results1 = option.calculateScen(outputs=[npv, discount_cf, delta, gamma, test_output], shm=shm, reduce='aver',
                                    path_kwargs={'s1': 'gbmconst', 'discount': 'hw1f_discountFactor'},
                                    calc_kwargs={'calc_arg1': 10})

    # calculate from model
    basescen = shm.getScenario('basescen')
    gbmconst_basescen = basescen.getModel('gbmconst')
    arg_d = { 'x0': gbmconst_basescen._x0, 'rf': gbmconst_basescen._rf, 'div': gbmconst_basescen._div, 'vol': gbmconst_basescen._vol }
    assert option.setPricingParams_GBMConst(**arg_d).NPV() == option.setPricingParams_Model(gbmconst_basescen).NPV()

    # calendar holiday
    mydates = [mx.Date(2022, 10, 11), mx.Date(2022, 10, 12), mx.Date(2022, 10, 13), mx.Date(2022, 11, 11)]

    kr_cal = mx.SouthKorea()
    user_cal = mx.UserCalendar('testcal')

    for cal in [kr_cal, user_cal]:
        repo.addHolidays(cal, mydates, onlyrepo=False)
        # repo.removeHolidays(cal, mydates, onlyrepo=False)

    # graph
    # rfCurve.graph_view(show=False)

    # report
    html_template = '''
        <!DOCTYPE html>
        <html>
        <head><title>{{ name }}</title></head>
        <body>
            <h1>Scenario Summary - Custom Template</h1>
            <p>models : {{ models_num }} - {{ model_names }}</p>
            <p>calcs : {{ calcs_num }} - {{ calc_names }}</p>
            <p>corr : {{ corr }}</p>
            <p>timegrid : {{ timegrid_items }}</p>
            <p>filename : {{ scen.filename }}</p>
            <p>ismomentmatch : {{ scen.isMomentMatching }}</p>
        </body>
        '''

    html = scen.report(typ='html', html_template=html_template, browser_isopen=False)


if __name__ == "__main__":
    test()
