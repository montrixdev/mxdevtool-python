import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.data as mx_d

# 시나리오 template 만들어서 저장하기

ref_date = mx.Date.todaysDate()

xfm_config = {
    'ip': '192.168.0.1',
    'hostname': 'test',
    'dbname': 'testdb',
    'type': 'database',
    'id': 'tekjkd',
    'pw': 'testt',
    'location': 'd:/'
}

xm = xen.XenarixFileManager(xfm_config)
# xm = xen.XenarixDbManager(config)

mdm_config = {
    'ip': '192.168.0.1',
    'id': 'tekjkd',
    'pw': 'testt',
}

mdp = mx_d.MarketDataManager(mdm_config)

# template 을 만듬 - 추 후 editor engine 이 될거임
mrk_d = mdp.get_data(name='name1', refDate=ref_date)

# mrk 사용법?
quote = mrk_d.get_quote('kospi2')
curve = mrk_d.get_yieldcurve('irskrw')s
spread_curve = curve.zero_spread(0.001)
surface = mrk_d.get_surface('swaptionvol')

stb = xen.ScenarioTemplateBuilder(mrk_d)

gbmconst_template1 = stb.gbmconst(name='gbm1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')
gbmconst1 = gbmconst_template1.make_model()
gbmconst_template2 = gbmconst1.make_template(x0=150)

models = [gbmconst_template1]

linearOper1 = stb.linearOper(name='linearOper1', multiple=1.0, spread=0.0)

calcs = [linearOper1]

# corr = mx.Matrix()
timegrid = stb.timegrid(refDate=ref_date)

scen_template1 = stb.make_template(name, models, calcs, corr, timegrid, rsg, isMomentMatching)
scen_template1.make_scenario()

# 저장하고, 로드함.
template_name1 = 'template_name1'
xm.save(name=template_name1, scen=[scen1, scen_template])
scen_template2 = xm.load(name=template_name1)

# 끝?



