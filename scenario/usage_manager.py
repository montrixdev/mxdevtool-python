import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.data as mx_d

# xenarix manager - 이거 먼저.
xfm_config = {
    'location': 'd:/mxdevtool'
}

xm = xen.XenarixFileManager(xfm_config)


# 저장하고, 로드함.
name = 'name1'
xm.save(name=name, scen=[scen1])
scen = xm.load(name=name)


# data manager
mdm_config = {
    'location': 'd:/mxdevtool/data',
}

mdp = mx_d.DataManager(mdm_config)

# template 을 만듬 - 추 후 editor engine 이 될거임
mrk_d = mdp.get_marketdata(name='name1', refDate=ref_date)

# mrk 사용법?
quote = mrk_d.get_quote('kospi2')
curve = mrk_d.get_yieldcurve('irskrw')s
spread_curve = curve.zero_spread(0.001)
surface = mrk_d.get_surface('swaptionvol')