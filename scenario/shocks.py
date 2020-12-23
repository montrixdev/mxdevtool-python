# shock
import mxdevtool as mx
import mxdevtool.xenarix as xen

class GenerateHelper:
    pass

gh = GenerateHelper()

scen1 = xen.Scenario(filename1, models, calcs)
scen2 = xen.Scenario(filename2, models, calcs)

scen1 = xen.ScenarioGenerator(models, calcs, corr, timegrid, rsg, filename, isMomentMatching)

# 그냥 만든다

def build_gbm(name, x0, rfCurve, divCurve, volTs):

    x0 = 100
    gbm = xen.GBM('gbm1', x0=x0, rfCurve=[], divCurve=[], volTs=[])

    scen1 = xen.ScenarioGenerator(models, calcs, corr, timegrid, rsg, filename, isMomentMatching)

    return scen_list


scen_list = []

marketdata = get_shocked_marketdata()

for data in marketdata:
    # upshock = marketdata['upshock']

    kospi2 = build_gbm('kospi2', data['kospi2'])
    sp500 = build_gbm('sp500', data['sp500'])

    models = [kospi2, sp500]
    scen = xen.ScenarioGenerator()
    scen_list.append(scen)


# 완성되면 이건 여러가지 시나리오등을 한꺼번에 생성함

# -----------------------------------------------------------------

mrk_d = mx.MarketData()

scen1 = xen.Scenario(name, models, calcs, corr, timegrid, rsg, isMomentMatching)
scen1.toJson()

def toJson(self):
    json = dict()

    json['models'] = [m.toJson() for m in models]
    json['calcs'] = [c.toJson() for c in calcs]

    return json


sb = xen.ScenarioBuilder(json)


# 하고 싶은게 뭐냐면.

# 만들어진 시나리오를 그냥 또는 묶어서 저장하고 싶다.

xm = xen.XenarixManager(config)

## 그냥
scen1 = xen.Scenario(name, models, calcs, corr, timegrid, rsg, isMomentMatching)
xm.save(name, scen1) # save 내부에서 json 을 이용하여, serialize 한다음에 저장함. 밖에서는 알필요 없음.

## 묶은거
scenset1 = xen.ScenarioSet(name, [scen1, scen2])
xm.save(name, scenset1)

# 저장된 시나리오를 불러오고 싶다.
scen1 = xm.load(name) # load 내부에서 json을 불러와서 다시 생성해냄.

# 저장된 시나리오를 보고 싶다.
xm.get_list(filter='test')


# -----------------------------------
# MarketData랑 연동 된 모델(Shock같은거 처리나, mrk_d만 업데이트 해서 쓰는 용도)은 builder를 이용함.

sb = xen.ScenarioBuilder()

curve = mrk_d.get_curve('name')
curve.spread(0.0001)

gbm = xen.GBM(x0=100, rfCurve='irskrw', div='irskrw')
sb.add_model(gbm)




