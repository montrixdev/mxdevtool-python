# shock
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