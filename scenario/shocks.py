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


# 하고 싶은게 뭐냐면 -----------------------------------------------
# 전체적으로 모양을 그리고 차근차근함.

# 첫번째 단계

# 1. 만들어진 시나리오를 그냥 또는 묶어서 저장하고 싶다.
xm = xen.XenarixManager(config)

## 하나만 저장 1
scen1 = xen.Scenario(name, models, calcs, corr, timegrid, rsg, isMomentMatching)
xm.save(name, scen1) # save 내부에서 json 을 이용하여, serialize 한다음에 저장함. 밖에서는 알필요 없음.

## 하나만 저장 2 - market data 같이 들어있는 것
stb = ScenarioTemplateBuilder(marketdata)

gbmconst1 = stb.gbmconst(name='gbm1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')
gbm1 = stb.gbm(name='gbm1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')

models = [gbmconst1, gbm1]
linearOper1 = stb.calc.linearOper(name='linearOper1', multiple=1.0, spread=0.0)

calcs = [linearOper1]

scen2 = stb.make_template(name, models, calcs, corr, timegrid, rsg, isMomentMatching)

scen2.get_scenario()


## 묶은거 저장
xm.save(name, [scen1, scen2])

# 2. 저장된 시나리오를 불러오고 싶다.
scen1 = xm.load(name) # load 내부에서 json을 불러와서 다시 생성해냄.

# 3. 저장된 시나리오를 보고 싶다.
xm.get_list(filter='test')

# 이건 두번째 단계

# 1. template을 만들어서 market 데이터만 넣고 하고 싶다.

# 2. market 데이터를 가져오고 싶다. (이건 좀 큼)


# -----------------------------------
# MarketData랑 연동 된 모델(Shock같은거 처리나, mrk_d만 업데이트 해서 쓰는 용도)은 builder를 이용함.
# 목적 : market데이터만 바꿔서 넣으려는 것 
# 처리방법 선택의 기준 (직관적이고, 단순함(목적만 딱 해결), 지금 필요함? )

# 처리방법 1. 이런 builder class 로 작업하는 것. (복잡 사용법 알아야함)
# json을 입력 데이터로 사용함. builder로 만드는 건... 2번 처리방법이랑 같은 거 아닌가...?
#
# 목적 세부 : 만들어지면 어케 되지...? 뭘 하지 이걸로...? scenario template을 
#             만들어서 mrk 데이터를 바꿀꺼다. 그게 자동으로 된다...?
#             한번 만들어논 template 코드를 만드는데 편하다...? 다시 template을 만드는데 편하다.
#             그 editor를 만들 수 있다. 
#             그냥 json으로 한번 만들어주고, 사용하는게 editor의 중간단계임.
#             중간단계를 먼저 하고, viewer하고 editor로 가는 단계에서 작업을 하면 됨.
#             그래서 여기서 먼저 할 것은 build_from_json 만 작성하면 될 듯하다.
# 

sb = xen.ScenarioBuilder(mrk_d)

curve = mrk_d.get_curve('name')
curve.spread(-0.0001)

# 이거는 방법이 잘 안보임... 
# 두개 기능을 넣으면 나중에 헷갈릴 수도 있으니...
gbm = xen.GBM(x0=100, rfCurve='irskrw', div='irskrw')
sb.add_model(gbm)

# 이거는 json을 생성함 내부에서 가지고 있다가
# 아 add 방식이 아니고 
# args
gbmconst1 = xen.GBMConst.makeJson(name='gbmconst1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')
gbm1 = xen.GBM.makeJson(name='gbm1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')
gbm1 = xen.GBM.getJson(name='gbm1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')

model_templates = [gbmconst1, gbm1]

calc1 = sb.get_calc(name='calc1', x0=100, rf='irskrw{3m}', div='irskrw{3m}')

sb.add_model_gbm(name='gbm', x0=100, rfCurve='irskrw', divCurve='irskrw')

# 처리방법 2. 그냥 코드로 짜로 input에 market data 넣는 부분이랑 mapping 함. (사용자가 그냥 짜면됨)
# 목적 세부 : 그냥 구축할때 사용가능(그냥 이렇게 해도, 별 무리 없을 듯한데...? 시스템 구축이니까)
#             이거는 실제로 지금 할 필요는 없고, 필요할 때 그냥 만들면됨.
#             1번 처리방법을 자동화하는 작업이거든. 그래서 이것의 코드는 일반화 되어서 1번의 
#             처리방법에서 사용됨
#             
# market을 로드해
mrk_d = Marketdata()

# 내가 mapping 하고 싶은 데이터 예) gbm1 은 kospi2에 함
gbm = xen.GBMConst(name='gbmconst', x0=mrk_d['kospi2'], rf=mrk_d['irskrw']['3m'])
models = [gbm]

xen.generate1d('...')






