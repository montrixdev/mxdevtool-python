import mxdevtool as mx
import mxdevtool.xenarix as xen
import os, json
# 목적?
# 시장데이터? , ㄴㄴ
# xeninput.json 을 만들고, 그거를 통해서 클래스를 만드는 것( serialization )
# 필요함? ㄴㄴ - 나중에

# shock을 주고 하는 것.
# 필요함? 음...
# helper가 하는 일이 뭐지...? 

# 이거 할일이 생각보다 많은데...?
# shock

# shocked 된 거라던지 한꺼번에 가지고 있는 놈
# serialize, deserialize
# 여기서 모델 수정하고 자시고 ㄱㄱ
class ScenarioInputJson:
    def __init__(self, json_str):
        self.__dict__ = json.loads(json_str)


class MarketDataJson:
    def __init__(self, json_str):
        self.__dict__ = json.loads(json_str)


class ScenarioTemplate:
    def __init__(self, json_str):
        self.json_str = json_str
        self.initialize()

    def initialize():
        # 두부분으로 나눔 (scen set or scen ? 무조건 set으로 들어옴?)
        ## scenario 와 market_data
        input_json = ScenarioInputJson(self.json_str)
        
        scen_list = []

        for scen in input_json.scen_list:
            marketData = scen.marketData

            # models
            models = []
            for m in scen.models:
                models.append(self._model(m, marketData))
            
            calcs = []
            for c in scen.calcs:
                calcs.append(self._calc(c, marketData))
            
            corr = mx.Matrix(scen.corr)

            timegrid = _timegrid(scen.timegrid, marketData)

            scen = xen.ScenarioGenerator(models, calcs, corr, timegrid, rsg, filename, isMomentMatching)

            scen_list.append(scen)

        return scen_list
    
    def _data_read(self, mrk_d):
        pass

    def _model(self, m, mrk_d):
        mrk_d = self.marketData
        typ_lower = m.type.lower()
        model = None

        if typ_lower == xen.GBMConst.__name__.lower():
            x0 = mrk_d.get_data(m.x0)
            rf = mrk_d.get_data(m.rf)
            div = mrk_d.get_data(m.div)
            vol = mrk_d.get_data(m.vol)

            model = xen.GBMConst(x0, rf, div, vol)
        elif typ_lower == xen.GBM.__name__.lower():
            x0 = mrk_d.get_data(m.x0)
            rfCurve = mrk_d.get_data(m.rfCurve)
            divCurve = mrk_d.get_data(m.divCurve)
            volTs = mrk_d.get_data(m.volTs)

            model = xen.GBMConst(x0, rfCurve, divCurve, volTs)
        elif typ_lower == xen.HullWhite1F.__name__.lower():
            fittingCurve = mrk_d.get_data(m.fittingCurve)
            alphaPara = mrk_d.get_data(m.alphaPara)
            sigmaPara = mrk_d.get_data(m.sigmaPara)

            model = xen.HullWhite1F(fittingCurve, alphaPara, sigmaPara)
        else:
            raise Exception('unknown model - {0}'.format(m.type))

        return model
            
    def _calc(self, c, mrk_d):
        def get_model_or_calc(name):
            pv = None
            if c.pv in self.models:
                pv = self.models[c.pv]
            elif c.pv in self.calcs:
                pv = self.calcs[c.pv]
            else:
                Exception('no exist model or calc : '.format(c.pv))

            return pv

        typ_lower = c.type.lower()
        calc = None

        if typ_lower == xen.LinearOper.__name__.lower():
            pv = get_model_or_calc(c.pv)
            calc = xen.LinearOper(c.name, pv, c.multiple, c.spread)
        # elif c.type == xen.Shift.__name__.lower():
        #     pv = get_model_or_calc(c.pv)
        #     calc = xen.Shift(c.name, pv, c.shift, c.fill)
        else:
            raise Exception('unknown calc - {0}'.format(c.type))

        return calc

    def _timegrid(self, tg, mrk_d):
        timegrid = None
        typ_lower = tg.type.lower()
        refDate = mx.Date(tg.refDate)
        if typ_lower == xen.TimeGrid.__name__.lower():
            timegrid = xen.TimeGrid(refDate, tg.maxYear, tg.frequency, tg.frequency_month, tg.frequency_day)
        elif typ_lower == xen.TimeEqualGrid.__name__.lower():
            timegrid = xen.TimeEqualGrid(refDate, tg.maxYear, tg.nPerYear)
        elif typ_lower == xen.TimeArrayGrid.__name__.lower():
            timegrid = xen.TimeArrayGrid(refDate, tg.times)
        else:

    def make_scenario(self):
        pass


# 이거는 생각보다 구현할게 많다요.
# 모델 더하고 빼고, 기존꺼 name exist 등..
# 이거는 하나의 scenario 임 
# list는?

class Template:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def make_model(self):
        _typ = self.kwargs['type'].lower()

        if _typ == xen.GBMConst.__name__.lower():
            return 
        pass

    def make_calc(self):
        pass


class ScenarioTemplateBuilder:
    def __init__(self, marketData):
        self.marketData = marketData

    def gbmconst(self, name, x0, rf, div, sigma):
        _type = xen.GBMConst.__name__.lower()
        template = Template(name=name, type=_type, x0=x0, rf=rf, div=div, sigma=sigma)

        return template

    def gbm(self, name, x0, rfCurve, divCurve, sigmaCurve):
        pass

    def make_scenario(self):



# history, connect to db, upload, download
# db or file ?
class XenarixManager:
    def load(self, name, key, refDate):
        pass

    def save(self, name, scen):
        pass

    def get_list(self, filter=None):
        pass


# market template 처리?
# 저장할 때 
class XenarixFileManager(XenarixManager):
    def __init__(self, config):
        self.config = config

    # 저장할때는 toJson 만 있으면 처리됨
    # toJson을 어떻게 하지...? 
    # 기존꺼 ProcessValue에는 넣는게 가능
    # template_builder 클래스(toJson을 뚫음)를 만들어야함. 이건 나중에
    def save(self, name, scen):
        json_str = scen.toJson()
        path = os.path.join(self.config['location'], name + '.xen')

        f = open(path, "a")
        f.write(json_str)
        f.close()

    # 로드 할때는 template 이던 아니던 같이 처리가 됨.
    def load(self, name, key, refDate):
        path = os.path.join(self.config['location'], name + '.xen')
        f = open(path, "r")
        json_str = f.read()

        sb = ScenarioBuilder(json_str)

        scen = sb.get_scenario()

        # 기존의 market이랑 합치는거는 builder에서 함.
        return scen
        
        mrk_d = Markdata(refDate) # 걍 전체를 떠. 거기서 가져와

        # kospi2 = mrk_d['kospi2']
        
        scen_json = {}

        pass

    def load_scenario_set(self):
        pass    


class DataManager:
    def __init__(self, config):
        self.location = config['location']

    def initialize(self):
        

    def get_marketdata(self, name, refDate):

        pass


    


# 자 니가 여러개를 set로 돌릴거야... 
# 그러면 어떻게 해야함?  

# 머머 돌릴건데...?, 아 기본은 이거고 여기다가 요렇게 shock 준거랑, 요래 shock준거랑

# 기본?

config = {
    'ip': '192.168.0.1',
    'hostname': 'test',
    'dbname': 'testdb',
    'type': 'database',
    'id': 'tekjkd',
    'pw': 'testt'
}

xm = XenarixTestManager(config=config) # helper를 만들어
scen = xm.load(name='hedge', key='test1', refDate=mx.Date(2020, 12, 21)) # 오늘자 시나리오를 가져와(미리 세팅된)
xm.save(name, key, scen)
xm.save(name, key, scenSet)

shock = { 
    'kospi2': {
        'type': 'mul', 
        'value': 0.1
            }
        }

scen1 = scen.clone(shock) # scenario에다가 shock을 주는 형식 ?

mrk_d = Markdata()
mrk_d1 mrk_data.clone(shock) # 시장데이터에다가 shock을 주는 형식 ? 

# 시나리오 세트를 만들어
ss = ScenarioSet(baseScen=scen)
ss.add(name='scen1', scen1)


scen1 = xen.Scenario(filename1, models, calcs)
scen2 = xen.Scenario(filename2, models, calcs)

scen1 = xen.ScenarioGenerator(models, calcs, corr, timegrid, rsg, filename, isMomentMatching)


for data in marketdata:
    # upshock = marketdata['upshock']

    kospi2 = build_gbm('kospi2', data['kospi2'])
    sp500 = build_gbm('sp500', data['sp500'])

    models = [kospi2, sp500]
    scen = xen.ScenarioGenerator()
    scen_list.append(scen)


for scen in scen_list:
    scen.generate()


# 완성되면 이건 여러가지 시나리오등을 한꺼번에 생성함

