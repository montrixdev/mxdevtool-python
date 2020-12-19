import mxdevtool as mx
import mxdevtool.xenarix as xen
import numpy as np

# 기준일 설정
ref_date = mx.Date.todaysDate()

# 모델 생성
gbmconst1 = xen.GBMConst('gbmconst1', x0=100, rf=0.02, div=0.005, vol=0.25)
gbmconst2 = xen.GBMConst('gbmconst2', x0=200, rf=0.015, div=0.003, vol=0.2)
vasicek = xen.Vasicek1F('vasicek', r0=0.015, alpha=0.1, longterm=0.04, sigma=0.01)

models = [gbmconst1, gbmconst2, vasicek]

# 모델간 상관계수 설정
corrMatrix = mx.IdentityMatrix(len(models))

gbmconst1_gbmconst2_corr = 0.3
corrMatrix[1][0] = gbmconst1_gbmconst2_corr 
corrMatrix[0][1] = gbmconst1_gbmconst2_corr

gbmconst1_vasicek_corr = 0.1
corrMatrix[2][0] = gbmconst1_vasicek_corr 
corrMatrix[0][2] = gbmconst1_vasicek_corr

# 시간 간격 및 최대 생성 구간 설정
timeGrid = mx.TimeEqualGrid(ref_date, 3, 365)

# random
filename = './multipleassets.npz'
rsg = xen.Rsg(sampleNum=5000)
results = xen.generate(models, None, corrMatrix, timeGrid, rsg, filename, False)

# 결과 로드
results = xen.ScenarioResults(filename)
data = results.toNumpyArr()

print(data)



