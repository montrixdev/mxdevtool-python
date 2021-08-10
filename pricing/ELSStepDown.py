import numpy as np
import mxdevtool as mx
import mxdevtool.xenarix as xen
import mxdevtool.termstructures as ts

filename = './test_stepdown.npz'
refDate = mx.Date(2012,8,22)
riskFree = 0.0307

class StepDownPayoff:
	def __init__(self, notional, issue_date, maturity_date, initial_values, ki, ki_flag, coupons):
		self.notional = notional
		self.issue_date = issue_date
		self.maturity_date = maturity_date
		self.initial_values = initial_values
		self.ki = ki
		self.ki_flag = ki_flag
		self.coupons = coupons

		# t_pos for calculation
		self.coupon_tpos = []
		self.discount_factors = []

	def initialize_timeGrid(self, timeGrid):
		if not isinstance(timeGrid, mx.core_TimeGrid):
			raise Exception('timeGrid is required')

		for cpn in self.coupons:
			d = cpn[0]
			t_pos = timeGrid.closestIndex_Date(d)
			self.coupon_tpos.append(t_pos)

	def precalculation_discountFactors(self, discountCurve):
		if not isinstance(discountCurve, mx.YieldTermStructure):
			return

		if len(self.discount_factors) > 0:
			return

		for cpn in self.coupons:
			d = cpn[0]
			self.discount_factors.append(discountCurve.discount(d))

	def get_min_return(self, multi_path, t_pos):
		min_return = 1.0

		for i, initial_value in enumerate(self.initial_values):
			min_return = min(min_return, multi_path[i][t_pos] / initial_value)

		return min_return

	def check_ki(self, multi_path):
		if self.ki_flag:
			return True

		for i, initial_value in enumerate(self.initial_values):
			min_return = np.min(np.array(multi_path[i]) / initial_value)
			if min_return <= self.ki:
				return True

		return False

	def value(self, multi_path, discount):
		self.precalculation_discountFactors(discount)

		for cpn, t_pos, disc in zip(self.coupons[:-1], self.coupon_tpos[:-1], self.discount_factors[:-1]):
			min_return = self.get_min_return(multi_path, t_pos)
			ex_level = cpn[1]
			if min_return >= ex_level: # early exercise
				rate = cpn[2]
				return self.notional * (1.0 + rate) * disc



		last_cpn = self.coupons[-1]
		last_t_pos = self.coupon_tpos[-1]
		last_disc = self.discount_factors[-1]

		min_return = self.get_min_return(multi_path, last_t_pos)

		if min_return >= last_cpn[1]: # last exercise
			return self.notional * (1.0 + last_cpn[2]) * last_disc
		else:
			if self.check_ki(multi_path):
				return self.notional * min_return * last_disc
			else:
				return self.notional * (1.0 + last_cpn[2]) * last_disc


def build_stepdown():
	notional = 10000
	issue_date = mx.Date(2012,8,22)
	maturity_date = mx.Date(2012,8,22)

	initial_values = [387833, 27450]

	ki = 0.35
	ki_flag = False

	coupons = [(mx.Date(2013,2,13), 0.9, 0.06),
			(mx.Date(2013,8,13), 0.9, 0.12),
			(mx.Date(2014,2,13), 0.85, 0.18),
			(mx.Date(2014,8,13), 0.85, 0.24),
			(mx.Date(2015,2,13), 0.8, 0.30),
			(mx.Date(2015,8,13), 0.8, 0.36)]

	return StepDownPayoff(notional, issue_date, maturity_date, initial_values, ki, ki_flag, coupons)


def build_scenario(overwrite=True):
	print('stepdown test...', filename)

	if not overwrite:
		return

	initialValues = [387833, 27450]
	dividends = [0.0247, 0.0181]
	volatilities = [0.2809, 0.5795]

	gbmconst1 = xen.GBMConst('gbmconst1', initialValues[0], riskFree, dividends[0], volatilities[0])
	gbmconst2 = xen.GBMConst('gbmconst2', initialValues[1], riskFree, dividends[1], volatilities[1])

	models = [gbmconst1, gbmconst2]
	corr = 0.6031

	corrMatrix = mx.IdentityMatrix(len(models))
	corrMatrix[0][1] = corr
	corrMatrix[1][0] = corr

	timeGrid = mx.TimeEqualGrid(refDate, 3, 365)

	# random
	rsg = xen.Rsg(sampleNum=5000)
	xen.generate(models, None, corrMatrix, timeGrid, rsg, filename, False)


def pricing():
	results = xen.ScenarioResults(filename)

	payoff = build_stepdown()
	payoff.initialize_timeGrid(results.timegrid)

	simulNum = results.simulNum
	discount_curve = ts.FlatForward(refDate, 0.0307)

	v = 0

	for i in range(simulNum):
		path = results[i]

		v += payoff.value(path, discount_curve)

		if i != 0 and i % 5000 == 0:
			print(i, v / i)

	print(simulNum, v / simulNum)


def test():
	mx.Settings.instance().setEvaluationDate(refDate)

	build_scenario(overwrite=True)
	pricing()

if __name__ == "__main__":
	test()

	#mx.npzee_view(filename)