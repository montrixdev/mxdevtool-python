# excel link : https://blog.naver.com/montrix/221371425358

import mxdevtool as mx
import mxdevtool.xenarix as xen

def test():
	print('sobol random test...')

	scenario_num = 1000
	dimension = 365
	seed = 0
	skip = 1024
	isMomentMatching = False
	randomType = "sobol"
	subType = "joekuod7"
	randomTransformType = "invnormal"

	rsg = xen.Rsg(scenario_num, dimension, seed, skip, isMomentMatching, 
				randomType, subType, randomTransformType)

	print(randomType, len(rsg.nextSequence()))
	

if __name__ == "__main__":
    test()