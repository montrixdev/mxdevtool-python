# excel link : https://blog.naver.com/montrix/221359670804

import mxdevtool as mx
import mxdevtool.xenarix as xen

def test():
	print('pseudo random test...')

	scenario_num = 1000
	dimension = 100
	seed = 1
	skip = 7
	isMomentMatching = False
	randomType = "pseudo"
	subType = "mersennetwister"
	randomTransformType = "boxmullernormal"

	rsg = xen.Rsg(scenario_num, dimension, seed, skip, isMomentMatching, 
				randomType, subType, randomTransformType)
    
	print(randomType, len(rsg.nextSequence()))

if __name__ == "__main__":
    test()