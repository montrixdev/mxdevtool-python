import scenario.models.BK1F as bk1f
import scenario.models.CIR1F as cir1f
import scenario.models.GBM as gbm
import scenario.models.GBMConst as gbmconst
import scenario.models.GTwoExt as gtwoext
import scenario.models.Heston as heston
import scenario.models.HullWhite1F as hw1f
import scenario.models.MultipleModels as multimodels
import scenario.models.Vasicek1F as vasicek1f


if __name__ == "__main__":
    bk1f.test()
    cir1f.test()
    gbm.test()
    gbmconst.test()
    gtwoext.test()
    heston.test()
    hw1f.test()
    multimodels.test()
    vasicek1f.test()

    