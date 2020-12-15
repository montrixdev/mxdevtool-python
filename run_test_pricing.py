import pricing.ELSStepDown as stepdown
import pricing.ExoticOption as exotic
import pricing.CCP_SwapCurve as ccpswap
import pricing.Interpolation as interp
import pricing.IRS_Calculator as irscalc
import pricing.Swaption as swaption
import pricing.VanillaOption as vanillaoptioncalc
import pricing.VanillaOptionGraph as vanillaoptiongraphcalc


if __name__ == "__main__":
    stepdown.test()
    exotic.test()
    ccpswap.test()
    interp.test()
    irscalc.test()
    swaption.test()
    vanillaoptioncalc.test()
    vanillaoptiongraphcalc.test()




