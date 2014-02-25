from ROOT import TMath
from ROOT import TH1

class SignificanceHist:
    """A class to compare an expected and an observed histrogram"""
    def __init__(self,h1,h2):
        self._hobs = h1
        self._hexp = h2

    # --------------------------------------------------------------
    def GetSignificanceHist(self,sigma):
        # ---> from arXiv:1111.2062v2
        hh_signi = self._hobs.Clone("hh_signi")
        for ibin in range(1,hh_signi.GetNbinsX()):
            obs  = self._hobs.GetBinContent(ibin)
            exp  = self._hexp.GetBinContent(ibin)
            sig  = -9999999.
            if obs>exp:
                Q = ROOT.Math.inc_gamma_c(obs,exp)
                pval = 1-Q
                if(pval>0.5): sig = 0
                else: sig = sqrt(2.)*TMath.ErfInverse(1.-2*pval)
            else:
                Q = ROOT.Math.inc_gamma_c(obs+1,exp)
                pval = Q
            if(pval>0.5): sig = 0
            else: sig = -sqrt(2.)*TMath.ErfInverse(1.-2*pval)
            hh_signi.SetBinContent(ibin,sig)
            hh_signi.SetBinError(ibin,0)
        # --> End of the for loop
        hh_signi.GetYaxis().SetTitle("Significance")
        hh_signi.GetYaxis().SetRangeUser(-1*sigma,1*sigma)
        hh_signi.SetFillColor(2) # Color convention for the stat only error

        # --> Some drawing commands to fit a low panel plot
        hh_signi.SetTitleSize(0.11,"Y")
        hh_signi.SetTitleOffset(0.60,"Y")
        hh_signi.SetTitleSize(0.11,"X")
        hh_signi.SetLabelSize(0.11,"X")
        hh_signi.SetLabelSize(0.11,"Y")
        hh_signi.GetYaxis().SetNdivisions(6)
        return hh_signi
    # --------------------------------------------------------------
    def GetSignificanceHist_WithSyst(self,sigma):
        # ---> from arXiv:1111.2062v2
        # ---> The expected histogram should have correct error bars
        # ---> TO BE IMPLEMENTED FROM THE C++ DIPHOTON ANALYSIS 
        print "TO BE IMPLEMENTED FROM THE C++ DIPHOTON ANALYSIS" 
        hh_signi = self._hobs.Clone("hh_signi")
        return hh_signi
    # --------------------------------------------------------------
    def GetChiHist(self,sigma):
        hh_chi =  self._hobs.Clone("hh_chi")
        for ibin in range (1,hh_chi.GetNbinsX()):
            obs       = self._hobs.GetBinContent(ibin)
            exp       = self._hexp.GetBinContent(ibin)
            sigma_exp = sqrt(exp)
            chi       = (obs-exp)/sigma_exp
            if sigma_exp !=0 : hh_chi.SetBinContent(ibin,chi)
            else: hh_chi.SetBinContent(ibin,0)
        hh_chi.SetBinError(ibin,0)
        hh_chi.GetYaxis().SetTitle("#chi")
        hh_chi.GetYaxis().SetRangeUser(-1*sigma,1*sigma)
        hh_chi.SetFillColor(2)
        hh_chi.SetTitleSize(0.11,"Y")
        hh_chi.SetTitleOffset(0.60,"Y")
        hh_chi.SetTitleSize(0.11,"X")
        hh_chi.SetLabelSize(0.11,"X")
        hh_chi.SetLabelSize(0.11,"Y")
        hh_chi.GetYaxis().SetNdivisions(6)
        return hh_chi
    # --------------------------------------------------------------
    def GetRatioHist(self):
        hh_ratio =  self._hobs.Clone("hh_ratio")
        hh_ratio.Sumw2()
        hh_ratio.Divide(self._hexp)
        hh_ratio.GetYaxis().SetTitle("Data/Background")
        # hh_ratio.GetYaxis().SetRangeUser(1-deviation,1+deviation)
        hh_ratio.SetTitleSize(0.11,"Y")
        hh_ratio.SetTitleOffset(0.60,"Y")
        hh_ratio.SetTitleSize(0.11,"X")
        hh_ratio.SetLabelSize(0.11,"X")
        hh_ratio.SetLabelSize(0.11,"Y")
        hh_ratio.GetYaxis().SetNdivisions(6)
        min_range = hh_ratio.GetBinContent(hh_ratio.GetMinimumBin())
        max_range = hh_ratio.GetBinContent(hh_ratio.GetMaximumBin())
        hh_ratio.GetYaxis().SetRangeUser( min_range*(0.9), max_range*(1.1) )
        return hh_ratio

