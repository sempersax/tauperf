from rootpy.plotting import Efficiency, Graph, Canvas
from rootpy import asrootpy

import ROOT

# -------------------------------------------------------------------------------------------------
class RejectionCurve(Efficiency):
    """ A class to convert a Efficiency graph into a rejection graph """
    def __init__(self, eff, **kwargs):
        hnotpass = eff.GetTotalHistogram().Clone()
        hnotpass.Add(eff.GetTotalHistogram(), eff.GetPassedHistogram(), 1., -1.)
        super(RejectionCurve, self).__init__(hnotpass, eff.GetTotalHistogram(), **kwargs)


# -------------------------------------------------------------------------------------------------
class RoC(Graph):
    """ A class to compute the RoC curve fron 2 Efficiency object """
    def __init__(self, eff_s, eff_b, **kwargs):
        if len(eff_s)!=len(eff_b):
            raise RuntimeError('the lenghts of the two eff graphs are differents !')
        super(RoC, self).__init__(len(eff_s), **kwargs)
        for i, (s, b) in enumerate(zip(eff_s.efficiencies(), eff_b.efficiencies())):
            self.SetPoint(i, s, 1-b)
            
# -------------------------------------------------------------------------------------------------
class SvsB_Canvas(Canvas):
    """ A class to plot Signal efficiency and backg rejection on the same canvas """
    def __init__(self, teff_s, teff_b, xtitle='BDT score', **kwargs):
        super(SvsB_Canvas, self).__init(**kwargs)
        self.eff = teff_s
        self.rej = RejectionCurve(teff_b)
        self.eff.linecolor = 'black' 
        self.eff.markercolor = 'black'
        self.eff.markerstyle = 'square'
        self.rej.linecolor = 'red' 
        self.rej.linestyle = 'dashed' 
        self.rej.markercolor = 'red'
        self.rej.markerstyle = 'triangleup'
        self.eff.Draw("AP")
        self.rej.Draw("sameP")
        ROOT.gPad.Update()
        self.eff.GetPaintedGraph().GetXaxis().SetTitle(xtitle)
        self.eff.GetPaintedGraph().GetYaxis().SetTitle("Efficiency")
        self.eff.GetPaintedGraph().GetYaxis().SetRangeUser(0,1.05)
        ROOT.gPad.Update()
        self.right_axis = ROOT.TGaxis(ROOT.gPad.GetUxmax(), ROOT.gPad.GetUymin(),
                                      ROOT.gPad.GetUxmax(), ROOT.gPad.GetUymax(), 0, 1.05, 510, "+L")
        self.right_axis.SetLineColor(ROOT.kRed)
        self.right_axis.SetLabelColor(ROOT.kRed)
        self.right_axis.SetTextColor(ROOT.kRed)
        self.right_axis.SetTitle("Rejection = 1 - #epsilon_{B}")
        self.right_axis.Draw("same")
        ROOT.gStyle.SetPadTickY(0)
        ROOT.gPad.Update()
        ROOT.gStyle.SetPadTickY(1)

# -------------------------------------------------------------------
class EfficiencyPlot(Canvas):
    """ A class to draw several TEFFiciency graph on a TCanvas"""
    def __init__(self, eff_list, **kwargs):
        super(EfficiencyPlot, self).__init__(**kwargs)
        self._eff_list = eff_list
        if isinstance(eff_list, (list, tuple)):
            self._eff_list = eff_list
        else:
            self._eff_list = [eff_list]
        
        # TODO: CONTINUE TO ROOTPY-IFY THIS
        self.legend = ROOT.TLegend(0.70,0.77,0.90,0.94)
        self.legend.SetFillColor(0)
        self.label = ROOT.TLatex()
        self.label.SetNDC()
        self.cd()
        n_plotted_eff = 0
        for eff_key in self._eff_list:
            if n_plotted_eff==0:
                self._eff_list[eff_key][0].Draw("AP")
                n_plotted_eff += 1
            else: self._eff_list[eff_key][0].Draw("SAMEP")
            ROOT.gPad.Update()
            self._eff_list[eff_key][0].SetLineColor(self._eff_list[eff_key][1])
            self._eff_list[eff_key][0].SetMarkerColor(self._eff_list[eff_key][1])
            self._eff_list[eff_key][0].SetMarkerStyle(self._eff_list[eff_key][2])
            self._eff_list[eff_key][0].GetPaintedGraph().GetXaxis().SetTitle(self._eff_list[eff_key][3])
            self._eff_list[eff_key][0].GetPaintedGraph().GetYaxis().SetTitle(self._eff_list[eff_key][4])
            self._eff_list[eff_key][0].GetPaintedGraph().GetYaxis().SetRangeUser(0,1.05)
            self.legend.AddEntry( self._eff_list[eff_key][0] , self._eff_list[eff_key][5],"lp")
        self.legend.Draw("same")
        self.RedrawAxis()
        self.label.DrawLatex(0.5,0.5,label_title)
        ROOT.gPad.Update()

        return None
    def SetXtitle(self, title):
        for eff in eff_list:
            self._eff_list.GetPaintedGraph().GetXaxis().SetTitle(title)
        ROOT.gPad.Update()


# -------------------------------------------------------------------------------------------------
class DiscriVar_Canvas(ROOT.TCanvas):
    """ A class to plot histogram distribution of sig and bkg """

    def __init__(self, args):
        """ args = [sig_file,bkg_file,histname,hist_title]"""
        ROOT.TCanvas.__init__(self,args[2],args[3])
        self.hsig = args[0].Get(args[2])
        self.hbkg = args[1].Get(args[2])
        # --> hist style 
        self.hsig .SetLineColor( ROOT.kRed  ) 
        self.hbkg .SetLineColor( ROOT.kBlue ) 
        self.hbkg .SetLineStyle( ROOT.kDashed ) 
        # --> hist axis labels
        self.hsig .GetXaxis().SetTitle( args[3] )
        self.hbkg .GetXaxis().SetTitle( args[3] )
        self.hsig .GetYaxis().SetTitle( "Arbitrary Units" )
        self.hbkg .GetYaxis().SetTitle( "Arbitrary Units" )
        # --> hist normalization
        self.hsig.Sumw2()
        self.hbkg.Sumw2()
        self.hsig.Scale ( 1./self.hsig.Integral() )
        self.hbkg.Scale ( 1./self.hbkg.Integral() )
        AnalysisTools.SetMaximumHist([self.hsig,self.hbkg])
        # --> hist legend        
        self.legend = ROOT.TLegend(0.14,0.18,0.55,0.40)
        self.legend.SetFillColor(0)
        self.legend.AddEntry( self.hsig,"Signal" ,"l")
        self.legend.AddEntry( self.hbkg,'Background', 'l')
        self.cd()
        self.hsig.Draw("HIST")
        self.hbkg.Draw("SAMEHIST")
        self.legend.Draw("same")
        ROOT.gPad.Update()
        return None
