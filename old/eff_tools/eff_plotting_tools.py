import ROOT

# -------------------------------------------------------------------------------------------------
class RejectionCurve(ROOT.TEfficiency):
    """ A class to convert a TEfficiency graph into a rejection graph """
    def __init__(self,eff):
        hnotpass = eff.GetTotalHistogram().Clone( eff.GetTotalHistogram().GetName()+"_notpass" )
        hnotpass.Add( eff.GetTotalHistogram(), eff.GetPassedHistogram(),1.,-1.)
        ROOT.TEfficiency.__init__(self,hnotpass,eff.GetTotalHistogram())
        return None


# -------------------------------------------------------------------------------------------------
class RoCcurve(ROOT.TGraph):
    """ A class to compute the RoC curve fron 2 TEfficiency object """
    def __init__(self, teff_s, teff_b):
        super(RoCcurve, self).__init__()#ROOT.TGraph.__init__(self)
        self._teff_s = teff_s
        self._teff_b = teff_b
        self.SetName(self._teff_s.GetName()+"_roc_curve")
        nbins_s = self._teff_s.GetTotalHistogram().GetNbinsX()
        nbins_b = self._teff_b.GetTotalHistogram().GetNbinsX()

        if nbins_s != nbins_b:
            print 'Signal and Bakground efficiencies are different !!!'

        for bin in range(1,nbins_s):
            self.SetPoint(bin-1, self._teff_s.GetEfficiency(bin),
                           1-self._teff_b.GetEfficiency(bin))
        self.GetXaxis().SetTitle("#epsilon_{S}")
        self.GetYaxis().SetTitle("1-#epsilon_{B}")
        self.SetLineWidth(4)
        return None


class SvsB_Perf_Canvas(ROOT.TCanvas):
    """ A class to plot Signal efficiency and backg rejection on the same canvas """

    def __init__(self,teff_s,teff_b,xtitle='BDT score'):
        super(SvsB_Perf_Canvas, self).__init__()
        #ROOT.TCanvas.__init__(self)
        self.eff = teff_s
        self.rej = RejectionCurve(teff_b)

        self.eff.SetLineColor(ROOT.kBlack)
        self.eff.SetMarkerColor(ROOT.kBlack)
        self.eff.SetMarkerStyle(ROOT.kFullSquare)

        self.rej.SetLineColor(ROOT.kRed)
        self.rej.SetMarkerColor(ROOT.kRed)
        self.rej.SetMarkerStyle(ROOT.kFullTriangleUp)
        self.rej.SetLineStyle(ROOT.kDashed)
        self.cd()
        self.eff.Draw("AP")
        self.rej.Draw("sameP")
        ROOT.gPad.Update()
        self.eff.GetPaintedGraph().GetXaxis().SetTitle(xtitle)
        self.eff.GetPaintedGraph().GetYaxis().SetTitle("Efficiency")
        self.eff.GetPaintedGraph().GetYaxis().SetRangeUser(0,1.05)
        ROOT.gPad.Update()
        self.right_axis = ROOT.TGaxis( ROOT.gPad.GetUxmax(),ROOT.gPad.GetUymin(),
                                       ROOT.gPad.GetUxmax(),ROOT.gPad.GetUymax(),0,1.05,510,"+L")
        self.right_axis.SetLineColor(ROOT.kRed)
        self.right_axis.SetLabelColor(ROOT.kRed)
        self.right_axis.SetTextColor(ROOT.kRed)
        self.right_axis.SetTitle("Rejection = 1 - #epsilon_{B}")
        self.right_axis.Draw("same")
        ROOT.gStyle.SetPadTickY(0)
        ROOT.gPad.Update()
        ROOT.gStyle.SetPadTickY(1)
        return None


    def GetOptimalCutValue(self, requested_bkg_rej_value):
        # --> more efficient with fine granularity TEfficiency object
        # --> need a TEfficiency object where high score correspond to signal-like events
        cut_value = 0
        bin = 1
        rej_bin = self.rej.GetEfficiency(bin)
        while rej_bin < requested_bkg_rej_value:
            rej_bin = self.rej.GetEfficiency(bin)
            bin +=1
        cut_value = 0.5*(self.rej.GetTotalHistogram().GetBinLowEdge(bin+1)+
                         self.rej.GetTotalHistogram().GetBinLowEdge(bin) )
        print cut_value,rej_bin
        return cut_value

# -------------------------------------------------------------------------------------------------
class EfficiencyPlot(ROOT.TCanvas):
    """ A class to draw several TEFFiciency graph on a TCanvas"""
    def __init__(self,name,title,eff_list,label_title):
        ROOT.TCanvas.__init__(self,name,title)
        self._eff_list = eff_list
        print str(eff_list)
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
    def SetXtitle(self,title):
        for eff_key in eff_list:
            self._eff_list.GetPaintedGraph().GetXaxis().SetTitle(title)
        ROOT.gPad.Update()


# -------------------------------------------------------------------------------------------------
class DiscriVar_Canvas(ROOT.TCanvas):
    """ A class to plot histogram distribution of sig and bkg """

    def __init__(self,args):
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
