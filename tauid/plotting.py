# root/rootpy imports
from rootpy import ROOT
from rootpy.io import root_open
from rootpy.plotting import Hist, Efficiency, Graph
# local imports
from . import VARIABLES
from .variables import get_label
from tools.datasets import SIGNALS_14TEV, VERSION, DATASETS

def get_hist_array():
    hist_array = {}
    for var in VARIABLES['plotting']+VARIABLES['plotting_id']:
        hist_array[var['name']] = Hist(var['bins'], var['range'][0], var['range'][1])
        hist_array[var['name']].xaxis.title = get_label(var)
    return hist_array

def get_efficiency_array():
    eff_array = {}
    for var in VARIABLES['plotting']:
        h = Hist(10, 0, 1)
        eff_array[var['name']] = ROOT.TEfficiency(h.name, h.title, var['bins'], var['range'][0], var['range'][1])
    return eff_array

def get_mean_rms(category, var):
    gr_mean = Graph(len(SIGNALS_14TEV))
    gr_rms = Graph(len(SIGNALS_14TEV))
    for ip, signal in enumerate(SIGNALS_14TEV):
        with root_open('efficiencies/eff_presel_{0}_v{1}.root'.format(signal, VERSION)) as fsig:
            h_s = fsig[category].Get('h_'+category+'_'+var['name'])
            gr_mean.SetPoint(ip, DATASETS[signal]['mu'], h_s.GetMean())
            gr_mean.SetPointError(ip, 0, 0, h_s.GetMeanError(), h_s.GetMeanError())
            gr_rms.SetPoint(ip, DATASETS[signal]['mu'], h_s.GetRMS())
            gr_rms.SetPointError(ip, 0, 0, h_s.GetRMSError(), h_s.GetRMSError())
    gr_mean.xaxis.title = 'Average Interactions Per Bunch Crossing'
    gr_mean.yaxis.title = 'Mean of '+get_label(var)
    gr_rms.xaxis.title = 'Average Interactions Per Bunch Crossing'
    gr_rms.yaxis.title = 'RMS of '+get_label(var)
    return gr_mean, gr_rms

class RejectionCurve(ROOT.TEfficiency):
    """ A class to convert a TEfficiency graph into a rejection graph """
    def __init__(self,eff):
        hnotpass = eff.GetTotalHistogram().Clone( eff.GetTotalHistogram().GetName()+"_notpass" )
        hnotpass.Add( eff.GetTotalHistogram(), eff.GetPassedHistogram(),1.,-1.)
        ROOT.TEfficiency.__init__(self,hnotpass,eff.GetTotalHistogram())
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
