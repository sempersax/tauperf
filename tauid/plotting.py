# root/rootpy imports
import ROOT
from rootpy import asrootpy
from rootpy.io import root_open
from rootpy.memory import keepalive
from rootpy.plotting.utils import draw
from rootpy.plotting import Hist, Efficiency, Graph, Canvas
# local imports
from . import VARIABLES, log
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
        eff_array[var['name']] = ROOT.TEfficiency(h.name, get_label(var), var['bins'], var['range'][0], var['range'][1])
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

def rejection(eff):
    htot = asrootpy(eff.GetTotalHistogram()).Clone()
    hpass = asrootpy(eff.GetPassedHistogram())
    hnotpass =  htot - hpass
    rej = Efficiency(hnotpass, htot, name='Rej_{0}'.format(eff.name), title=eff.title)
    return rej

