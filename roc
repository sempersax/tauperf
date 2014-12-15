#!/usr/bin/env python
from tauperf.categories.hlt import Category_1P_HLT, Category_MP_HLT

import os
import logging
import rootpy
from rootpy.plotting import Graph, Canvas, Hist, Legend, Efficiency
from rootpy.plotting.style import set_style
from rootpy.extern.tabulartext import PrettyTable
from rootpy import ROOT

from tauperf import NTUPLE_PATH
from tauperf.analysis import Analysis
from tauperf.variables import VARIABLES
from tauperf.plotting import draw_shape, draw_efficiencies

log = logging.getLogger(os.path.basename(__file__))
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)
set_style('ATLAS', shape='rect')


ana = Analysis(ntuple_path=os.path.join(NTUPLE_PATH, 'training_11_12_2014'))

# TARGET_REJECTION = 0.9 # rej of 10
TARGET_REJECTION = 0.8571428571428572 # rej of 7

class working_point(object):
    def __init__(self, cut, eff_s, eff_b):
        self.cut = cut
        self.eff_s = eff_s
        self.eff_b = eff_b
    

def roc(category):
    sig_tot = ana.tau.events(category)[1].value
    bkg_tot = ana.jet.events(category, weighted=True)[1].value

    cut_vals = [0.02*i for i in xrange(50)]
    roc_gr = Graph(len(cut_vals))
    roc_list = []

    for i, val in enumerate(cut_vals):
        cut = 'hlt_bdt_score_pileup_corrected>{0}'.format(val)
        eff_sig = ana.tau.events(category, cut)[1].value
        eff_sig /= sig_tot
        eff_bkg = ana.jet.events(category, cut, weighted=True)[1].value
        eff_bkg /= bkg_tot
        rej_bkg = 1 - eff_bkg
        log.info((val, eff_sig, rej_bkg))
        roc_list.append(working_point(
                val, eff_sig, eff_bkg))
        roc_gr.SetPoint(i, eff_sig, rej_bkg)
    return roc_gr, roc_list

gr_sp, wp_sp = roc(Category_1P_HLT)    
gr_sp.color = 'red'
gr_sp.legendstyle = 'l'
gr_sp.linewidth = 3
gr_sp.title = '1 prong'
gr_sp.xaxis.title = '#epsilon_{S}' 
gr_sp.yaxis.title = '1 - #epsilon_{B}' 

gr_mp, wp_mp = roc(Category_MP_HLT)    
gr_mp.color = 'blue'
gr_mp.title = 'Multi prongs'
gr_mp.linewidth = 3
gr_mp.legendstyle = 'l'
gr_mp.xaxis.title = '#epsilon_{S}' 
gr_mp.yaxis.title = '1 - #epsilon_{B}' 


c = Canvas()
c.SetGridx()
c.SetGridy()
gr_sp.Draw('AL')
gr_mp.Draw('SAMEL')
leg = Legend([gr_sp, gr_mp])
# leg.SetNDC()
leg.Draw('same')
c.SaveAs('plots/roc.png')

TARGET_REJECTION = [6, 8, 10, 12, 14]
TARGET_1P = []
TARGET_MP = []
for target in TARGET_REJECTION:
    target_eff_b = 1. / target

    sorted_list = sorted(wp_sp, key=lambda wp: abs(wp.eff_b - target_eff_b))
    log.info(sorted_list[0])
    TARGET_1P.append(sorted_list[0])

    sorted_list = sorted(wp_mp, key=lambda wp: abs(wp.eff_b - target_eff_b))
    log.info(sorted_list[0])
    TARGET_MP.append(sorted_list[0])
    
    


def efficiencies_plot(category, working_points):
    vars = {'npv': VARIABLES['npv']}
    efficiencies = []
    for wp in working_points:
        cut = 'hlt_bdt_score_pileup_corrected>={0}'.format(wp.cut)
        hist_samples = ana.get_hist_samples_array(vars, 'hlt', category)
        hist_samples_cut = ana.get_hist_samples_array(vars, 'hlt', category, cut)
        eff = Efficiency(
            hist_samples_cut['npv']['tau'], hist_samples['npv']['tau'])
        eff.title = 'Rejection of {0:1.2f}'.format(1. / wp.eff_b)
        efficiencies.append(eff)
    c = draw_efficiencies(efficiencies, 'npv', category)
    return c

def score_plot(category):
    sig = ana.tau.get_hist_array(
        {'hlt_bdt_score': Hist(20, 0, 1)},
        category=category)
    bkg = ana.jet.get_hist_array(
        {'hlt_bdt_score': Hist(20, 0, 1)},
        category=category)
    hsig = sig['hlt_bdt_score']
    hbkg = bkg['hlt_bdt_score']
    plot = draw_shape(hsig, hbkg, 'BDT Score', category)
    return plot

plot_1P = score_plot(Category_1P_HLT)
plot_1P.SaveAs('plots/scores_1p.png')

plot_MP = score_plot(Category_MP_HLT)
plot_MP.SaveAs('plots/scores_mp.png')

plot_eff_1P = efficiencies_plot(Category_1P_HLT, TARGET_1P)
plot_eff_1P.SaveAs('plots/efficiencies_npv_optim_1p.png')

plot_eff_MP = efficiencies_plot(Category_MP_HLT, TARGET_MP)
plot_eff_MP.SaveAs('plots/efficiencies_npv_optim_mp.png')

table = PrettyTable(['Category', 'cut', 'signal efficiency', 'background rejection (1/eff_b)'])
for t in TARGET_1P:
    table.add_row([
            Category_1P_HLT.name, 
            t.cut, t.eff_s, 
            1. / t.eff_b if t.eff_b != 0. else -9999.])
for t in TARGET_MP:
    table.add_row([
            Category_MP_HLT.name, 
            t.cut, t.eff_s, 
            1. / t.eff_b if t.eff_b != 0. else -9999.])
log.info(40 * '=')
print table
log.info(40 * '=')
