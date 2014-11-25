#!/usr/bin/env python
from tauperf.categories.hlt import Category_1P_HLT, Category_MP_HLT

import os
import logging
import rootpy
from rootpy.plotting import Graph, Canvas, Hist, Legend
from rootpy.plotting.style import set_style
from rootpy.extern.tabulartext import PrettyTable
from rootpy import ROOT

from tauperf import NTUPLE_PATH
from tauperf.analysis import Analysis
from tauperf.variables import VARIABLES
from tauperf.plotting import draw_shape

log = logging.getLogger(os.path.basename(__file__))
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)
set_style('ATLAS', shape='rect')


ana = Analysis(ntuple_path=os.path.join(NTUPLE_PATH, 'training_24_11_2014'))

TARGET_REJECTION = 0.9

def roc(category):
    sig_tot = ana.tau.events(category)[1].value
    bkg_tot = ana.jet.events(category)[1].value

    cut_vals = [0.02*i for i in xrange(50)]
    working_point = (0, 1.0, 0.0) # cut, eff, rej
    gr = Graph(len(cut_vals))
    for i, val in enumerate(cut_vals):
        cut = 'hlt_bdt_score>{0}'.format(val)
        eff_sig = ana.tau.events(category, cut)[1].value
        eff_sig /= sig_tot
        eff_bkg = ana.jet.events(category, cut)[1].value
        eff_bkg /= bkg_tot
        rej_bkg = 1 - eff_bkg
        if abs(rej_bkg - TARGET_REJECTION) < abs(working_point[2] - TARGET_REJECTION):
            working_point = (val, eff_sig, rej_bkg)

        log.info((val, eff_sig, rej_bkg))
        gr.SetPoint(i, eff_sig, rej_bkg)
    return gr, working_point

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

table = PrettyTable(['Category', 'cut', 'signal efficiency', 'background rejection (1/eff_b)'])
table.add_row([Category_1P_HLT.name, wp_sp[0], wp_sp[1], 1. / (1. - wp_sp[2]) if (1. - wp_sp[2]) != 0. else -9999.])
table.add_row([Category_MP_HLT.name, wp_mp[0], wp_mp[1], 1./ (1. - wp_mp[2]) if (1. - wp_mp[2]) != 0. else -9999.])
log.info(40 * '=')
print table
log.info(40 * '=')

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

