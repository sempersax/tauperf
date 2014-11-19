#!/usr/bin/env python
from tauperf.categories.hlt import Category_1P_HLT, Category_MP_HLT

import os
import logging
import rootpy
from rootpy.plotting import Graph, Canvas, Hist
from rootpy.plotting.style import set_style
from rootpy import ROOT

from tauperf.analysis import Analysis
from tauperf.variables import VARIABLES
from tauperf.plotting import draw_shape

log = logging.getLogger(os.path.basename(__file__))
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)
set_style('ATLAS', shape='rect')


ana = Analysis(ntuple_path='/Users/quentin/Desktop/sandbox')

def roc(category):
    sig_tot = ana.tau.events(category)[1].value
    bkg_tot = ana.jet.events(category)[1].value

    cut_vals = [0.02*i for i in xrange(50)]

    gr = Graph(len(cut_vals))
    for i, val in enumerate(cut_vals):
        cut = 'hlt_bdt_score>{0}'.format(val)
        eff_sig = ana.tau.events(category, cut)[1].value
        eff_sig /= sig_tot

        eff_bkg = ana.jet.events(category, cut)[1].value
        eff_bkg /= bkg_tot
        log.info((val, eff_sig, 1-eff_bkg))
        gr.SetPoint(i, eff_sig, 1-eff_bkg)

    return gr

# gr_sp = roc(Category_1P_HLT)    
# gr_sp.color = 'red'
# gr_sp.linewidth = 3
# gr_sp.title = '1 prong'
# gr_sp.xaxis.title = '#epsilon_{S}' 
# gr_sp.yaxis.title = '1 - #epsilon_{B}' 
# gr_mp = roc(Category_MP_HLT)    
# gr_mp.color = 'blue'
# gr_mp.linewidth = 3
# gr_mp.xaxis.title = '#epsilon_{S}' 
# gr_mp.yaxis.title = '1 - #epsilon_{B}' 

# c = Canvas()
# gr_sp.Draw('AL')
# gr_mp.Draw('SAMEL')
# c.SaveAs('roc.png')

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
plot_1P.SaveAs('scores_1p.png')

plot_MP = score_plot(Category_MP_HLT)
plot_MP.SaveAs('scores_mp.png')

# log.info(list(sig.y()))
# log.info(list(bkg.y()))

# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument('--categories', default='plotting')
#     parser.add_argument('--var', default=None, help='Specify a particular variable')
#     args = parser.parse_args()
