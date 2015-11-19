#!/usr/bin/env python

import os
import logging
import rootpy
from rootpy.plotting import Graph, Canvas, Hist, Legend, Efficiency
from rootpy.plotting.style import set_style
from prettytable import PrettyTable
from rootpy import ROOT

from tauperf import UNMERGED_NTUPLE_PATH
from tauperf.analysis import Analysis
from tauperf.variables import VARIABLES
from tauperf.plotting import draw_shape, draw_efficiencies
from tauperf.parallel import run_pool, FuncWorker
from tauperf.categories import Category_1P_HLT, Category_MP_HLT
from tauperf.samples.db import cleanup

log = logging.getLogger(os.path.basename(__file__))
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)
set_style('ATLAS', shape='rect')


class working_point(object):
    def __init__(self, cut, eff_s, eff_b, name='wp'):
        self.name = name
        self.cut = cut
        self.eff_s = eff_s
        self.eff_b = eff_b
    
def get_sig_bkg(ana, cat, cut):
    """small function to calculate sig and bkg yields"""
    y_sig = ana.tau.events(cat, cut, force_reopen=True)[1].value
    y_bkg = ana.jet.events(cat, cut, weighted=True, force_reopen=True)[1].value
    

    return y_sig, y_bkg

def roc(
    ana, 
    category,
    discr_var):
    """
    Calculates the ROC curve
    Returns the sorted list of wp and a TGraph
    """
    cut_vals = [0.01 * (i + 1) for i in xrange(98)]
    roc_gr = Graph(len(cut_vals))
    roc_list = []
    
    log.info('create the workers')
    workers = [FuncWorker(
            get_sig_bkg, ana,
            category, '{0} > {1}'.format(discr_var, val))
               for val in cut_vals]
    log.info('run the pool')
    run_pool(workers, n_jobs=-1)
    yields = [w.output for w in workers]

    log.info('--> Calculate the total yields')
    sig_tot = ana.tau.events(category)[1].value
    bkg_tot = ana.jet.events(category, weighted=True)[1].value
    for i, (val, yields) in enumerate(zip(cut_vals, yields)):
        eff_sig = yields[0] / sig_tot
        eff_bkg = yields[1] / bkg_tot
        rej_bkg = 1. / eff_bkg if eff_bkg !=0 else 0
        roc_list.append(working_point(
                val, eff_sig, eff_bkg))
        roc_gr.SetPoint(i, eff_sig, rej_bkg)
    return roc_gr, roc_list


def old_working_points(ana, category, wp_level):
    log.info('create the workers')

    cuts = [
        wp_level + '_is_loose == 1', 
        wp_level + '_is_medium == 1',
        wp_level + '_is_tight == 1' 
        ]
    
    workers = [FuncWorker(
            get_sig_bkg, 
            ana, category, 
            cut) for cut in cuts]
    run_pool(workers, n_jobs=-1)
    yields = [w.output for w in workers]

    log.info('--> Calculate the total yields')
    sig_tot = ana.tau.events(category)[1].value
    bkg_tot = ana.jet.events(category, weighted=True)[1].value
    gr = Graph(len(cuts))
    wps = []
    for i, (val, yields) in enumerate(zip(cuts, yields)):
        eff_sig = yields[0] / sig_tot
        eff_bkg = yields[1] / bkg_tot
        rej_bkg = 1. / eff_bkg if eff_bkg != 0 else 0
        wps.append(working_point(
                val, eff_sig, eff_bkg))
        gr.SetPoint(i, eff_sig, rej_bkg)
    return gr, wps
    

def efficiencies_plot(
    ana, 
    category, 
    discri_var,
    plot_level,
    working_points):

    vars = {
        'pt': VARIABLES['pt'],
        'eta': VARIABLES['eta'],
        'good_npv': VARIABLES['good_npv'],
        'averageintpercrossing': VARIABLES['averageintpercrossing'],
        }
    canvases = {}

    efficiencies = {}
    for v in vars.keys():
        efficiencies[v] = []

    for wp in working_points:
        cut = wp.cut if isinstance(wp.cut, str) else '{0} >= {1}'.format(discri_var, wp.cut)
        hist_samples = ana.get_hist_samples_array(vars, plot_level, category=category)
        hist_samples_cut = ana.get_hist_samples_array(vars, plot_level, category=category, cuts=cut)
        for v in vars.keys():
            efficiencies[v].append(Efficiency(
                    hist_samples_cut[v]['tau'], 
                    hist_samples[v]['tau'],
                    title=wp.name))

    for v, effs in efficiencies.items():
        canvases[v] = draw_efficiencies(
            effs, plot_level + '_' + v, category)

    return canvases

def score_plot(ana, category, discri_var):
    sig = ana.tau.get_hist_array(
        {discri_var: Hist(20, 0, 1)},
        category=category)
    bkg = ana.jet.get_hist_array(
        {discri_var: Hist(20, 0, 1)},
        category=category)
    hsig = sig[discri_var]
    hbkg = bkg[discri_var]
    plot = draw_shape(hsig, hbkg, 'BDT Score', category)
    return plot


if __name__ == '__main__':
    from tauperf.cmd import get_parser
    parser = get_parser()
    parser.add_argument('--no-roc', action='store_true', default=False)
    parser.add_argument('--score-var', default=None, type=str)
    args = parser.parse_args()
    ana = Analysis(
        trigger=args.trigger)

    if args.trigger:
        score_var = 'hlt_bdtjetscore'
        wp_level = 'hlt'
    else:
        score_var = 'off_bdtjetscore'
        wp_level = 'off'
        
    if args.score_var is not None:
        score_var = args.score_var

    table = PrettyTable([
            'Category', 
            'cut', 
            'signal efficiency', 
            'background rejection (1/eff_b)'])


    for cat in ana.iter_categories(args.categories):

        gr_old, wp_old = old_working_points(ana, cat, wp_level)
        if not args.no_roc:
            roc_new, wp_new = roc(ana, cat, score_var)    
            
            roc_new.color = 'red'
            roc_new.legendstyle = 'l'
            roc_new.linewidth = 3
            roc_new.title = '1 prong'
            roc_new.xaxis.title = '#epsilon_{S}' 
            roc_new.yaxis.title = '1 / #epsilon_{B}' 
            roc_new.yaxis.SetRangeUser(1., 1e4)
            c = Canvas()
            c.SetGridx()
            c.SetGridy()
            c.SetLogy(True)
            roc_new.Draw('AL')
            gr_old.Draw('sameP')
            c.SaveAs('plots/roc_cat_{0}.png'.format(cat.name))
        
        TARGET = wp_old
        
        plot_score = score_plot(ana, cat, score_var)
        plot_score.SaveAs('plots/scores_cat_{0}.png'.format(cat.name))
        

        plot_effs = efficiencies_plot(ana, cat, score_var, wp_level, TARGET)
        for v, plt in plot_effs.items():
            plt.SaveAs('plots/efficiencies_var_{0}_cat_{1}.png'.format(v, cat.name))
        

        for t in TARGET:
            table.add_row([
                cat.name, 
                t.cut, t.eff_s, 
                1. / t.eff_b if t.eff_b != 0. else -9999.])

    print
    log.info(40 * '=')
    print table
    log.info(40 * '=')
    print
