#!/usr/bin/env python

import os
import logging
import rootpy
from rootpy.plotting import Canvas
from rootpy.plotting.style import set_style
from prettytable import PrettyTable
from rootpy import ROOT

from tauperf.analysis import Analysis
from tauperf.plotting.roc import roc, score_plot, old_working_points

log = logging.getLogger(os.path.basename(__file__))
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)
set_style('ATLAS', shape='rect')




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
            c.SaveAs('plots/roc_cat_{0}_score_{1}.png'.format(cat.name, score_var))
        
        TARGET = wp_old
        
        plot_score = score_plot(ana, cat, score_var)
        plot_score.SaveAs('plots/scores_cat_{0}_score_{1}.png'.format(cat.name, score_var))
        

        # plot_effs = efficiencies_plot(ana, cat, score_var, wp_level, TARGET)
        # for v, plt in plot_effs.items():
        #     plt.SaveAs('plots/efficiencies_var_{0}_cat_{1}_score_{2}.png'.format(v, cat.name, score_var))
        

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
