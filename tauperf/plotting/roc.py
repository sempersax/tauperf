from rootpy.plotting import Graph, Canvas, Hist, Legend, Efficiency

from . import draw_shape, draw_efficiencies
from ..parallel import run_pool, FuncWorker
from ..classify import working_point
from .. import log; log = log[__name__]

def roc(
    ana, 
    category,
    discr_var):
    """
    Calculates the ROC curve
    Returns the sorted list of wp and a TGraph
    """
    h_template = Hist(1000, 0, 1)

    h_sig = ana.tau.get_hist_array(
        {discr_var: h_template},
        category=category)
    h_sig = h_sig[discr_var]

    h_bkg = ana.jet.get_hist_array(
        {discr_var: h_template},
        category=category)
    h_bkg = h_bkg[discr_var]

    roc_gr = Graph(h_sig.GetNbinsX())
    roc_list = []
    for i in range(1, h_sig.GetNbinsX()):
        eff_sig_i = (h_sig.Integral() - h_sig.Integral(0, i)) / h_sig.Integral()
        eff_bkg_i = (h_bkg.Integral() - h_bkg.Integral(0, i)) / h_bkg.Integral()
        rej_bkg_i = 1. / eff_bkg_i if eff_bkg_i != 0 else 0.
        roc_list.append(working_point(
                h_sig.GetBinLowEdge(i), eff_sig_i, eff_bkg_i))

        roc_gr.SetPoint(i, eff_sig_i, rej_bkg_i)
    return roc_gr, roc_list

def old_working_points(ana, category, wp_level):
    log.info('create the workers')

    names = ['loose', 'medium', 'tight']

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
    for i, (val, yields, name) in enumerate(zip(cuts, yields, names)):
        eff_sig = yields[0] / sig_tot
        eff_bkg = yields[1] / bkg_tot
        rej_bkg = 1. / eff_bkg if eff_bkg != 0 else 0
        wps.append(working_point(
                val, eff_sig, eff_bkg, name=name))
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


def get_sig_bkg(ana, cat, cut):
    """small function to calculate sig and bkg yields"""
    y_sig = ana.tau.events(cat, cut, force_reopen=True)[1].value
    y_bkg = ana.jet.events(cat, cut, weighted=True, force_reopen=True)[1].value
    return y_sig, y_bkg
