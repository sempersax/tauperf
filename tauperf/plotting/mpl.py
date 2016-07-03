from root_numpy import fill_hist
from rootpy.plotting import Hist, Canvas, Efficiency
from rootpy.plotting import root2matplotlib as rmpl
from rootpy.plotting.style import set_style
from rootpy import asrootpy
import matplotlib.pyplot as plt

from ..variables import VARIABLES, get_label
from . import log; log = log[__name__]

set_style('ATLAS', mpl=True)

def score_plot(sig_arr, bkg_arr, sig_weight, bkg_weight):
    '''
    make a plot of the score for bkg and signal
    '''
    
    hsig = Hist(100, -0.5, 0.5)
    hbkg = Hist(100, -0.5, 0.5)

    fill_hist(hsig, sig_arr, sig_weight)
    fill_hist(hbkg, bkg_arr, bkg_weight)
    hsig /= hsig.Integral()
    hbkg /= hbkg.Integral()
    hsig.color = 'red'
    hbkg.color = 'blue'
    hsig.title = 'signal'
    hbkg.title = 'background'
    fig = plt.figure()
    rmpl.hist([hsig, hbkg], stacked=False)
    plt.ylabel('Arbitrary Unit')
    plt.xlabel('BDT Score')
    plt.legend(loc='upper right')
    return fig


def eff_curve(accept, total, var, weight_field=None, prefix='off'):
    """
    Draw the efficiency curve from two record arrays
    (selection already applied so computation should be fast over several vars)
    """

    if var not in VARIABLES.keys():
        log.error('Wrong variable name (see variables.py)')
        raise ValueError('Wrong variable name (see variables.py)')

    var_info = VARIABLES[var]
    hnum = Hist(
        var_info['bins'],
        var_info['range'][0],
        var_info['range'][1])

    hden = Hist(
        var_info['bins'],
        var_info['range'][0],
        var_info['range'][1])
    
    log.info('filling')
    if prefix is not None:
        field = prefix + '_' + var_info['name']
    else:
        field = var_info['name']

    num =  accept[field]
    den = total[field]

    if 'scale' in var_info.keys():
        num *= var_info['scale']
        den *= var_info['scale']

    fill_hist(hnum, accept[prefix + '_' + var], accept[weight_field])
    fill_hist(hden, total[prefix + '_' + var], total[weight_field])
    eff = Efficiency(
        hnum, hden, 
        name='eff_{0}'.format(var),
        title=get_label(VARIABLES[var]))

    return eff



def eff_plot(eff):

    log.info(eff.painted_graph)
    fig = plt.figure()
    rmpl.errorbar([eff.painted_graph])
    plt.ylabel('Efficiency')
    plt.xlabel(eff.title)
    plt.legend(loc='upper right')
    return fig
