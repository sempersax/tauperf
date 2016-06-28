from root_numpy import fill_hist
from rootpy.plotting import Hist, Canvas
from rootpy.plotting import root2matplotlib as rmpl
from rootpy.plotting.style import set_style
import matplotlib.pyplot as plt


set_style('ATLAS', mpl=True)

def score_plot(sig_arr, bkg_arr, sig_weight, bkg_weight):

    
    hsig = Hist(20, 0, 1)
    hbkg = Hist(20, 0, 1)

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
    plt.xlabel('Arbitrary Unit')
    plt.xlabel('BDT Score')
    plt.legend(loc='upper right')
    return fig
