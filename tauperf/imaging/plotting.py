import math
import numpy as np
import itertools
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm

from . import log; log = log[__name__]

def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi

def get_wp(true_pos, false_pos, thresh, method='corner', target_value=0.8):
    if (true_pos.ndim, false_pos.ndim, thresh.ndim) != (1, 1, 1):
        raise ValueError('wrong dimension')
    if len(true_pos) != len(false_pos) or len(true_pos) != len(thresh):
        raise ValueError('wrong size')


    if method == 'corner':
        # compute the distance to the (0, 1) point
        dr_square = true_pos * true_pos + (false_pos - 1) * (false_pos - 1)
        # get the index in the dr_square array
        index_min = np.argmin(dr_square)
        # return optimal true positive eff, false positive eff and threshold for cut
    elif method == 'target_eff':
        val = target_value
        target_eff = np.abs(true_pos - val)
        index_min = np.argmin(target_eff)
    elif method == 'target_rej':
        target_rej = np.abs(false_pos - val)
        index_min = np.argmin(target_rej)
    else:
        raise ValueError('wrong method argument')

    return true_pos[index_min], false_pos[index_min], thresh[index_min]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          name='plots/imaging/confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes[::-1])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)#[:, np.newaxis]
    else:
        print
        print('Confusion matrix, without normalization')
        plt.colorbar()

    log.info('Confusion matrix')
    log.info('')
    print(cm)
    log.info('')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('True Tau Decay Mode')
    plt.ylabel('Reconstructed Tau Decay Mode')
    plt.tight_layout()
    plt.savefig(name)


# pt efficiency
def calc_eff(accept, total, binning=(20, 0, 100), name='1p1n'):
    from root_numpy import fill_hist
    from rootpy.plotting import Hist, Efficiency
    hnum = Hist(binning[0], binning[1], binning[2])
    hden = Hist(binning[0], binning[1], binning[2])
    fill_hist(hnum, accept)
    fill_hist(hden, total)
    eff = Efficiency(hnum, hden, name='eff_' + name, title=name)
    return eff

def get_eff(arr, pred, scale=1., binning=(20, 0, 100), color='red', name='1p1n'):
    total = arr * scale
    accept = total[pred == 1]
    eff = calc_eff(accept, total, binning=binning, name=name)
    eff.color = color
    return eff


def plot_image(rec, eta, phi, ene, irec, cal_layer, suffix):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    rect = mpatches.Rectangle(
        (-0.2, -0.2), 0.4, 0.4, 
        fill=False, linewidth=3, label='selection')
    plt.scatter(
        eta, phi, c=ene, marker='s', s=40,
        label= 'Number of cells = {0}'.format(len(eta)))
    plt.colorbar()
    ax.add_patch(rect)
    plt.plot(
        rec['true_charged_eta'] - rec['true_eta'], 
        dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
        label='charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))
    if not '0n' in suffix:
        plt.plot(
            rec['true_neutral_eta'] - rec['true_eta'], 
            dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
            label='neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('{0}: image {1} sampling {2}'.format(suffix, irec, cal_layer))
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    fig.savefig('plots/imaging/images/image_{0}_s{1}_{2}.pdf'.format(
            irec, cal_layer, suffix))
    fig.clf()
    plt.close()

def plot_heatmap(image, rec, pos_central_cell, irec, cal_layer, suffix, fixed_scale=False):
    fig = plt.figure()
    if fixed_scale:
        image[image <= 0] = 0.00001

    plt.imshow(
        image, 
        extent=[-0.2, 0.2, -0.2, 0.2], 
        interpolation='nearest',  
        cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
        norm=None if fixed_scale is False else LogNorm(0.0001, 1))

    plt.colorbar()
    plt.plot(
        rec['true_charged_eta'] - rec['true_eta'], 
        dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
        label='true charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))

    plt.plot(
        rec['off_tracks_eta'][0] - pos_central_cell['eta'], 
        dphi(rec['off_tracks_phi'][0], pos_central_cell['phi']), 'bo', 
        label='reco charge pi, pT = %1.2f GeV' % (rec['off_tracks_pt'][0] / 1000.))


    if not '0n' in suffix:
        plt.plot(
            rec['true_neutral_eta'] - rec['true_eta'], 
            dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
            label='true neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('{0}: image {1} sampling {2}'.format(suffix, irec, cal_layer))
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    fig.savefig('plots/imaging/images/image_{0}_s{1}_{2}.pdf'.format(
            irec, cal_layer, suffix))
    fig.clf()
    plt.close()
