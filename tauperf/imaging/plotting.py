import numpy as np
import itertools
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from . import log; log = log[__name__]

def get_wp(true_pos, false_pos, thresh, method='corner'):
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
        target_eff = np.abs(true_pos - 0.12)
        index_min = np.argmin(target_eff)
    elif method == 'target_rej':
        target_rej = np.abs(false_pos - 0.860)
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

    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.tight_layout()
    plt.savefig(name)


# pt efficiency
from root_numpy import fill_hist
from rootpy.plotting import Hist, Efficiency
def calc_eff(accept, total, binning=(20, 0, 100), name='1p1n'):
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
