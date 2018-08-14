import math
import numpy as np
import itertools
import sys
import socket
import matplotlib as mpl;
import os
from mpl_toolkits.mplot3d import Axes3D

techlab_hosts = [
    'techlab-gpu-nvidiak20-03.cern.ch',
    'techlab-gpu-nvidiagtx1080-07.cern.ch'
]
if socket.gethostname() in techlab_hosts:
    mpl.use('PS')
else:
    mpl.use('TkAgg')


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm

from . import log; log = log.getChild(__name__)

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
    plt.figure()
    np.set_printoptions(precision=2)
    diagonal = float(np.trace(cm)) / float(np.sum(cm))
    log.info('Diag / Total = {0} / {1}'.format(np.trace(cm), np.sum(cm)))
    cm = cm.T.astype('float') / cm.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes[::-1])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)#[:, np.newaxis]

    plt.colorbar()

    log.info('Confusion matrix')
    print
    print(cm)
    print 
    print 'Diagonal'
    print diagonal * 100
    print

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:1.2f}'.format(100 * cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 100 * thresh else "black")

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


def plot_roc(y_test, y_pred, y_pant):
    from sklearn.metrics import roc_curve

    y_test_1p = y_test[np.logical_or(y_test == 0, y_test == 1, y_test == 2)]
    y_pred_1p = y_pred[np.logical_or(y_test == 0, y_test == 1, y_test == 2)]
    y_pred_1p = y_pred_1p[:,0] / (y_pred_1p[:,0] + y_pred_1p[:,1] + y_pred_1p[:,2])
    fpr_1p0n, tpr_1p0n, _ = roc_curve(y_test_1p, y_pred_1p, pos_label=0)

    y_pant_1p0n = y_pant[y_test == 0]
    y_pant_1pXn = y_pant[np.logical_or(y_test == 1, y_test == 2)]
    eff_pant_1p0n = float(len(y_pant_1p0n[y_pant_1p0n == 0])) / float(len(y_pant_1p0n))
    rej_pant_1p0n = float(len(y_pant_1pXn[y_pant_1pXn != 0])) / float(len(y_pant_1pXn))
    
    y_test_1pXn = y_test[np.logical_or(y_test == 1, y_test == 2)]
    y_pred_1pXn = y_pred[np.logical_or(y_test == 1, y_test == 2)]
    y_pred_1pXn = y_pred_1pXn[:,1] / (y_pred_1pXn[:,1] + y_pred_1pXn[:,2])
    fpr_1p1n, tpr_1p1n, _ = roc_curve(y_test_1pXn, y_pred_1pXn, pos_label=1)

    y_pant_1p1n = y_pant[y_test == 1]
    y_pant_1p2n = y_pant[y_test == 2]
    eff_pant_1p1n = float(len(y_pant_1p1n[y_pant_1p1n == 1])) / float(len(y_pant_1p1n))
    rej_pant_1p1n = float(len(y_pant_1p2n[y_pant_1p2n != 1])) / float(len(y_pant_1p2n))

    y_test_3p = y_test[np.logical_or(y_test == 3, y_test == 4)]
    y_pred_3p = y_pred[np.logical_or(y_test == 3, y_test == 4)]
    y_pant_3p = y_pant[np.logical_or(y_test == 3, y_test == 4)]

    y_pred_3p = y_pred_3p[:,3] / (y_pred_3p[:,3] + y_pred_3p[:,4])
    fpr_3p0n, tpr_3p0n, _ = roc_curve(y_test_3p, y_pred_3p, pos_label=3)

    y_pant_3p0n = y_pant[y_test == 3]
    y_pant_3pXn = y_pant[y_test == 4]
    eff_pant_3p0n = float(len(y_pant_3p0n[y_pant_3p0n == 3])) / float(len(y_pant_3p0n))
    rej_pant_3p0n = float(len(y_pant_3pXn[y_pant_3pXn != 3])) / float(len(y_pant_3pXn))

    plt.figure()
    plt.plot(tpr_1p0n, 1 - fpr_1p0n, label='1p0n vs 1p(>0)n', color='red')
    plt.plot(tpr_1p1n, 1 - fpr_1p1n, label='1p1n vs 1pXn', color='blue')
    plt.plot(tpr_3p0n, 1 - fpr_3p0n, label='3p0n vs 3pXn', color='purple')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.scatter(
        [eff_pant_1p0n, eff_pant_1p1n, eff_pant_3p0n],
        [rej_pant_1p0n, rej_pant_1p1n, rej_pant_3p0n],
        s=100,
        marker='v',
        c=['red', 'blue', 'purple'],
        label='pantau')

    
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0, 1, 0.1))
    axes.yaxis.set_ticks(np.arange(0, 1, 0.1))
    axes.grid(True)
    
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/roc_curve.pdf')

def plot_event(test, evts, y_pred, decay_mode, suffix='dummy'):
# decay_type classification
    if decay_mode == '1p0n':
        decay_type = 0
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '1p1n':
        decay_type = 1
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '1pXn':
        decay_type = 2
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '3p0n':
        decay_type = 3
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '3pXn':
        decay_type = 4
        decay_compare = 0
        decay_compare_mode = '1p0n'
    newpath = r'./plots/imaging/' + decay_mode 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i in range(len(evts)):
        evt = test[int(evts[i])]
        fig = plt.figure()
        fixed_scale = True
        if fixed_scale:
            evt['s3'][evt['s3'] <= 0] = 0.00001
            evt['s2'][evt['s2'] <= 0] = 0.00001
            evt['s1'][evt['s1'] <= 0] = 0.00001

        plt.imshow(
            evt['s2'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            #print evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], int(evt['tracks'][i_track, 3])
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], int(evt['tracks'][i_track, 3]), fontsize=10)
        plt.colorbar()
        plt.title('True Mode: ' + decay_mode + ' Event number: ' + str(int(evts[i])) + '\n' + decay_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_type]) + '\n' + decay_compare_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_compare]))
        plt.savefig('./plots/imaging/' + decay_mode + '/' + suffix + '_' + decay_mode + '_evt_' + str(int(evts[i])) + '_s2_heatmap.pdf')
        plt.close()


        fig = plt.figure()
        plt.imshow(
            evt['s1'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], int(evt['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + str(int(evts[i])) + '\n' + decay_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_type]) + '\n' + decay_compare_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + suffix + '_' + decay_mode + '_evt_' + str(int(evts[i])) + '_s1_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt['s3'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Blues if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], int(evt['tracks'][i_track, 3]), fontsize=10)
        plt.title('True Mode: ' + decay_mode + ' Event number: ' + str(int(evts[i])) + '\n' + decay_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_type]) + '\n' + decay_compare_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + suffix + '_' + decay_mode + '_evt_' + str(int(evts[i])) + '_s3_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt['s1'], 
            alpha = 1.,
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))
        plt.imshow(
            evt['s2'], 
            alpha = 0.65,
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))
        plt.imshow(
            evt['s3'], 
            alpha = 0.65,
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Blues if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], int(evt['tracks'][i_track, 3]), fontsize=10)
        plt.title('True Mode: ' + decay_mode + ' Event number: ' + str(int(evts[i])) + '\n' + decay_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_type]) + '\n' + decay_compare_mode + 'Score: ' + str(y_pred[int(evts[i])][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + suffix + '_' + decay_mode + '_evt_' + str(int(evts[i])) + '_s1_s2_s3_overlay.pdf')
        plt.close()

#######################################################################
def score_plots(y_pred, y_truth, decay_mode):        
# decay_type classification
    if decay_mode == '1p0n':
        decay_type = 0
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '1p1n':
        decay_type = 1
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '1pXn':
        decay_type = 2
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '3p0n':
        decay_type = 3
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '3pXn':
        decay_type = 4
        decay_compare = 0
        decay_compare_mode = '1p0n'
    newpath = r'./plots/imaging/' + decay_mode 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    scores_true = y_pred[y_truth == decay_type]
    scores_false = y_pred[y_truth == decay_compare]
    scores_true = scores_true[:,decay_type]
    scores_false = scores_false[:,decay_type]
    fig = plt.figure()
    plt.hist(scores_true, bins=100, range=(0,1), color = 'blue', label= decay_mode, density = True)
    plt.hist(scores_false, bins=100, range=(0,1), color = 'red', label=decay_compare_mode, alpha = 0.5, density = True)
    plt.yscale('log', nonposy='clip')
    plt.xlabel('scores')
    plt.ylabel('A.U.')
    plt.title(decay_mode + ' scores for true 1p0n and 1p1n decay modes')
    plt.legend(loc='upper right', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + 'True_' + decay_mode + '_scores.pdf')
    plt.close()

    scores_true = y_pred[y_truth == decay_type]
    scores_false = y_pred[y_truth == decay_compare]
    scores_true = scores_true[:,decay_type]
    scores_false = scores_false[:,decay_type]
    scores_1p0n = y_pred[y_truth == decay_type][:,0]
    scores_1p1n = y_pred[y_truth == decay_type][:,1]
    scores_1pXn = y_pred[y_truth == decay_type][:,2]
    scores_3p0n = y_pred[y_truth == decay_type][:,3]
    scores_3pXn = y_pred[y_truth == decay_type][:,4]

    fig = plt.figure()
    plt.hist(scores_1p0n, bins=100, range=(0,1), color = 'blue', label='1p0n positive', density = True)
    plt.hist(scores_1p1n, bins=100, range=(0,1), color = 'red', label='1p1n positive', alpha = 0.5, density = True)
    plt.hist(scores_1pXn, bins=100, range=(0,1), color = 'green', label='1pXn positive', alpha = 0.5, density = True)
    plt.hist(scores_3p0n, bins=100, range=(0,1), color = 'orange', label='3p0n positive', alpha = 0.5, density = True)
    plt.hist(scores_3pXn, bins=100, range=(0,1), color = 'pink', label='3pXn positive', alpha = 0.5, density = True)
    plt.yscale('log', nonposy='clip')
    plt.xlabel('scores')
    plt.ylabel('A.U.')
    plt.title('True Mode: ' + decay_mode)
    plt.legend(loc='upper right', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + 'True_' + decay_mode + '_all_scores.pdf')
    plt.close()

##############################################################################
def score_outliers(test, y_pred, y_truth, decay_mode):
# decay_type classification
    if decay_mode == '1p0n':
        decay_type = 0
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '1p1n':
        decay_type = 1
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '1pXn':
        decay_type = 2
        decay_compare = 1
        decay_compare_mode = '1p1n'
    if decay_mode == '3p0n':
        decay_type = 3
        decay_compare = 0
        decay_compare_mode = '1p0n'
    if decay_mode == '3pXn':
        decay_type = 4
        decay_compare = 0
        decay_compare_mode = '1p0n'
    newpath = r'./plots/imaging/' + decay_mode 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    worst_outliers = np.intersect1d(np.where(y_truth == decay_type)[0], np.where(y_pred[:,decay_type] <= 0.01)[0])
    best_outliers = np.intersect1d(np.where(y_truth == decay_type)[0], np.where(y_pred[:,decay_type] >= 0.99)[0])
    print len(worst_outliers)
    print len(best_outliers)

# delta eta tracks
#    best_positive_deta_leadtrack = best_outliers['tracks'][:, 0, 1]
#    worst_positive_deta_leadtrack = worst_outliers['tracks'][:, 0, 1]

# delta phi tracks
#    best_positive_dphi_leadtrack = best_outliers['tracks'][:, 0, 2]
#    worst_positive_dphi_leadtrack = worst_outliers['tracks'][:, 0, 2]

# deta plots
    plt.figure() 
    plt.hist(test[best_outliers[0]]['tracks'][0, 1], bins=60, range=(-0.3, 0.3), color = 'blue', label='>= 99%', density = True)
    plt.hist(test[worst_outliers[0]]['tracks'][0, 1], bins=60, range=(-0.3, 0.3), color = 'red', label='<= 1%', density = True, alpha = 0.4)
    plt.xlabel('dEta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_outliers_' + 'dEta.pdf')
    plt.close()

# dphi plots
    plt.figure() 
    plt.hist(test[best_outliers[0]]['tracks'][0, 2], bins=60, range=(-0.3, 0.3), color = 'blue', label='> 99%', density = True)
    plt.hist(test[worst_outliers[0]]['tracks'][0, 2], bins=60, range=(-0.3, 0.3), color = 'red', label='<= 1%', density = True, alpha = 0.4)
    plt.xlabel('dPhi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_outliers_' + 'dPhi.pdf')
    plt.close()

#heatmaps

    fixed_scale = True

    for i in range(10):
        evt = test[best_outliers[i]]
        evt2 = test[worst_outliers[i]]
        if fixed_scale:
            evt['s3'][evt['s3'] <= 0.] = 0.00001
            evt['s2'][evt['s2'] <= 0.] = 0.00001
            evt['s1'][evt['s1'] <= 0.] = 0.00001
    

            evt2['s3'][evt2['s3'] <= 0.] = 0.00001
            evt2['s2'][evt2['s2'] <= 0.] = 0.00001
            evt2['s1'][evt2['s1'] <= 0.] = 0.00001

        fig = plt.figure()
        plt.imshow(
            evt['s1'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], 
                     int(evt['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(best_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[best_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[best_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'best_outliers' + '_' + decay_mode + '_evt_' + str(best_outliers[i]) + '_s1_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt['s2'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], 
                     int(evt['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(best_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[best_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[best_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'best_outliers' + '_' + decay_mode + '_evt_' + str(best_outliers[i]) + '_s2_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt['s3'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt['tracks'][i_track, 1], evt['tracks'][i_track, 2], 'bo')
            plt.text(evt['tracks'][i_track, 1] + 0.02, evt['tracks'][i_track, 2], 
                     int(evt['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(best_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[best_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[best_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'best_outliers' + '_' + decay_mode + '_evt_' + str(best_outliers[i]) + '_s3_heatmap.pdf')
        plt.close()
    
        fig = plt.figure()
        plt.imshow(
            evt2['s1'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt2['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt2['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt2['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt2['tracks'][i_track, 1], evt2['tracks'][i_track, 2], 'bo')
            plt.text(evt2['tracks'][i_track, 1] + 0.02, evt2['tracks'][i_track, 2], 
                     int(evt2['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(worst_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[worst_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[worst_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'worst_outliers' + '_' + decay_mode + '_evt_' + str(worst_outliers[i]) + '_s1_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt2['s2'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt2['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt2['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt2['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt2['tracks'][i_track, 1], evt2['tracks'][i_track, 2], 'bo')
            plt.text(evt2['tracks'][i_track, 1] + 0.02, evt2['tracks'][i_track, 2], 
                     int(evt2['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(worst_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[worst_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[worst_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'worst_outliers' + '_' + decay_mode + '_evt_' + str(worst_outliers[i]) + '_s2_heatmap.pdf')
        plt.close()

        fig = plt.figure()
        plt.imshow(
            evt2['s3'], 
            extent=[-0.2, 0.2, -0.2, 0.2], 
            interpolation='nearest',  
            cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
            norm=None if fixed_scale is False else LogNorm(0.0001, 1))

        for i_track in xrange(len(evt['tracks'])):
            if evt2['tracks'][i_track, 0] == 0.:
                continue
            if math.fabs(evt2['tracks'][i_track, 2]) > 0.2:
                continue
            if math.fabs(evt2['tracks'][i_track, 1]) > 0.2:
                continue
            plt.plot(evt2['tracks'][i_track, 1], evt2['tracks'][i_track, 2], 'bo')
            plt.text(evt2['tracks'][i_track, 1] + 0.02, evt2['tracks'][i_track, 2], 
                     int(evt2['tracks'][i_track, 3]), fontsize=10)

        plt.title('True Mode: ' + decay_mode + ' Event number: ' + 
                   str(int(worst_outliers[i])) + '\n' + decay_mode + 
                   'Score: ' + str(y_pred[worst_outliers[i]][decay_type]) + 
                   '\n' + decay_compare_mode + 'Score: ' + str(y_pred[worst_outliers[i]][decay_compare]))
        plt.colorbar()
        plt.savefig('./plots/imaging/' + decay_mode + '/' + 'worst_outliers' + '_' + decay_mode + '_evt_' + str(worst_outliers[i]) + '_s3_heatmap.pdf')
        plt.close()
    





def compare_bins(test, y_pred, y_truth, decay_mode):
###########################################################################
#new stuff
###########################################################################
# decay_type classification
    if decay_mode == '1p0n':
        decay_type = 0
        decay_compare = 1
    if decay_mode == '1p1n':
        decay_type = 1
        decay_compare = 0
    if decay_mode == '1pXn':
        decay_type = 2
        decay_compare = 1
    if decay_mode == '3p0n':
        decay_type = 3
        decay_compare = 0
    if decay_mode == '3pXn':
        decay_type = 4
        decay_compare = 0
    newpath = r'./plots/imaging/' + decay_mode 
    if not os.path.exists(newpath):
        os.makedirs(newpath)


# Grab just the TRUE decay_mode decays, same as done originally
    y_pred_decay_mode = y_pred[y_truth == decay_type]
    test_decay_mode = test[y_truth == decay_type]
    true_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_type]
    false_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_compare]

# Maximum Locations
    locationS1_x_true = np.argmax(np.amax(true_positive['s1'], axis=2), axis=1) - 2
    locationS1_y_true = np.argmax(np.amax(true_positive['s1'], axis=1), axis=1) - 60
    locationS1_x_false = np.argmax(np.amax(false_positive['s1'], axis=2), axis=1) - 2
    locationS1_y_false = np.argmax(np.amax(false_positive['s1'], axis=1), axis=1) - 60

    locationS2_x_true = np.argmax(np.amax(true_positive['s2'], axis=2), axis=1) - 16
    locationS2_y_true = np.argmax(np.amax(true_positive['s2'], axis=1), axis=1) - 16
    locationS2_x_false = np.argmax(np.amax(false_positive['s2'], axis=2), axis=1) - 16
    locationS2_y_false = np.argmax(np.amax(false_positive['s2'], axis=1), axis=1) - 16

    locationS3_x_true = np.argmax(np.amax(true_positive['s3'], axis=2), axis=1) - 16
    locationS3_y_true = np.argmax(np.amax(true_positive['s3'], axis=1), axis=1) - 8
    locationS3_x_false = np.argmax(np.amax(false_positive['s3'], axis=2), axis=1) - 16
    locationS3_y_false = np.argmax(np.amax(false_positive['s3'], axis=1), axis=1) - 8

    locationS4_x_true = np.argmax(np.amax(true_positive['s4'], axis=2), axis=1) - 8
    locationS4_y_true = np.argmax(np.amax(true_positive['s4'], axis=1), axis=1) - 8
    locationS4_x_false = np.argmax(np.amax(false_positive['s4'], axis=2), axis=1) - 8
    locationS4_y_false = np.argmax(np.amax(false_positive['s4'], axis=1), axis=1) - 8

    locationS5_x_true = np.argmax(np.amax(true_positive['s5'], axis=2), axis=1) - 8
    locationS5_y_true = np.argmax(np.amax(true_positive['s5'], axis=1), axis=1) - 8
    locationS5_x_false = np.argmax(np.amax(false_positive['s5'], axis=2), axis=1) - 8
    locationS5_y_false = np.argmax(np.amax(false_positive['s5'], axis=1), axis=1) - 8
# end maximum block

# minimum locations - 1000000 used to prevent minimum from being zero.
    locationS1_x_true_min = true_positive['s1'] 
    locationS1_x_true_min[locationS1_x_true_min == 0.] = 1000000
    locationS2_x_true_min = true_positive['s2']
    locationS2_x_true_min[locationS2_x_true_min == 0.] = 1000000
    locationS3_x_true_min = true_positive['s3']
    locationS3_x_true_min[locationS3_x_true_min == 0.] = 1000000
    locationS4_x_true_min = true_positive['s4']
    locationS4_x_true_min[locationS4_x_true_min == 0.] = 1000000
    locationS5_x_true_min = true_positive['s5']
    locationS5_x_true_min[locationS5_x_true_min == 0.] = 1000000

    locationS1_y_true_min = true_positive['s1']
    locationS1_y_true_min[locationS1_y_true_min == 0.] = 1000000
    locationS2_y_true_min = true_positive['s2']
    locationS2_y_true_min[locationS2_y_true_min == 0.] = 1000000
    locationS3_y_true_min = true_positive['s3']
    locationS3_y_true_min[locationS3_y_true_min == 0.] = 1000000
    locationS4_y_true_min = true_positive['s4']
    locationS4_y_true_min[locationS4_y_true_min == 0.] = 1000000
    locationS5_y_true_min = true_positive['s5']
    locationS5_y_true_min[locationS5_y_true_min == 0.] = 1000000

    locationS1_x_false_min = false_positive['s1']
    locationS1_x_false_min[locationS1_x_false_min == 0.] = 1000000
    locationS2_x_false_min = false_positive['s2']
    locationS2_x_false_min[locationS2_x_false_min == 0.] = 1000000
    locationS3_x_false_min = false_positive['s3']
    locationS3_x_false_min[locationS3_x_false_min == 0.] = 1000000
    locationS4_x_false_min = false_positive['s4']
    locationS4_x_false_min[locationS4_x_false_min == 0.] = 1000000
    locationS5_x_false_min = false_positive['s5']
    locationS5_x_false_min[locationS5_x_false_min == 0.] = 1000000

    locationS1_y_false_min = false_positive['s1']
    locationS1_y_false_min[locationS1_y_false_min == 0.] = 1000000
    locationS2_y_false_min = false_positive['s2']
    locationS2_y_false_min[locationS2_y_false_min == 0.] = 1000000
    locationS3_y_false_min = false_positive['s3']
    locationS3_y_false_min[locationS3_y_false_min == 0.] = 1000000
    locationS4_y_false_min = false_positive['s4']
    locationS4_y_false_min[locationS4_y_false_min == 0.] = 1000000
    locationS5_y_false_min = false_positive['s5']
    locationS5_y_false_min[locationS5_y_false_min == 0.] = 1000000

    locationS1_x_true_min = np.argmin(np.nanmin(true_positive['s1'], axis=2), axis=1) - 2
    locationS1_y_true_min = np.argmin(np.nanmin(true_positive['s1'], axis=1), axis=1) - 60
    locationS1_x_false_min = np.argmin(np.nanmin(false_positive['s1'], axis=2), axis=1) - 2
    locationS1_y_false_min = np.argmin(np.nanmin(false_positive['s1'], axis=1), axis=1) - 60

    locationS2_x_true_min = np.argmin(np.nanmin(true_positive['s2'], axis=2), axis=1) - 16
    locationS2_y_true_min = np.argmin(np.nanmin(true_positive['s2'], axis=1), axis=1) - 16
    locationS2_x_false_min = np.argmin(np.nanmin(false_positive['s2'], axis=2), axis=1) - 16
    locationS2_y_false_min = np.argmin(np.nanmin(false_positive['s2'], axis=1), axis=1) - 16

    locationS3_x_true_min = np.argmin(np.nanmin(true_positive['s3'], axis=2), axis=1) - 16
    locationS3_y_true_min = np.argmin(np.nanmin(true_positive['s3'], axis=1), axis=1) - 8
    locationS3_x_false_min = np.argmin(np.nanmin(false_positive['s3'], axis=2), axis=1) - 16
    locationS3_y_false_min = np.argmin(np.nanmin(false_positive['s3'], axis=1), axis=1) - 8

    locationS4_x_true_min = np.argmin(np.nanmin(true_positive['s4'], axis=2), axis=1) - 8
    locationS4_y_true_min = np.argmin(np.nanmin(true_positive['s4'], axis=1), axis=1) - 8
    locationS4_x_false_min = np.argmin(np.nanmin(false_positive['s4'], axis=2), axis=1) - 8
    locationS4_y_false_min = np.argmin(np.nanmin(false_positive['s4'], axis=1), axis=1) - 8

    locationS5_x_true_min = np.argmin(np.nanmin(true_positive['s5'], axis=2), axis=1) - 8
    locationS5_y_true_min = np.argmin(np.nanmin(true_positive['s5'], axis=1), axis=1) - 8
    locationS5_x_false_min = np.argmin(np.nanmin(false_positive['s5'], axis=2), axis=1) - 8
    locationS5_y_false_min = np.argmin(np.nanmin(false_positive['s5'], axis=1), axis=1) - 8
 
    test_decay_mode = test[y_truth == decay_type]
    true_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_type]
    false_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_compare]
# end minimum block

# Deltas between layers
    deltaS1S2_y_true = (locationS1_y_true) - (locationS2_y_true)
    deltaS1S2_x_true = (locationS1_x_true) - (locationS2_x_true)

    deltaS2S3_y_true = locationS2_y_true - locationS3_y_true
    deltaS2S3_x_true = locationS2_x_true - locationS3_x_true

    deltaS1S2_y_false = locationS1_y_false - locationS2_y_false 
    deltaS1S2_x_false = locationS1_x_false - locationS2_x_false 

    deltaS2S3_y_false = locationS2_y_false - locationS3_y_false 
    deltaS2S3_x_false = locationS2_x_false - locationS3_x_false 

    deltaS1S2_y_true_min = locationS1_y_true_min - locationS2_y_true_min
    deltaS1S2_x_true_min = locationS1_x_true_min - locationS2_x_true_min

    deltaS2S3_y_true_min = locationS2_y_true_min - locationS3_y_true_min
    deltaS2S3_x_true_min = locationS2_x_true_min - locationS3_x_true_min

    deltaS1S2_y_false_min = locationS1_y_false_min - locationS2_y_false_min 
    deltaS1S2_x_false_min = locationS1_x_false_min - locationS2_x_false_min 

    deltaS2S3_y_false_min = locationS2_y_false_min - locationS3_y_false_min 
    deltaS2S3_x_false_min = locationS2_x_false_min - locationS3_x_false_min 
# end delta block

# Selection of the best events
    test_decay_mode = test[y_truth == decay_type]
    true_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_type]
    false_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_compare]

    best_Tau_y = np.where(deltaS1S2_y_true == 0)[0]
    best_Tau_x = np.where(deltaS1S2_x_true == 0)[0]

    best = np.intersect1d(best_Tau_x,best_Tau_y)

    if len(best) > 10:
        RANGE = 10
    elif len(best) <= 10:
        RANGE = len(best)

    best_true = np.zeros(RANGE)
    for i in range(RANGE):
        best = np.intersect1d(best_Tau_x,best_Tau_y)
        best = best[i]
        best_event1 = np.where(test_decay_mode['s1'] == true_positive['s1'][best,locationS1_x_true[best]+2,locationS1_y_true[best]+60])[0][0]
        best_event2 = np.where(test_decay_mode['s1'] == true_positive['s1'][best,locationS1_x_true[best]+2,locationS1_y_true[best]+60])[1][0]
        best_event3 = np.where(test_decay_mode['s1'] == true_positive['s1'][best,locationS1_x_true[best]+2,locationS1_y_true[best]+60])[2][0]
        best_true_event = np.where(test['s1'] == test_decay_mode['s1'][best_event1, best_event2, best_event3])[0][0]
        best_true[i] = best_true_event
#    print 'best true tau is event number ', best_true_event

    best_Tau_y = np.where(deltaS1S2_y_false == 0)[0]
    best_Tau_x = np.where(deltaS1S2_x_false == 0)[0]

    best = np.intersect1d(best_Tau_x,best_Tau_y)


    best_false = np.zeros(len(best))
    best_false_scores = np.zeros((len(best),3))


    for i in range(len(best)):
        best = np.intersect1d(best_Tau_x,best_Tau_y)
        best = best[i]
        best_event1 = np.where(test_decay_mode['s1'] == false_positive['s1'][best,locationS1_x_false[best]+2,locationS1_y_false[best]+60])[0][0]
        best_event2 = np.where(test_decay_mode['s1'] == false_positive['s1'][best,locationS1_x_false[best]+2,locationS1_y_false[best]+60])[1][0]
        best_event3 = np.where(test_decay_mode['s1'] == false_positive['s1'][best,locationS1_x_false[best]+2,locationS1_y_false[best]+60])[2][0]
        best_false_event = np.where(test['s1'] == test_decay_mode['s1'][best_event1, best_event2, best_event3])[0][0]
        best_false[i] = best_false_event

        best_false_scores[i][0] = int(best_false[i])
        best_false_scores[i][1] = y_pred[int(best_false[i])][decay_compare]
        best_false_scores[i][2] = y_pred[int(best_false[i])][decay_type]
    best_false_scores = best_false_scores[best_false_scores[:,1].argsort()]

    best_false = best_false_scores[:,0]
    best_false = np.flip(best_false,0)
    best_false = best_false[0:10]

    #arr = y_pred[y_truth == decay_compare]
    #arr = arr[arr[:,decay_compare].argsort()]
    #arr = np.flip(arr,0)
    #print np.where(arr)[0]
    #arr = arr[:,decay_compare]
    #print arr, len(arr)
    #arr = np.where(np.argmax(arr, axis=1) == decay_compare)[0]
    #print arr, len(arr)
    #print arr[0:10]
    #best_false = best_false.astype(int)
    #print np.sort(best_false)

    #sys.exit()

#    print 'best false tau is event number ', best_false_event

# Selection of the worst events
    worst_Tau_y = np.where(deltaS1S2_y_true >= 60)[0]
    worst_Tau_x = np.where(deltaS1S2_y_true >= 8)[0]

    worst = np.intersect1d(worst_Tau_x,worst_Tau_y)
    if len(worst) > 10:
        RANGE = 10
    elif len(worst) <= 10:
        RANGE = len(worst)

    worst_true = np.zeros(RANGE)
    for i in range(RANGE):
        worst= np.intersect1d(worst_Tau_x,worst_Tau_y)
        worst = worst[i]
        worst_event1 = np.where(test_decay_mode['s1'] == true_positive['s1'][worst,locationS1_x_true[worst]+2,locationS1_y_true[worst]+60])[0][0]
        worst_event2 = np.where(test_decay_mode['s1'] == true_positive['s1'][worst,locationS1_x_true[worst]+2,locationS1_y_true[worst]+60])[1][0]
        worst_event3 = np.where(test_decay_mode['s1'] == true_positive['s1'][worst,locationS1_x_true[worst]+2,locationS1_y_true[worst]+60])[2][0]
        worst_true_event = np.where(test['s1'] == test_decay_mode['s1'][worst_event1, worst_event2, worst_event3])[0][0]
        worst_true[i] = worst_true_event
#    print 'worst true tau is event number ', worst_true_event

    worst_Tau_y = np.where(abs(deltaS1S2_y_false) >= 40)[0]
    worst_Tau_x = np.where(abs(deltaS1S2_x_false) >= 8)[0]

    worst = np.intersect1d(worst_Tau_x,worst_Tau_y)

    worst_false = np.zeros(len(worst))
    worst_false_scores = np.zeros((len(worst),3))
    for i in range(len(worst)):
        worst = np.intersect1d(worst_Tau_x,worst_Tau_y)
        worst = worst[i]
        worst_event1 = np.where(test_decay_mode['s1'] == false_positive['s1'][worst,locationS1_x_false[worst]+2,locationS1_y_false[worst]+60])[0][0]
        worst_event2 = np.where(test_decay_mode['s1'] == false_positive['s1'][worst,locationS1_x_false[worst]+2,locationS1_y_false[worst]+60])[1][0]
        worst_event3 = np.where(test_decay_mode['s1'] == false_positive['s1'][worst,locationS1_x_false[worst]+2,locationS1_y_false[worst]+60])[2][0]
        worst_false_event = np.where(test['s1'] == test_decay_mode['s1'][worst_event1, worst_event2, worst_event3])[0][0]
        worst_false[i] = worst_false_event

        worst_false_scores[i][0] = int(worst_false[i])
        worst_false_scores[i][1] = y_pred[int(worst_false[i])][decay_compare]
        worst_false_scores[i][2] = y_pred[int(worst_false[i])][decay_type]
    worst_false_scores = worst_false_scores[worst_false_scores[:,1].argsort()]
    
    worst_false = worst_false_scores[:,0]
    worst_false = np.flip(worst_false,0)
    worst_false = worst_false[0:10]
       
#    print 'worst false tau is event number ', worst_false_event

###########################################################################

    # select only the true 'decay_mode' tau decays
    test_decay_mode = test[y_truth == decay_type]
    y_pred_decay_mode = y_pred[y_truth == decay_type]
    true_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_type]
    false_positive = test_decay_mode[np.argmax(y_pred_decay_mode, axis=1) == decay_compare]
    plt.figure() 
    plt.hist(true_positive['eta'], bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(false_positive['eta'], bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_compare_eta.pdf')

# Isolate the S1 layer information
    testS1_decay_mode = test_decay_mode['s1']
    testS1_decay_mode = np.sum(testS1_decay_mode, axis = (1, 2))
	
    yS1_pred_decay_mode = y_pred[y_truth == decay_type]
    trueS1_positive = testS1_decay_mode[np.argmax(yS1_pred_decay_mode, axis=1) == decay_type]
    falseS1_positive = testS1_decay_mode[np.argmax(yS1_pred_decay_mode, axis=1) == decay_compare]


# Isolate the S2 layer information
    testS2_decay_mode = test_decay_mode['s2']
    testS2_decay_mode = np.sum(testS2_decay_mode, axis = (1, 2))
	
    yS2_pred_decay_mode = y_pred[y_truth == decay_type]
    trueS2_positive = testS2_decay_mode[np.argmax(yS2_pred_decay_mode, axis=1) == decay_type]
    falseS2_positive = testS2_decay_mode[np.argmax(yS2_pred_decay_mode, axis=1) == decay_compare]

# Isolate the S3 layer information
    testS3_decay_mode = test_decay_mode['s3']
    testS3_decay_mode = np.sum(testS3_decay_mode, axis = (1, 2))
    	
    yS3_pred_decay_mode = y_pred[y_truth == decay_type]
    trueS3_positive = testS3_decay_mode[np.argmax(yS3_pred_decay_mode, axis=1) == decay_type]
    falseS3_positive = testS3_decay_mode[np.argmax(yS3_pred_decay_mode, axis=1) == decay_compare]

# Isolate the S4 layer information
    testS4_decay_mode = test_decay_mode['s4']
    testS4_decay_mode = np.sum(testS4_decay_mode, axis = (1, 2))
    	
    yS4_pred_decay_mode = y_pred[y_truth == decay_type]
    trueS4_positive = testS4_decay_mode[np.argmax(yS4_pred_decay_mode, axis=1) == decay_type]
    falseS4_positive = testS4_decay_mode[np.argmax(yS4_pred_decay_mode, axis=1) == decay_compare]

# Isolate the S5 layer information
    testS5_decay_mode = test_decay_mode['s5']
    testS5_decay_mode = np.sum(testS5_decay_mode, axis = (1, 2))
    	
    yS5_pred_decay_mode = y_pred[y_truth == decay_type]
    trueS5_positive = testS5_decay_mode[np.argmax(yS5_pred_decay_mode, axis=1) == decay_type]
    falseS5_positive = testS5_decay_mode[np.argmax(yS5_pred_decay_mode, axis=1) == decay_compare]

# Initialize the ratio arrays    
    
    S1S2ratio_True = np.divide(trueS1_positive,trueS2_positive)
    S1S2ratio_False = np.divide(falseS1_positive,falseS2_positive)
    S2S3ratio_True = np.divide(trueS3_positive,trueS2_positive)
    S2S3ratio_False = np.divide(falseS3_positive,falseS2_positive)
    S4S3ratio_True = np.divide(trueS4_positive,trueS3_positive)
    S4S3ratio_False = np.divide(falseS4_positive,falseS3_positive)
    S5S4ratio_True = np.divide(trueS5_positive,trueS4_positive)
    S5S4ratio_False = np.divide(falseS5_positive,falseS4_positive) 

# Deletes nan and inf instance from array, maybe shouldn't be done?  Done to assist with histogram plotting
    S1S2ratio_True = S1S2ratio_True[np.logical_not(np.isnan(S1S2ratio_True))]
    S1S2ratio_True = S1S2ratio_True[np.logical_not(np.isinf(S1S2ratio_True))]
    S1S2ratio_False = S1S2ratio_False [np.logical_not(np.isnan(S1S2ratio_False ))]
    S1S2ratio_False = S1S2ratio_False [np.logical_not(np.isinf(S1S2ratio_False ))]
   
    S2S3ratio_True = S2S3ratio_True[np.logical_not(np.isnan(S2S3ratio_True))]
    S2S3ratio_True = S2S3ratio_True[np.logical_not(np.isinf(S2S3ratio_True))]
    S2S3ratio_False = S2S3ratio_False [np.logical_not(np.isnan(S2S3ratio_False ))]
    S2S3ratio_False = S2S3ratio_False [np.logical_not(np.isinf(S2S3ratio_False ))]

    S4S3ratio_True = S4S3ratio_True[np.logical_not(np.isnan(S4S3ratio_True))]
    S4S3ratio_True = S4S3ratio_True[np.logical_not(np.isinf(S4S3ratio_True))]
    S4S3ratio_False = S4S3ratio_False [np.logical_not(np.isnan(S4S3ratio_False ))]
    S4S3ratio_False = S4S3ratio_False [np.logical_not(np.isinf(S4S3ratio_False ))]
   
    S5S4ratio_True = S5S4ratio_True[np.logical_not(np.isnan(S5S4ratio_True))]
    S5S4ratio_True = S5S4ratio_True[np.logical_not(np.isinf(S5S4ratio_True))]
    S5S4ratio_False = S5S4ratio_False [np.logical_not(np.isnan(S5S4ratio_False ))]
    S5S4ratio_False = S5S4ratio_False [np.logical_not(np.isinf(S5S4ratio_False ))]

# delta eta tracks
    true_positive_deta_leadtrack = true_positive['tracks'][:, 0, 1]
    false_positive_deta_leadtrack = false_positive['tracks'][:, 0, 1]

# delta phi tracks
    true_positive_dphi_leadtrack = true_positive['tracks'][:, 0, 2]
    false_positive_dphi_leadtrack = false_positive['tracks'][:, 0, 2]

# Creating the histogram plots of the energy ratio.
    plt.figure() 
    plt.hist(S1S2ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S1S2ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s1/s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s1s2ratio.pdf')
    plt.close()

    plt.figure() 
    plt.hist(S2S3ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S2S3ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s3/s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s2s3ratio.pdf')
    plt.close()

    plt.figure() 
    plt.hist(S4S3ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S4S3ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s4/s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s4s3ratio.pdf')
    plt.close()

    plt.figure() 
    plt.hist(S5S4ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S5S4ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s5/s4')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s5s4ratio.pdf')
    plt.close()


# Maximum location Plots
    plt.figure() 
    plt.hist(locationS1_x_true, bins=61, range=(-1, 60), color = 'blue', label='s1 true peak location', density = True)
    plt.hist(locationS2_x_true, bins=61, range=(-1, 60), color = 'red', label='s2 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS3_x_true, bins=61, range=(-1, 60), color = 'green', label='s3 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_x_true, bins=61, range=(-1, 60), color = 'yellow', label='s4 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_x_true, bins=61, range=(-1, 60), color = 'brown', label='s5 true peak location', density = True, alpha = 0.4)
    plt.xlabel('x location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_all_X_true_Location.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_y_true, bins=121, range=(-1, 120), color = 'blue', label='s1 true peak locations', density = True)
    plt.hist(locationS2_y_true, bins=121, range=(-1, 120), color = 'red', label='s2 true peak locations', density = True, alpha = 0.4)
    plt.hist(locationS3_y_true, bins=121, range=(-1, 120), color = 'green', label='s3 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_y_true, bins=121, range=(-1, 120), color = 'yellow', label='s4 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_y_true, bins=121, range=(-1, 120), color = 'brown', label='s5 true peak location', density = True, alpha = 0.4)
    plt.xlabel('y location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_all_Y_true_Location.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_x_false, bins=4, range=(-2, 2), color = 'blue', label='s1 false peak location', density = True)
    plt.hist(locationS2_x_false, bins=32, range=(-16, 16), color = 'red', label='s2 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS3_x_false, bins=32, range=(-16, 16), color = 'green', label='s3 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_x_false, bins=16, range=(-8, 8), color = 'yellow', label='s4 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_x_false, bins=16, range=(-8, 8), color = 'brown', label='s5 false peak location', density = True, alpha = 0.4)
    plt.xlabel('x location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_all_X_false_Location.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_y_false, bins=120, range=(-60, 60), color = 'blue', label='s1 false peak locations', density = True)
    plt.hist(locationS2_y_false, bins=32, range=(-16, 16), color = 'red', label='s2 false peak locations', density = True, alpha = 0.4)
    plt.hist(locationS3_y_false, bins=32, range=(-16, 16), color = 'green', label='s3 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_y_false, bins=16, range=(-8, 8), color = 'yellow', label='s4 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_y_false, bins=16, range=(-8, 8), color = 'brown', label='s5 false peak location', density = True, alpha = 0.4)
    plt.xlabel('y location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_all_Y_false_Location.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_x_true, bins=4, range=(-2, 2), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_x_false, bins=4, range=(-2, 2), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s1_peak_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_y_true, bins=120, range=(-60, 60), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_y_false, bins=120, range=(-60, 60), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s1_peak_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS2_x_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_x_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s2_peak_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS2_y_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_y_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s2_peak_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS3_x_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_x_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s3_peak_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS3_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s3_peak_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS4_x_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_x_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s4_peak_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS4_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s4_peak_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS5_x_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_x_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s5_peak_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS5_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s5_peak_y_locations.pdf')
    plt.close()

# Minimum Plots
    plt.figure() 
    plt.hist(locationS1_x_true_min, bins=16, range=(-2, 2), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_x_false_min, bins=16, range=(-2, 2), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s1_min_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS1_y_true_min, bins=120, range=(-60, 60), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_y_false_min, bins=120, range=(-60, 60), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s1_min_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS2_x_true_min, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_x_false_min, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 min x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s2_min_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS2_y_true_min, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_y_false_min, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 min y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s2_min_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS3_x_true_min, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_x_false_min, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s3_min_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS3_y_true_min, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_y_false_min, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s3_min_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS4_x_true_min, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_x_false_min, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s4_min_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS4_y_true_min, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_y_false_min, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s4_min_y_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS5_x_true_min, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_x_false_min, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak phi')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s5_min_x_locations.pdf')
    plt.close()

    plt.figure() 
    plt.hist(locationS5_y_true_min, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_y_false_min, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + '_s5_min_y_locations.pdf')
    plt.close()
# end minimum plots

# delta plots
    plt.figure() 
    plt.hist(deltaS1S2_y_true, bins=152, range=(-76, 76), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_y_false , bins=152, range=(-76, 76), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS1S2_y.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS1S2_x_true, bins=40, range=(-20, 20), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_x_false , bins=40, range=(-20, 20), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS1S2_x.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS2S3_y_true, bins=64, range=(-32, 32), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_y_false , bins=64, range=(-32, 32), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS2S3_y.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS2S3_x_true, bins=48, range=(-24, 24), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_x_false , bins=48, range=(-24, 24), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS2S3_x.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS1S2_y_true_min, bins=152, range=(-76, 76), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_y_false_min , bins=152, range=(-76, 76), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS1S2_y_min.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS1S2_x_true_min, bins=40, range=(-20, 20), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_x_false_min , bins=40, range=(-20, 20), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS1S2_x_min.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS2S3_y_true_min, bins=64, range=(-32, 32), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_y_false_min, bins=64, range=(-32, 32), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS2S3_y_min.pdf')
    plt.close()

    plt.figure() 
    plt.hist(deltaS2S3_x_true_min, bins=48, range=(-24, 24), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_x_false_min, bins=48, range=(-24, 24), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'deltaS2S3_x_min.pdf')
    plt.close()

# deta plots
    plt.figure() 
    plt.hist(true_positive_deta_leadtrack , bins=60, range=(-0.3, 0.3), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(false_positive_deta_leadtrack , bins=60, range=(-0.3, 0.3), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('dEta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'dEta.pdf')
    plt.close()

# dphi plots
    plt.figure() 
    plt.hist(true_positive_dphi_leadtrack , bins=60, range=(-0.3, 0.3), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(false_positive_dphi_leadtrack , bins=60, range=(-0.3, 0.3), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('dEta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/' + decay_mode + '/' + decay_mode + 'dPhi.pdf')
    plt.close()

# return best and worst true/false events
    return(best_true, best_false, worst_true, worst_false)
