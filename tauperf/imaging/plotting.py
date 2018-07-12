import math
import numpy as np
import itertools
import socket
import matplotlib as mpl;
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

def compare_bins(test, y_pred, y_truth):
###########################################################################
#new stuff
#np.unravel_index() obtains the indice of the array for where a value is located
###########################################################################

# Grab just the TRUE 1p0n decays, same as done originally
    y_pred_1p0n = y_pred[y_truth == 0]
    test_1p0n = test[y_truth == 0]
    true_positive = test_1p0n[np.argmax(y_pred_1p0n, axis=1) == 0]
    false_positive = test_1p0n[np.argmax(y_pred_1p0n, axis=1) == 1]
    RANGE = max(len(true_positive), len(false_positive))

# initialize arrays for each calorimeter layer cell location
    locationS1_x_true = []
    locationS2_x_true = []
    locationS3_x_true = []
    locationS4_x_true = []
    locationS5_x_true = []

    locationS1_y_true = []
    locationS2_y_true = []
    locationS3_y_true = []
    locationS4_y_true = []
    locationS5_y_true = []

    locationS1_x_false = []
    locationS2_x_false = []
    locationS3_x_false = []
    locationS4_x_false = []
    locationS5_x_false = []

    locationS1_y_false = []
    locationS2_y_false = []
    locationS3_y_false = []
    locationS4_y_false = []
    locationS5_y_false = []

# Loop over the maximum amount of true / false positive


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

    deltaS1S2_y_true = locationS1_y_true - locationS2_y_true
    deltaS1S2_x_true = locationS1_x_true - locationS2_x_true

    deltaS2S3_y_true = locationS2_y_true - locationS3_y_true
    deltaS2S3_x_true = locationS2_x_true - locationS3_x_true

    deltaS1S2_y_false = locationS1_y_false - locationS2_y_false 
    deltaS1S2_x_false = locationS1_x_false - locationS2_x_false 

    deltaS2S3_y_false = locationS2_y_false - locationS3_y_false 
    deltaS2S3_x_false = locationS2_x_false - locationS3_x_false 

###########################################################################

    # select only the true 1p0n tau decays
    test_1p0n = test[y_truth == 0]
    y_pred_1p0n = y_pred[y_truth == 0]
    true_positive = test_1p0n[np.argmax(y_pred_1p0n, axis=1) == 0]
    false_positive = test_1p0n[np.argmax(y_pred_1p0n, axis=1) == 1]
    plt.figure() 
    plt.hist(true_positive['eta'], bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(false_positive['eta'], bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('eta')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_compare_eta.pdf')

# Isolate the S1 layer information
    testS1_1p0n = test_1p0n['s1']
    testS1_1p0n = np.sum(testS1_1p0n, axis = (1, 2))
	
    yS1_pred_1p0n = y_pred[y_truth == 0]
    trueS1_positive = testS1_1p0n[np.argmax(yS1_pred_1p0n, axis=1) == 0]
    falseS1_positive = testS1_1p0n[np.argmax(yS1_pred_1p0n, axis=1) == 1]


# Isolate the S2 layer information
    testS2_1p0n = test_1p0n['s2']
    testS2_1p0n = np.sum(testS2_1p0n, axis = (1, 2))
	
    yS2_pred_1p0n = y_pred[y_truth == 0]
    trueS2_positive = testS2_1p0n[np.argmax(yS2_pred_1p0n, axis=1) == 0]
    falseS2_positive = testS2_1p0n[np.argmax(yS2_pred_1p0n, axis=1) == 1]

# Isolate the S3 layer information
    testS3_1p0n = test_1p0n['s3']
    testS3_1p0n = np.sum(testS3_1p0n, axis = (1, 2))
    	
    yS3_pred_1p0n = y_pred[y_truth == 0]
    trueS3_positive = testS3_1p0n[np.argmax(yS3_pred_1p0n, axis=1) == 0]
    falseS3_positive = testS3_1p0n[np.argmax(yS3_pred_1p0n, axis=1) == 1]

# Isolate the S4 layer information
    testS4_1p0n = test_1p0n['s4']
    testS4_1p0n = np.sum(testS4_1p0n, axis = (1, 2))
    	
    yS4_pred_1p0n = y_pred[y_truth == 0]
    trueS4_positive = testS4_1p0n[np.argmax(yS4_pred_1p0n, axis=1) == 0]
    falseS4_positive = testS4_1p0n[np.argmax(yS4_pred_1p0n, axis=1) == 1]

# Isolate the S3 layer information
    testS5_1p0n = test_1p0n['s5']
    testS5_1p0n = np.sum(testS5_1p0n, axis = (1, 2))
    	
    yS5_pred_1p0n = y_pred[y_truth == 0]
    trueS5_positive = testS5_1p0n[np.argmax(yS5_pred_1p0n, axis=1) == 0]
    falseS5_positive = testS5_1p0n[np.argmax(yS5_pred_1p0n, axis=1) == 1]

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

# Creating the histogram plots of the energy ratio.
    plt.figure() 
    plt.hist(S1S2ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S1S2ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s1/s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s1s2ratio.pdf')

    plt.figure() 
    plt.hist(S2S3ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S2S3ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s3/s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s2s3ratio.pdf')

    plt.figure() 
    plt.hist(S4S3ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S4S3ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s4/s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s4s3ratio.pdf')

    plt.figure() 
    plt.hist(S5S4ratio_True, bins=44, range=(-1.1, 1.1), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(S5S4ratio_False, bins=44, range=(-1.1, 1.1), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('energy ratio s5/s4')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s5s4ratio.pdf')

    plt.figure() 
    plt.hist(locationS1_x_true, bins=61, range=(-1, 60), color = 'blue', label='s1 true peak location', density = True)
    plt.hist(locationS2_x_true, bins=61, range=(-1, 60), color = 'red', label='s2 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS3_x_true, bins=61, range=(-1, 60), color = 'green', label='s3 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_x_true, bins=61, range=(-1, 60), color = 'yellow', label='s4 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_x_true, bins=61, range=(-1, 60), color = 'brown', label='s5 true peak location', density = True, alpha = 0.4)
    plt.xlabel('x location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_all_X_true_Location.pdf')

    plt.figure() 
    plt.hist(locationS1_y_true, bins=121, range=(-1, 120), color = 'blue', label='s1 true peak locations', density = True)
    plt.hist(locationS2_y_true, bins=121, range=(-1, 120), color = 'red', label='s2 true peak locations', density = True, alpha = 0.4)
    plt.hist(locationS3_y_true, bins=121, range=(-1, 120), color = 'green', label='s3 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_y_true, bins=121, range=(-1, 120), color = 'yellow', label='s4 true peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_y_true, bins=121, range=(-1, 120), color = 'brown', label='s5 true peak location', density = True, alpha = 0.4)
    plt.xlabel('y location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_all_Y_true_Location.pdf')

    plt.figure() 
    plt.hist(locationS1_x_false, bins=4, range=(-2, 2), color = 'blue', label='s1 false peak location', density = True)
    plt.hist(locationS2_x_false, bins=32, range=(-16, 16), color = 'red', label='s2 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS3_x_false, bins=32, range=(-16, 16), color = 'green', label='s3 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_x_false, bins=16, range=(-8, 8), color = 'yellow', label='s4 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_x_false, bins=16, range=(-8, 8), color = 'brown', label='s5 false peak location', density = True, alpha = 0.4)
    plt.xlabel('x location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_all_X_false_Location.pdf')

    plt.figure() 
    plt.hist(locationS1_y_false, bins=120, range=(-60, 60), color = 'blue', label='s1 false peak locations', density = True)
    plt.hist(locationS2_y_false, bins=32, range=(-16, 16), color = 'red', label='s2 false peak locations', density = True, alpha = 0.4)
    plt.hist(locationS3_y_false, bins=32, range=(-16, 16), color = 'green', label='s3 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS4_y_false, bins=16, range=(-8, 8), color = 'yellow', label='s4 false peak location', density = True, alpha = 0.4)
    plt.hist(locationS5_y_false, bins=16, range=(-8, 8), color = 'brown', label='s5 false peak location', density = True, alpha = 0.4)
    plt.xlabel('y location')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_all_Y_false_Location.pdf')

    plt.figure() 
    plt.hist(locationS1_x_true, bins=4, range=(-2, 2), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_x_false, bins=4, range=(-2, 2), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s1_peak_x_locations.pdf')

    plt.figure() 
    plt.hist(locationS1_y_true, bins=120, range=(-60, 60), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS1_y_false, bins=120, range=(-60, 60), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s1_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(locationS2_x_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_x_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 peak x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s2_peak_x_locations.pdf')

    plt.figure() 
    plt.hist(locationS2_y_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS2_y_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s2_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(locationS3_x_true, bins=32, range=(-16, 16), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_x_false, bins=32, range=(-16, 16), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s3_peak_x_locations.pdf')

    plt.figure() 
    plt.hist(locationS3_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS3_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s3 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s3_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(locationS4_x_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_x_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s4_peak_x_locations.pdf')

    plt.figure() 
    plt.hist(locationS4_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS4_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s4 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s4_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(locationS5_x_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_x_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak x')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s5_peak_x_locations.pdf')

    plt.figure() 
    plt.hist(locationS5_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s5_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(locationS5_y_true, bins=16, range=(-8, 8), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(locationS5_y_false, bins=16, range=(-8, 8), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s5 peak y')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/1p0n_s5_peak_y_locations.pdf')

    plt.figure() 
    plt.hist(deltaS1S2_y_true, bins=152, range=(-76, 76), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_y_false , bins=152, range=(-76, 76), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/deltaS1S2_y.pdf')

    plt.figure() 
    plt.hist(deltaS1S2_x_true, bins=40, range=(-20, 20), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS1S2_x_false , bins=40, range=(-20, 20), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s1 - s2')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/deltaS1S2_x.pdf')

    plt.figure() 
    plt.hist(deltaS2S3_y_true, bins=64, range=(-32, 32), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_y_false , bins=64, range=(-32, 32), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/deltaS2S3_y.pdf')

    plt.figure() 
    plt.hist(deltaS2S3_x_true, bins=48, range=(-24, 24), color = 'blue', label='1p0n identified as 1p0n', density = True)
    plt.hist(deltaS2S3_x_false , bins=48, range=(-24, 24), color = 'red', label='1p0n identified as 1p1n', density = True, alpha = 0.4)
    plt.xlabel('s2 - s3')
    plt.ylabel('A. U.')
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/deltaS2S3_x.pdf')
