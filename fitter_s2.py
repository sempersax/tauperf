import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import roc_curve
from keras.models import load_model

from tauperf import log; log = log['/fitter']
from tauperf.imaging.models import single_layer_model_s2
from tauperf.imaging.utils import fit_model_single_layer


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--no-train', default=False, action='store_true')
parser.add_argument(
    '--no-train-pi0', default=False, action='store_true')
parser.add_argument(
    '--no-train-twopi0', default=False, action='store_true')
parser.add_argument(
    '--overwrite', default=False, action='store_true')
args = parser.parse_args()


log.info('loading data...')
data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v6')
                        

images_1p0n  = np.load(os.path.join(data_dir, 'images_new_1p0n.npy'))
images_1p1n  = np.load(os.path.join(data_dir, 'images_new_1p1n.npy'))
images_1p2n  = np.load(os.path.join(data_dir, 'images_new_1p2n.npy'))


log.info('splitting')

train_1p0n, test_1p0n = model_selection.train_test_split(
    images_1p0n, test_size=0.3, random_state=42)
val_1p0n, test_1p0n = np.split(test_1p0n, [len(test_1p0n) / 2])

train_1p1n, test_1p1n = model_selection.train_test_split(
    images_1p1n, test_size=0.3, random_state=42)
val_1p1n, test_1p1n = np.split(test_1p1n, [len(test_1p1n) / 2])

train_1p2n, test_1p2n = model_selection.train_test_split(
    images_1p2n, test_size=0.3, random_state=42)
val_1p2n, test_1p2n = np.split(test_1p2n, [len(test_1p2n) / 2])


log.info('samples ..')
headers = ["sample", "Total", "Training", "Validation", "Testing"]
sample_size_table = [
    ['1p0n', len(images_1p0n), len(train_1p0n), len(val_1p0n), len(test_1p0n)],
    ['1p1n', len(images_1p1n), len(train_1p1n), len(val_1p1n), len(test_1p1n)],
    ['1p2n', len(images_1p2n), len(train_1p2n), len(val_1p2n), len(test_1p2n)],
]

log.info('')
print tabulate(sample_size_table, headers=headers, tablefmt='simple')
log.info('')

# training/testing samples for 1p1n against 1p0n
train_pi0 = np.concatenate((train_1p0n, train_1p1n, train_1p2n))
test_pi0  = np.concatenate((test_1p0n, test_1p1n, test_1p2n))
val_pi0   = np.concatenate((val_1p0n, val_1p1n, val_1p2n))

y_train_pi0 = np.concatenate((
    np.zeros(train_1p0n.shape, dtype=np.uint8),
    np.ones(train_1p1n.shape, dtype=np.uint8),
    np.ones(train_1p2n.shape, dtype=np.uint8)))

y_test_pi0 = np.concatenate((
    np.zeros(test_1p0n.shape, dtype=np.uint8),
    np.ones(test_1p1n.shape, dtype=np.uint8),
    np.ones(test_1p2n.shape, dtype=np.uint8)))

y_val_pi0 = np.concatenate((
    np.zeros(val_1p0n.shape, dtype=np.uint8),
    np.ones(val_1p1n.shape, dtype=np.uint8),
    np.ones(val_1p2n.shape, dtype=np.uint8)))


# training/testing samples for 1p1n against 1p2n
train_twopi0 = np.concatenate((train_1p1n, train_1p2n))
test_twopi0  = np.concatenate((test_1p1n, test_1p2n))       
val_twopi0   = np.concatenate((val_1p1n, val_1p2n))

y_train_twopi0 = np.concatenate((
    np.ones(train_1p1n.shape, dtype=np.uint8),
    np.zeros(train_1p2n.shape, dtype=np.uint8)))

y_test_twopi0 = np.concatenate((
    np.ones(test_1p1n.shape, dtype=np.uint8),
    np.zeros(test_1p2n.shape, dtype=np.uint8)))

y_val_twopi0 = np.concatenate((
    np.ones(val_1p1n.shape, dtype=np.uint8),
    np.zeros(val_1p2n.shape, dtype=np.uint8)))


# ##############################################
log.info('training stuff')
model_pi0_filename = 'cache/crackpot_s2_only_pi0.h5'
if args.no_train or args.no_train_pi0:
    model_pi0 = load_model(model_pi0_filename)
else:
    model_pi0 = single_layer_model_s2(train_pi0['s2'])
    fit_model_single_layer(
        model_pi0,
        train_pi0['s2'], y_train_pi0,
        val_pi0['s2'], y_val_pi0,
        filename=model_pi0_filename,
        overwrite=args.overwrite,
        no_train=args.no_train or args.no_train_pi0)

model_twopi0_filename = 'cache/crackpot_s2_only_twopi0.h5'
if args.no_train or args.no_train_twopi0:
    model_twopi0 = load_model(model_twopi0_filename)
else:
    model_twopi0 = single_layer_model_s2(train_twopi0['s2'])
    fit_model_single_layer(
        model_twopi0,
        train_twopi0['s2'], y_train_twopi0,
        val_twopi0['s2'], y_val_twopi0,
        filename=model_twopi0_filename,
        overwrite=args.overwrite,
        no_train=args.no_train or args.no_train_twopi0)



# ##############################################
log.info('testing stuff')

log.info('compute 1p1n vs 1p0n classifier scores')
y_pred_pi0 = model_pi0.predict(
        test_pi0['s2'], 
        batch_size=32, verbose=1)

log.info('compute 1p1n vs 1p2n classifier scores')
y_pred_twopi0 = model_twopi0.predict(
        test_twopi0['s2'], 
        batch_size=32, verbose=1)
print
# ######################
log.info('Drawing the roc curve')
from tauperf.imaging.plotting import plot_confusion_matrix, get_eff, get_wp

fptr_1p1n, tpr_1p1n, thresh_1p1n = roc_curve(y_test_pi0, y_pred_pi0)
opt_fptr_1p1n, opt_tpr_1p1n, opt_thresh_1p1n = get_wp(
    fptr_1p1n, tpr_1p1n, thresh_1p1n, method='target_eff')
log.info('1p1n vs 1p0n: cutting on the score at {0}'.format(opt_thresh_1p1n))

fptr_1p2n, tpr_1p2n, thresh_1p2n = roc_curve(y_test_twopi0, y_pred_twopi0)
opt_fptr_1p2n, opt_tpr_1p2n, opt_thresh_1p2n = get_wp(
    fptr_1p2n, tpr_1p2n, thresh_1p2n, method='target_rej')
log.info('1p1n vs 1p2n: cutting on the score at {0}'.format(opt_thresh_1p2n))



plt.figure()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fptr_1p1n, tpr_1p1n, color='red', label='1pXn vs 1p0n')
plt.plot(fptr_1p2n, tpr_1p2n, color='blue', label='1p1n vs 1p2n')
plt.plot([opt_fptr_1p1n, opt_fptr_1p2n],
          [opt_tpr_1p1n, opt_tpr_1p2n], 'go',
         label='working points')
plt.xlabel('miss-classification rate')
plt.ylabel('classification efficiency')
plt.title('classification with calo sampling s1, s2 and s3')
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.savefig('./plots/imaging/roc_curve_s2.pdf')


# ######################
log.info('Drawing the confusion matrix')
X_test = np.concatenate((test_1p0n, test_1p1n, test_1p2n))

score_pi0 = model_pi0.predict(
    X_test['s2'], 
    batch_size=32, verbose=1)

score_twopi0 = model_twopi0.predict(
    X_test['s2'], 
    batch_size=32, verbose=1)

pred_pi0 = score_pi0 > opt_thresh_1p1n
pred_twopi0 = score_twopi0 < opt_thresh_1p2n

y_true = np.concatenate((
        np.zeros(test_1p0n.shape, dtype=np.uint8),
        np.ones(test_1p1n.shape, dtype=np.uint8),
        np.ones(test_1p2n.shape, dtype=np.uint8) + 1))

from tauperf.imaging.evaluate import matrix_1p
cm = matrix_1p(y_true, pred_pi0, pred_twopi0)
class_names = ['1p0n', '1p1n', '1p2n']
plt.figure()
plot_confusion_matrix(
    cm, classes=class_names, 
    title='Confusion matrix with sampling s1, s2 and s3',
    name='plots/imaging/confusion_matrix_s2.pdf')


# ######################
log.info('Drawing the efficiency plots')
from rootpy.plotting import root2matplotlib as rmpl
from rootpy.plotting.style import set_style
set_style('ATLAS', mpl=True)


# pred_pi0 = pred_pi0.reshape((whapred_pi0.shape[0],))
# pred_twopi0 = pred_twopi0.reshape((pred_twopi0.shape[0],))


# 1p1n vs 1p0n - pt 
eff_pi0_pt_1p1n = get_eff(
    test_1p1n['pt'].reshape((test_1p1n.shape[0])), pred_pi0[y_true == 1], 
    scale=0.001, color='red', name='1p1n')

eff_pi0_pt_1p0n = get_eff(
    test_1p0n['pt'].reshape((test_1p0n.shape[0])), pred_pi0[y_true == 0] == 0, 
    scale=0.001, color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_pt_1p1n.painted_graph, eff_pi0_pt_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Transverse Momentum [GeV]')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_pt_s2.pdf')

# 1p1n vs 1p0n - eta
eff_pi0_eta_1p1n = get_eff(
    test_1p1n['eta'].reshape((test_1p1n.shape[0])), pred_pi0[y_true == 1], 
    binning=(10, -1.1, 1.1), color='red', name='1p1n')

eff_pi0_eta_1p0n = get_eff(
    test_1p0n['eta'].reshape((test_1p0n.shape[0])), pred_pi0[y_true == 0] == 0, 
    binning=(10, -1.1, 1.1), color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_eta_1p1n.painted_graph, eff_pi0_eta_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Pseudorapidity')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_eta_s2.pdf')

# 1p1n vs 1p0n - mu
eff_pi0_mu_1p1n = get_eff(
    test_1p1n['mu'].reshape((test_1p1n.shape[0])), pred_pi0[y_true == 1], 
    binning=(10, 0, 40), color='red', name='1p1n')

eff_pi0_mu_1p0n = get_eff(
    test_1p0n['mu'].reshape((test_1p0n.shape[0])), pred_pi0[y_true == 0] == 0, 
    binning=(10, 0, 40), color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_mu_1p1n.painted_graph, eff_pi0_mu_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Average Interaction Per Bunch Crossing')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_mu_s2.pdf')

# 1p1n vs 1p2n - pt 
eff_twopi0_pt_1p1n = get_eff(
    test_1p1n['pt'].reshape((test_1p1n.shape[0])), pred_twopi0[y_true == 1] == 0, 
    scale=0.001, color='red', name='1p1n')

eff_twopi0_pt_1p2n = get_eff(
    test_1p2n['pt'].reshape((test_1p2n.shape[0])), pred_twopi0[y_true == 2], 
    scale=0.001, color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_pt_1p1n.painted_graph, eff_twopi0_pt_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Transverse Momentum [GeV]')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_pt_s2.pdf')

# 1p1n vs 1p2n - eta
eff_twopi0_eta_1p1n = get_eff(
    test_1p1n['eta'].reshape((test_1p1n.shape[0])), pred_twopi0[y_true == 1] == 0, 
    binning=(10, -1.1, 1.1), color='red', name='1p1n')

eff_twopi0_eta_1p2n = get_eff(
    test_1p2n['eta'].reshape((test_1p2n.shape[0])), pred_twopi0[y_true == 2], 
    binning=(10, -1.1, 1.1), color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_eta_1p1n.painted_graph, eff_twopi0_eta_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Pseudorapidity')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_eta_s2.pdf')

# 1p1n vs 1p2n - mu
eff_twopi0_mu_1p1n = get_eff(
    test_1p1n['mu'].reshape((test_1p1n.shape[0])), pred_twopi0[y_true == 1] == 0, 
    binning=(10, 0, 40), color='red', name='1p1n')

eff_twopi0_mu_1p2n = get_eff(
    test_1p2n['mu'].reshape((test_1p2n.shape[0])), pred_twopi0[y_true == 2], 
    binning=(10, 0, 40), color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_mu_1p1n.painted_graph, eff_twopi0_mu_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Average Interaction Per Bunch Crossing')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_mu_s2.pdf')
