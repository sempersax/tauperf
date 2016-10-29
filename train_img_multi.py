import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import roc_curve

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

from tauperf import log; log = log['/train-img-multi']
from tauperf.imaging.plotting import plot_confusion_matrix, get_eff, get_wp
from tauperf.imaging.models import binary_2d_model, dense_merged_model
from tauperf.imaging.processing import prepare_train_test, create_record

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--no-train', default=False, action='store_true')
args = parser.parse_args()


log.info('loading data...')
data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v6')
                        
arr_1p0n_s1 = np.load(os.path.join(data_dir, 'images_S1_1p0n.npy'))
arr_1p0n_s2 = np.load(os.path.join(data_dir, 'images_S2_1p0n.npy'))
arr_1p0n_s3 = np.load(os.path.join(data_dir, 'images_S3_1p0n.npy'))

arr_1p1n_s1 = np.load(os.path.join(data_dir, 'images_S1_1p1n.npy'))
arr_1p1n_s2 = np.load(os.path.join(data_dir, 'images_S2_1p1n.npy'))
arr_1p1n_s3 = np.load(os.path.join(data_dir, 'images_S3_1p1n.npy'))

arr_1p2n_s1 = np.load(os.path.join(data_dir, 'images_S1_1p2n.npy'))
arr_1p2n_s2 = np.load(os.path.join(data_dir, 'images_S2_1p2n.npy'))
arr_1p2n_s3 = np.load(os.path.join(data_dir, 'images_S3_1p2n.npy'))

arr_1p0n_ki = np.load(os.path.join(data_dir, 'kinematics_1p0n.npy'))
arr_1p1n_ki = np.load(os.path.join(data_dir, 'kinematics_1p1n.npy'))
arr_1p2n_ki = np.load(os.path.join(data_dir, 'kinematics_1p2n.npy'))


log.info('splitting')

(train_1p0n_s1, test_1p0n_s1,
 train_1p0n_s2, test_1p0n_s2,
 train_1p0n_s3, test_1p0n_s3,
 train_1p0n_ki, test_1p0n_ki) = prepare_train_test(arr_1p0n_s1, arr_1p0n_s2, arr_1p0n_s3, arr_1p0n_ki)

(train_1p1n_s1, test_1p1n_s1,
 train_1p1n_s2, test_1p1n_s2,
 train_1p1n_s3, test_1p1n_s3,
 train_1p1n_ki, test_1p1n_ki) = prepare_train_test(arr_1p1n_s1, arr_1p1n_s2, arr_1p1n_s3, arr_1p1n_ki)

(train_1p2n_s1, test_1p2n_s1,
 train_1p2n_s2, test_1p2n_s2,
 train_1p2n_s3, test_1p2n_s3,
 train_1p2n_ki, test_1p2n_ki) = prepare_train_test(arr_1p2n_s1, arr_1p2n_s2, arr_1p2n_s3, arr_1p2n_ki)


# training samples for 1p1n agains 1p0n
train_pi0_s1 = np.concatenate((train_1p0n_s1, train_1p1n_s1))
train_pi0_s2 = np.concatenate((train_1p0n_s2, train_1p1n_s2))
train_pi0_s3 = np.concatenate((train_1p0n_s3, train_1p1n_s3))

test_pi0_s1 = np.concatenate((test_1p0n_s1, test_1p1n_s1))
test_pi0_s2 = np.concatenate((test_1p0n_s2, test_1p1n_s2))
test_pi0_s3 = np.concatenate((test_1p0n_s3, test_1p1n_s3))

target_pi0 = np.concatenate((
        np.zeros(train_1p0n_s1.shape[0], dtype=np.uint8),
        np.ones(train_1p1n_s1.shape[0], dtype=np.uint8)))
target_pi0 = target_pi0.reshape((target_pi0.shape[0], 1))
test_pi0 = np.concatenate((
        np.zeros(test_1p0n_s1.shape[0], dtype=np.uint8),
        np.ones(test_1p1n_s1.shape[0], dtype=np.uint8)))
test_pi0 = test_pi0.reshape((test_pi0.shape[0], 1))

# training samples for 1p1n agains 1p2n
train_twopi0_s1 = np.concatenate((train_1p2n_s1, train_1p1n_s1))
train_twopi0_s2 = np.concatenate((train_1p2n_s2, train_1p1n_s2))
train_twopi0_s3 = np.concatenate((train_1p2n_s3, train_1p1n_s3))

test_twopi0_s1 = np.concatenate((test_1p2n_s1, test_1p1n_s1))
test_twopi0_s2 = np.concatenate((test_1p2n_s2, test_1p1n_s2))
test_twopi0_s3 = np.concatenate((test_1p2n_s3, test_1p1n_s3))

target_twopi0 = np.concatenate((
        np.zeros(train_1p2n_s1.shape[0], dtype=np.uint8),
        np.ones(train_1p1n_s1.shape[0], dtype=np.uint8)))
target_twopi0 = target_twopi0.reshape((target_twopi0.shape[0], 1))
test_twopi0 = np.concatenate((
        np.zeros(test_1p2n_s1.shape[0], dtype=np.uint8),
        np.ones(test_1p1n_s1.shape[0], dtype=np.uint8)))
test_twopi0 = test_twopi0.reshape((test_twopi0.shape[0], 1))



model_pi0_filename = 'cache/crackpot_pi0.h5'
model_twopi0_filename = 'cache/crackpot_twopi0.h5'
if args.no_train:
    from keras.models import load_model
    model_pi0 = load_model(model_pi0_filename)
    model_twopi0 = load_model(model_twopi0_filename)

else:

    model_pi0_s1 = binary_2d_model(train_pi0_s1)
    model_pi0_s2 = binary_2d_model(train_pi0_s2)
    model_pi0_s3 = binary_2d_model(train_pi0_s3)
    model_pi0 = dense_merged_model([model_pi0_s1, model_pi0_s2, model_pi0_s3])
    log.info('compiling model for 1 pi0')
    model_pi0.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])


    model_twopi0_s1 = binary_2d_model(train_twopi0_s1)
    model_twopi0_s2 = binary_2d_model(train_twopi0_s2)
    model_twopi0_s3 = binary_2d_model(train_twopi0_s3)
    model_twopi0 = dense_merged_model([model_twopi0_s1, model_twopi0_s2, model_twopi0_s3])
    log.info('compiling model for 2 pi0')
    model_twopi0.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])


    log.info('starting training...')
    try:
        model_pi0.fit([train_pi0_s1, train_pi0_s2, train_pi0_s3], target_pi0, 
                      nb_epoch=50, 
                      batch_size=128,                
                      validation_data=([
                    test_pi0_s1, test_pi0_s2, test_pi0_s3], test_pi0),
                      callbacks=[
                EarlyStopping(
                    verbose=True, patience=5, 
                    monitor='val_loss'),
                ModelCheckpoint(
                    model_pi0_filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)])
        model_pi0.save(model_pi0_filename)

        model_twopi0.fit([train_twopi0_s1, train_twopi0_s2, train_twopi0_s3], target_twopi0, 
                      nb_epoch=50, 
                      batch_size=128,                
                      validation_data=([
                    test_twopi0_s1, test_twopi0_s2, test_twopi0_s3], test_twopi0),
                      callbacks=[
                EarlyStopping(
                    verbose=True, patience=5, 
                    monitor='val_loss'),
                ModelCheckpoint(
                    model_twopi0_filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)])
        model_twopi0.save(model_twopi0_filename)
    except KeyboardInterrupt:
        log.info('Ended early..')

log.info('testing stuff')

y_pred_pi0 = model_pi0.predict([test_pi0_s1, test_pi0_s2, test_pi0_s3], batch_size=32, verbose=0)
y_pred_twopi0 = model_twopi0.predict([test_twopi0_s1, test_twopi0_s2, test_twopi0_s3], batch_size=32, verbose=0)

fptr_1p1n, tpr_1p1n, thresh_1p1n = roc_curve(test_pi0, y_pred_pi0)
opt_fptr_1p1n, opt_tpr_1p1n, opt_thresh_1p1n = get_wp(
    fptr_1p1n, tpr_1p1n, thresh_1p1n, method='target_eff')
log.info((opt_fptr_1p1n, opt_tpr_1p1n, opt_thresh_1p1n))
log.info('Cutting on the score at {0}'.format(opt_thresh_1p1n))

fptr_1p2n, tpr_1p2n, thresh_1p2n = roc_curve(test_twopi0, y_pred_twopi0)
opt_fptr_1p2n, opt_tpr_1p2n, opt_thresh_1p2n = get_wp(
    fptr_1p2n, tpr_1p2n, thresh_1p2n, method='target_rej')
log.info((opt_fptr_1p2n, opt_tpr_1p2n, opt_thresh_1p2n))
log.info('Cutting on the score at {0}'.format(opt_thresh_1p2n))



# ROC curve
plt.figure()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fptr_1p1n, tpr_1p1n, color='red', label='1p1n vs 1p0n')
plt.plot(fptr_1p2n, tpr_1p2n, color='blue', label='1p1n vs 1p2n')
plt.plot([opt_fptr_1p1n, opt_fptr_1p2n],
          [opt_tpr_1p1n, opt_tpr_1p2n], 'go',
         label='working points')
plt.xlabel('miss-classification rate')
plt.ylabel('classification efficiency')
plt.title('classification with calo sampling s1, s2 and s3')
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.savefig('./plots/imaging/roc_curve.pdf')

# # confusion matrix
test_s1 = np.concatenate((test_1p0n_s1, test_1p1n_s1, test_1p2n_s1))
test_s2 = np.concatenate((test_1p0n_s2, test_1p1n_s2, test_1p2n_s2))
test_s3 = np.concatenate((test_1p0n_s3, test_1p1n_s3, test_1p2n_s3))
test_ki = np.concatenate((test_1p0n_ki, test_1p1n_ki, test_1p2n_ki))


score_pi0 = model_pi0.predict([test_s1, test_s2, test_s3], batch_size=32, verbose=0)
score_twopi0 = model_twopi0.predict([test_s1, test_s2, test_s3], batch_size=32, verbose=0)

y_true = np.concatenate((
        np.zeros((test_1p0n_s1.shape[0]), dtype=np.uint8),
        np.ones((test_1p1n_s1.shape[0]), dtype=np.uint8),
        np.ones((test_1p2n_s1.shape[0]), dtype=np.uint8) + 1))

from tauperf.imaging.evaluate import matrix_1p
y_pred_pi0 = score_pi0 > opt_thresh_1p1n
y_pred_twopi0 = score_twopi0 < opt_thresh_1p2n
cm = matrix_1p(y_true, y_pred_pi0, y_pred_twopi0)
class_names = ['1p0n', '1p1n', '1p2n']
plt.figure()
plot_confusion_matrix(
    cm, classes=class_names, 
    title='Confusion matrix with sampling s1, s2 and s3',
    name='plots/imaging/confusion_matrix.pdf')


# pt efficiency
from rootpy.plotting import root2matplotlib as rmpl
from rootpy.plotting.style import set_style
set_style('ATLAS', mpl=True)


y_pred_pi0 = y_pred_pi0.reshape((y_pred_pi0.shape[0],))
y_pred_twopi0 = y_pred_twopi0.reshape((y_pred_twopi0.shape[0],))

kin_1p0n = test_ki[y_true == 0]
kin_1p1n = test_ki[y_true == 1]
kin_1p2n = test_ki[y_true == 2]

# 1p1n vs 1p0n - pt 
eff_pi0_pt_1p1n = get_eff(
    kin_1p1n['pt'], y_pred_pi0[y_true == 1], 
    scale=0.001, color='red', name='1p1n')

eff_pi0_pt_1p0n = get_eff(
    kin_1p0n['pt'], y_pred_pi0[y_true == 0] == 0, 
    scale=0.001, color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_pt_1p1n.painted_graph, eff_pi0_pt_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Transverse Momentum [GeV]')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_pt.pdf')

# 1p1n vs 1p0n - eta
eff_pi0_eta_1p1n = get_eff(
    kin_1p1n['eta'], y_pred_pi0[y_true == 1], 
    binning=(10, -1.1, 1.1), color='red', name='1p1n')

eff_pi0_eta_1p0n = get_eff(
    kin_1p0n['eta'], y_pred_pi0[y_true == 0] == 0, 
    binning=(10, -1.1, 1.1), color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_eta_1p1n.painted_graph, eff_pi0_eta_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Pseudorapidity')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_eta.pdf')

# 1p1n vs 1p0n - mu
eff_pi0_mu_1p1n = get_eff(
    kin_1p1n['mu'], y_pred_pi0[y_true == 1], 
    binning=(10, 0, 40), color='red', name='1p1n')

eff_pi0_mu_1p0n = get_eff(
    kin_1p0n['mu'], y_pred_pi0[y_true == 0] == 0, 
    binning=(10, 0, 40), color='black', name='1p0n')

fig = plt.figure()
rmpl.errorbar([eff_pi0_mu_1p1n.painted_graph, eff_pi0_mu_1p0n.painted_graph])
plt.title('1p1n vs 1p0n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Average Interaction Per Bunch Crossing')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pi0_mu.pdf')

# 1p1n vs 1p2n - pt 
eff_twopi0_pt_1p1n = get_eff(
    kin_1p1n['pt'], y_pred_twopi0[y_true == 1] == 0, 
    scale=0.001, color='red', name='1p1n')

eff_twopi0_pt_1p2n = get_eff(
    kin_1p2n['pt'], y_pred_twopi0[y_true == 2], 
    scale=0.001, color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_pt_1p1n.painted_graph, eff_twopi0_pt_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Transverse Momentum [GeV]')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_pt.pdf')

# 1p1n vs 1p2n - eta
eff_twopi0_eta_1p1n = get_eff(
    kin_1p1n['eta'], y_pred_twopi0[y_true == 1] == 0, 
    binning=(10, -1.1, 1.1), color='red', name='1p1n')

eff_twopi0_eta_1p2n = get_eff(
    kin_1p2n['eta'], y_pred_twopi0[y_true == 2], 
    binning=(10, -1.1, 1.1), color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_eta_1p1n.painted_graph, eff_twopi0_eta_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Pseudorapidity')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_eta.pdf')

# 1p1n vs 1p2n - mu
eff_twopi0_mu_1p1n = get_eff(
    kin_1p1n['mu'], y_pred_twopi0[y_true == 1] == 0, 
    binning=(10, 0, 40), color='red', name='1p1n')

eff_twopi0_mu_1p2n = get_eff(
    kin_1p2n['mu'], y_pred_twopi0[y_true == 2], 
    binning=(10, 0, 40), color='black', name='1p2n')

fig = plt.figure()
rmpl.errorbar([eff_twopi0_mu_1p1n.painted_graph, eff_twopi0_mu_1p2n.painted_graph])
plt.title('1p1n vs 1p2n classifier')
plt.ylabel('Efficiency')
plt.xlabel('Average Interaction Per Bunch Crossing')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_twopi0_mu.pdf')
