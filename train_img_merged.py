import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from keras.layers import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tauperf import log; log = log['/train-img-2']
from tauperf.imaging.plotting import plot_confusion_matrix, get_wp


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--cal-layer', default=2, choices=[None, 1, 2], type=int,
    help='select layer for the image selection')
parser.add_argument(
    '--no-train', default=False, action='store_true')
args = parser.parse_args()

cal_layer = args.cal_layer


log.info('loading data...')

arr_1p1n_s1 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S1_1p1n.npy'))

arr_1p0n_s1 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S1_1p0n.npy'))

arr_1p1n_s2 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S2_1p1n.npy'))

arr_1p0n_s2 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S2_1p0n.npy'))

arr_1p1n_s3 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S3_1p1n.npy'))

arr_1p0n_s3 = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'images_S3_1p0n.npy'))

kin_1p0n = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'kinematics_1p0n.npy'))

kin_1p1n = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v6', 
    'kinematics_1p1n.npy'))




data_s1 = np.concatenate((
        arr_1p1n_s1, arr_1p0n_s1))

data_s2 = np.concatenate((
        arr_1p1n_s2, arr_1p0n_s2))

data_s3 = np.concatenate((
        arr_1p1n_s3, arr_1p0n_s3))

kin = np.concatenate((
        kin_1p1n, kin_1p0n))


reshaped_data_s1 = data_s1.reshape((data_s1.shape[0], data_s1.shape[1] * data_s1.shape[2]))
reshaped_data_s2 = data_s2.reshape((data_s2.shape[0], data_s2.shape[1] * data_s2.shape[2]))
reshaped_data_s3 = data_s3.reshape((data_s3.shape[0], data_s3.shape[1] * data_s3.shape[2]))

target = np.concatenate((
    np.ones(arr_1p1n_s1.shape[0],dtype=np.uint8),
    np.zeros(arr_1p0n_s1.shape[0], dtype=np.uint8)))

log.info('splitting')
d_train_s1, d_test_s1, d_train_s2, d_test_s2, d_train_s3, d_test_s3, kin_train, kin_test, y_train, y_test = model_selection.train_test_split(
    reshaped_data_s1, reshaped_data_s2, reshaped_data_s3, kin, target, test_size=0.2, random_state=42)

log.info('Training stat: 1p1n = {0} images, 1p0n = {1} images'.format(
        len(y_train[y_train==1]), len(y_train[y_train==0])))


y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))



d_train_s1 = d_train_s1.reshape((d_train_s1.shape[0], data_s1.shape[1], data_s1.shape[2]))
d_test_s1  = d_test_s1.reshape((d_test_s1.shape[0], data_s1.shape[1], data_s1.shape[2]))

d_train_s2 = d_train_s2.reshape((d_train_s2.shape[0], data_s2.shape[1], data_s2.shape[2]))
d_test_s2  = d_test_s2.reshape((d_test_s2.shape[0], data_s2.shape[1], data_s2.shape[2]))

d_train_s3 = d_train_s3.reshape((d_train_s3.shape[0], data_s3.shape[1], data_s3.shape[2]))
d_test_s3  = d_test_s3.reshape((d_test_s3.shape[0], data_s3.shape[1], data_s3.shape[2]))

d_train_s1 = np.expand_dims(d_train_s1, axis=1)
d_test_s1  = np.expand_dims(d_test_s1, axis=1)

d_train_s2 = np.expand_dims(d_train_s2, axis=1)
d_test_s2  = np.expand_dims(d_test_s2, axis=1)

d_train_s3 = np.expand_dims(d_train_s3, axis=1)
d_test_s3  = np.expand_dims(d_test_s3, axis=1)

model_filename = 'cache/crackpot.h5'
if args.no_train:
    from keras.models import load_model
    model = load_model(model_filename)

else:
    log.info(data_s1.shape)
    model_s1 = Sequential()
    model_s1.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=(1, data_s1.shape[1], data_s1.shape[2])))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s1.add(Dropout(0.2))
    model_s1.add(Flatten())
    model_s1.add(Dense(128))
    model_s1.add(Activation('relu'))
    model_s1.add(Dropout(0.2))
    
    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=(1, data_s2.shape[1], data_s2.shape[2])))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Dropout(0.2))
    model_s2.add(Flatten())
    model_s2.add(Dense(128))
    model_s2.add(Activation('relu'))
    model_s2.add(Dropout(0.2))

    model_s3 = Sequential()
    model_s3.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=(1, data_s3.shape[1], data_s3.shape[2])))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s3.add(Dropout(0.2))
    model_s3.add(Flatten())
    model_s3.add(Dense(128))
    model_s3.add(Activation('relu'))
    model_s3.add(Dropout(0.2))

    
    merged = Merge([model_s1, model_s2, model_s3], mode='sum')
    model = Sequential()
    model.add(merged)
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    log.info('compiling model...')
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    log.info('starting training...')
    try:
        model.fit([d_train_s1, d_train_s2, d_train_s3], y_train, 
                  nb_epoch=40, 
                  batch_size=128,                
                  validation_data=([d_test_s1, d_test_s2, d_test_s3], y_test),
                  callbacks=[
                EarlyStopping(
                    verbose=True, patience=5, 
                    monitor='val_loss'),
                ModelCheckpoint(
                    model_filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)])
        model.save(model_filename)
    except KeyboardInterrupt:
        log.info('Ended early..')

log.info('testing stuff')
log.info('Testing stat: 1p1n = {0} images, 1p0n = {1} images'.format(
        len(y_test[y_test==1]), len(y_test[y_test==0])))

loss = model.evaluate([d_test_s1, d_test_s2, d_test_s3], y_test, batch_size=128)
print
log.info(loss)

y_pred = model.predict([d_test_s1, d_test_s2, d_test_s3], batch_size=32, verbose=0)



fptr, tpr, thresh = roc_curve(y_test, y_pred)
opt_fptr, opt_tpr, opt_thresh = get_wp(fptr, tpr, thresh, method='target_rej')
log.info((opt_fptr, opt_tpr, opt_thresh))
log.info('Cutting on the score at {0}'.format(opt_thresh))
log.info('roc auc = {0}'.format(
        roc_auc_score(y_test, y_pred)))

# ROC curve
plt.figure()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fptr, tpr, color='red', label='1p1n vs 1p0n')
plt.plot(
    opt_fptr, opt_tpr, 'bx',
    label='Optimal working point')
plt.plot([0, opt_fptr], [1, opt_tpr], linestyle='--', linewidth=2)
plt.xlabel('1p0n miss-classification efficiency')
plt.ylabel('1p1n classification efficiency')
plt.title('1p1n vs 1p0n classification with calo sampling s1, s2 and s3')
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.savefig('./plots/imaging/roc_curve.pdf')
plt.figure()

# confusion matrix
cnf_mat = confusion_matrix(y_test, y_pred > opt_thresh)
np.set_printoptions(precision=2)
class_names = ['1p0n', '1p1n']
cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
cm = np.rot90(cm.T, 1)
plot_confusion_matrix(
    cm, classes=class_names, 
    title='Confusion matrix with sampling s1, s2 and s3',
    name='plots/imaging/confusion_matrix.pdf')



# pt efficiency
from root_numpy import fill_hist
from rootpy.plotting import Hist, Efficiency, Canvas
from rootpy.plotting import root2matplotlib as rmpl
from rootpy.plotting.style import set_style
set_style('ATLAS', mpl=True)
def get_eff(accept, total, binning=(20, 0, 100), name='1p1n'):
    hnum = Hist(binning[0], binning[1], binning[2])
    hden = Hist(binning[0], binning[1], binning[2])
    fill_hist(hnum, accept)
    fill_hist(hden, total)
    eff = Efficiency(hnum, hden, name='eff_' + name, title=name)
    return eff

y_1p1n = y_test == 1
score_1p1n = y_pred[y_1p1n]

y_1p0n = y_test == 0
score_1p0n = y_pred[y_1p0n]

total_pt_1p1n = kin_test['pt'][y_1p1n.reshape((y_1p1n.shape[0],))]
total_pt_1p1n /= 1000.
accept_pt_1p1n = total_pt_1p1n[score_1p1n > opt_thresh]
eff_pt_1p1n = get_eff(accept_pt_1p1n, total_pt_1p1n, name='1p1n')
eff_pt_1p1n.color = 'red'

total_pt_1p0n = kin_test['pt'][y_1p0n.reshape((y_1p0n.shape[0],))]
total_pt_1p0n /= 1000.
accept_pt_1p0n = total_pt_1p0n[score_1p0n < opt_thresh]
eff_pt_1p0n = get_eff(accept_pt_1p0n, total_pt_1p0n, name='1p0n')


fig = plt.figure()
rmpl.errorbar([eff_pt_1p1n.painted_graph, eff_pt_1p0n.painted_graph])
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Transverse Momentum [GeV]')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_pt.pdf')



total_eta_1p1n = kin_test['eta'][y_1p1n.reshape((y_1p1n.shape[0],))]
accept_eta_1p1n = total_eta_1p1n[score_1p1n > opt_thresh]
eff_eta_1p1n = get_eff(accept_eta_1p1n, total_eta_1p1n, binning=(20, -1.1, 1.1), name='1p1n')
eff_eta_1p1n.color = 'red'

total_eta_1p0n = kin_test['eta'][y_1p0n.reshape((y_1p0n.shape[0],))]
accept_eta_1p0n = total_eta_1p0n[score_1p0n < opt_thresh]
eff_eta_1p0n = get_eff(accept_eta_1p0n, total_eta_1p0n, binning=(20, -1.1, 1.1), name='1p0n')


fig = plt.figure()
rmpl.errorbar([eff_eta_1p1n.painted_graph, eff_eta_1p0n.painted_graph])
plt.ylabel('Efficiency')
plt.xlabel('Visible Tau Pseudorapidity')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_eta.pdf')


total_mu_1p1n = kin_test['mu'][y_1p1n.reshape((y_1p1n.shape[0],))]
accept_mu_1p1n = total_mu_1p1n[score_1p1n > opt_thresh]
eff_mu_1p1n = get_eff(accept_mu_1p1n, total_mu_1p1n, binning=(10, 0, 40), name='1p1n')
eff_mu_1p1n.color = 'red'

total_mu_1p0n = kin_test['mu'][y_1p0n.reshape((y_1p0n.shape[0],))]
accept_mu_1p0n = total_mu_1p0n[score_1p0n < opt_thresh]
eff_mu_1p0n = get_eff(accept_mu_1p0n, total_mu_1p0n, binning=(10, 0, 40), name='1p0n')


fig = plt.figure()
rmpl.errorbar([eff_mu_1p1n.painted_graph, eff_mu_1p0n.painted_graph])
plt.ylabel('Efficiency')
plt.xlabel('Average interaction per Bunch Crossing')
plt.legend(loc='lower left')
fig.savefig('plots/imaging/efficiency_mu.pdf')

