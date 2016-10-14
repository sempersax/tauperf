import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from tauperf import log; log = log['/train-img']
from tauperf.imaging.plotting import plot_confusion_matrix, get_wp


log.info('loading data...')

arr_1p1n = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v5', 
    'images_1p1n.npy'))

arr_1p0n = np.load(os.path.join(
    os.getenv('DATA_AREA'), 
    'tauid_ntuples', 'v5', 
    'images_1p0n.npy'))


data = np.concatenate((
        arr_1p1n, arr_1p0n))

reshaped_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

target = np.concatenate((
    np.ones(arr_1p1n.shape[0],dtype=np.uint8),
    np.zeros(arr_1p0n.shape[0], dtype=np.uint8)))

log.info('splitting')
data_train, data_test, y_train, y_test = model_selection.train_test_split(
    reshaped_data, target, test_size=0.2, random_state=42)

log.info('Training stat: 1p1n = {0} images, 1p0n = {1} images'.format(
        len(y_train[y_train==1]), len(y_train[y_train==0])))

# data_train = data_train.reshape((data_train.shape[0], data.shape[1], data.shape[2]))
# data_test = data_test.reshape((data_test.shape[0], data.shape[1], data.shape[2]))

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

# -- build the model
model = Sequential()
model.add(Dense(128, input_dim=data_train.T.shape[0]))
model.add(Activation('tanh'))
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
    model.fit(data_train, y_train, nb_epoch=60, batch_size=32)
except KeyboardInterrupt:
    log.info('Ended early..')

log.info('testing stuff')
log.info('Testing stat: 1p1n = {0} images, 1p0n = {1} images'.format(
        len(y_test[y_test==1]), len(y_test[y_test==0])))

loss = model.evaluate(data_test, y_test, batch_size=32)
print
log.info(loss)

y_pred = model.predict(data_test, batch_size=32, verbose=0)



fptr, tpr, thresh = roc_curve(y_test, y_pred)
opt_fptr, opt_tpr, opt_thresh = get_wp(fptr, tpr, thresh)

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
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.savefig('./plots/imaging/roc_curve.pdf')
plt.figure()

# confusion matrix
cnf_mat = confusion_matrix(y_test, y_pred > opt_thresh)
np.set_printoptions(precision=2)
class_names = ['1p0n', '1p1n']
cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
plot_confusion_matrix(cm, classes=class_names)



