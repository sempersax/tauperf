import numpy as np
import logging
import os

from sklearn import model_selection


# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU


def get_wp(true_pos, false_pos, thresh):
    if (true_pos.ndim, false_pos.ndim, thresh.ndim) != (1, 1, 1):
        raise ValueError('wrong dimension')
    if len(true_pos) != len(false_pos) or len(true_pos) != len(thresh):
        raise ValueError('wrong size')

    # compute the distance to the (0, 1) point
    dr_square = true_pos * true_pos + (false_pos - 1) * (false_pos - 1)
    # get the index in the dr_square array
    index_min = np.argmin(dr_square)
    # return optimal true positive eff, false positive eff and threshold for cut
    return true_pos[index_min], false_pos[index_min], thresh[index_min]

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))

log.info('loading data...')

arr_1p1n = np.load('data_test/images_1p1n_dr0.2.npy')
# remove dummies
arrs = [arr for arr in arr_1p1n]
arrs = filter(lambda a: a != None, arrs)
arr_1p1n = np.array(arrs)

arr_1p0n = np.load('data_test/images_1p0n_dr0.2.npy')


data = np.concatenate((
        arr_1p1n, arr_1p0n))

reshaped_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

target = np.concatenate((
    np.ones(arr_1p1n.shape[0],dtype=np.uint8),
    np.zeros(arr_1p0n.shape[0], dtype=np.uint8)))

log.info('splitting')
data_train, data_test, y_train, y_test = model_selection.train_test_split(
    reshaped_data, target, test_size=0.2, random_state=42)

# data_train = data_train.reshape((data_train.shape[0], data.shape[1], data.shape[2]))
# data_test = data_test.reshape((data_test.shape[0], data.shape[1], data.shape[2]))
print y_train.shape, y_train.T.shape

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

# -- build the model
model = Sequential()
# model.add(Dense(1, input_dim=data_train.T.shape[0], activation='sigmoid'))
model.add(Dense(32, input_dim=data_train.T.shape[0]))
model.add(Activation('tanh'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


log.info('compiling model...')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

NAME = './FIRST_TRAIN'

log.info('starting training...')
try:
    model.fit(data_train, y_train, nb_epoch=30, batch_size=32)
except KeyboardInterrupt:
    log.info('Ended early..')

log.info('testing stuff')
loss = model.evaluate(data_test, y_test, batch_size=32)
print
print loss

y_pred = model.predict(data_test, batch_size=32, verbose=0)

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tauperf.plotting.mpl import plot_confusion_matrix

fptr, tpr, thresh = roc_curve(y_test, y_pred)

opt_fptr, opt_tpr, opt_thresh = get_wp(fptr, tpr, thresh)
log.info('Cutting on the score at {0}'.format(opt_thresh))
log.info('roc auc = {0}'.format(
        roc_auc_score(y_test, y_pred)))

plt.figure()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fptr, tpr, color='red', label='1p1n vs 1p0n')
plt.plot(
    opt_fptr, opt_tpr, 'bx',
    label='Optimal working point')
plt.xlabel('1p0n miss-classification efficiency')
plt.ylabel('1p1n classification efficiency')
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.figure()

cnf_mat = confusion_matrix(y_test, y_pred > opt_thresh)
np.set_printoptions(precision=2)
class_names = ['1p0n', '1p1n']

cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
plot_confusion_matrix(cm, classes=class_names)
plt.show()
