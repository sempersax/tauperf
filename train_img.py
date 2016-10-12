import numpy as np
import logging
import os

from sklearn import cross_validation


# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))

log.info('loading data...')

arr_1p1n = np.load('data_test/images_1p1n.npy')
# remove dummies
arrs = [arr for arr in arr_1p1n]
arrs = filter(lambda a: a != None, arrs)
arr_1p1n = np.array(arrs)

arr_1p0n = np.load('data_test/images_1p0n.npy')


data = np.concatenate((
        arr_1p1n, arr_1p0n))

reshaped_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

target = np.concatenate((
    np.ones(arr_1p1n.shape[0],dtype=np.uint8),
    np.zeros(arr_1p0n.shape[0], dtype=np.uint8)))

log.info('splitting')
data_train, data_test, y_train, y_test = cross_validation.train_test_split(
    reshaped_data, target, test_size=0.2, random_state=42)

# data_train = data_train.reshape((data_train.shape[0], data.shape[1], data.shape[2]))
# data_test = data_test.reshape((data_test.shape[0], data.shape[1], data.shape[2]))
print y_train.shape, y_train.T.shape

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

# -- build the model
model = Sequential()
model.add(Dense(1, input_dim=data_train.T.shape[0], activation='sigmoid'))
log.info('compiling model...')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

NAME = './FIRST_TRAIN'

log.info('starting training...')
try:
    model.fit(data_train, y_train, nb_epoch=10, batch_size=32)

except KeyboardInterrupt:
    log.info('Ended early..')


with open(NAME + '.yaml', 'w') as f:
    f.write(model.to_yaml())
