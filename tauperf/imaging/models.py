import os
from keras.models import Sequential, load_model
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from . import log; log = log[__name__]

def binary_2d_model(data):
    """
    """
    model = Sequential()
    model.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=data[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    return model


def dense_merged_model(data, mode='sum'):
    """
    """
    log.info('build 2d convolutional model for s1')
    model_s1 = binary_2d_model(data['s1'])

    log.info('build 2d convolutional model for s2')
    model_s2 = binary_2d_model(data['s2'])

    log.info('build 2d convolutional model for s3')
    model_s3 = binary_2d_model(data['s3'])

    models = [model_s1, model_s2, model_s3]

    log.info('Merge the models to a dense model')
    merged_model = Merge(models, mode=mode)
    model = Sequential()
    model.add(merged_model)
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def load_or_fit(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    overwrite=False,
    no_train=False):

    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to false')

    if no_train:
        model = load_model(filename)
        return True

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

        log.info('Start training ...')
        model.fit(
            [X_train['s1'], X_train['s2'], X_train['s3']],
            y_train,
            nb_epoch=50,
            batch_size=128,
            validation_data=(
                [X_test['s1'], X_test['s2'], X_test['s3']], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
        return True
    

    except KeyboardInterrupt:
        log.info('Ended early..')
        return False

