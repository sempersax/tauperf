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

def binary_3d_model(data):
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


def merged_3d_model(data):
    log.info(data['s1'][0].shape)
    model_s1 = Sequential()
    model_s1.add(Convolution2D(
            32, 2, 8, border_mode='same', input_shape=data['s1'][0].shape, dim_ordering='th'))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((1, 2), dim_ordering='th'))
    model_s1.add(Convolution2D(
            16, 2, 4, border_mode='same',dim_ordering='th'))
    model_s1.add(Activation('relu'))
1    model_s1.add(MaxPooling2D((1, 3), dim_ordering='th'))
    model_s1.add(Dropout(0.2))

    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            32, 4, 4, border_mode='same', input_shape=data['s2'][0].shape, dim_ordering='th'))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Convolution2D(
            16, 3, 3, border_mode='same', dim_ordering='th'))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 1), dim_ordering='th'))
    model_s2.add(Dropout(0.2))

    model_s3 = Sequential()
    model_s3.add(Convolution2D(
            32, 4, 4, border_mode='same', input_shape=data['s3'][0].shape, dim_ordering='th'))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((2, 1), dim_ordering='th'))
    model_s3.add(Convolution2D(
            16, 2, 4, border_mode='same', dim_ordering='th'))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((2, 1), dim_ordering='th'))
    model_s3.add(Dropout(0.2))

    model = Sequential()
    model.add(Merge([model_s1, model_s2, model_s3], mode='concat', concat_axis=1))
    model.add(Convolution2D(
            8, 2, 2, border_mode='same', dim_ordering='th'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
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

def dense_merged_model_categorical(data, mode='sum'):
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
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model


def load_or_fit(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

    if no_train:
        log.info('loading model {0}'.format(os.path.basename(filename)))
        model = load_model(filename)
        return True

    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to false')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
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

