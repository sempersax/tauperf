import os
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D

from . import log; log = log[__name__]


def single_layer_model_s2(data):
    """
    """
    log.info('build 2d convolutional model for s2 with shape {0}'.format(
            data[0].shape))
    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=data[0].shape))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Dropout(0.2))
    model_s2.add(Flatten())
    model_s2.add(Dense(128))
    model_s2.add(Activation('relu'))
    model_s2.add(Dropout(0.2))
    model_s2.add(Dense(16))
    model_s2.add(Activation('relu'))
    model_s2.add(Dense(1))
    model_s2.add(Activation('sigmoid'))
    return model_s2


def binary_2d_model(data):
    """
    """
    log.info('build 2d convolutional model with shape {0}'.format(
            data[0].shape))
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
    print data['s1'][0].shape
    print data['s2'][0].shape
    print data['s3'][0].shape
    model_s1.add(Convolution2D(
            30, 2, 8, border_mode='same', input_shape=data['s1'][0].shape, dim_ordering='th'))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((4, 1), dim_ordering='th'))
    model_s1.add(Convolution2D(
            15, 2, 4, border_mode='same',dim_ordering='th'))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((4, 1), dim_ordering='th'))
    model_s1.add(Dropout(0.2))

    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            30, 4, 4, border_mode='same', input_shape=data['s2'][0].shape, dim_ordering='th'))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Convolution2D(
            15, 3, 3, border_mode='same', dim_ordering='th'))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((1, 2), dim_ordering='th'))
    model_s2.add(Dropout(0.2))

    model_s3 = Sequential()
    model_s3.add(Convolution2D(
            32, 4, 4, border_mode='same', input_shape=data['s3'][0].shape, dim_ordering='th'))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((1, 2), dim_ordering='th'))
    model_s3.add(Convolution2D(
            16, 2, 4, border_mode='same', dim_ordering='th'))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((1, 2), dim_ordering='th'))
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
    log.info('build 2d convolutional model with shape {0}'.format(
            data['s1'][0].shape))
    model_s1 = Sequential()
    model_s1.add(Convolution2D(
            64, 4, 24, border_mode='same', 
            input_shape=data['s1'][0].shape))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s1.add(Dropout(0.2))
    model_s1.add(Flatten())
    model_s1.add(Dense(128))
    model_s1.add(Activation('relu'))
    model_s1.add(Dropout(0.2))


    log.info('build 2d convolutional model for s2')
    log.info('build 2d convolutional model with shape {0}'.format(
            data['s2'][0].shape))
    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            64, 5, 5, border_mode='same', 
            input_shape=data['s2'][0].shape))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Dropout(0.2))
    model_s2.add(Flatten())
    model_s2.add(Dense(128))
    model_s2.add(Activation('relu'))
    model_s2.add(Dropout(0.2))

    log.info('build 2d convolutional model for s3')
    log.info('build 2d convolutional model with shape {0}'.format(
            data['s3'][0].shape))
    model_s3 = Sequential()
    model_s3.add(Convolution2D(
            64, 3, 6, border_mode='same', 
            input_shape=data['s3'][0].shape))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s3.add(Dropout(0.2))
    model_s3.add(Flatten())
    model_s3.add(Dense(128))
    model_s3.add(Activation('relu'))
    model_s3.add(Dropout(0.2))

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
    log.info('build the track classification model (only ntracks for now)')
    model_kin = Sequential()
    model_kin.add(Dense(256, input_dim=1))
    model_kin.add(Activation('relu'))
    model_kin.add(Dropout(0.2))
    model_kin.add(Dense(128))
    model_kin.add(Activation('relu'))
    model_kin.add(Dropout(0.2))

    log.info('build 2d convolutional model for s1')
    model_s1 = Sequential()
    model_s1.add(Convolution2D(
            64, 6, 2, border_mode='same', 
            input_shape=data[0]['s1'].shape))
    model_s1.add(Activation('relu'))
    model_s1.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s1.add(Dropout(0.2))
    model_s1.add(Flatten())
    model_s1.add(Dense(128))
    model_s1.add(Activation('relu'))
    model_s1.add(Dropout(0.2))

    log.info('build 2d convolutional model for s2')
    model_s2 = Sequential()
    model_s2.add(Convolution2D(
            64, 2, 2, border_mode='same', 
            input_shape=data[0]['s2'].shape))
    model_s2.add(Activation('relu'))
    model_s2.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s2.add(Dropout(0.2))
    model_s2.add(Flatten())
    model_s2.add(Dense(128))
    model_s2.add(Activation('relu'))
    model_s2.add(Dropout(0.2))

    log.info('build 2d convolutional model for s3')
    model_s3 = Sequential()
    model_s3.add(Convolution2D(
            64, 4, 6, border_mode='same', 
            input_shape=data[0]['s3'].shape))
    model_s3.add(Activation('relu'))
    model_s3.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model_s3.add(Dropout(0.2))
    model_s3.add(Flatten())
    model_s3.add(Dense(128))
    model_s3.add(Activation('relu'))
    model_s3.add(Dropout(0.2))

    models = [model_kin, model_s1, model_s2, model_s3]

    log.info('Merge the models to a dense model')
    merged_model = Merge(models, mode=mode)
    model = Sequential()
    model.add(merged_model)
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    return model

