import os
import sys

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Masking
from keras.layers.convolutional import Convolution1D, Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import LSTM

#from keras import backend as K

from . import log ; log = log.getChild(__name__)

"""attempted to implement a 'Noisy ReLU', which is an extension of ReLU to include
Gaussian Noise.  According to Wikipedia, Noisy ReLUs have been used with some 
success in restricted Boltzmann machines for computer vision tasks.  As our problem
is computer vision based, I thought it might be an avenue of interest.  However,
Noisy ReLUs are not built into Keras, and the function must therefore be constructed
by the user.  So far, I have failed to get said function to work."""

#def relu_noise(x):
#    isPositive = K.greater(x,0)
#    print isPositive
#    noise = K.random_normal(K.shape(x), mean=0.5, stddev=0.5)
    
#    return (x * isPositive) + noise

#relu_noise(True)
#sys.exit()

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
    model_s1.add(Dense(130))
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
    model_s2.add(Dense(130))
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
    model_s3.add(Dense(130))
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
    log.info('build the tracks classification model')

    model_kin = Sequential()
    model_kin.add(Dense(128, input_dim=4))
    model_kin.add(Activation('relu'))
    model_kin.add(Dropout(0.2))
    model_kin.add(Dense(64, input_dim=4))
    model_kin.add(Activation('relu'))
    model_kin.add(Dropout(0.2))
    model_kin.add(Dense(32))
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
    model_s1.add(Dense(130))
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
    model_s2.add(Dense(130))
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
    model_s3.add(Dense(130))
    model_s3.add(Activation('relu'))
    model_s3.add(Dropout(0.2))

    log.info('merge calo layers')
    models = [model_s1, model_s2, model_s3]
    merge_calo = Merge(models, mode=mode)
    model_calo = Sequential()
    model_calo.add(merge_calo)
    model_calo.add(Dense(32))
    model_calo.add(Activation('relu'))


    log.info('Merge the models to a dense model')
    merged_model = Merge([model_kin, model_calo], mode=mode)
#     merged_model = Merge(models, mode=mode)
    model = Sequential()
    model.add(merged_model)
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    return model

# This is the definition used as of 25 June 18.
def dense_merged_model_topo(data, n_classes=3, final_activation='softmax'):
    """
    """

    log.info('* Tracks: build rnn model')
    tracks_input = Input(shape=data[0]['tracks'].shape)
    tracks_x = Reshape((15, 4))(tracks_input)
    log.info('\t tracks input shape = {0}'.format(tracks_x._keras_shape))
    tracks_x = Masking()(tracks_x)
    log.info('\t tracks input shape = {0}'.format(tracks_x._keras_shape))
    tracks_x = LSTM(32)(tracks_x)
    log.info('\t tracks lstm shape  = {0}'.format(tracks_x._keras_shape))
    tracks_x = Dense(128, activation='relu')(tracks_x)
    log.info('\t tracks dense shape = {0}'.format(tracks_x._keras_shape))
    tracks_out = Reshape((1, 128))(tracks_x)
    log.info('\t tracks final shape = {0}'.format(tracks_x._keras_shape))

    log.info('* S1: build 2d convolutional model')
    s1_input = Input(shape=data[0]['s1'].shape)
    s1_shape = data[0]['s1'].shape
    s1_x = Reshape((s1_shape[0], s1_shape[1], 1))(s1_input)
    log.info('\t s1 input shape   = {0}'.format(s1_x._keras_shape))
    s1_x = Conv2D(64, (6, 2), padding='same', activation='relu')(s1_x)
    log.info('\t s1 convolu shape = {0}'.format(s1_x._keras_shape))
    s1_x = MaxPooling2D((2, 2))(s1_x)
    log.info('\t s1 maxpool shape = {0}'.format(s1_x._keras_shape))
    s1_x = Dropout(0.2)(s1_x)
    s1_x = Flatten()(s1_x)
    log.info('\t s1 flatten shape = {0}'.format(s1_x._keras_shape))
    s1_x = Dense(128, activation='relu')(s1_x)
    log.info('\t s1 dense shape   = {0}'.format(s1_x._keras_shape))
    s1_x = Dropout(0.2)(s1_x)
    s1_out = Reshape((1, 128))(s1_x)
    log.info('\t s1 final shape   = {0}'.format(s1_out._keras_shape))

    log.info('* S2: build 2d convolutional model')
    s2_input = Input(shape=data[0]['s2'].shape)
    s2_shape = data[0]['s2'].shape
    s2_x = Reshape((s2_shape[0], s2_shape[1], 1))(s2_input)
    log.info('\t s2 input shape   = {0}'.format(s2_x._keras_shape))
    s2_x = Conv2D(64, (3, 3), padding='same', activation='relu')(s2_x)
    log.info('\t s2 convolu shape = {0}'.format(s2_x._keras_shape))
    s2_x = MaxPooling2D((2, 2))(s2_x)
    log.info('\t s2 maxpool shape = {0}'.format(s2_x._keras_shape))
    s2_x = Dropout(0.2)(s2_x)
    s2_x = Flatten()(s2_x)
    log.info('\t s2 flatten shape = {0}'.format(s2_x._keras_shape))
    s2_x = Dense(128, activation='relu')(s2_x)
    log.info('\t s2 dense shape   = {0}'.format(s2_x._keras_shape))
    s2_x = Dropout(0.2)(s2_x)
    s2_out = Reshape((1, 128))(s2_x)
    log.info('\t s2 final shape   = {0}'.format(s2_out._keras_shape))

    log.info('* S3: build 2d convolutional model')
    s3_input = Input(shape=data[0]['s3'].shape)
    s3_shape = data[0]['s3'].shape
    s3_x = Reshape((s3_shape[0], s3_shape[1], 1))(s3_input)
    log.info('\t s3 input shape = {0}'.format(s3_x._keras_shape))
    s3_x = Conv2D(64, (3, 2), padding='same', activation='relu')(s3_x)
    log.info('\t s3 convolu shape = {0}'.format(s3_x._keras_shape))
    s3_x = MaxPooling2D((2, 2))(s3_x)
    log.info('\t s3 maxpool shape = {0}'.format(s3_x._keras_shape))
    s3_x = Dropout(0.2)(s3_x)
    s3_x = Flatten()(s3_x)
    log.info('\t s3 flatten shape = {0}'.format(s3_x._keras_shape))
    s3_x = Dense(128, activation='relu')(s3_x)
    log.info('\t s3 dense shape   = {0}'.format(s3_x._keras_shape))
    s3_x = Dropout(0.2)(s3_x)
    s3_out = Reshape((1, 128))(s3_x)
    log.info('\t s3 final shape   = {0}'.format(s3_out._keras_shape))

    log.info('* S4: build 2d convolutional model')
    s4_input = Input(shape=data[0]['s4'].shape)
    s4_shape = data[0]['s4'].shape
    s4_x = Reshape((s4_shape[0], s4_shape[1], 1))(s4_input)
    log.info('\t s4 input shape   = {0}'.format(s4_x._keras_shape))
    s4_x = Conv2D(64, (2, 2), padding='same', activation='relu')(s4_x)
    log.info('\t s4 convolu shape = {0}'.format(s4_x._keras_shape))
    s4_x = MaxPooling2D((2, 2))(s4_x)
    log.info('\t s4 maxpool shape = {0}'.format(s4_x._keras_shape))
    s4_x = Dropout(0.2)(s4_x)
    s4_x = Flatten()(s4_x)
    log.info('\t s4 flatten shape = {0}'.format(s4_x._keras_shape))
    s4_x = Dense(128, activation='relu')(s4_x)
    log.info('\t s4 dense shape   = {0}'.format(s4_x._keras_shape))
    s4_x = Dropout(0.2)(s4_x)
    s4_out = Reshape((1, 128))(s4_x)
    log.info('\t s4 final shape   = {0}'.format(s4_out._keras_shape))

    log.info('* S5: build 2d convolutional model')
    s5_input = Input(shape=data[0]['s5'].shape)
    s5_shape = data[0]['s5'].shape
    s5_x = Reshape((s5_shape[0], s5_shape[1], 1))(s5_input)
    log.info('\t s5 input shape   = {0}'.format(s5_x._keras_shape))
    s5_x = Conv2D(64, (2, 2), padding='same', activation='relu')(s5_x)
    log.info('\t s5 convolu shape = {0}'.format(s5_x._keras_shape))
    s5_x = MaxPooling2D((2, 2))(s5_x)
    log.info('\t s5 maxpool shape = {0}'.format(s5_x._keras_shape))
    s5_x = Dropout(0.2)(s5_x)
    s5_x = Flatten()(s5_x)
    log.info('\t s5 flatten shape = {0}'.format(s5_x._keras_shape))
    s5_x = Dense(128, activation='relu')(s5_x)
    log.info('\t s5 dense shape   = {0}'.format(s5_x._keras_shape))
    s5_x = Dropout(0.2)(s5_x)
    s5_out = Reshape((1, 128))(s5_x)
    log.info('\t s5 final shape   = {0}'.format(s5_out._keras_shape))


    log.info('merge layers')
    layers = [
        tracks_out,
        s1_out,
        s2_out,
        s3_out,
        s4_out,
        s5_out
        ]

    merge = concatenate(layers, axis=1)
    # log.info('\t merged layers shape = {0}'.format(merge._keras_shape))
    # merge_x = LSTM(32)(merge)
    merge_x = Flatten()(merge)
    log.info('\t merged lstm shape   = {0}'.format(merge_x._keras_shape))
    merge_x = Dense(512, activation='relu')(merge_x)
    log.info('\t merged dense shape  = {0}'.format(merge_x._keras_shape))
    output_mod = Dense(n_classes, activation=final_activation)
    output_x = output_mod(merge_x)
    log.info('\t final shape         = {0}'.format(output_x._keras_shape))


    model_input = [
        tracks_input,
        s1_input,
        s2_input,
        s3_input,
        s4_input,
        s5_input
        ]

    model = Model(inputs=model_input, outputs=output_x)

    return model

