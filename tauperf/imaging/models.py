from keras.layers import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D



def binary_2d_model(data):
    """
    """
    model = Sequential()
    model.add(Convolution2D(
            64, 6, 6, border_mode='same', 
            input_shape=(1, data.shape[2], data.shape[3])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    return model


def dense_merged_model(models, mode='sum'):
    """
    """
    merged_model = Merge(models, mode=mode)
    model = Sequential()
    model.add(merged_model)
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
