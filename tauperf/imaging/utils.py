import os
from . import log; log = log[__name__]
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np



def fit_model_multi(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

 
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')

        model.fit(
            [
                X_train['tracks'], 
                X_train['s1'], 
                X_train['s2'], 
                X_train['s3'], 
                X_train['s4'], 
                X_train['s5']
             ],
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                [
                    X_test['tracks'], 
                    X_test['s1'], 
                    X_test['s2'], 
                    X_test['s3'], 
                    X_test['s4'], 
                    X_test['s5']
                    ], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'



def fit_model(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')
        model.fit(
            [
                # X_train['tracks'], 
                X_train['s1'], 
                X_train['s2'], 
                X_train['s3'], 
                X_train['s4'], 
                X_train['s5']
                ],
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                [
                    # X_test['tracks'], 
                    X_test['s1'], 
                    X_test['s2'], 
                    X_test['s3'], 
                    X_test['s4'], 
                    X_test['s5']
                    ], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'


def fit_model_single_layer(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

 
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
        log.info(X_train.shape)
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                X_test, y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        log.info('Ended early..')

