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

        kin_train = np.hstack([
#             X_train['pt'],
#             X_train['eta'],
            X_train['ntracks'],
            X_train['empovertrksysp']
            ])
        kin_test =np.hstack([
#             X_test['pt'],
#             X_test['eta'],
            X_test['ntracks'],
            X_test['empovertrksysp']
            ])
        
        
        model.fit(
            [kin_train, X_train['s1'], X_train['s2'], X_train['s3']],
            y_train,
            nb_epoch=100,
            batch_size=128,
            validation_data=(
                [kin_test, X_test['s1'], X_test['s2'], X_test['s3']], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=10, monitor='val_acc'),
                ModelCheckpoint(
                    filename, monitor='val_acc', 
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
            nb_epoch=50,
            batch_size=128,
            validation_data=(
                X_test, y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        log.info('Ended early..')

