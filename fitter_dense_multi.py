import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
# import tables
# from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import roc_curve
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from tauperf import log; log = log['/fitter']
from tauperf.imaging.models import dense_merged_model_categorical
from tauperf.imaging.models import dense_merged_model_rnn
from tauperf.imaging.models import dense_merged_model_topo
from tauperf.imaging.utils import fit_model_multi
from tauperf.imaging.load import load_data

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--no-train', default=False, action='store_true')
parser.add_argument(
    '--overwrite', default=False, action='store_true')
parser.add_argument(
    '--equal-size', default=False, action='store_true')
parser.add_argument(
    '--debug', default=False, action='store_true')

args = parser.parse_args()


log.info('loading data...')
data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v12/test')
                        

filenames = [
    os.path.join(data_dir, "images_new_1p0n.h5"),
    os.path.join(data_dir, "images_new_1p1n.h5"),
    os.path.join(data_dir, "images_new_1p2n.h5"),
    os.path.join(data_dir, "images_new_3p0n.h5"),
    os.path.join(data_dir, "images_new_3p1n.h5"),
]
labels = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']

train, test, val, y_train, y_test, y_val = load_data(
    filenames, labels, equal_size=args.equal_size, debug=args.debug)



y_train_cat = to_categorical(y_train, 5)
y_test_cat  = to_categorical(y_test, 5)
y_val_cat   = to_categorical(y_val, 5)


# ##############################################
model_filename = 'cache/crackpot_dense_multi.h5'
if args.no_train:
    log.info('loading model')
    model = load_model(model_filename)
else:
    log.info('training...')
    model = dense_merged_model_topo(train, n_classes=5, final_activation='softmax')
#     model = dense_merged_model_rnn(train)
    fit_model_multi(
        model,
        train, y_train_cat,
        val, y_val_cat,
        filename=model_filename,
        loss='categorical_crossentropy',
        overwrite=args.overwrite,
        no_train=args.no_train)




# ##############################################
log.info('testing stuff')

log.info('compute classifier scores')

y_pred = model.predict(
        [test['tracks'], test['s1'], test['s2'], test['s3'], test['s4'], test['s5']], 
        batch_size=32, verbose=1)


print
log.info('drawing the confusion matrix')
from sklearn.metrics import confusion_matrix
from tauperf.imaging.plotting import plot_confusion_matrix
cnf_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
plt.figure()
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_categorical.pdf')

cnf_mat = confusion_matrix(y_test, test['pantau'])
plt.figure()
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_reference.pdf')
