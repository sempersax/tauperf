import os
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from keras.utils.np_utils import to_categorical
from tauperf import log; log = log['/fitter']
from tauperf.imaging.load import load_test_data, print_sample_size
from tauperf.imaging.plotting import plot_confusion_matrix, plot_roc
import tables
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
parser.add_argument(
    '--training-chunks', default=3, type=int)
parser.add_argument(
    '--one-prong-only', default=False, action='store_true')

args = parser.parse_args()


data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v12/test')
                        
if args.one_prong_only:
    filenames = [
        os.path.join(data_dir, "images_new_1p0n.h5"),
        os.path.join(data_dir, "images_new_1p1n.h5"),
        os.path.join(data_dir, "images_new_1p2n.h5"),
        ]
    labels = ['1p0n', '1p1n', '1pXn']
    n_classes = 3
else: 
    filenames = [
        os.path.join(data_dir, "images_new_1p0n.h5"),
        os.path.join(data_dir, "images_new_1p1n.h5"),
        os.path.join(data_dir, "images_new_1p2n.h5"),
        os.path.join(data_dir, "images_new_3p0n.h5"),
        os.path.join(data_dir, "images_new_3p1n.h5"),
        ]
    labels = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']
    n_classes = 5


print_sample_size(filenames, labels)

features = ['tracks', 's1', 's2', 's3', 's4', 's5']

test, val, y_test, y_val = load_test_data(
    filenames, features)

X_test  = [test[feat] for feat in features]

X_val   = [val[feat] for feat in features]
y_val_cat   = to_categorical(y_val, n_classes)


# ##############################################
model_filename = 'cache/multi_{0}_classes.h5'.format(n_classes)
if args.no_train:
    log.info('loading model')
    from keras.models import load_model
    model = load_model(model_filename)
else:
    log.info('training...')
    from tauperf.imaging.models import dense_merged_model_topo
    model = dense_merged_model_topo(test, n_classes=n_classes, final_activation='softmax')

    from tauperf.imaging.utils import fit_model_gen
    fit_model_gen(
        model,
        filenames, features,
        X_val, y_val_cat,
        n_chunks=args.training_chunks,
        use_multiprocessing=False,
        filename=model_filename,
        loss='categorical_crossentropy',
        overwrite=args.overwrite,
        no_train=args.no_train, 
        equal_size=args.equal_size, 
        debug=args.debug)




# ##############################################
log.info('testing stuff')

log.info('compute classifier scores')

y_pred = model.predict(X_test, batch_size=32, verbose=1)
print

log.info('drawing the confusion matrix')
cnf_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_categorical.pdf')

log.info('drawing the pantau confusion matrix')
cnf_mat = confusion_matrix(y_test, test['pantau'])
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Pantau confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_reference.pdf')

log.info('drawing the roc curves and pantau WP')
plot_roc(y_test, y_pred, test['pantau'])

