import os
import numpy as np
import sys
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, confusion_matrix
from keras.utils.np_utils import to_categorical
from tauperf.imaging.load import load_test_data, print_sample_size


import logging
log = logging.getLogger(os.path.basename(__file__))
log.setLevel(logging.INFO)

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
parser.add_argument(
    '--dev', default=False, action='store_true')
parser.add_argument(
    '--epochs', default=100, type=int)

args = parser.parse_args()


data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'v13/test')
print data_dir
sys.exit()
                        
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
if args.debug:
    test = test[:1000]
    y_test = y_test[:1000]

X_test  = [test[feat] for feat in features]

X_val   = [val[feat] for feat in features]
y_val_cat   = to_categorical(y_val, n_classes)


# ##############################################
if args.dev:
    model_filename = 'cache/multi_{0}_classes_{1}_chunks'.format(
        n_classes, args.training_chunks)
    model_filename += '_{epoch:02d}_epochs.h5'
else:
    model_filename = 'cache/multi_{0}_classes_original.h5'.format(n_classes)

if args.no_train:
    log.info('loading model')
    from keras.models import load_model
    model = load_model(model_filename)
else:
    log.info('training...')
    from tauperf.imaging.models import dense_merged_model_topo,RGB_like_merged_model
#    model = dense_merged_model_topo(test, n_classes=n_classes, final_activation='softmax')
    model = RGB_like_merged_model(test, n_classes=n_classes, final_activation='softmax')

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
        dev=args.dev,
        debug=args.debug,
        epochs=args.epochs)




# ##############################################
log.info('testing stuff')

log.info('compute classifier scores')

y_pred = model.predict(X_test, batch_size=32, verbose=1)
print

log.info('drawing the confusion matrix')
cnf_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))

# import plotting here in case there is an import issue unrelated to training
from tauperf.imaging.plotting import plot_confusion_matrix, plot_roc, compare_bins, plot_event, score_plots,score_outliers
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

print
log.info('drawing the score histograms')
score_plots(y_pred, y_test, '1p0n')
score_plots(y_pred, y_test, '1p1n')
score_outliers(test, y_pred, y_test, '1p0n')
score_outliers(test, y_pred, y_test, '1p1n')
sys.exit()

print
log.info('drawing the comparison histogram')
best_true, best_false, worst_true, worst_false = compare_bins(test, y_pred, y_test, '1p0n')

log.info('drawing the heatmaps of selected events')
plot_event(test, best_true, y_pred, '1p0n', 'best_true_positive')
plot_event(test, best_false, y_pred, '1p0n', 'best_false_positive')
plot_event(test, worst_true, y_pred, '1p0n', 'worst_true_positive')
plot_event(test, worst_false, y_pred, '1p0n', 'worst_false_positive')

print
log.info('drawing the comparison histogram')
best_true, best_false, worst_true, worst_false = compare_bins(test, y_pred, y_test, '1p1n')

print
log.info('drawing the heatmaps of selected events')
plot_event(test, best_true, y_pred, '1p1n', 'best_true_positive')
plot_event(test, best_false, y_pred, '1p1n', 'best_false_positive')
plot_event(test, worst_true, y_pred, '1p1n', 'worst_true_positive')
plot_event(test, worst_false, y_pred, '1p1n', 'worst_false_positive')


print 'Conv2D s1 = 64 (6,2), s2 = 64 (3,3)'
newpath = r'./plots/imaging/' + 's1_64_6_2.s2_64_3_3' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
