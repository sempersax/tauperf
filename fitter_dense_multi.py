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
labels = ['1p0n', '1p1n', '1p2n', '3p0n', '3p1n']

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
cnf_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
log.info('Diag / Total = {0} / {1}'.format(np.trace(cnf_mat), np.sum(cnf_mat)))
cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
cm = np.rot90(cm.T, 1)
np.set_printoptions(precision=2)
from tauperf.imaging.plotting import plot_confusion_matrix
class_names = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']
plt.figure()
plot_confusion_matrix(
    cm, classes=class_names, 
    title='Confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_categorical.pdf')

# h5file_1p0n.close() 
# h5file_1p1n.close() 
# h5file_1p2n.close() 
# h5file_3p0n.close() 
# h5file_3p1n.close() 

# ######################
# log.info('drawing the roc curve')
# from tauperf.imaging.plotting import plot_confusion_matrix, get_eff, get_wp

# fptr_1p0n, tpr_1p0n, thresh_1p0n = roc_curve(y_test, y_pred[:,0], pos_label=0)
# fptr_1p1n, tpr_1p1n, thresh_1p1n = roc_curve(y_test, y_pred[:,1], pos_label=1)
# fptr_1p2n, tpr_1p2n, thresh_1p2n = roc_curve(y_test, y_pred[:,2], pos_label=2)

# plt.figure()
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.plot(fptr_1p0n, tpr_1p0n, color='red', label='1p0n vs others')
# plt.plot(fptr_1p1n, tpr_1p1n, color='blue', label='1p1n vs others')
# plt.plot(fptr_1p2n, tpr_1p2n, color='green', label='1p2n vs others')
# plt.xlabel('miss-classification rate')
# plt.ylabel('classification efficiency')
# axes = plt.gca()
# axes.xaxis.set_ticks(np.arange(0, 1, 0.1))
# axes.yaxis.set_ticks(np.arange(0, 1, 0.1))
# axes.grid(True)
# plt.title('classification with calo sampling s1, s2 and s3')
# plt.legend(loc='lower right', fontsize='small', numpoints=1)
# plt.savefig('./plots/imaging/roc_curve_categorical.pdf')

# y_pred_1p0n = y_pred.take(np.where(y_test == 0)[0], axis=0)
# y_pred_1p1n = y_pred.take(np.where(y_test == 1)[0], axis=0)
# y_pred_1p2n = y_pred.take(np.where(y_test == 2)[0], axis=0)

# plt.figure()
# plt.hist(y_pred_1p0n[:,[0]], 20, normed=1. / 20, facecolor='red', label='1p0n')
# plt.hist(y_pred_1p1n[:,[0]], 20, normed=1. / 20, facecolor='blue', alpha=0.75, label='1p1n')
# plt.hist(y_pred_1p2n[:,[0]], 20, normed=1. / 20., facecolor='green', alpha=0.75, label='1p2n')
# plt.legend(loc='upper right', fontsize='small', numpoints=1)
# plt.xlabel('score 0')

# # plt.plot([-1, 1], [-1, 1], '--')
# # plt.scatter(y_pred[:,[0]][y_test == 0], y_pred[:,[1]][y_test == 0], c='red')
# # plt.scatter(y_pred[:,[0]][y_test == 1], y_pred[:,[1]][y_test == 1], c='blue')
# # plt.scatter(y_pred[:,[0]][y_test == 2], y_pred[:,[1]][y_test == 2], c='green')
# plt.savefig('./plots/imaging/scatter_0_1_categorical.pdf')

# plt.figure()
# plt.hist(y_pred_1p0n[:,[1]], 20, normed=1, facecolor='red', label='1p0n')
# plt.hist(y_pred_1p1n[:,[1]], 20, normed=1, facecolor='blue', alpha=0.75, label='1p1n')
# plt.hist(y_pred_1p2n[:,[1]], 20, normed=1, facecolor='green', alpha=0.75, label='1p2n')
# plt.legend(loc='upper right', fontsize='small', numpoints=1)
# plt.xlabel('score 1')
# # plt.plot([-1, 1], [-1, 1], '--')
# # plt.scatter(y_pred_1p0n[:,[0]], y_pred_1p0n[:,[2]], c='red')
# # plt.scatter(y_pred_1p1n[:,[0]], y_pred_1p1n[:,[2]], c='blue')
# # plt.scatter(y_pred_1p2n[:,[0]], y_pred_1p2n[:,[2]], c='green')
# plt.savefig('./plots/imaging/scatter_0_2_categorical.pdf')

# plt.figure()
# plt.hist(y_pred_1p0n[:,[2]], 20, normed=1, facecolor='red', label='1p0n')
# plt.hist(y_pred_1p1n[:,[2]], 20, normed=1, facecolor='blue', alpha=0.75, label='1p1n')
# plt.hist(y_pred_1p2n[:,[2]], 20, normed=1, facecolor='green', alpha=0.75, label='1p2n')
# plt.legend(loc='upper right', fontsize='small', numpoints=1)
# plt.xlabel('score 2')
# # plt.plot([-1, 1], [-1, 1], '--')
# # plt.scatter(y_pred_1p0n[:,[1]], y_pred_1p0n[:,[2]], c='red')
# # plt.scatter(y_pred_1p1n[:,[1]], y_pred_1p1n[:,[2]], c='blue')
# # plt.scatter(y_pred_1p2n[:,[1]], y_pred_1p2n[:,[2]], c='green')
# plt.savefig('./plots/imaging/scatter_1_2_categorical.pdf')


# from root_numpy import fill_hist
# from rootpy.plotting import Hist2D
# h_2d_1p0n = Hist2D(200, 0, 1, 200, 0, 1)
# fill_hist(h_2d_1p0n, y_pred_1p0n[:, [0, 1]])
# h_2d_1p0n = h_2d_1p0n / h_2d_1p0n.integral()
# contours = np.linspace(h_2d_1p0n.min(), h_2d_1p0n.max(), 4,
#                        endpoint=False)[1:]
# h_2d_1p0n.SetContour(len(contours), np.asarray(contours, dtype=float))
# h_2d_1p1n = Hist2D(20, 0, 1, 20, 0, 1)
# fill_hist(h_2d_1p1n, y_pred_1p1n[:, [0, 1]])
# h_2d_1p1n = h_2d_1p1n / h_2d_1p1n.integral()
# from rootpy.plotting import Canvas
# c = Canvas()
# h_2d_1p0n.Draw('CONT LIST')
# # h_2d_1p1n.Draw('SAMECONT0')
# c.SaveAs('./plots/imaging/contour.pdf')
