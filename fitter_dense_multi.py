import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import roc_curve
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from tauperf import log; log = log['/fitter']
from tauperf.imaging.models import dense_merged_model_categorical
from tauperf.imaging.utils import fit_model

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--no-train', default=False, action='store_true')
parser.add_argument(
    '--overwrite', default=False, action='store_true')

args = parser.parse_args()


log.info('loading data...')
data_dir = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v6')
                        

images_1p0n  = np.load(os.path.join(data_dir, 'images_new_1p0n.npy'))
images_1p1n  = np.load(os.path.join(data_dir, 'images_new_1p1n.npy'))
images_1p2n  = np.load(os.path.join(data_dir, 'images_new_1p2n.npy'))


log.info('splitting')

train_1p0n, test_1p0n = model_selection.train_test_split(
    images_1p0n, test_size=0.3, random_state=42)
val_1p0n, test_1p0n = np.split(test_1p0n, [len(test_1p0n) / 2])

train_1p1n, test_1p1n = model_selection.train_test_split(
    images_1p1n, test_size=0.3, random_state=42)
val_1p1n, test_1p1n = np.split(test_1p1n, [len(test_1p1n) / 2])

train_1p2n, test_1p2n = model_selection.train_test_split(
    images_1p2n, test_size=0.3, random_state=42)
val_1p2n, test_1p2n = np.split(test_1p2n, [len(test_1p2n) / 2])

headers = ["sample", "Total", "Training", "Validation", "Testing"]
sample_size_table = [
    ['1p0n', len(images_1p0n), len(train_1p0n), len(val_1p0n), len(test_1p0n)],
    ['1p1n', len(images_1p1n), len(train_1p1n), len(val_1p1n), len(test_1p1n)],
    ['1p2n', len(images_1p2n), len(train_1p2n), len(val_1p2n), len(test_1p2n)],
]

log.info('')
print tabulate(sample_size_table, headers=headers, tablefmt='simple')
log.info('')

train = np.concatenate((train_1p0n, train_1p1n, train_1p2n))
test  = np.concatenate((test_1p0n, test_1p1n, test_1p2n))
val   = np.concatenate((val_1p0n, val_1p1n, val_1p2n))

y_train = np.concatenate((
        np.zeros(train_1p0n.shape, dtype=np.uint8),
        np.ones(train_1p1n.shape, dtype=np.uint8),
        np.ones(train_1p2n.shape, dtype=np.uint8) + 1))

y_test = np.concatenate((
        np.zeros(test_1p0n.shape, dtype=np.uint8),
        np.ones(test_1p1n.shape, dtype=np.uint8),
        np.ones(test_1p2n.shape, dtype=np.uint8) + 1))

y_val = np.concatenate((
        np.zeros(val_1p0n.shape, dtype=np.uint8),
        np.ones(val_1p1n.shape, dtype=np.uint8),
        np.ones(val_1p2n.shape, dtype=np.uint8) + 1))


y_train_cat = to_categorical(y_train, 3)
y_test_cat  = to_categorical(y_test, 3)
y_val_cat   = to_categorical(y_val, 3)


# ##############################################
log.info('training stuff')
model_filename = 'cache/crackpot_dense_multi.h5'
if args.no_train:
    model = load_model(model_filename)
else:
    model = dense_merged_model_categorical(train)
    fit_model(
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
        [test['s1'], test['s2'], test['s3']], 
        batch_size=32, verbose=1)


print
# ######################
log.info('Drawing the roc curve')
from tauperf.imaging.plotting import plot_confusion_matrix, get_eff, get_wp

fptr_1p0n, tpr_1p0n, thresh_1p0n = roc_curve(y_test, y_pred[:,0], pos_label=0)
fptr_1p1n, tpr_1p1n, thresh_1p1n = roc_curve(y_test, y_pred[:,1], pos_label=1)
fptr_1p2n, tpr_1p2n, thresh_1p2n = roc_curve(y_test, y_pred[:,2], pos_label=2)

print fptr_1p2n, tpr_1p2n
plt.figure()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fptr_1p0n, tpr_1p0n, color='red', label='1p0n vs 1pXn')
plt.plot(fptr_1p1n, tpr_1p1n, color='blue', label='1p1n vs others')
plt.plot(fptr_1p2n, tpr_1p2n, color='green', label='1p2n vs others')
# plt.plot(fptr_1p2n, tpr_1p2n, color='blue', label='1p1n vs 1p2n')
# plt.plot([opt_fptr_1p1n, opt_fptr_1p2n],
#           [opt_tpr_1p1n, opt_tpr_1p2n], 'go',
#          label='working points')
plt.xlabel('miss-classification rate')
plt.ylabel('classification efficiency')
plt.title('classification with calo sampling s1, s2 and s3')
plt.legend(loc='lower right', fontsize='small', numpoints=1)
plt.savefig('./plots/imaging/roc_curve_categorical.pdf')

