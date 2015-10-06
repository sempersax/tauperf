from tauperf.analysis import Analysis

import numpy as np
fields = [
    "off_cells_e",
    "off_cells_eta",
    "off_cells_phi",
    "off_cells_z"
]


def insert_zeros(a):
    TOTAL_SIZE = 3300
    if a.size > TOTAL_SIZE:
        raise RuntimeError
    else:
        b = a.copy()
        b.resize(TOTAL_SIZE)
        return b

def flatten(a):
    # print a.shape, type(a)
    return a


v_insert_zeros = np.vectorize(insert_zeros, otypes=[np.ndarray])
def prepare_dataset(rec):
    arrs = []
    for f in fields:
        arr = v_insert_zeros(rec[f])
        arr = np.dstack(arr[:])[0].T
        arrs.append(arr)
    final_arr = np.hstack(arrs)
    return final_arr

ana = Analysis()

from root_numpy import rec2array
tau_rec = ana.tau.records()
jet_rec = ana.jet.records()

tau_arr = prepare_dataset(tau_rec)
jet_arr = prepare_dataset(jet_rec)

train_data = np.vstack([tau_arr[:len(tau_arr) / 2], jet_arr[:len(jet_arr) /  2]])
target_data = np.hstack([np.ones(len(tau_arr) / 2), np.zeros(len(jet_arr) / 2)])
print target_data.shape, train_data.shape
from sklearn import svm, metrics


classifier = svm.SVC(gamma=0.001)
print 'start the fit'
classifier.fit(train_data, target_data)
print 'start the prediction'
predicted_tau = classifier.predict(tau_arr[len(tau_arr) / 2:])
predicted_jet = classifier.predict(jet_arr[len(jet_arr) / 2:])
