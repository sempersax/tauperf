import numpy as np
import h5py
import tauperf
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree, ensemble
from sklearn.cross_validation import cross_val_score


def transform(score, n_estimators=100, learning_rate=0.1):
    res = -1 + 2.0 / (1.0 +
                      np.exp(
            -n_estimators *
             learning_rate * score / 1.5))
    return res

def labels(s, b):
    ones = np.ones(len(s), dtype=np.int)
    zeros = np.zeros(len(b), dtype=np.int)
    return np.concatenate((ones, zeros))




f = h5py.File('/cluster/warehouse/qbuat/crackpotauid_ntuples/v3/tables.h5', 'r')
tau_rec = f.get('tau')
jet_rec = f.get('jet')

fields = 'off_cells_e_normalized'
# fields = ('off_centFracCorrected', 'off_nwidetracks', 'off_dRmax')


if isinstance(fields, (tuple, list)):
    s_trains = []
    s_tests = []
    b_trains = []
    b_tests = []
    for field in fields:
        s_trains.append(tau_rec[field][:len(tau_rec) / 2])
        s_tests.append(tau_rec[field][len(tau_rec) / 2:])
        b_trains.append(jet_rec[field][:len(tau_rec) / 2])
        b_tests.append(jet_rec[field][len(tau_rec) / 2:])
        
    S_train = np.dstack(s_trains)[0]
    S_test  = np.dstack(s_tests)[0]
    
    B_train = np.dstack(b_trains)[0]
    B_test  = np.dstack(b_tests)[0]
else:
    S_tot = tau_rec[fields]
    B_tot = tau_rec[fields]

    S_train = tau_rec[fields][:len(tau_rec) / 2]
    S_test  = tau_rec[fields][len(tau_rec) / 2:]
    B_train = jet_rec[fields][:len(tau_rec) / 2]
    B_test  = jet_rec[fields][len(tau_rec) / 2:]


X_train = np.concatenate((S_train, B_train))
y_train = labels(S_train, B_train)



# y_train = y_train.astype(int)
classifier = ensemble.AdaBoostClassifier(
    tree.DecisionTreeClassifier(),
    learning_rate=0.1,
    n_estimators=100)

# classifier = ensemble.ExtraTreesClassifier(
#     n_estimators=100,
#     n_jobs=-1,
#     verbose=1)

scores = cross_val_score(classifier, X_train, y_train)

# classifier = Classifier(
#     layers=[
#         Layer("Maxout", units=100, pieces=2),
#         Layer("Softmax")],
#     learning_rate=0.001,
#     n_iter=25)

# # from sklearn import datasets, svm, metrics
# # # Create a classifier: a support vector classifier
# # classifier = svm.SVC(gamma=0.001, verbose=True)


# Fit on training sample
# classifier.fit(X_train, y_train)

# s = classifier.predict(S_test)
# b = classifier.predict(B_test)

# # score = nn.score(X_test, y_test)
