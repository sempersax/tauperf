import numpy as np
from sklearn.metrics import confusion_matrix



def matrix_1p(y_true, y_pred_pi0, y_pred_twopi0):

    def classify(pred_pi0, pred_twopi0):
        if pred_pi0:
            if pred_twopi0:
                return 2
            else:
                return 1
        else:
            return 0
    v_classify = np.vectorize(classify)

    y_pred = v_classify(y_pred_pi0, y_pred_twopi0)
    cnf_mat = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=1)
    cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)
    return cm
