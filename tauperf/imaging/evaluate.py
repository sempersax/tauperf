import numpy as np
from sklearn.metrics import confusion_matrix

from . import log; log = log[__name__]

def matrix_decays(y_true, y_pred_pi0, y_pred_twopi0, y_pred_3p_pi0, y_is_1p):

    def classify(pred_pi0, pred_twopi0, pred_3p_pi0, is_1p):
        # 1p0n = 0, 1p1n = 1, 1p2n = 2, 3p0n = 3, 3p1n = 4
        if is_1p:
            if pred_pi0:
                if pred_twopi0:
                    return 2
                else:
                    return 1
            else:
                return 0
        else:
            if pred_3p_pi0:
                return 4
            else:
                return 3

    v_classify = np.vectorize(classify)

    y_pred = v_classify(y_pred_pi0, y_pred_twopi0, y_pred_3p_pi0, y_is_1p)
    cnf_mat = confusion_matrix(y_true, y_pred)
    diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
    log.info('Diag / Total = {0} / {1}'.format(np.trace(cnf_mat), np.sum(cnf_mat)))
    cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)
    np.set_printoptions(precision=2)
    return cm, diagonal


def matrix_decays_1p(y_true, y_pred_pi0, y_pred_twopi0):

    def classify(pred_pi0, pred_twopi0):
        # 1p0n = 0, 1p1n = 1, 1p2n = 2, 3p0n = 3, 3p1n = 4
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
    diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
    log.info('Diag / Total = {0} / {1}'.format(np.trace(cnf_mat), np.sum(cnf_mat)))
    cm = cnf_mat.T.astype('float') / cnf_mat.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)
    np.set_printoptions(precision=2)
    return cm, diagonal
