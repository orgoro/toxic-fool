from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import roc_auc_score
import numpy as np

def calc_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(y_true)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def calc_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def calc_f1(y_true, y_pred):
    precision = calc_precision(y_true, y_pred)
    recall = calc_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class RocCallback(Callback):
    def __init__(self, dataset):
        # type: (data.Dataset) -> None
        self.x = dataset.train_seq
        self.y = dataset.train_lbl
        self.x_val = dataset.val_seq
        self.y_val = dataset.val_lbl
        super(RocCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        idx = np.random.randint(self.x.shape[0], size=10000)
        x = self.x[idx]
        y = self.y[idx]
        y_pred = self.model.predict(x)
        roc = roc_auc_score(y, y_pred[:])
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('roc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return
