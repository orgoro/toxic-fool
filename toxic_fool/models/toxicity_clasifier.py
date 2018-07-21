from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras

from data import data_processor as process


class ToxicityClassifier(object):

    def __init__(self, session, max_seq):
        # type: (tf.Session, np.int) -> None
        self._session = session
        self._max_seq = max_seq

        self._model = self._build_graph()  # type: keras.Model

    def _build_graph(self):
        raise NotImplementedError('implemented by child')

    def train(self, dataset):
        # type: (process.Dataset) -> None
        raise NotImplementedError('implemented by child')

    def classify(self, seq):
        # type: (np.ndarray) -> np.ndarray
        raise NotImplementedError('implemented by child')

    def get_f1_score(self, seqs, lbls):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        raise NotImplementedError('implemented by child')

    def get_gradient(self, seq):
        # type: (np.ndarray) -> np.ndarray
        """ claculate gradients of d classes/ d input
        :param seq: text seq tokenized
        :return: returns array of gradients per class for each seq element
        """
        raise NotImplementedError('implemented by child')