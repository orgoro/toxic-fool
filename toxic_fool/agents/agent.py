from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf

from models import ToxicityClassifier


class AgentConfig(object):

    def __init__(self):
        # should be called after child class init
        self.vars = vars(self)

    def __str__(self):
        print('________________________')
        for k, v in self.vars.items():
            print('|{:10} | {:10}|'.format(k, v))
        print('________________________')


class Agent(object):

    def __init__(self, sess, tox_model, config):
        # type: (tf.Session, ToxicityClassifier, AgentConfig) -> None
        self._sess = sess
        self._config = config
        self._tox_model = tox_model

    @abc.abstractmethod
    def _build_graph(self):
        pass

    @abc.abstractmethod
    def _train_step(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def restore(self, restore_path):
        pass

    @abc.abstractmethod
    def attack(self, seq, target_confidence):
        pass
